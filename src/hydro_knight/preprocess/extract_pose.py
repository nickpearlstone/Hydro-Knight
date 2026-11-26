"""Pose extraction (GPU step): video -> raw tiled YOLO-pose detections on disk, resumable in chunks."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .tiled_pose import detect_crops, tile_grid

KEYPOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]

KP_COLUMNS = [f"{axis}{i}" for i in range(17) for axis in ("x", "y", "c")]
# CAUTION[extract-pose-box-columns]: keep bx/by names; x1..y2 clash with keypoint columns.
BOX_COLUMNS = ["bx1", "by1", "bx2", "by2"]
DET_COLUMNS = ["frame", "crop"] + BOX_COLUMNS + ["conf"] + KP_COLUMNS
FRAME_COLUMNS = ["frame", "n_dets", "scene_diff"]


class VideoReadError(RuntimeError):
    """The video opened but yielded far fewer frames than it reports (often an unsupported codec)."""


_SETTING_KEYS = (
    "model",
    "tile",
    "overlap",
    "imgsz",
    "conf",
    "include_full",
    "max_frames",
    "frame_step",
    "stop_frame",
)
_SETTING_DEFAULTS = {"frame_step": 1, "stop_frame": None}


def _code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "code": _code_version(),
    }


def _fourcc(cap) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")


def _scene_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def write_chunk(
    out_dir: Path, index: int, dets: pd.DataFrame, frames: pd.DataFrame
) -> None:
    """Write one chunk's detections and frame rows (keypoints float16, boxes and conf float32)."""
    dets = dets[DET_COLUMNS].astype(
        {"frame": "int32", "crop": "int8"}
        | {c: "float32" for c in BOX_COLUMNS + ["conf"]}
        | {c: "float16" for c in KP_COLUMNS}
    )
    frames = frames[FRAME_COLUMNS].astype(
        {"frame": "int32", "n_dets": "int32", "scene_diff": "float32"}
    )
    dets.to_parquet(
        out_dir / f"det-{index:05d}.parquet", compression="zstd", index=False
    )
    frames.to_parquet(
        out_dir / f"frames-{index:05d}.parquet", compression="zstd", index=False
    )


def load_detections(det_dir: Path, require_complete: bool = True):
    """Read a clip folder written by `extract_raw`; returns (meta, dets, frames). Raises if incomplete unless `require_complete` is False."""
    det_dir = Path(det_dir)
    meta_path = det_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json in {det_dir}")
    meta = json.loads(meta_path.read_text())
    if require_complete and not meta.get("complete"):
        raise RuntimeError(f"extraction not complete for {det_dir}")
    n = meta["chunks_done"]
    dets = [pd.read_parquet(det_dir / f"det-{i:05d}.parquet") for i in range(n)]
    frames = [pd.read_parquet(det_dir / f"frames-{i:05d}.parquet") for i in range(n)]
    dets = (
        pd.concat(dets, ignore_index=True)
        if dets
        else pd.DataFrame(columns=DET_COLUMNS)
    )
    frames = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=FRAME_COLUMNS)
    )
    return meta, dets, frames.sort_values("frame", ignore_index=True)


def extract_raw(
    video_path: Path,
    out_dir: Path,
    model_name: str = "yolo11n-pose.pt",
    tile: int = 480,
    overlap: float = 0.25,
    imgsz: int = 1280,
    conf: float = 0.25,
    include_full: bool = True,
    device=None,
    chunk_frames: int = 1000,
    max_frames: int | None = None,
    frame_step: int = 1,
    stop_frame: int | None = None,
) -> dict:
    """Extract raw tiled detections for one clip into `out_dir`, resuming if partly done; `tile` in pixels, `conf` is a floor below which nothing is saved."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    settings = {
        "model": model_name,
        "tile": tile,
        "overlap": overlap,
        "imgsz": imgsz,
        "conf": conf,
        "include_full": include_full,
        "max_frames": max_frames,
        "frame_step": frame_step,
        "stop_frame": stop_frame,
    }
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is not None:
        old = {
            k: meta["settings"].get(k, _SETTING_DEFAULTS.get(k)) for k in _SETTING_KEYS
        }
        if old != settings:
            raise ValueError(
                f"{out_dir} was extracted with different settings {old}; "
                "delete the folder to re-extract"
            )
        if meta.get("complete"):
            return meta

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    origins = tile_grid(width, height, tile, overlap)
    codec = _fourcc(cap)

    if meta is not None:
        if meta["frames_processed"] and (width, height) != (
            meta["video"]["width"],
            meta["video"]["height"],
        ):
            raise ValueError(
                f"{video_path.name} is now {width}x{height} but frames were extracted at "
                f"{meta['video']['width']}x{meta['video']['height']}; delete {out_dir}"
            )
        meta["video"].update(path=str(video_path), codec=codec)
    else:
        meta = {
            "clip_id": video_path.stem,
            "video": {
                "path": str(video_path),
                "width": width,
                "height": height,
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "codec": codec,
            },
            "settings": settings,
            "tile_origins": [list(o) for o in origins],
            "chunk_frames": chunk_frames,
            "versions": _versions(),
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunks_done": 0,
            "frames_processed": 0,
            "complete": False,
        }
        _write_json_atomic(meta_path, meta)

    chunk_frames = meta["chunk_frames"]
    # CAUTION[extract-pose-resume-progress]: frames_processed must be written atomically with chunks_done.
    start = meta["frames_processed"]

    prev_hist = None
    for i in range(start):
        if not cap.grab():
            break
        if i == start - 1:
            ok, prev = cap.retrieve()
            prev_hist = _scene_hist(prev) if ok else None

    model = YOLO(model_name)
    total = meta["video"]["frame_count"] or None
    if max_frames is not None:
        total = max_frames
    if stop_frame is not None:
        total = min(total or stop_frame, stop_frame)
    bar = tqdm(total=total, initial=start, desc=f"extract {video_path.stem}")

    det_parts, frame_rows = [], []
    frame_idx = start
    while max_frames is None or frame_idx < max_frames:
        if stop_frame is not None and frame_idx >= stop_frame:
            break
        if frame_idx % frame_step:
            if not cap.grab():
                break
            frame_idx += 1
            bar.update(1)
            continue
        ok, frame = cap.read()
        if not ok:
            break
        boxes, confs, kpts, crops = detect_crops(
            model, frame, origins, tile, imgsz, conf, device, include_full
        )
        hist = _scene_hist(frame)
        diff = (
            np.nan
            if prev_hist is None
            else cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        )
        prev_hist = hist
        frame_rows.append((frame_idx, len(confs), diff))
        if len(confs):
            block = np.column_stack(
                [
                    np.full(len(confs), frame_idx),
                    crops,
                    boxes,
                    confs,
                    kpts.reshape(len(confs), -1),
                ]
            )
            det_parts.append(block)
        frame_idx += 1
        bar.update(1)

        if len(frame_rows) >= chunk_frames:
            _flush(out_dir, meta, det_parts, frame_rows, frame_idx)
            det_parts, frame_rows = [], []

    if frame_rows:
        _flush(out_dir, meta, det_parts, frame_rows, frame_idx)
    bar.close()
    cap.release()

    # frame-count tolerance is 1%
    expected = meta["video"]["frame_count"]
    if stop_frame is not None and expected:
        expected = min(expected, stop_frame)
    short = max_frames is None and expected and frame_idx < 0.99 * expected
    if frame_idx == 0 or short:
        meta["error"] = (
            f"read {frame_idx} of {expected} frames; "
            f"the {codec!r} video may not be decodable in this environment"
        )
        _write_json_atomic(meta_path, meta)
        raise VideoReadError(f"{video_path.name}: {meta['error']}")
    meta.pop("error", None)
    meta["complete"] = True
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    if (
        max_frames is None
        and stop_frame is None
        and expected
        and frame_idx > 1.01 * expected
    ):
        meta["warning"] = f"read {frame_idx} frames but video reports {expected}"
        print(f"WARNING {video_path.name}: {meta['warning']}")
    _write_json_atomic(meta_path, meta)
    return meta


def _flush(out_dir, meta, det_parts, frame_rows, frame_idx) -> None:
    dets = pd.DataFrame(
        np.concatenate(det_parts) if det_parts else np.empty((0, len(DET_COLUMNS))),
        columns=DET_COLUMNS,
    )
    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    write_chunk(out_dir, meta["chunks_done"], dets, frames)
    meta["chunks_done"] += 1
    meta["frames_processed"] = frame_idx
    _write_json_atomic(out_dir / "meta.json", meta)


if __name__ == "__main__":
    import sys

    video = Path(sys.argv[1])
    extract_raw(video, Path("data/detections") / video.stem)
