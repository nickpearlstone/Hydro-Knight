"""
Pose extraction (the GPU step): video -> raw tiled YOLO-pose detections on disk.

YOLO is the only expensive stage of the pipeline, so this step saves its output
*before* any decision that might change: every detection from every tile and
the whole-frame pass, un-merged and un-tracked. Merging duplicates and assigning
track ids happen later, on CPU, in `build_tracks.py` — so a better merge rule or
tracker never requires another pass over the video.

Output, one folder per clip:

    <out_dir>/
      det-00000.parquet     raw detections: DET_COLUMNS, one row per detection per crop
      frames-00000.parquet  one row per processed frame: FRAME_COLUMNS
      ...                   (one numbered pair per chunk of `chunk_frames` frames)
      meta.json             video facts, settings, tile origins, versions, progress

`frames` has a row for every frame read, including frames with no detections.
That is what separates "a swimmer vanished" from "that frame was never
processed", and it is how a finished clip is verified against the video's
frame count — an empty keypoint file used to be indistinguishable from a clip
with no swimmers.

Chunks make extraction resumable: `meta.json` records completed chunks, and a
rerun continues after the last one (a crash loses at most one chunk).
"""

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

# 17 COCO keypoints, in the order YOLO returns them.
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
# Box corners are bx/by: plain x1,y1,x2,y2 would collide with keypoint 1 and 2 columns.
BOX_COLUMNS = ["bx1", "by1", "bx2", "by2"]
DET_COLUMNS = ["frame", "crop"] + BOX_COLUMNS + ["conf"] + KP_COLUMNS
FRAME_COLUMNS = ["frame", "n_dets", "scene_diff"]

# Settings that must match for a resumed or skipped folder to be trusted.
_SETTING_KEYS = (
    "model",
    "tile",
    "overlap",
    "imgsz",
    "conf",
    "include_full",
    "max_frames",
)


def _code_version() -> str | None:
    """Current git commit of the repo, or None outside a git checkout."""
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


def _scene_hist(frame: np.ndarray) -> np.ndarray:
    """Normalized hue/saturation histogram of a downscaled frame (cheap scene fingerprint)."""
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
    """Write one chunk's detections and frame rows with the on-disk dtypes.

    Keypoints are stored as float16 (under 0.13 px error below 2048 px); boxes and conf stay float32.
    Args:
        out_dir: clip folder.
        index: chunk number (file suffix).
        dets: rows with DET_COLUMNS.
        frames: rows with FRAME_COLUMNS.
    """
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
    """Read a clip folder written by `extract_raw`.

    Args:
        det_dir: clip folder containing meta.json and the chunk files.
        require_complete: raise if extraction did not finish.
    Returns:
        (meta, dets, frames): meta dict; dets DataFrame (DET_COLUMNS); frames DataFrame
        (FRAME_COLUMNS), one row per processed frame, sorted by frame.
    Raises:
        FileNotFoundError: no meta.json in det_dir.
        RuntimeError: require_complete and the folder is unfinished.
    """
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
) -> dict:
    """Extract raw tiled detections for one clip into `out_dir`, resuming if partly done.

    A folder already marked complete with the same settings is skipped without loading the model.
    Args:
        video_path: input clip.
        out_dir: clip output folder (created).
        model_name: YOLO-pose weights.
        tile: tile size in pixels.
        overlap: fractional tile overlap.
        imgsz: per-crop inference resolution (must exceed `tile` for tiling to help).
        conf: detection confidence floor; detections below it are never saved.
        include_full: also run a whole-frame pass per frame.
        device: torch device for YOLO (e.g. 0 for the first GPU), or None.
        chunk_frames: frames per chunk file (resume granularity).
        max_frames: stop after this many frames (debugging only; real runs extract full clips).
    Returns:
        The final meta dict (also written to out_dir/meta.json).
    Raises:
        ValueError: the folder holds a previous run made with different settings.
        OSError: the video cannot be opened.
    """
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
    }

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is not None:
        old = {k: meta["settings"].get(k) for k in _SETTING_KEYS}
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

    if meta is None:
        meta = {
            "clip_id": video_path.stem,
            "video": {
                "path": str(video_path),
                "width": width,
                "height": height,
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
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
    start = meta["frames_processed"]  # updated in the same write as chunks_done

    # Resume: step past finished frames without inference; keep the last one for scene_diff.
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
    bar = tqdm(total=total, initial=start, desc=f"extract {video_path.stem}")

    det_parts, frame_rows = [], []
    frame_idx = start
    while max_frames is None or frame_idx < max_frames:
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

        if (frame_idx - start) % chunk_frames == 0:
            _flush(out_dir, meta, det_parts, frame_rows, frame_idx)
            start, det_parts, frame_rows = frame_idx, [], []

    if frame_rows:
        _flush(out_dir, meta, det_parts, frame_rows, frame_idx)
    bar.close()
    cap.release()

    expected = meta["video"]["frame_count"]
    meta["complete"] = True
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    if max_frames is None and expected and abs(frame_idx - expected) > 0.01 * expected:
        # Container frame counts are estimates, so only a large gap is suspicious.
        meta["warning"] = f"read {frame_idx} frames but video reports {expected}"
        print(f"WARNING {video_path.name}: {meta['warning']}")
    _write_json_atomic(meta_path, meta)
    return meta


def _flush(out_dir, meta, det_parts, frame_rows, frame_idx) -> None:
    """Write the pending chunk, then advance meta's progress counters on disk."""
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
