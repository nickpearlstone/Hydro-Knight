"""
Pose-backend benchmark: missed swimmers, duplicate detections, and speed, scored
against hand-clicked swimmer positions.

Steps, in order:

    uv run python scripts/benchmark_pose.py frames   # pick reference frames from local clips
    uv run python -m hydro_knight.annotate            # Swimmer count tab: click every person
    uv run python scripts/benchmark_pose.py run      # score every backend, write results
    uv run python scripts/benchmark_pose.py sweep    # recall vs false positives across cutoffs

Ground truth is one click per person, in the water or out of it (deck, stairs, guard stand);
the file format lives in hydro_knight/annotate/count.py. A detection is matched to a click that lies inside
its box (padded 10%), one detection per click, highest confidence first:
  - found:     a water click that some detection claimed
  - duplicate: a detection whose box holds only clicks already claimed by another detection
  - false pos: a detection whose box holds no click at all
  - deck hits: detections on out-of-water people, reported but not scored either way
Missed = water clicks no detection claimed. Speed is median wall time per frame on this machine.

MediaPipe needs `uv sync --extra spike`. On an 8 GB M1 Mac, crops go to YOLO 4 at a time
(--batch): all 16 at once wedged the GPU for the s/m models.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydro_knight.annotate import count  # noqa: E402
from hydro_knight.preprocess.tiled_pose import (  # noqa: E402
    FULL_FRAME,
    merge_detections,
    tile_grid,
)

MANIFEST = Path("data/manifests/pool_footage.jsonl")
GT_PATH = count.COUNT_PATH
OUT_DIR = Path("runs/pose_bench")
MP_MODEL = "raw_local/pose_landmarker.task"

# Production extraction settings (extract_pose.extract_raw defaults).
TILE, OVERLAP, IMGSZ, CONF = 480, 0.25, 1280, 0.25
BOX_PAD = 0.10

YOLO_MODELS = [
    "yolo11n-pose",
    "yolo11s-pose",
    "yolo11m-pose",
    "yolo26n-pose",
    "yolo26s-pose",
    "yolo26m-pose",
]


# ---------------------------------------------------------------- frames


def cmd_frames(args) -> None:
    """Pick one frame per clip at drowning onset (distress) or mid-clip (others), seeded."""
    if GT_PATH.exists() and not args.force:
        raise SystemExit(f"{GT_PATH} exists (has your clicks); pass --force to re-pick")
    rows = [json.loads(line) for line in MANIFEST.open()]
    rows = [
        r
        for r in rows
        if "[HOLD" not in r["notes"] and Path(f"raw_local/{r['clip_id']}.mp4").exists()
    ]
    rng = random.Random(args.seed)
    distress = rng.sample([r for r in rows if r["label"] == "distress"], args.n - 2)
    other = rng.sample([r for r in rows if r["label"] != "distress"], 2)
    frames = []
    for r in distress + other:
        if r["events"]:
            idx = int(round(r["events"][0]["start"] * r["fps"]))
        else:  # start_sec can be a livestream offset, so take the local file's midpoint
            cap = cv2.VideoCapture(f"raw_local/{r['clip_id']}.mp4")
            idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            cap.release()
        item = {"clip_id": r["clip_id"], "frame": idx, "label": r["label"]}
        count.read_frame(item)
        frames.append(item | {"water": [], "deck": []})
    count.save({"frames": frames}, GT_PATH)
    print(f"picked {len(frames)} frames -> {GT_PATH}, images in {count.frame_dir()}/")


# ---------------------------------------------------------------- backends


class YoloBackend:
    """YOLO-pose, fed `batch` images per call: all 16 crops at once hangs the 8 GB M1 for s/m.

    A "+nms" suffix (e.g. "yolo26n-pose+nms") runs YOLO26 through its one-to-many head with
    classic NMS instead of the default NMS-free end-to-end head.
    """

    def __init__(self, name: str, device: str, batch: int, conf: float = CONF):
        from ultralytics import YOLO

        self.name, self.device, self.batch, self.conf = name, device, batch, conf
        self.extra = {"end2end": False} if name.endswith("+nms") else {}
        self.model = YOLO(f"{name.removesuffix('+nms')}.pt")

    def __call__(self, images: list[np.ndarray]):
        """Per image: (boxes (n,4) xyxy in image pixels, confs (n,))."""
        results = []
        for k in range(0, len(images), self.batch):
            chunk = images[k : k + self.batch]
            results += self.model(
                chunk,
                imgsz=IMGSZ,
                conf=self.conf,
                verbose=False,
                device=self.device,
                **self.extra,
            )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.empty((0, 4), np.float32), np.empty(0, np.float32)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        return out


class MediaPipeBackend:
    """MediaPipe PoseLandmarker; box = landmark extent, confidence = mean landmark presence."""

    name = "mediapipe"

    def __init__(self, num_poses: int = 50):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=CONF,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, images: list[np.ndarray]):
        out = []
        for img in images:
            h, w = img.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = self.lm.detect(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            )
            boxes, confs = [], []
            for pose in res.pose_landmarks:
                xs = np.clip([p.x for p in pose], 0, 1) * w
                ys = np.clip([p.y for p in pose], 0, 1) * h
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                confs.append(np.mean([p.presence or 0.0 for p in pose]))
            out.append(
                (
                    np.asarray(boxes, np.float32).reshape(-1, 4),
                    np.asarray(confs, np.float32),
                )
            )
        return out


def detect(backend, frame: np.ndarray, tiled: bool):
    """Whole-frame pass, or production tiling (tiles + whole frame, IoS merge). Returns boxes, confs."""
    if not tiled:
        return backend([frame])[0]
    h, w = frame.shape[:2]
    origins = tile_grid(w, h, TILE, OVERLAP)
    crops_img = [frame[oy : oy + TILE, ox : ox + TILE] for ox, oy in origins] + [frame]
    crop_ids = list(range(len(origins))) + [FULL_FRAME]
    boxes, confs, crops = [], [], []
    for (b, c), (ox, oy), cid in zip(
        backend(crops_img), origins + [(0, 0)], crop_ids, strict=True
    ):
        b = b.copy()
        b[:, [0, 2]] += ox
        b[:, [1, 3]] += oy
        boxes.append(b)
        confs.append(c)
        crops.append(np.full(len(c), cid, np.int8))
    boxes, confs, crops = (
        np.concatenate(boxes),
        np.concatenate(confs),
        np.concatenate(crops),
    )
    keep = merge_detections(boxes, confs, crops)
    return boxes[keep], confs[keep]


# ---------------------------------------------------------------- scoring


def _inside(boxes: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(n_boxes, n_pts) bool: point inside box padded by BOX_PAD of its size."""
    if len(pts) == 0 or len(boxes) == 0:
        return np.zeros((len(boxes), len(pts)), bool)
    pw = (boxes[:, 2] - boxes[:, 0]) * BOX_PAD
    ph = (boxes[:, 3] - boxes[:, 1]) * BOX_PAD
    x, y = pts[None, :, 0], pts[None, :, 1]
    return (
        (x >= (boxes[:, 0] - pw)[:, None])
        & (x <= (boxes[:, 2] + pw)[:, None])
        & (y >= (boxes[:, 1] - ph)[:, None])
        & (y <= (boxes[:, 3] + ph)[:, None])
    )


def score_frame(boxes, confs, water, deck) -> dict:
    """Greedy one-detection-per-click match, highest confidence first, nearest click to box centre."""
    pts = np.asarray(water + deck, np.float32).reshape(-1, 2)
    is_water = np.arange(len(pts)) < len(water)
    inside = _inside(boxes, pts)
    taken = np.zeros(len(pts), bool)
    found = dup = fp = deck_hit = 0
    for i in np.argsort(-confs, kind="stable"):
        cand = np.flatnonzero(inside[i])
        if len(cand) == 0:
            fp += 1
            continue
        free = cand[~taken[cand]]
        if len(free) == 0:
            dup += 1
            continue
        cx, cy = (boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2
        j = free[np.argmin(np.hypot(pts[free, 0] - cx, pts[free, 1] - cy))]
        taken[j] = True
        if is_water[j]:
            found += 1
        else:
            deck_hit += 1
    return {
        "water": len(water),
        "found": found,
        "missed": len(water) - found,
        "duplicate": dup,
        "false_pos": fp,
        "deck_hit": deck_hit,
        "detections": len(boxes),
    }


def cmd_run(args) -> None:
    gt = json.loads(GT_PATH.read_text())
    frames = [f for f in gt["frames"] if f["water"] or f["deck"]]
    if not frames:
        raise SystemExit(
            "no clicked frames yet: label them in the Swimmer count tab of "
            "`uv run python -m hydro_knight.annotate`"
        )
    images = [count.read_frame(f) for f in frames]
    print(
        f"{len(frames)} labeled frames, {sum(len(f['water']) for f in frames)} water clicks"
    )

    backends = [
        lambda m=m: YoloBackend(m, args.device, args.batch) for m in args.models
    ]
    if not args.no_mediapipe:
        backends.append(MediaPipeBackend)

    rows, per_frame = [], []
    for make in backends:
        backend = make()
        for tiled in (False, True):
            mode = "tiled" if tiled else "whole"
            for img in images[:2]:  # warm-up (model load, MPS kernel compile)
                detect(backend, img, tiled)
            totals, times = {}, []
            for f, img in zip(frames, images, strict=True):
                t0 = time.perf_counter()
                boxes, confs = detect(backend, img, tiled)
                times.append(time.perf_counter() - t0)
                s = score_frame(boxes, confs, f["water"], f["deck"])
                per_frame.append(
                    {
                        "model": backend.name,
                        "mode": mode,
                        "clip_id": f["clip_id"],
                        "frame": f["frame"],
                    }
                    | s
                )
                for k, v in s.items():
                    totals[k] = totals.get(k, 0) + v
            row = {
                "model": backend.name,
                "mode": mode,
                "recall": totals["found"] / max(1, totals["water"]),
                **totals,
                "ms_per_frame": 1000 * statistics.median(times),
            }
            rows.append(row)
            print(
                f"{backend.name:14s} {mode:5s} recall {row['recall']:.2f}  "
                f"missed {row['missed']:4d}  dup {row['duplicate']:3d}  fp {row['false_pos']:3d}  "
                f"{row['ms_per_frame']:7.0f} ms/frame"
            )
        del backend
        if args.device == "mps":
            import torch

            torch.mps.empty_cache()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_frame).to_csv(out / "per_frame.csv", index=False)
    n_water = rows[0]["water"]
    lines = [
        f"{len(frames)} hand-labeled frames, {n_water} swimmers in the water, "
        f"device `{args.device}`, batch {args.batch}.",
        "",
        "| Model | Mode | Found | Missed | Duplicates | False pos | ms/frame |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['found']}/{n_water} ({r['recall']:.0%}) | "
            f"{r['missed']} | {r['duplicate']} | {r['false_pos']} | {r['ms_per_frame']:.0f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}/summary.md, summary.csv, per_frame.csv")


def cmd_sweep(args) -> None:
    """Fair threshold comparison: detect once at a low floor, then score every cutoff offline.

    Thresholding after the merge gives the same boxes as detecting at that cutoff, because the
    greedy merge only lets a box be removed by a higher-confidence one.
    """
    import pandas as pd

    frames = [
        f for f in json.loads(GT_PATH.read_text())["frames"] if f["water"] or f["deck"]
    ]
    images = [count.read_frame(f) for f in frames]
    thresholds = np.round(np.arange(args.floor, 0.501, 0.025), 3)
    rows = []
    for name in args.models:
        backend = YoloBackend(name, args.device, args.batch, conf=args.floor)
        dets = [detect(backend, img, tiled=True) for img in images]
        for t in thresholds:
            tot = {}
            for f, (b, c) in zip(frames, dets, strict=True):
                keep = c >= t
                for k, v in score_frame(
                    b[keep], c[keep], f["water"], f["deck"]
                ).items():
                    tot[k] = tot.get(k, 0) + v
            rows.append(
                {"model": name, "threshold": t, "recall": tot["found"] / tot["water"]}
                | tot
            )
        del backend
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.csv", index=False)
    print(
        df[
            ["model", "threshold", "found", "recall", "duplicate", "false_pos"]
        ].to_string(index=False)
    )
    print(f"\nwrote {out}/sweep.csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames", help="pick reference frames")
    f.add_argument("--n", type=int, default=10)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--force", action="store_true", help="overwrite existing clicks")
    r = sub.add_parser("run", help="score every backend")
    r.add_argument("--device", default="mps")
    r.add_argument("--models", nargs="+", default=YOLO_MODELS)
    r.add_argument("--no-mediapipe", action="store_true")
    r.add_argument("--batch", type=int, default=4, help="crops per YOLO call")
    r.add_argument("--out", default=str(OUT_DIR))
    sw = sub.add_parser("sweep", help="tiled recall vs false positives across cutoffs")
    sw.add_argument("--device", default="mps")
    sw.add_argument("--models", nargs="+", default=["yolo11n-pose", "yolo26n-pose"])
    sw.add_argument("--batch", type=int, default=4)
    sw.add_argument(
        "--floor", type=float, default=0.05, help="lowest cutoff detected at"
    )
    sw.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()
    {"frames": cmd_frames, "run": cmd_run, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
