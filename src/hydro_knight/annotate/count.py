"""
Swimmer-count labels: every person clicked on a handful of still frames.

This is the ground truth behind scripts/benchmark_pose.py. Each frame record is
{"clip_id", "frame", "label", "water": [[x, y], ...], "deck": [[x, y], ...]}:
one point per person, in original-frame pixels, "water" for people in the pool and
"deck" for people out of it (deck, stairs, guard stand). The file holds points only,
no pixels, so it is safe to commit. The frame images are cached under raw_local/
and are never committed.

The benchmark script picks the frames (`benchmark_pose.py frames`); the labeler's
Swimmer count tab edits the points.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from ..ingest import download

COUNT_PATH = Path("data/annotations/pose_bench_gt.json")
KINDS = ("water", "deck")


def frame_dir() -> Path:
    """Cache folder for the frame images (inside raw_local/, so gitignored)."""
    return download.RAW_LOCAL / "pose_bench"


def frame_path(item: dict) -> Path:
    """Cached image path for one frame record."""
    return frame_dir() / f"{item['clip_id']}_{item['frame']}.png"


def read_frame(item: dict) -> np.ndarray:
    """The frame as a BGR array: from the cache, or decoded from raw_local/<clip_id>.mp4 and cached.

    Raises:
        FileNotFoundError: the frame isn't cached and the video can't supply it.
    """
    path = frame_path(item)
    if path.exists():
        return cv2.imread(str(path))
    cap = cv2.VideoCapture(str(download.RAW_LOCAL / f"{item['clip_id']}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(
            f"can't read frame {item['frame']} of {item['clip_id']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return frame


def load(path: Path = COUNT_PATH) -> dict:
    """The whole label file: {"frames": [...]}."""
    return json.loads(Path(path).read_text())


def save(data: dict, path: Path = COUNT_PATH) -> None:
    """Write the label file atomically (temp file + rename), so a crash never truncates it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def clean_points(body: dict) -> dict[str, list[list[float]]]:
    """Validate {"water": [[x, y]...], "deck": [...]} posted by the UI; rounds to 0.1 px.

    Raises:
        ValueError: a missing kind, or a point that isn't two finite, non-negative numbers.
    """
    out = {}
    for kind in KINDS:
        pts = body.get(kind)
        if not isinstance(pts, list):
            raise ValueError(f"missing {kind!r} point list")
        clean = []
        for p in pts:
            if not (isinstance(p, list | tuple) and len(p) == 2):
                raise ValueError(f"bad point {p!r}")
            x, y = float(p[0]), float(p[1])
            if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0:
                raise ValueError(f"bad point {p!r}")
            clean.append([round(x, 1), round(y, 1)])
        out[kind] = clean
    return out
