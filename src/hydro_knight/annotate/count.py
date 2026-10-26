"""
Swimmer-count labels: every person clicked on a handful of still frames.

This is the ground truth behind hydroknight benchmark pose. Each frame record is
{"clip_id", "frame", "label", "water": [[x, y], ...], "deck": [[x, y], ...]}:
one point per person, in original-frame pixels, "water" for people in the pool and
"deck" for people out of it (deck, stairs, guard stand). An optional "hidden" list marks
people whose head isn't visible in the still (fully under water with only a shadow showing,
only an arm or back visible, the head blocked by a tube, person or splash, or not sure it's
a person). Scoring ignores deck and hidden points alike (`ignored`): a detector is neither
penalised for missing them nor for boxing them. The file holds points only,
no pixels, so it is safe to commit. The frame images are cached under raw_local/
and are never committed.

The benchmark script picks the frames (`benchmark_pose.py frames`); the labeler's
Swimmer count tab edits the points.

Head boxes (the tab's head-box mode) add optional keys to a frame record:
"heads": {"water": [[x1, y1, x2, y2], ...], "deck": [...], "no_head": [[x, y], ...]}
("no_head" = a person is visible but their head isn't, so a head detector can't be
asked to find them), "heads_reviewed" (a person checked every box on the frame) and
"heads_source" (who drew the boxes first, e.g. a model's pre-labels).

Two frame sets share this format: the benchmark frames (COUNT_PATH, the test set)
and the head validation frames (VAL_PATH, picked by `hydroknight heads frames` from
other clips, for choosing checkpoints and thresholds without touching the test set).
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
VAL_PATH = Path("data/annotations/head_val_gt.json")
KINDS = ("water", "deck")
OPTIONAL_KINDS = ("hidden",)
HEAD_BOX_KINDS = ("water", "deck")
OPTIONAL_KINDS = ("hidden",)


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


def _point(p) -> list[float]:
    """One [x, y] point: two finite, non-negative numbers, rounded to 0.1 px."""
    if not (isinstance(p, list | tuple) and len(p) == 2):
        raise ValueError(f"bad point {p!r}")
    x, y = float(p[0]), float(p[1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0:
        raise ValueError(f"bad point {p!r}")
    return [round(x, 1), round(y, 1)]


def clean_points(body: dict) -> dict[str, list[list[float]]]:
    """Validate {"water": [[x, y]...], "deck": [...], "hidden": [...]} posted by the UI.

    "hidden" is optional (older clients and files don't send it). Rounds to 0.1 px.

    Raises:
        ValueError: a missing kind, or a point that isn't two finite, non-negative numbers.
    """
    out = {}
    for kind in KINDS + OPTIONAL_KINDS:
        pts = body.get(kind)
        if pts is None and kind in OPTIONAL_KINDS:
            continue
        if not isinstance(pts, list):
            raise ValueError(f"missing {kind!r} point list")
        out[kind] = [_point(p) for p in pts]
    return out


def ignored(frame: dict) -> list:
    """Points a detector is scored on neither way: people on the deck and hidden heads."""
    return list(frame.get("deck", [])) + list(frame.get("hidden", []))


def clean_heads(body: dict) -> dict[str, list[list[float]]]:
    """Validate {"water": [[x1, y1, x2, y2]...], "deck": [...], "no_head": [[x, y]...]}.

    Boxes are rounded to 0.1 px and must have x1 < x2 and y1 < y2.

    Raises:
        ValueError: a missing list, or a box or point that isn't finite and non-negative.
    """
    if not isinstance(body, dict):
        raise ValueError("heads must be an object")
    out = {}
    for kind in HEAD_BOX_KINDS:
        boxes = body.get(kind)
        if not isinstance(boxes, list):
            raise ValueError(f"missing heads {kind!r} box list")
        clean = []
        for b in boxes:
            if not (isinstance(b, list | tuple) and len(b) == 4):
                raise ValueError(f"bad box {b!r}")
            x1, y1, x2, y2 = (round(float(v), 1) for v in b)
            ok = all(math.isfinite(v) and v >= 0 for v in (x1, y1, x2, y2))
            if not ok or x1 >= x2 or y1 >= y2:
                raise ValueError(f"bad box {b!r}")
            clean.append([x1, y1, x2, y2])
        out[kind] = clean
    pts = body.get("no_head")
    if not isinstance(pts, list):
        raise ValueError("missing heads 'no_head' point list")
    out["no_head"] = [_point(p) for p in pts]
    return out
