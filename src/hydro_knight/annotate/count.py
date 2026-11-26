"""Swimmer-count labels: people clicked on still frames (points only, original-frame pixels; images cached in raw_local/)."""

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
    """The frame as a BGR array, from the cache or decoded from raw_local/<clip_id>.mp4."""
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
    """Write the label file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _point(p) -> list[float]:
    if not (isinstance(p, list | tuple) and len(p) == 2):
        raise ValueError(f"bad point {p!r}")
    x, y = float(p[0]), float(p[1])
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0:
        raise ValueError(f"bad point {p!r}")
    return [round(x, 1), round(y, 1)]


def clean_points(body: dict) -> dict[str, list[list[float]]]:
    """Validate posted water/deck points ("hidden" optional); rounded to 0.1 px."""
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


def head_truth(frame: dict) -> tuple[list, list]:
    """(heads to find, points ignored): reviewed head boxes if present, else the dots."""
    h = frame.get("heads")
    if not (frame.get("heads_reviewed") and h):
        return list(frame["water"]), ignored(frame)

    def centres(boxes):
        return [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes]

    return centres(h["water"]), centres(h["deck"]) + list(h["no_head"])


def clean_heads(body: dict) -> dict[str, list[list[float]]]:
    """Validate posted head boxes (x1 < x2, y1 < y2) and no_head points; rounded to 0.1 px."""
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
