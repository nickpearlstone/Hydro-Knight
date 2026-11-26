"""Coverage zones: one polygon per clip (original-frame pixels) for the patch of water its camera covers."""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

import numpy as np

ZONES_PATH = Path("data/annotations/zones.json")


def load(path: Path = ZONES_PATH) -> dict:
    """The {clip_id: zone} file, or {} if it doesn't exist yet."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save(data: dict, path: Path = ZONES_PATH) -> None:
    """Write the zone file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def clean_zone(body: dict) -> dict:
    """Validate a posted zone; points rounded to 0.1 px, at least 3 unless camera_moves."""
    if not isinstance(body, dict):
        raise ValueError("expected a JSON object")
    moves = bool(body.get("camera_moves", False))
    pts = body.get("points", [])
    if not isinstance(pts, list):
        raise ValueError("points must be a list")
    clean = []
    for p in pts:
        if not (isinstance(p, list | tuple) and len(p) == 2):
            raise ValueError(f"bad point {p!r}")
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0:
            raise ValueError(f"bad point {p!r}")
        clean.append([round(x, 1), round(y, 1)])
    if not moves and len(clean) < 3:
        raise ValueError("a zone needs at least 3 points")
    frame = body.get("frame", 0)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError(f"bad frame {frame!r}")
    return {
        "points": clean,
        "frame": frame,
        "camera_moves": moves,
        "date": datetime.date.today().isoformat(),
    }


def in_zone(zone: dict | None, x, y) -> np.ndarray:
    """Bool array: which (x, y) points lie inside the zone polygon (even-odd rule).

    A missing or moving-camera zone contains everything.
    """
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    if zone is None or zone.get("camera_moves") or len(zone.get("points", [])) < 3:
        return np.ones(x.shape, bool)
    poly = np.asarray(zone["points"], float)
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    inside = np.zeros(x.shape, bool)
    for x1, y1, x2, y2 in zip(px, py, qx, qy, strict=True):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xc)
    return inside
