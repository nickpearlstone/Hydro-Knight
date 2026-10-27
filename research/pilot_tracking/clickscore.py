"""
Score a track table against victim clicks, without using track ids as the answer key.

A click is held by a track if one of its boxes within `frame_tol` frames contains the click
or has its center within `radius_h` box heights, skipping boxes taller than `max_ratio` times
the local typical height at that moment or as their track's median (the matcher's rule in
eval/victims.py, which keeps a lifeguard's or a merged crowd box from holding the victim).
Coverage = share of clicks held by any track. Continuity = largest share held by one id.

Detector-agnostic: needs frame, track_id, bx1, by1, bx2, by2 only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

from hydro_knight.eval.victims import local_height


@dataclass(frozen=True)
class ScoreSettings:
    frame_tol: int = 2
    radius_h: float = 0.75
    max_ratio: float = 3.0


@dataclass
class Clicks:
    """One clip's scored clicks: file frames and pixel positions (trim start .. contact, one per frame)."""

    frame: np.ndarray
    x: np.ndarray
    y: np.ndarray


def clicks_for(record, fps: float) -> Clicks:
    ev = record.events[0]
    lo = 0.0 if record.trim_start is None else record.to_file_time(record.trim_start)
    hi = record.to_file_time(ev["contact"])
    seen, f, x, y = set(), [], [], []
    for c in sorted(ev.get("victim", []), key=lambda c: c["t"]):
        t = record.to_file_time(c["t"])
        fr = round(t * fps)
        if t < lo or t > hi or fr in seen:
            continue
        seen.add(fr)
        f.append(fr), x.append(c["x"]), y.append(c["y"])
    return Clicks(np.array(f), np.array(x, float), np.array(y, float))


def local_height_fn(reference: pd.DataFrame, fps: float) -> Callable:
    """Cached local_height(frame, y) on a reference table (the clip's original tracks).

    Swimmer size near a point is a property of the scene, so every relinked variant of a
    clip is checked against the same reference.
    """
    det = reference[reference["track_id"] >= 0]

    @lru_cache(maxsize=None)
    def ref(frame: int, y: int):
        return local_height(det, frame, y, fps)

    return lambda frame, y: ref(int(frame), int(round(y)))


class ClickScorer:
    """Precomputes a frame-sorted view of one clip's confirmed track rows."""

    def __init__(
        self, table: pd.DataFrame, ref_height: Callable | None = None, s: ScoreSettings = ScoreSettings()
    ):
        t = table[table["track_id"] >= 0].sort_values("frame", kind="stable")
        self.f = t["frame"].to_numpy()
        self.id = t["track_id"].to_numpy()
        self.b = t[["bx1", "by1", "bx2", "by2"]].to_numpy(float)
        h = t["by2"] - t["by1"]
        self.track_h = t["track_id"].map(h.groupby(t["track_id"]).median()).to_numpy(float)
        self.ref_height = ref_height
        self.s = s

    def held(self, frame: int, x: float, y: float) -> set[int]:
        lo = np.searchsorted(self.f, frame - self.s.frame_tol, side="left")
        hi = np.searchsorted(self.f, frame + self.s.frame_tol, side="right")
        b = self.b[lo:hi]
        if not len(b):
            return set()
        h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
        inside = (b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])
        d = np.hypot((b[:, 0] + b[:, 2]) / 2 - x, (b[:, 1] + b[:, 3]) / 2 - y) / h
        ok = inside | (d <= self.s.radius_h)
        ref = self.ref_height(frame, y) if self.ref_height else None
        if ref is not None:
            ok &= np.maximum(h, self.track_h[lo:hi]) <= self.s.max_ratio * ref
        return {int(i) for i in self.id[lo:hi][ok]}

    def score(self, c: Clicks) -> dict:
        """coverage and continuity for one path of clicks (NaN if no clicks)."""
        n = len(c.frame)
        if n == 0:
            return {"coverage": np.nan, "continuity": np.nan}
        held = [self.held(f, x, y) for f, x, y in zip(c.frame, c.x, c.y, strict=True)]
        counts: dict[int, int] = {}
        for h in held:
            for i in h:
                counts[i] = counts.get(i, 0) + 1
        return {
            "coverage": sum(bool(h) for h in held) / n,
            "continuity": max(counts.values(), default=0) / n,
        }


def translated_paths(
    c: Clicks, zone: np.ndarray, n: int, rng: np.random.Generator, tries: int = 2000
) -> list[Clicks]:
    """The click path moved by random offsets that keep every point inside the pool zone."""
    poly = Polygon(zone)
    (x0, y0), (x1, y1) = zone.min(0), zone.max(0)
    out = []
    for _ in range(tries):
        if len(out) == n:
            break
        dx = rng.uniform(x0 - c.x.min(), x1 - c.x.max()) if x1 - x0 > np.ptp(c.x) else None
        dy = rng.uniform(y0 - c.y.min(), y1 - c.y.max()) if y1 - y0 > np.ptp(c.y) else None
        if dx is None or dy is None:
            break
        if poly.contains_points(np.c_[c.x + dx, c.y + dy]).all():
            out.append(Clicks(c.frame, c.x + dx, c.y + dy))
    return out


def shifted_paths(
    c: Clicks, lo_frame: int, hi_frame: int, n: int, rng: np.random.Generator
) -> list[Clicks]:
    """The click path moved to a random other moment, wrapping around [lo_frame, hi_frame)."""
    span = hi_frame - lo_frame
    out = []
    for _ in range(n):
        off = rng.integers(1, span) if span > 1 else 0
        out.append(Clicks(lo_frame + (c.frame - lo_frame + off) % span, c.x, c.y))
    return out
