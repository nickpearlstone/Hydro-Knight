"""Splash beside the head: water motion next to each row's head, relative to the water around it.

Method (from the stack review, where it ranked the victim above same-moment swimmers at
AUC ~0.65): grey frames at 960 px wide; motion = |frame(t) - frame(t - ~1/15 s)|; mean
motion in three head-width patches (left, right, just below the head) divided by the mean
motion in a ring 3-5 head widths away, inside the coverage zone. Waves move the whole
neighbourhood, so the ring cancels them; a swimmer thrashing in place moves only the
patches beside the head. Whitewater brightness and zone-wide normalisation were dropped
(they failed or added nothing in the review). Causal: only the current and an earlier
frame are used. Ghost rows are measured at the spot where the head was last seen.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .data import VIDEOS, Clip

WIDTH = 960
EPS = 0.5  # grey levels; keeps the ratio finite over still water
HEAD_W_PER_BOX_W = 0.3  # patch size = 0.3 x box width, as in the review


def add_splash(rows: pd.DataFrame, clip: Clip) -> pd.DataFrame:
    """rows + a `splash` column (NaN where it can't be measured)."""
    out = rows.assign(splash=np.nan)
    if rows.empty:
        return out
    cap = cv2.VideoCapture(str(VIDEOS / f"{clip.clip_id}.mp4"))
    if not cap.isOpened():
        return out
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sc = WIDTH / W0
    Wd, Hd = WIDTH, int(round(H0 * sc))
    zmask = np.ones((Hd, Wd), bool)
    if clip.zone and clip.zone.get("points") and not clip.zone.get("camera_moves"):
        zm = np.zeros((Hd, Wd), np.uint8)
        pts = (np.asarray(clip.zone["points"], float) * sc).round().astype(np.int32)
        cv2.fillPoly(zm, [pts], 1)
        zmask = zm.astype(bool)
    k = max(1, round(clip.fps / 15))
    frames = np.rint(rows["t"].to_numpy() * clip.fps).astype(int)
    want_now = set(np.unique(frames).tolist())
    want = want_now | {f - k for f in want_now}
    first, last = min(want), max(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    if abs(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - first) > 1:
        cap.release()
        return out
    idx_by_frame = {f: np.flatnonzero(frames == f) for f in want_now}
    x = rows["x"].to_numpy() * sc
    y = rows["y"].to_numpy() * sc
    hw = (
        np.maximum(5.0, HEAD_W_PER_BOX_W * rows["w"].to_numpy()) * sc
    )  # full-res px -> 960 px
    val = out["splash"].to_numpy().copy()
    prev = {}
    for f in range(first, last + 1):
        if not cap.grab():
            break
        if f not in want:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        g = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            (Wd, Hd),
            interpolation=cv2.INTER_AREA,
        )
        prev[f] = g
        if f in want_now and (f - k) in prev:
            M = np.abs(g.astype(np.float32) - prev[f - k].astype(np.float32))
            integ = cv2.integral(M, sdepth=cv2.CV_64F)
            for i in idx_by_frame[f]:
                val[i] = _ratio(M, integ, x[i], y[i], hw[i], zmask)
        for old in [p for p in prev if p < f - k]:
            del prev[old]
    cap.release()
    out["splash"] = val
    return out


def _patch(integ: np.ndarray, cx: float, cy: float, s: float) -> float:
    H, W = integ.shape[0] - 1, integ.shape[1] - 1
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = min(W, x0 + int(s)), min(H, y0 + int(s))
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    return (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]) / (
        (x1 - x0) * (y1 - y0)
    )


def _ratio(M, integ, cx, cy, hw, zmask) -> float:
    s = max(3.0, round(hw))
    side = [
        _patch(integ, cx - 1.25 * hw, cy, s),
        _patch(integ, cx + 1.25 * hw, cy, s),
        _patch(integ, cx, cy + 1.25 * hw, s),
    ]
    side = [v for v in side if np.isfinite(v)]
    R = int(np.ceil(5 * hw)) + 1
    Hd, Wd = M.shape
    x0, x1 = max(0, int(cx) - R), min(Wd, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(Hd, int(cy) + R + 1)
    if not side or x1 <= x0 or y1 <= y0:
        return np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (r2 >= (3 * hw) ** 2) & (r2 <= (5 * hw) ** 2) & zmask[y0:y1, x0:x1]
    if ring.sum() < 10:
        return np.nan
    return float(np.mean(side) / (M[y0:y1, x0:x1][ring].mean() + EPS))
