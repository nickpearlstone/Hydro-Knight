"""Place-anchored features: a head's recent history taken from the boxes around it, not one track id.

Why (2026-10-28): v3 scores the victim only 48% of the catch window, because features need 2 s of
one track and the victim's head track keeps breaking. A third of the breaks overlap in time (two
boxes on the victim at once) and the rest reappear a median 1.4 head heights away, so stitching
track ends (stack.stitch) caught 5% of them.

Here every anchor box walks BACKWARD through the earlier sampled frames: at each one it moves to
the nearest box (any track, or none) within RADIUS head heights of where it just was and within
MAX_RATIO in size; a frame with no such box is skipped. The spot follows a drifting swimmer, and
duplicates or id changes stop mattering. Features are then the same as stack.heads.add_features,
computed on that path: churn and size / confidence flicker over WINDOW_S, churn over 4 s, aspect
over the last 1 s. Causal: only boxes at or before the anchor's frame are used.

Result (2026-10-28, whole-frame heads, full K-fold protocol, 46 clips, 3 seeds): victim scored
48% -> 58% of the catch window; ranker AUC 0.79-0.80 -> 0.81; with appearance AUC 0.89 -> 0.91-0.92,
victim top 3 for 36-37% -> 44% of the window, top 1 24% -> 31%, reaches top 1 in 96% of clips.
Neighbours don't inflate place churn in crowds (lower than per-track churn at every crowding level).

Cost: ~3.5 ms per 0.5 s step on one M1 core on the busiest clip (67 heads in the zone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RADIUS = 1.5  # head heights of the anchor box
MAX_RATIO = 1.5  # larger / smaller head height for a box to join the path
LOOKBACK_S = 4.0  # longest feature window (churn_4)


def _sd(v: np.ndarray) -> float:
    return float(np.std(v)) if len(v) else np.nan


def _churn(x, y, h, t, now, w, fps):
    """Smoothed path length - net displacement over the last w s, in head heights; NaN if short."""
    k = t >= now - w - 0.5 / fps
    if k.sum() < 4 or t[k][0] > now - w + 0.15:
        return np.nan
    sx = pd.Series(x[k]).rolling(3, min_periods=1).mean().to_numpy()
    sy = pd.Series(y[k]).rolling(3, min_periods=1).mean().to_numpy()
    path = np.hypot(np.diff(sx), np.diff(sy)).sum()
    return (path - np.hypot(sx[-1] - sx[0], sy[-1] - sy[0])) / h


def place_features(
    pool: pd.DataFrame, anchors: pd.DataFrame, fps: float, window_s: float = 2.0
) -> pd.DataFrame:
    """Features for each anchor row (index kept): churn, size_flick, conf_flick, churn_4, aspect, path_n.

    pool / anchors: frame, cx, cy, hh, x0, x1, conf (pool = every kept box, any track id).
    """
    pool = pool.sort_values("frame")
    by_frame = {
        f: g[["cx", "cy", "hh", "x0", "x1", "conf"]].to_numpy(np.float64)
        for f, g in pool.groupby("frame")
    }
    frames = np.array(sorted(by_frame))
    cols = ["churn", "size_flick", "conf_flick", "churn_4", "aspect", "path_n"]
    out = pd.DataFrame(np.nan, index=anchors.index, columns=cols)
    for f, a in anchors.groupby("frame"):
        n = len(a)
        cur = a[["cx", "cy"]].to_numpy(np.float64).copy()
        h0 = a.hh.to_numpy(np.float64)
        first = np.column_stack(
            [cur, h0, (a.x1 - a.x0).to_numpy(np.float64), a.conf.to_numpy(np.float64)]
        )
        paths = [[(f, *first[i])] for i in range(n)]
        lo = np.searchsorted(frames, f - LOOKBACK_S * fps - 0.5)
        for g in frames[lo : np.searchsorted(frames, f)][::-1]:
            b = by_frame[g]
            d = np.hypot(
                cur[:, None, 0] - b[None, :, 0], cur[:, None, 1] - b[None, :, 1]
            )
            ratio = np.maximum(h0[:, None], b[None, :, 2]) / np.maximum(
                np.minimum(h0[:, None], b[None, :, 2]), 1e-6
            )
            d[(d > RADIUS * h0[:, None]) | (ratio > MAX_RATIO)] = np.inf
            j = d.argmin(1)
            hit = np.isfinite(d[np.arange(n), j])
            for i in np.flatnonzero(hit):
                bx = b[j[i]]
                paths[i].append((g, bx[0], bx[1], bx[2], bx[4] - bx[3], bx[5]))
                cur[i] = bx[:2]
        now = f / fps
        for i, idx in enumerate(a.index):
            p = np.array(paths[i][::-1])  # oldest first
            t, x, y, h, w, c = p.T
            t = t / fps
            hm = float(np.median(h))
            last = t >= now - window_s - 0.5 / fps
            full = last.sum() >= 4 and t[last][0] <= now - window_s + 0.15
            one = t >= now - 1.0
            out.loc[idx] = (
                _churn(x, y, hm, t, now, window_s, fps),
                _sd(np.log(np.maximum(h[last], 1))) if full else np.nan,
                _sd(c[last]) if full else np.nan,
                _churn(x, y, hm, t, now, 4.0, fps),
                float(
                    np.median(
                        np.log(np.maximum(w[one], 1)) - np.log(np.maximum(h[one], 1))
                    )
                ),
                len(p),
            )
    return out
