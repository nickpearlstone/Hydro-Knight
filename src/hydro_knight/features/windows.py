"""Slice per-frame pose features into fixed-length temporal windows (sequences)."""

from __future__ import annotations

import numpy as np

from .normalize import features_from_dataframe


def make_windows(
    df,
    window: int = 32,
    stride: int = 8,
    min_ref_conf: float = 0.3,
    velocity: bool = True,
):
    """Slice each swimmer's normalized poses into overlapping fixed-length windows."""
    n_feat = 70 if velocity else 34
    feats, meta = features_from_dataframe(df, min_ref_conf=min_ref_conf)
    meta = meta.copy()
    meta["row"] = np.arange(len(meta))
    windows, info = [], []
    for tid, g in meta.groupby("track_id"):
        if tid < 0:
            continue
        g = g.sort_values("frame")
        rows = g["row"].to_numpy()
        frames = g["frame"].to_numpy()
        pos = feats[rows]
        if velocity:
            cen = g[["cx", "cy"]].to_numpy()
            gaps = np.diff(frames)[:, None]
            vel = np.vstack([np.zeros((1, 34)), np.diff(pos, axis=0) / gaps])
            cvel = np.vstack([np.zeros((1, 2)), np.diff(cen, axis=0) / gaps])
            track_feats = np.hstack([pos, vel, cvel])
        else:
            track_feats = pos
        for s in range(0, len(rows) - window + 1, stride):
            windows.append(track_feats[s : s + window])
            info.append((int(tid), int(frames[s])))

    arr = (
        np.stack(windows).astype(np.float32)
        if windows
        else np.empty((0, window, n_feat), np.float32)
    )
    return arr, info
