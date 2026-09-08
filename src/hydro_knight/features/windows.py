"""
Slice per-frame pose features into fixed-length temporal windows (sequences).

A single pose is one frame's snapshot. To learn *motion* (Rung 3) the model
needs short sequences. For each tracked swimmer we sort their poses by frame
and slide a window over them.

Output windows have shape (N, window, 70): N sequences, each `window` frames of
the 70-dim per-frame feature vector (34 normalized-pose coords + 34 per-keypoint
velocities + 2 centroid velocities). One window = "how this swimmer moved over ~W
frames" — the unit the temporal autoencoder reconstructs.
"""

from __future__ import annotations

import numpy as np

from .normalize import features_from_dataframe


def make_windows(df, window: int = 32, stride: int = 8, min_ref_conf: float = 0.3):
    """Slice each swimmer's normalized poses into overlapping fixed-length windows.

    Per track (track_id < 0 skipped), poses are frame-sorted and windowed with a sliding
    stride; each frame carries 70 features (pos + gap-normalized velocity + centroid velocity).
    Args:
        df: keypoint-Parquet DataFrame (see features_from_dataframe).
        window: frames per window.
        stride: step between window starts (overlap = window - stride).
        min_ref_conf: min reference-joint confidence, passed through to feature extraction.
    Returns:
        (windows, info): windows is (N, window, 70) float32; info is a list of
        (track_id, start_frame), one per window. Tracks with fewer than `window` usable poses yield none.
    """
    feats, meta = features_from_dataframe(df, min_ref_conf=min_ref_conf)
    meta = meta.copy()
    meta["row"] = np.arange(len(meta))
    windows, info = [], []
    for tid, g in meta.groupby("track_id"):
        if tid < 0:
            continue  # unconfirmed tracks have no stable identity
        g = g.sort_values("frame")
        rows = g["row"].to_numpy()
        frames = g["frame"].to_numpy()
        pos = feats[rows]
        cen = g[["cx","cy"]].to_numpy()
        gaps = np.diff(frames)[:, None]
        vel = np.vstack([np.zeros((1, 34)), np.diff(pos, axis=0) / gaps])
        cvel = np.vstack([np.zeros((1, 2)), np.diff(cen, axis=0) / gaps])
        track_feats = np.hstack([pos, vel, cvel])  # (T, 70): 34 pos + 34 kp-vel + 2 centroid-vel
        for s in range(0, len(rows) - window + 1, stride):
            windows.append(track_feats[s: s+window])
            info.append((int(tid), int(frames[s])))

    arr = (
        np.stack(windows).astype(np.float32)
        if windows
        else np.empty((0, window, 70), np.float32)
    )
    return arr, info
