"""
Turn raw keypoint rows into normalized, position/scale-invariant features.

A swimmer's pose is encoded relative to their own body, not the frame:
  1. translate so the hip-center is the origin  (removes WHERE they are)
  2. scale by torso length (shoulder-center -> hip-center)  (removes size/distance)

Output per pose: 34 numbers = 17 keypoints x normalized (x, y). Same body
configuration -> same vector, regardless of pool position or camera distance.
That invariance is what lets the autoencoder learn "normal swimming shape"
instead of "normal pixel location".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# COCO keypoint indices we use as anatomical references.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# Column groups in the keypoint Parquet (x0,y0,c0 ... x16,y16,c16).
_XY_COLS = [f"{ax}{i}" for i in range(17) for ax in ("x", "y")]
_C_COLS = [f"c{i}" for i in range(17)]


def normalize_pose(kpts: np.ndarray, min_ref_conf: float = 0.3) -> np.ndarray | None:
    """
    kpts: (17, 3) array of (x, y, confidence). Returns a 34-vector or None.

    None means the pose is unusable — the reference joints (both hips and both
    shoulders) weren't confidently detected, so we can't define a reliable body
    frame to normalize against.
    """
    lhip, rhip = kpts[L_HIP], kpts[R_HIP]
    lsh, rsh = kpts[L_SHOULDER], kpts[R_SHOULDER]

    if min(lhip[2], rhip[2], lsh[2], rsh[2]) < min_ref_conf:
        return None

    hip_center = (lhip[:2] + rhip[:2]) / 2.0
    shoulder_center = (lsh[:2] + rsh[:2]) / 2.0
    torso = np.linalg.norm(shoulder_center - hip_center)
    if torso < 1e-3:
        return None  # degenerate (points collapsed) — unreliable

    xy = kpts[:, :2] - hip_center  # translate to hip-centered frame
    xy = xy / torso  # scale by torso length
    return xy.flatten().astype(np.float32)  # (34,)


def features_from_dataframe(df: pd.DataFrame, min_ref_conf: float = 0.3):
    """
    Convert a keypoint Parquet (loaded as a DataFrame) into a feature matrix.

    Returns (features, meta):
      features : (N_valid, 34) float32 — one row per usable pose
      meta     : DataFrame with the frame/track_id of each kept row (so we can
                 trace a feature back to which swimmer/frame it came from)
    """
    xy = df[_XY_COLS].to_numpy(dtype=np.float32).reshape(len(df), 17, 2)
    conf = df[_C_COLS].to_numpy(dtype=np.float32).reshape(len(df), 17, 1)
    kpts = np.concatenate([xy, conf], axis=2)  # (N, 17, 3)

    feats, keep_idx = [], []
    for i in range(len(df)):
        v = normalize_pose(kpts[i], min_ref_conf=min_ref_conf)
        if v is not None:
            feats.append(v)
            keep_idx.append(i)

    features = (
        np.array(feats, dtype=np.float32) if feats else np.empty((0, 34), np.float32)
    )
    meta = df.iloc[keep_idx][["frame", "track_id"]].reset_index(drop=True)
    return features, meta
