"""Turn raw keypoint rows into normalized, position/scale-invariant features."""

from __future__ import annotations

import numpy as np
import pandas as pd

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

_XY_COLS = [f"{ax}{i}" for i in range(17) for ax in ("x", "y")]
_C_COLS = [f"c{i}" for i in range(17)]


def normalize_pose(kpts: np.ndarray, min_ref_conf: float = 0.3) -> np.ndarray | None:
    """Encode one pose as a hip-centered, torso-scaled (position/scale-invariant) vector."""
    lhip, rhip = kpts[L_HIP], kpts[R_HIP]
    lsh, rsh = kpts[L_SHOULDER], kpts[R_SHOULDER]

    if min(lhip[2], rhip[2], lsh[2], rsh[2]) < min_ref_conf:
        return None

    hip_center = (lhip[:2] + rhip[:2]) / 2.0
    shoulder_center = (lsh[:2] + rsh[:2]) / 2.0
    torso = np.linalg.norm(shoulder_center - hip_center)
    if torso < 1e-3:
        return None

    xy = kpts[:, :2] - hip_center
    xy = xy / torso
    return xy.flatten().astype(np.float32)


def features_from_dataframe(df: pd.DataFrame, min_ref_conf: float = 0.3):
    """Convert a keypoint-Parquet DataFrame into normalized pose features plus per-row metadata."""
    xy = df[_XY_COLS].to_numpy(dtype=np.float32).reshape(len(df), 17, 2)
    conf = df[_C_COLS].to_numpy(dtype=np.float32).reshape(len(df), 17, 1)
    kpts = np.concatenate([xy, conf], axis=2)

    feats, keep_idx, centers = [], [], []
    for i in range(len(df)):
        v = normalize_pose(kpts[i], min_ref_conf=min_ref_conf)
        if v is not None:
            feats.append(v)
            keep_idx.append(i)
            centers.append((kpts[i][11][:2] + kpts[i][12][:2]) / 2)

    features = (
        np.array(feats, dtype=np.float32) if feats else np.empty((0, 34), np.float32)
    )
    meta = df.iloc[keep_idx][["frame", "track_id"]].reset_index(drop=True)
    meta["cx"] = [c[0] for c in centers]
    meta["cy"] = [c[1] for c in centers]
    return features, meta
