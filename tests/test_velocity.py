"""
Tests for the velocity features added to make_windows (features/windows.py).

Each window frame is now 70-dim, laid out as three blocks:
    cols  0:34  — normalized pose position (17 keypoints x, y)
    cols 34:68  — per-keypoint velocity (in the normalized, hip-centered frame)
    cols 68:70  — centroid velocity (raw hip-center motion through the pool)

These tests pin the *behavior* of that velocity math:
  - the first frame of a track has zero velocity (no previous frame to diff),
  - a stationary swimmer has zero velocity everywhere,
  - pure translation is invisible to per-keypoint velocity (normalization removes
    it) but shows up in centroid velocity — the core design intent,
  - a genuine change of body shape *does* produce per-keypoint velocity,
  - velocity is normalized by the frame gap, so skipped frames don't inflate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro_knight.features.normalize import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from hydro_knight.features.windows import make_windows

# Feature-block column ranges within the 70-dim frame vector.
POS = slice(0, 34)  # normalized pose position
KP_VEL = slice(34, 68)  # per-keypoint velocity
CEN_VEL = slice(68, 70)  # centroid velocity
L_WRIST = 9  # COCO index; its x sits at flat position 2*9 = 18


def _pose_row(frame: int, track_id: int, tx=0.0, ty=0.0, ref_conf=1.0, lw=None) -> dict:
    """
    One keypoint row. The body is translated by (tx, ty); `ref_conf` sets the
    reference-joint confidence (< 0.3 makes normalize drop the pose, simulating a
    submerged/low-confidence frame). `lw` optionally overrides the left-wrist
    absolute position to create a change of body *shape* rather than translation.
    """
    xy = np.zeros((17, 2), dtype=float)
    xy[L_HIP] = [90.0 + tx, 100.0 + ty]
    xy[R_HIP] = [110.0 + tx, 100.0 + ty]
    xy[L_SHOULDER] = [90.0 + tx, 60.0 + ty]
    xy[R_SHOULDER] = [110.0 + tx, 60.0 + ty]
    refs = (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)
    for i in range(17):
        if i not in refs:
            xy[i] = [tx, ty]  # other joints ride along with the translation
    if lw is not None:
        xy[L_WRIST] = lw

    row = {"frame": frame, "track_id": track_id}
    for i in range(17):
        row[f"x{i}"], row[f"y{i}"] = float(xy[i, 0]), float(xy[i, 1])
        row[f"c{i}"] = 1.0
    for j in refs:
        row[f"c{j}"] = ref_conf  # only the reference joints gate the drop
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_first_frame_velocity_is_zero():
    # The first frame of a track has no predecessor, so every velocity channel
    # (keypoint + centroid) is padded to 0.
    rows = [_pose_row(f, 1, tx=5.0 * f, ty=3.0 * f) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    first_frame_velocity = windows[0][0, 34:70]  # both velocity blocks, frame 0
    assert np.allclose(first_frame_velocity, 0.0)


def test_stationary_swimmer_has_zero_velocity():
    # Never moves -> every velocity channel is 0 in every frame of every window.
    rows = [_pose_row(f, 1) for f in range(40)]  # tx = ty = 0
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    assert np.allclose(windows[:, :, 34:70], 0.0)


def test_pure_translation_invisible_to_keypoint_velocity_but_moves_centroid():
    # Design intent: a swimmer sliding uniformly (tx=5, ty=3 per frame) with a
    # fixed body shape. Normalization removes absolute position, so per-keypoint
    # velocity ~ 0 — but the raw centroid moves, so centroid velocity = (5, 3).
    rows = [_pose_row(f, 1, tx=5.0 * f, ty=3.0 * f) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    w = windows[0]
    assert np.allclose(w[:, KP_VEL], 0.0, atol=1e-4)  # translation-invariant
    assert np.allclose(w[0, CEN_VEL], [0.0, 0.0])  # first frame padded
    assert np.allclose(w[1:, CEN_VEL], [5.0, 3.0], atol=1e-4)  # net displacement


def test_shape_change_produces_nonzero_keypoint_velocity():
    # Body stays put; only the left wrist moves (2 px/frame). The normalized pose
    # genuinely changes, so the left-wrist velocity channel is nonzero.
    rows = [_pose_row(f, 1, lw=[90.0 + 2.0 * f, 80.0]) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    w = windows[0]
    lwrist_x_vel = 34 + 2 * L_WRIST  # velocity block offset + flat x index (=52)
    assert np.any(np.abs(w[1:, lwrist_x_vel]) > 1e-6)


def test_gap_normalization_divides_by_frame_gap():
    # Frame 2 is dropped (low-confidence reference joints), so usable frames are
    # 0,1,3,4. The 1->3 step spans a 2-frame gap; with tx=6/frame the centroid
    # moves 12 across it. Gap-normalized velocity must be 6 (12 / 2), not 12.
    rows = [
        _pose_row(0, 1, tx=0.0),
        _pose_row(1, 1, tx=6.0),
        _pose_row(2, 1, tx=12.0, ref_conf=0.0),  # dropped by normalize
        _pose_row(3, 1, tx=18.0),
        _pose_row(4, 1, tx=24.0),
    ]
    windows, _ = make_windows(_df(rows), window=3, stride=1)
    # first window = usable poses [f0, f1, f3]; row 2 (f3) is the across-gap step.
    across_gap_centroid_vx = windows[0][2, 68]
    assert np.isclose(across_gap_centroid_vx, 6.0)
