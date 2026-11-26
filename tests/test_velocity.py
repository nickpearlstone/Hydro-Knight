from __future__ import annotations

import numpy as np
import pandas as pd

from hydro_knight.features.normalize import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from hydro_knight.features.windows import make_windows

POS = slice(0, 34)
KP_VEL = slice(34, 68)
CEN_VEL = slice(68, 70)
L_WRIST = 9


def _pose_row(frame: int, track_id: int, tx=0.0, ty=0.0, ref_conf=1.0, lw=None) -> dict:
    xy = np.zeros((17, 2), dtype=float)
    xy[L_HIP] = [90.0 + tx, 100.0 + ty]
    xy[R_HIP] = [110.0 + tx, 100.0 + ty]
    xy[L_SHOULDER] = [90.0 + tx, 60.0 + ty]
    xy[R_SHOULDER] = [110.0 + tx, 60.0 + ty]
    refs = (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)
    for i in range(17):
        if i not in refs:
            xy[i] = [tx, ty]
    if lw is not None:
        xy[L_WRIST] = lw

    row = {"frame": frame, "track_id": track_id}
    for i in range(17):
        row[f"x{i}"], row[f"y{i}"] = float(xy[i, 0]), float(xy[i, 1])
        row[f"c{i}"] = 1.0
    for j in refs:
        row[f"c{j}"] = ref_conf
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_first_frame_velocity_is_zero():
    rows = [_pose_row(f, 1, tx=5.0 * f, ty=3.0 * f) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    first_frame_velocity = windows[0][0, 34:70]
    assert np.allclose(first_frame_velocity, 0.0)


def test_stationary_swimmer_has_zero_velocity():
    rows = [_pose_row(f, 1) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    assert np.allclose(windows[:, :, 34:70], 0.0)


def test_pure_translation_invisible_to_keypoint_velocity_but_moves_centroid():
    rows = [_pose_row(f, 1, tx=5.0 * f, ty=3.0 * f) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    w = windows[0]
    assert np.allclose(w[:, KP_VEL], 0.0, atol=1e-4)
    assert np.allclose(w[0, CEN_VEL], [0.0, 0.0])
    assert np.allclose(w[1:, CEN_VEL], [5.0, 3.0], atol=1e-4)


def test_shape_change_produces_nonzero_keypoint_velocity():
    rows = [_pose_row(f, 1, lw=[90.0 + 2.0 * f, 80.0]) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8)
    w = windows[0]
    lwrist_x_vel = 34 + 2 * L_WRIST
    assert np.any(np.abs(w[1:, lwrist_x_vel]) > 1e-6)


def test_velocity_off_gives_pose_only_windows():
    rows = [_pose_row(f, 1, tx=5.0 * f, ty=3.0 * f) for f in range(40)]
    windows, _ = make_windows(_df(rows), window=32, stride=8, velocity=False)
    assert windows.shape[-1] == 34


def test_velocity_off_matches_the_position_block_exactly():
    rows = [_pose_row(f, 1, lw=[90.0 + 2.0 * f, 80.0]) for f in range(40)]
    full, info_full = make_windows(_df(rows), window=32, stride=8, velocity=True)
    pose, info_pose = make_windows(_df(rows), window=32, stride=8, velocity=False)

    assert info_full == info_pose
    assert full.shape[:2] == pose.shape[:2]
    assert np.array_equal(full[:, :, POS], pose)


def test_gap_normalization_divides_by_frame_gap():
    # 1->3 spans 2 frames (centroid moves 12): velocity must be 6, not 12
    rows = [
        _pose_row(0, 1, tx=0.0),
        _pose_row(1, 1, tx=6.0),
        _pose_row(2, 1, tx=12.0, ref_conf=0.0),
        _pose_row(3, 1, tx=18.0),
        _pose_row(4, 1, tx=24.0),
    ]
    windows, _ = make_windows(_df(rows), window=3, stride=1)
    across_gap_centroid_vx = windows[0][2, 68]
    assert np.isclose(across_gap_centroid_vx, 6.0)
