from __future__ import annotations

import numpy as np
import pandas as pd

from hydro_knight.features.normalize import (
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
)
from hydro_knight.features.windows import make_windows

_COLS = [f"{ax}{i}" for i in range(17) for ax in ("x", "y")] + [
    f"c{i}" for i in range(17)
]


def _pose_row(frame: int, track_id: int) -> dict:
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)
    xy[L_HIP] = [90.0, 100.0]
    xy[R_HIP] = [110.0, 100.0]
    xy[L_SHOULDER] = [90.0, 60.0]
    xy[R_SHOULDER] = [110.0, 60.0]
    row = {"frame": frame, "track_id": track_id}
    for i in range(17):
        row[f"x{i}"], row[f"y{i}"] = float(xy[i, 0]), float(xy[i, 1])
        row[f"c{i}"] = float(conf[i])
    return row


def _frames_df(track_id: int, n_frames: int) -> pd.DataFrame:
    return pd.DataFrame([_pose_row(f, track_id) for f in range(n_frames)])


def test_window_shape_and_count():
    df = _frames_df(track_id=1, n_frames=40)
    windows, info = make_windows(df, window=32, stride=8)
    assert windows.shape == (2, 32, 70)
    assert len(info) == 2
    assert info[0] == (1, 0)
    assert info[1] == (1, 8)


def test_too_few_frames_yields_no_windows():
    df = _frames_df(track_id=1, n_frames=10)
    windows, info = make_windows(df, window=32, stride=8)
    assert windows.shape == (0, 32, 70)
    assert info == []


def test_unconfirmed_tracks_excluded():
    df = _frames_df(track_id=-1, n_frames=40)
    windows, _ = make_windows(df, window=32, stride=8)
    assert windows.shape[0] == 0


def test_windows_are_per_track():
    df = pd.concat([_frames_df(1, 40), _frames_df(2, 40)], ignore_index=True)
    windows, info = make_windows(df, window=32, stride=8)
    track_ids = {tid for tid, _ in info}
    assert track_ids == {1, 2}
    assert windows.shape[0] == 4
