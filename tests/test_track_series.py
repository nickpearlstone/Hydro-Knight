from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from v2_fakes import track_rows

from hydro_knight.features.track_series import (
    DETECTED,
    N_BASE,
    TARGET_FPS,
    resample_track,
    track_series,
    trim_series,
    zero_when_missing,
)


def test_60fps_track_with_a_gap_keeps_missing_steps():
    g = track_rows(1, [f for f in range(60) if not 20 <= f <= 29])
    ts = resample_track(g, fps_src=60.0, frame_h=1080)
    assert len(ts) == 30 and ts.feats.shape == (30, N_BASE)
    assert ts.detected.tolist() == [not 10 <= k <= 14 for k in range(30)]
    missing = ~ts.detected
    assert (ts.feats[missing][:, zero_when_missing(N_BASE)] == 0).all()
    assert (ts.feats[missing][:, :34] == ts.feats[9, :34]).all()
    assert (ts.feats[:, DETECTED] == ts.detected).all()


def test_box_normalization_size_aspect_and_velocity():
    # 120 px/s = 1.5 box heights/s
    g = track_rows(1, range(60), box=(100, 100, 140, 180), dx_per_frame=2.0)
    ts = resample_track(g, fps_src=60.0, frame_h=1080)
    f = ts.feats
    assert np.allclose(f[:, :34], 0.0)
    assert np.allclose(f[:, 53], 0.5)
    assert np.allclose(f[:, 55], 80 / 1080)
    assert np.allclose(f[1:, 56], 1.5) and np.allclose(f[:, 57], 0.0)
    assert np.allclose(f[:, 54], 0.0)
    assert f[0, 56] == 0.0


def test_24fps_upsampling_and_degenerate_box_stay_finite():
    g = track_rows(1, range(24), box=(100, 100, 100, 100))
    ts = resample_track(g, fps_src=24.0, frame_h=1080)
    assert len(ts) == int(np.floor((23 / 24) * TARGET_FPS + 1e-9)) + 1
    assert ts.detected.all()
    assert np.isfinite(ts.feats).all()


def test_joint_velocity_switch_adds_34():
    g = track_rows(1, range(30))
    assert (
        resample_track(g, 30.0, 1080, joint_velocity=True).feats.shape[1] == N_BASE + 34
    )


def test_track_series_skips_unconfirmed_and_keeps_ids():
    df = pd.concat([track_rows(3, range(30)), track_rows(-1, range(30))])
    out = track_series(df, 30.0, 1080, clip_id="c1")
    assert [(s.clip_id, s.track_id) for s in out] == [("c1", 3)]


@pytest.mark.parametrize(
    ("t0", "t1", "kept"),
    [(0.0, np.inf, 30), (0.5, np.inf, 15), (0.0, 0.2, 7), (5.0, 9.0, None)],
)
def test_trim_series(t0, t1, kept):
    ts = resample_track(track_rows(1, range(30)), 30.0, 1080)
    out = trim_series(ts, t0, t1)
    if kept is None:
        assert out is None
    else:
        assert len(out) == kept and out.times()[0] >= t0 - 1e-9


@pytest.mark.parametrize("fps", [24.0, 59.94, 60.0])
def test_constant_velocity_is_constant_at_any_source_fps(fps):
    g = track_rows(1, range(int(fps * 4)), box=(100, 100, 140, 180), dx_per_frame=2.0)
    ts = resample_track(g, fps_src=fps, frame_h=1080)
    v = ts.feats[1:, 56]
    assert np.allclose(v, 2.0 * fps / 80.0, rtol=1e-3), (v.min(), v.max())


def test_zero_size_box_keeps_pose_values_moderate():
    g = track_rows(1, range(30), box=(100, 100, 100, 100))
    g["x3"] = g["x3"] + 10.0
    ts = resample_track(g, fps_src=30.0, frame_h=1080)
    assert np.abs(ts.feats[:, :34]).max() <= 100.0


@pytest.mark.parametrize("mode", ["raw", "capped", "off"])
def test_rate_modes(mode):
    from hydro_knight.features.track_series import RATE_CAP, RATE_COLS

    g = track_rows(1, range(30), box=(100, 100, 140, 180), dx_per_frame=40.0)
    f = resample_track(g, 30.0, 1080, rates=mode).feats
    vx = f[1:, 56]
    if mode == "raw":
        assert np.allclose(vx, 15.0)
    elif mode == "capped":
        assert np.allclose(vx, RATE_CAP)
    else:
        assert (f[:, RATE_COLS] == 0).all()
    assert f.shape[1] == N_BASE


def test_unknown_rate_mode_is_refused():
    with pytest.raises(ValueError, match="rates"):
        resample_track(track_rows(1, range(5)), 30.0, 1080, rates="smooth")


def test_frame_step_2_at_59_94_fps_never_reads_a_skipped_frame():
    # frame_step 2: odd frames are not missing
    fps = 59.94
    g = track_rows(1, range(0, int(60 * fps), 2))
    ts = resample_track(g, fps_src=fps, frame_h=1080, frame_step=2)
    assert len(ts) > 1700 and ts.detected.all()


def test_frame_step_2_still_marks_real_gaps_missing():
    g = track_rows(1, [f for f in range(0, 240, 2) if not 100 <= f <= 140])
    ts = resample_track(g, fps_src=59.94, frame_h=1080, frame_step=2)
    gone = ~ts.detected
    assert gone.any() and not gone[:45].any() and not gone[-45:].any()
