"""Track series v2: 30 fps resampling, missing steps, box normalization, rates."""

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
    # frames 0..59 at 60 fps = 1 s -> 30 steps; source frames 20-29 missing.
    g = track_rows(1, [f for f in range(60) if not 20 <= f <= 29])
    ts = resample_track(g, fps_src=60.0, frame_h=1080)
    assert len(ts) == 30 and ts.feats.shape == (30, N_BASE)
    # step k reads source frame 2k, so steps 10..14 are missing
    assert ts.detected.tolist() == [not 10 <= k <= 14 for k in range(30)]
    missing = ~ts.detected
    assert (ts.feats[missing][:, zero_when_missing(N_BASE)] == 0).all()
    assert (ts.feats[missing][:, :34] == ts.feats[9, :34]).all()  # coords held
    assert (ts.feats[:, DETECTED] == ts.detected).all()


def test_box_normalization_size_aspect_and_velocity():
    # 40x80 box moving right 2 px per source frame at 60 fps = 120 px/s = 1.5 box heights/s
    g = track_rows(1, range(60), box=(100, 100, 140, 180), dx_per_frame=2.0)
    ts = resample_track(g, fps_src=60.0, frame_h=1080)
    f = ts.feats
    assert np.allclose(f[:, :34], 0.0)  # every keypoint at the box center
    assert np.allclose(f[:, 53], 0.5)  # aspect w/h = 40/80
    assert np.allclose(f[:, 55], 80 / 1080)  # box size
    assert np.allclose(f[1:, 56], 1.5) and np.allclose(f[:, 57], 0.0)
    assert np.allclose(f[:, 54], 0.0)  # area constant
    assert f[0, 56] == 0.0  # first step has no previous detection


def test_24fps_upsampling_and_degenerate_box_stay_finite():
    g = track_rows(
        1, range(24), box=(100, 100, 100, 100)
    )  # zero-size box, 1 s at 24 fps
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
    ts = resample_track(track_rows(1, range(30)), 30.0, 1080)  # 1 s, steps at k/30 s
    out = trim_series(ts, t0, t1)
    if kept is None:
        assert out is None
    else:
        assert len(out) == kept and out.times()[0] >= t0 - 1e-9


@pytest.mark.parametrize("fps", [24.0, 59.94, 60.0])
def test_constant_velocity_is_constant_at_any_source_fps(fps):
    # 2 px per source frame, 80 px box: true speed = 2 * fps / 80 box heights per second
    g = track_rows(1, range(int(fps * 4)), box=(100, 100, 140, 180), dx_per_frame=2.0)
    ts = resample_track(g, fps_src=fps, frame_h=1080)
    v = ts.feats[1:, 56]
    assert np.allclose(v, 2.0 * fps / 80.0, rtol=1e-3), (v.min(), v.max())


def test_zero_size_box_keeps_pose_values_moderate():
    g = track_rows(1, range(30), box=(100, 100, 100, 100))
    g["x3"] = g["x3"] + 10.0  # one joint 10 px off a zero-size box
    ts = resample_track(g, fps_src=30.0, frame_h=1080)
    assert np.abs(ts.feats[:, :34]).max() <= 100.0


@pytest.mark.parametrize("mode", ["raw", "capped", "off"])
def test_rate_modes(mode):
    from hydro_knight.features.track_series import RATE_CAP, RATE_COLS

    # 40 px/frame at 30 fps with an 80 px box = 15 box heights/s: pure box jump speed.
    g = track_rows(1, range(30), box=(100, 100, 140, 180), dx_per_frame=40.0)
    f = resample_track(g, 30.0, 1080, rates=mode).feats
    vx = f[1:, 56]
    if mode == "raw":
        assert np.allclose(vx, 15.0)
    elif mode == "capped":
        assert np.allclose(vx, RATE_CAP)
    else:
        assert (f[:, RATE_COLS] == 0).all()
    assert f.shape[1] == N_BASE  # indices never move


def test_unknown_rate_mode_is_refused():
    with pytest.raises(ValueError, match="rates"):
        resample_track(track_rows(1, range(5)), 30.0, 1080, rates="smooth")


def test_frame_step_2_at_59_94_fps_never_reads_a_skipped_frame():
    # 60 s of a 59.94 fps clip extracted at frame_step 2: only even frames exist, all
    # detected. Nearest-frame rounding drifts onto odd (never processed) frames, which
    # must not read as "missing".
    fps = 59.94
    g = track_rows(1, range(0, int(60 * fps), 2))
    ts = resample_track(g, fps_src=fps, frame_h=1080, frame_step=2)
    assert len(ts) > 1700 and ts.detected.all()


def test_frame_step_2_still_marks_real_gaps_missing():
    g = track_rows(1, [f for f in range(0, 240, 2) if not 100 <= f <= 140])
    ts = resample_track(g, fps_src=59.94, frame_h=1080, frame_step=2)
    gone = ~ts.detected
    assert gone.any() and not gone[:45].any() and not gone[-45:].any()
