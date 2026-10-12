"""Window set v2: scored windows inside the track, real context, missing-step padding."""

from __future__ import annotations

import numpy as np
from v2_fakes import series_of

from hydro_knight.features.track_series import N_BASE, TrackSeries, zero_when_missing
from hydro_knight.features.windows_v2 import (
    WindowSet,
    context_for,
    dilations_for,
    make_window_set,
)


def _ts(n, track_id=1, clip="c", detected=None):
    feats, det = series_of(n, seed=track_id, detected=detected)
    return TrackSeries(clip, track_id, 0.0, feats, det)


def test_dilations_match_the_window_and_start_at_one():
    assert dilations_for(90) == [1, 2, 4, 8, 16, 13]
    assert context_for(90) == 44  # reach 89 <= 90
    for w in (30, 60, 90, 120):
        d = dilations_for(w)
        assert d[0] == 1 and 2 * sum(d) + 1 <= w


def test_scored_windows_stay_inside_the_track():
    ws = make_window_set([_ts(200), _ts(80, track_id=2)], window=90)
    # 200 steps: starts 0, 7.5, 15 … rounded, last start <= 110; the 80-step track gives none
    starts = ws.index[:, 1]
    assert (ws.index[:, 0] == 0).all() and starts.max() <= 110 and starts.min() == 0
    assert len(ws) == len(np.unique(np.rint(np.arange(0, 110 + 1e-9, 7.5))))
    assert ws.info(0) == ("c", 1, 0.0)


def test_context_is_real_inside_the_track_and_missing_beyond_it():
    ts = _ts(200)
    ws = make_window_set([ts], window=90)
    c = ws.context
    x, m = ws.gather([0])  # first scored step 0: left context is beyond the track start
    assert x.shape == (1, 90 + 2 * c, N_BASE)
    assert not m[0, :c].any() and m[0, c:].all()
    assert (x[0, :c][:, zero_when_missing(N_BASE)] == 0).all()
    assert (x[0, :c, :34] == ts.feats[0, :34]).all()  # coords held from the edge step
    mid = int(np.argmin(np.abs(ws.index[:, 1] - 60)))
    a = ws.index[mid, 1]
    x, m = ws.gather([mid])  # a middle window: context is real track data
    assert np.array_equal(x[0], ts.feats[a - c : a + 90 + c])


def test_concat_keeps_series_and_offsets():
    a = make_window_set([_ts(200, clip="a")])
    b = make_window_set([_ts(150, clip="b", track_id=5)])
    both = WindowSet.concat([a, b])
    assert len(both) == len(a) + len(b)
    assert both.info(len(a))[0:2] == ("b", 5)
    assert np.array_equal(both.gather([len(a)])[0], b.gather([0])[0])


def test_short_tracks_get_one_centered_padded_window_when_allowed():
    long, short, tiny = _ts(200), _ts(40, track_id=2), _ts(20, track_id=3)
    off = make_window_set([long, short, tiny], window=90)
    on = make_window_set([long, short, tiny], window=90, min_track_s=1.0)
    # 1 s = 30 steps: the 40-step track gets one window, the 20-step one does not
    assert len(on) == len(off) + 1
    s, a = on.index[-1]
    assert s == 1 and a == -25  # (90 - 40) // 2 padding before the track
    x, m = on.gather([len(on) - 1])
    c = on.context
    scored = m[0, c : c + 90]
    assert scored.sum() == 40 and scored[25:65].all()  # real steps in the middle
    pad = ~m[0]
    assert (x[0][pad][:, zero_when_missing(N_BASE)] == 0).all()
