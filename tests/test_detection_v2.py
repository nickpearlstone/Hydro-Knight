"""v2 detection metrics: causal alarm times, pre-contact scope, catches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from v2_fakes import series_of

from hydro_knight.eval.detection_v2 import alarm_table, clip_evals, operating_points
from hydro_knight.eval.metrics import event_catches
from hydro_knight.features.track_series import TrackSeries
from hydro_knight.features.windows_v2 import WindowSet, make_window_set


def _ws(track_id, n=300, clip="c"):
    feats, det = series_of(n, seed=track_id)
    return make_window_set([TrackSeries(clip, track_id, 0.0, feats, det)], window=90)


def _record(onset=5.0, contact=8.0):
    ev = {"start": onset, "contact": contact, "end": 9.0, "label": "distress"}
    return SimpleNamespace(
        clip_id="c", events=[ev], trim_start=None, to_file_time=lambda t: t
    )


def test_alarm_fires_after_the_window_and_its_context():
    ws = _ws(1)
    a = alarm_table(ws, np.zeros(len(ws)))
    assert np.isclose(a["t"].iloc[0], (90 + ws.context) / 30.0)  # first window, t0 = 0


def test_only_pre_contact_alarms_and_victim_catch():
    normal, victim = _ws(1), _ws(7)
    s_n, s_v = np.zeros(len(normal)), np.zeros(len(victim))
    av = alarm_table(victim, s_v)
    s_v[((av["t"] > 5.0) & (av["t"] < 8.0)).to_numpy()] = 10.0  # victim fires in event
    alarms = __import__("pandas").concat(
        [alarm_table(normal, s_n), alarm_table(victim, s_v)]
    )
    rep = SimpleNamespace(used_test=["c"], victim_tracks={"c": [7]})
    (clip,) = clip_evals(alarms, [_record()], rep)
    assert clip.duration_s == 8.0
    assert (clip.detections["frame"] / 30.0 <= 8.0 + 1e-9).all()
    (e,) = event_catches(clip, threshold=5.0)
    assert e["caught"] and 0.0 <= e["latency_s"] < 3.0
    pts = operating_points([clip], fa_targets=(0,))["points"]
    assert pts[0]["recall"] == 1.0 and pts[0]["fa_per_hour"] == 0.0


def test_victim_with_no_windows_is_a_miss():
    normal = _ws(1)
    alarms = alarm_table(normal, np.ones(len(normal)))
    rep = SimpleNamespace(used_test=["c"], victim_tracks={"c": []})
    (clip,) = clip_evals(alarms, [_record()], rep)
    assert not event_catches(clip, 0.0)[0]["caught"]


def test_empty_window_set_gives_empty_alarms():
    empty = WindowSet([], np.empty((0, 2), np.int64), 90, 44)
    assert alarm_table(empty, np.empty(0)).empty
