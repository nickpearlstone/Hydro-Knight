"""
Tests for the detector-agnostic eval harness (eval/metrics.py).

These pin the behaviors the whole reporting layer depends on: the neutral
detections format, event catch/latency logic, alarm-episode merging, and the
recall/false-alarm sweep. All synthetic — no model, no footage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydro_knight.eval.metrics import (
    ClipEval,
    _episodes,
    classify_detections,
    detections_from_windows,
    event_catches,
    false_alarm_episodes,
    falsely_flagged_tracks,
    footage_hours,
    is_scorable,
    label_detections,
    resolve_events,
    roc_pr,
    split_scores,
    sweep_recall_fa,
)

FPS = 30.0


def _clip(dets: list[tuple[int, int, float]], events=None, duration=60.0) -> ClipEval:
    """Clip from (track_id, frame, score) triples; windows span 32 frames."""
    df = pd.DataFrame(
        [{"track_id": t, "frame": f, "score": s, "span": 32} for t, f, s in dets],
        columns=["track_id", "frame", "score", "span"],
    )
    return ClipEval("testclip", df, events or [], FPS, duration)


def test_detections_from_windows_adapter():
    info = [(1, 0), (1, 8), (2, 0)]
    errors = np.array([0.1, 0.2, 0.3])
    det = detections_from_windows(info, errors, span=32)
    assert list(det.columns) == ["track_id", "frame", "score", "span"]
    assert len(det) == 3
    assert det["span"].tolist() == [32, 32, 32]
    # Empty input must still return a well-formed (empty) table.
    assert len(detections_from_windows([], np.array([]))) == 0


# Event: onset 10 s, guard contact 12 s, saved 14 s; the victim is track 1.
EV = {
    "start": 10.0,
    "contact": 12.0,
    "end": 14.0,
    "label": "distress",
    "victim_tracks": [1],
}


def test_only_the_victim_in_distress_is_positive():
    # Track 1 is the victim: frame 270 (9.0-10.07 s) overlaps onset..contact -> positive;
    # frame 390 (13.0-14.07 s) is the rescue after contact -> excluded. Track 2 swims
    # through the same seconds -> normal, not distress.
    clip = _clip([(1, 270, 0.5), (1, 390, 0.5), (2, 300, 0.5)], events=[EV])
    assert classify_detections(clip).tolist() == [1, -1, 0]
    assert label_detections(clip).tolist() == [True, False, False]


def test_catch_needs_the_victim_track_before_contact():
    # Another swimmer firing during the event is not a catch.
    other = _clip([(2, 300, 0.9)], events=[EV])
    assert event_catches(other, threshold=0.5)[0]["caught"] is False
    # The victim firing only after contact is the rescue, not a catch.
    late = _clip([(1, 375, 0.9)], events=[EV])  # 12.5-13.57 s
    assert event_catches(late, threshold=0.5)[0]["caught"] is False
    hit = _clip([(1, 300, 0.9), (1, 330, 0.9)], events=[EV])
    (ev,) = event_catches(hit, threshold=0.5)
    assert ev["caught"] is True and ev["latency_s"] == 0.0


def test_event_miss_below_threshold():
    (ev,) = event_catches(_clip([(1, 300, 0.2)], events=[EV]), threshold=0.5)
    assert ev["caught"] is False
    assert np.isnan(ev["latency_s"])


def test_latency_clipped_at_zero():
    # A window straddling the onset counts as latency 0, not negative.
    (ev,) = event_catches(_clip([(1, 280, 0.9)], events=[EV]), threshold=0.5)
    assert ev["caught"] is True
    assert ev["latency_s"] == 0.0


def test_episode_merging():
    # Firings at 1,2,3 s (one episode), then 10,10.5 s (second episode).
    times = np.array([1.0, 2.0, 3.0, 10.0, 10.5])
    eps = _episodes(times, merge_gap_s=2.0)
    assert eps == [(1.0, 3.0), (10.0, 10.5)]
    assert _episodes(np.array([]), 2.0) == []


def test_every_firing_swimmer_is_its_own_false_alarm():
    # The old count merged all tracks into one clip-wide episode that touched the
    # event, so 20 swimmers firing nonstop scored 0 false alarms.
    dets = [(t, f, 0.9) for t in range(2, 22) for f in range(0, 3600, 8)]
    clip = _clip(dets, events=[EV], duration=120.0)
    assert false_alarm_episodes(clip, threshold=0.5) == 20


def test_victim_alarms_before_onset_are_false_after_onset_are_not():
    clip = _clip(
        [
            (1, 60, 0.9),
            (1, 300, 0.9),
            (1, 380, 0.9),
        ],  # 2 s, 10 s (distress), 12.7 s (rescue)
        events=[EV],
        duration=120.0,
    )
    assert false_alarm_episodes(clip, threshold=0.5) == 1


def test_falsely_flagged_swimmers_never_drop_as_threshold_drops():
    # Episodes may fall (a flickering alarm merges into one); flagged swimmers can't.
    rng = np.random.default_rng(0)
    dets = [
        (t, f, float(rng.random())) for t in range(1, 15) for f in range(0, 1800, 8)
    ]
    clip = _clip(dets, events=[EV], duration=60.0)
    flagged = [falsely_flagged_tracks(clip, thr) for thr in (0.99, 0.9, 0.5, 0.1)]
    assert flagged == sorted(flagged)
    assert flagged[-1] == 14  # all 14, the victim too: it also fired before onset


def test_clips_without_matched_victims_cannot_be_scored():
    unmatched = _clip([(1, 300, 0.9)], events=[{"start": 10.0, "end": 14.0}])
    assert not is_scorable(unmatched)
    assert is_scorable(_clip([], events=[EV])) and is_scorable(_clip([]))
    with pytest.raises(ValueError, match="victim_tracks"):
        event_catches(unmatched, threshold=0.5)


def test_footage_hours_counts_whole_clips():
    assert abs(footage_hours([_clip([], events=[EV], duration=3600.0)]) - 1.0) < 1e-9


def test_sweep_recall_endpoints():
    clip = _clip([(1, 300, 0.9), (2, 1500, 0.1)], events=[EV], duration=120.0)
    sweep = sweep_recall_fa([clip], thresholds=np.array([0.0, 0.5, 2.0]))
    rec = dict(zip(sweep["threshold"], sweep["recall"], strict=True))
    assert rec[0.0] == 1.0
    assert rec[0.5] == 1.0  # the victim's 0.9 window still fires before contact
    assert rec[2.0] == 0.0


def test_split_scores_excludes_trained_on():
    # Rows flagged trained_on must not leak into the ROC/percentile pools —
    # scoring a model on its own training windows flatters the normal class.
    clip = _clip([(1, 300, 0.9), (2, 1500, 0.1), (2, 1600, 0.2)])
    clip.detections["trained_on"] = [False, True, False]
    err_n, err_d = split_scores([clip])
    assert sorted(err_n.tolist()) == [0.2, 0.9]  # the 0.1 trained-on row is gone
    assert len(err_d) == 0


def test_split_scores_and_perfect_roc():
    # Distress windows all score higher than normal -> ROC-AUC must be 1.0.
    clip = _clip(
        [(1, 300, 0.9), (1, 330, 0.8), (2, 1500, 0.1), (2, 1600, 0.2)],
        events=[EV],
    )
    err_n, err_d = split_scores([clip])
    assert len(err_n) == 2 and len(err_d) == 2
    res = roc_pr(err_n, err_d)
    assert res["roc_auc"] == 1.0
    # Degenerate inputs return None rather than crashing.
    assert roc_pr(np.array([]), err_d) is None


def test_resolve_events_closes_off_camera_saves_and_skips_unlabeled_onsets():
    events = [
        {"start": 5.0, "end": None, "saved_off_camera": True},  # saved after the clip
        {"start": 1.0, "end": 3.0},
        {"start": None, "end": None, "contact": 2.0},  # onset not marked yet
    ]
    out = resolve_events(events, clip_end_s=40.0)
    assert [(e["start"], e["end"]) for e in out] == [(5.0, 40.0), (1.0, 3.0)]
    assert events[0]["end"] is None  # input untouched
