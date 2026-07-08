"""
Tests for the detector-agnostic eval harness (eval/metrics.py).

These pin the behaviors the whole reporting layer depends on: the neutral
detections format, event catch/latency logic, alarm-episode merging, and the
recall/false-alarm sweep. All synthetic — no model, no footage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro_knight.eval.metrics import (
    ClipEval,
    _episodes,
    detections_from_windows,
    event_catches,
    false_alarm_episodes,
    label_detections,
    normal_hours,
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


def test_label_detections_time_overlap():
    # Event at 10-12 s. Window at frame 270 covers 9.0-10.07 s -> overlaps.
    # Window at frame 400 covers 13.3-14.4 s -> outside.
    clip = _clip(
        [(1, 270, 0.5), (1, 400, 0.5)],
        events=[{"start": 10.0, "end": 12.0, "label": "distress"}],
    )
    mask = label_detections(clip)
    assert mask.tolist() == [True, False]


def test_event_catch_and_latency():
    # Two above-threshold windows overlap the event; the FIRST one (frame 300
    # -> starts at 10.0 s, event onset 10.0 s) sets latency = 0.
    clip = _clip(
        [(1, 300, 0.9), (1, 330, 0.9), (1, 1200, 0.1)],
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
    )
    (ev,) = event_catches(clip, threshold=0.5)
    assert ev["caught"] is True
    assert ev["latency_s"] == 0.0


def test_event_miss_below_threshold():
    clip = _clip(
        [(1, 300, 0.2)],
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
    )
    (ev,) = event_catches(clip, threshold=0.5)
    assert ev["caught"] is False
    assert np.isnan(ev["latency_s"])


def test_latency_clipped_at_zero():
    # A window straddling the onset (starts before it) counts as latency 0,
    # not negative — overlap isn't precognition.
    clip = _clip(
        [(1, 280, 0.9)],  # 9.33-10.4 s vs onset 10.0
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
    )
    (ev,) = event_catches(clip, threshold=0.5)
    assert ev["caught"] is True
    assert ev["latency_s"] == 0.0


def test_episode_merging():
    # Firings at 1,2,3 s (one episode), then 10,10.5 s (second episode).
    times = np.array([1.0, 2.0, 3.0, 10.0, 10.5])
    eps = _episodes(times, merge_gap_s=2.0)
    assert eps == [(1.0, 3.0), (10.0, 10.5)]
    assert _episodes(np.array([]), 2.0) == []


def test_false_alarms_exclude_event_overlap():
    # One alarm burst inside the event (not false), one far outside (false).
    clip = _clip(
        [(1, 310, 0.9), (1, 320, 0.9), (2, 1500, 0.9)],
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
        duration=120.0,
    )
    assert false_alarm_episodes(clip, threshold=0.5) == 1


def test_normal_hours_subtracts_events():
    clip = _clip(
        [], events=[{"start": 0.0, "end": 360.0, "label": "distress"}], duration=3600.0
    )
    assert abs(normal_hours([clip]) - 0.9) < 1e-9


def test_sweep_recall_endpoints():
    # Threshold at min score -> everything fires -> recall 1. Above max -> 0.
    clip = _clip(
        [(1, 300, 0.9), (2, 1500, 0.1)],
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
        duration=120.0,
    )
    sweep = sweep_recall_fa([clip], thresholds=np.array([0.0, 0.5, 2.0]))
    rec = dict(zip(sweep["threshold"], sweep["recall"], strict=True))
    assert rec[0.0] == 1.0
    assert rec[0.5] == 1.0  # the 0.9 window still fires and overlaps the event
    assert rec[2.0] == 0.0


def test_split_scores_and_perfect_roc():
    # Distress windows all score higher than normal -> ROC-AUC must be 1.0.
    clip = _clip(
        [(1, 300, 0.9), (1, 330, 0.8), (2, 1500, 0.1), (2, 1600, 0.2)],
        events=[{"start": 10.0, "end": 14.0, "label": "distress"}],
    )
    err_n, err_d = split_scores([clip])
    assert len(err_n) == 2 and len(err_d) == 2
    res = roc_pr(err_n, err_d)
    assert res["roc_auc"] == 1.0
    # Degenerate inputs return None rather than crashing.
    assert roc_pr(np.array([]), err_d) is None
