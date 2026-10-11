"""Victim clicks -> tracks, by position and time."""

from __future__ import annotations

import pandas as pd
from v2_fakes import track_rows

from hydro_knight.eval.victims import is_labeled, match_victim_tracks
from hydro_knight.ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Setting,
    TimeOfDay,
    Weather,
)


def _rec(victim, start_sec=0.0, contact=3.0):
    ev = {
        "start": 1.0,
        "end": 5.0,
        "label": "distress",
        "contact": contact,
        "victim": victim,
    }
    return ClipRecord(
        clip_id="c",
        source_url="u",
        platform="youtube",
        start_sec=start_sec,
        end_sec=-1.0,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.DISTRESS,
        fps=30.0,
        events=[ev],
    )


DF = pd.concat(
    [
        track_rows(7, range(0, 90), box=(100, 100, 140, 180)),
        track_rows(9, range(0, 90), box=(300, 100, 340, 180)),
    ]
)


def test_click_inside_a_box_picks_that_track():
    click = {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"}
    m = match_victim_tracks(DF, _rec([click]), fps=30.0)
    assert m.tracks == {7} and m.unmatched == []


def test_click_with_no_detection_nearby_is_unmatched():
    gone = {
        "t": 10.0,
        "x": 120.0,
        "y": 140.0,
        "kind": "last_seen",
    }  # after both tracks end
    far = {"t": 1.0, "x": 900.0, "y": 900.0, "kind": "extra"}
    m = match_victim_tracks(DF, _rec([gone, far]), fps=30.0)
    assert m.tracks == set() and m.unmatched == [gone, far]


def test_section_download_converts_source_time():
    # local file starts at source 100 s: a click at source 101 s is file frame 30
    click = {"t": 101.0, "x": 320.0, "y": 140.0, "kind": "onset"}
    m = match_victim_tracks(DF, _rec([click], start_sec=100.0), fps=30.0)
    assert m.tracks == {9}


def test_is_labeled_needs_contact_and_both_clicks():
    onset = {"t": 1.0, "x": 1.0, "y": 1.0, "kind": "onset"}
    last = {"t": 2.0, "x": 1.0, "y": 1.0, "kind": "last_seen"}
    assert is_labeled(_rec([onset, last]))
    assert not is_labeled(_rec([onset]))
    assert not is_labeled(_rec([onset, last], contact=None))
