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


def test_fragment_between_clicks_is_found_and_bystander_is_not():
    # Victim path: (120,140) at 1 s -> (220,140) at 3 s. Track 7 is under the first
    # click; track 11 is the same victim later (between clicks, never clicked);
    # track 9 swims far away.
    from hydro_knight.eval.victims import near_path_tracks

    frag = track_rows(11, range(45, 76), box=(150, 100, 190, 180))
    df = pd.concat([DF, frag])
    clicks = [
        {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"},
        {"t": 3.0, "x": 220.0, "y": 140.0, "kind": "last_seen"},
    ]
    rec = _rec(clicks)
    m = match_victim_tracks(df, rec, fps=30.0)
    assert m.tracks == {7}
    assert near_path_tracks(df, rec, fps=30.0, exclude=m.tracks) == {11}


def test_fragment_needs_min_time_near_the_path():
    from hydro_knight.eval.victims import near_path_tracks

    blip = track_rows(11, range(60, 65), box=(150, 100, 190, 180))  # 5 frames < 0.3 s
    clicks = [
        {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"},
        {"t": 3.0, "x": 220.0, "y": 140.0, "kind": "last_seen"},
    ]
    assert near_path_tracks(pd.concat([DF, blip]), _rec(clicks), 30.0, {7}) == set()


# A crowd of 80-px swimmers around y=140, plus a lifeguard in the foreground whose
# 900-px box covers the victim clicks (the 7096dfde4791 failure).
CROWD = pd.concat(
    [
        track_rows(20 + i, range(0, 90), box=(400 + 60 * i, 100, 440 + 60 * i, 180))
        for i in range(4)
    ]
)
GUARD = track_rows(99, range(0, 90), box=(0, 0, 600, 900))


def test_oversized_box_containing_the_click_is_not_the_victim():
    click = {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"}
    m = match_victim_tracks(pd.concat([DF, CROWD, GUARD]), _rec([click]), fps=30.0)
    assert m.tracks == {7}
    assert m.oversize == {99}


def test_oversized_box_alone_leaves_the_click_unmatched():
    # the victim itself was not detected: the guard must not stand in for it
    click = {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"}
    m = match_victim_tracks(pd.concat([CROWD, GUARD]), _rec([click]), fps=30.0)
    assert m.tracks == set() and m.unmatched == [click]


def test_oversized_box_along_the_path_is_not_a_fragment():
    from hydro_knight.eval.victims import near_path_tracks

    clicks = [
        {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"},
        {"t": 3.0, "x": 220.0, "y": 140.0, "kind": "last_seen"},
    ]
    frag = track_rows(11, range(45, 76), box=(150, 100, 190, 180))
    df = pd.concat([DF, CROWD, GUARD, frag])
    assert near_path_tracks(df, _rec(clicks), 30.0, {7}) == {11}


def test_track_that_is_oversized_most_of_its_life_is_not_the_victim():
    # the guard's box shrinks to swimmer size only at the click (e.g. crouching)
    click = {"t": 2.0, "x": 120.0, "y": 140.0, "kind": "last_seen"}
    guard = pd.concat(
        [
            track_rows(99, range(0, 55), box=(0, 0, 600, 900)),
            track_rows(99, range(55, 65), box=(100, 100, 140, 180)),
        ]
    )
    m = match_victim_tracks(pd.concat([CROWD, guard]), _rec([click]), fps=30.0)
    assert m.tracks == set() and 99 in m.oversize


def test_fragment_time_counts_skipped_frames():
    # frame_step 2: 5 processed frames span ~0.33 s of real time, enough for min_s 0.3.
    from hydro_knight.eval.victims import near_path_tracks

    frag = track_rows(11, range(60, 70, 2), box=(150, 100, 190, 180))
    clicks = [
        {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"},
        {"t": 3.0, "x": 220.0, "y": 140.0, "kind": "last_seen"},
    ]
    df, rec = pd.concat([DF, frag]), _rec(clicks)
    assert near_path_tracks(df, rec, 30.0, {7}) == set()
    assert near_path_tracks(df, rec, 30.0, {7}, frame_step=2) == {11}
