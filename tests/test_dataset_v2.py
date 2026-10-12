"""Dataset v2: only labeled, active clips; victim tracks removed; whole-clip split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from v2_fakes import track_rows, write_keypoints

from hydro_knight.features.dataset_v2 import build_datasets
from hydro_knight.ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Setting,
    TimeOfDay,
    Weather,
)

ONSET = {"t": 1.0, "x": 120.0, "y": 140.0, "kind": "onset"}
LAST = {"t": 2.0, "x": 120.0, "y": 140.0, "kind": "last_seen"}
FULL = {
    "start": 1.0,
    "end": 5.0,
    "label": "distress",
    "contact": 3.0,
    "victim": [ONSET, LAST],
}


def _rec(cid, events, notes=""):
    return ClipRecord(
        clip_id=cid,
        source_url=f"u/{cid}",
        platform="youtube",
        start_sec=0.0,
        end_sec=-1.0,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.DISTRESS,
        notes=notes,
        fps=30.0,
        events=events,
    )


@pytest.fixture
def kp_dir(tmp_path):
    for cid in ("lab1", "lab2", "lab3", "unlab", "held"):
        df = pd.concat(
            [
                track_rows(7, range(0, 300), box=(100, 100, 140, 180)),  # victim
                track_rows(
                    9, range(0, 300), box=(300, 100, 340, 180)
                ),  # normal swimmer
            ]
        )
        write_keypoints(tmp_path / f"{cid}.parquet", df, fps=30.0)
    return tmp_path


RECORDS = [
    _rec("lab1", [FULL]),
    _rec("lab2", [FULL]),
    _rec("lab3", [FULL]),
    _rec("unlab", [{"start": 1.0, "end": 5.0, "label": "distress"}]),
    _rec("held", [FULL], notes="[HOLD: test]"),
    _rec("nofile", [FULL]),
]


def test_only_labeled_active_clips_and_no_victim_tracks(kp_dir):
    train, test, rep = build_datasets(kp_dir, RECORDS, test_frac=0.5, seed=0)
    assert rep.used_train and rep.used_test  # both sides exercised
    assert sorted(rep.used_train + rep.used_test) == ["lab1", "lab2", "lab3"]
    assert (
        rep.unlabeled == ["unlab"]
        and rep.held == ["held"]
        and rep.missing == ["nofile"]
    )
    assert rep.victim_tracks == {"lab1": [7], "lab2": [7], "lab3": [7]}
    for ws in (train, test):
        assert {s.track_id for s in ws.series} == {9}
    assert {s.clip_id for s in train.series}.isdisjoint(
        {s.clip_id for s in test.series}
    )


def test_trim_limits_the_windows(kp_dir):
    trimmed = _rec("lab1", [FULL])
    trimmed.trim_end = 4.0  # 4 s of a 10 s track -> scored windows end by 4 s
    train, test, _ = build_datasets(kp_dir, [trimmed], test_frac=0.0)
    ends = [train.info(i)[2] + 3.0 for i in range(len(train))]
    assert len(train) > 0 and max(ends) <= 4.0 + 1e-6


def test_no_labeled_clips_is_an_error_unless_smoke(kp_dir):
    unlabeled = [_rec("unlab", [{"start": 1.0, "end": 5.0, "label": "distress"}])]
    with pytest.raises(ValueError, match="label"):
        build_datasets(kp_dir, unlabeled)
    train, _, rep = build_datasets(
        kp_dir, unlabeled, require_labels=False, test_frac=0.0
    )
    assert rep.used_train == ["unlab"] and len(train) > 0


def test_reviewed_wrong_clips_are_skipped(kp_dir):
    _, _, rep = build_datasets(
        kp_dir, RECORDS, test_frac=0.5, seed=0, exclude_clips={"lab2"}
    )
    assert rep.rejected == ["lab2"]
    assert "lab2" not in rep.used_train + rep.used_test


def test_victim_windows_are_the_removed_tracks_inside_the_event(kp_dir):
    from hydro_knight.features.dataset_v2 import victim_windows

    _, test, rep = build_datasets(kp_dir, RECORDS, test_frac=0.5, seed=0)
    vic, in_event = victim_windows(kp_dir, RECORDS, rep)
    assert len(vic) and {s.track_id for s in vic.series} == {7}
    starts = np.array([vic.info(i)[2] for i in range(len(vic))])
    # event onset 1.0 s -> contact 3.0 s; a 3 s window overlaps it if it starts < 3.0
    assert in_event.any() and (starts[in_event] < 3.0).all()
    assert not in_event[starts >= 3.0].any()
