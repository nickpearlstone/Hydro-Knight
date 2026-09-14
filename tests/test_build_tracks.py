"""
Tests for preprocess/build_tracks.py: merge + ByteTrack over saved raw detections.

Each fix gets a before/after pair: the default settings must behave correctly,
and LEGACY must still reproduce the original bug, so a comparison run is honest.
Scenarios are synthetic detection folders written with extract_pose.write_chunk.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hydro_knight.preprocess.build_tracks import (
    COLUMNS,
    LEGACY,
    TrackSettings,
    build_tracks,
    read_provenance,
    track_detections,
)
from hydro_knight.preprocess.extract_pose import DET_COLUMNS, write_chunk


def _det(frame, box, conf=0.9, crop=-1):
    row = {"frame": frame, "crop": crop, "conf": conf}
    row.update(dict(zip(["bx1", "by1", "bx2", "by2"], box, strict=True)))
    row.update({f"{a}{i}": 1.0 for i in range(17) for a in "xyc"})
    return row


def _frames(n):
    return pd.DataFrame({"frame": range(n), "n_dets": 0, "scene_diff": 0.0})


def _dets(rows):
    return pd.DataFrame(rows, columns=DET_COLUMNS)


SWIMMER = [100.0, 100.0, 140.0, 200.0]


def test_seam_half_box_does_not_become_a_second_swimmer():
    # Every frame: a full-body box from tile 0 and its top half from tile 1.
    half = [100.0, 100.0, 140.0, 140.0]
    rows = [_det(f, SWIMMER, 0.9, crop=0) for f in range(10)]
    rows += [_det(f, half, 0.8, crop=1) for f in range(10)]

    fixed = track_detections(_dets(rows), _frames(10), fps=30, settings=TrackSettings())
    legacy = track_detections(_dets(rows), _frames(10), fps=30, settings=LEGACY)

    assert len(fixed) == 10  # one row per frame: the half box was merged away
    assert fixed.loc[fixed["track_id"] >= 0, "track_id"].nunique() == 1
    assert legacy.loc[legacy["track_id"] >= 0, "track_id"].nunique() == 2  # the phantom


def test_short_submersion_keeps_the_same_id_at_60fps():
    # Visible for frames 0-19, gone for 45 frames (0.75 s at 60 fps), back at 65-79 in place.
    rows = [_det(f, SWIMMER) for f in list(range(20)) + list(range(65, 80))]
    frames = _frames(80)

    fixed = track_detections(_dets(rows), frames, fps=60, settings=TrackSettings())
    legacy = track_detections(_dets(rows), frames, fps=60, settings=LEGACY)

    confirmed = fixed[fixed["track_id"] >= 0]
    assert confirmed["track_id"].nunique() == 1  # 1.0 s buffer = 60 frames > 45
    before = legacy[(legacy["frame"] < 20) & (legacy["track_id"] >= 0)][
        "track_id"
    ].unique()
    after = legacy[(legacy["frame"] >= 65) & (legacy["track_id"] >= 0)][
        "track_id"
    ].unique()
    assert set(before).isdisjoint(after)  # 30-frame buffer: comes back as a new swimmer


def test_frames_without_detections_still_count_toward_lost_time():
    # Gone for 45 frames at 30 fps (1.5 s) exceeds the 1.0 s buffer, so the id must change.
    # This only holds if empty frames are fed to the tracker.
    rows = [_det(f, SWIMMER) for f in list(range(20)) + list(range(65, 80))]
    out = track_detections(_dets(rows), _frames(80), fps=30, settings=TrackSettings())
    ids = out[out["track_id"] >= 0].groupby(out["frame"] >= 65)["track_id"].unique()
    assert set(ids[False]).isdisjoint(ids[True])


def test_mid_confidence_swimmer_can_start_a_track():
    # 0.55 clears the new 0.5 bar but not the original 0.6, which made activation=0.3 dead.
    rows = [_det(f, SWIMMER, conf=0.55) for f in range(5)]
    fixed = track_detections(_dets(rows), _frames(5), fps=30, settings=TrackSettings())
    legacy = track_detections(_dets(rows), _frames(5), fps=30, settings=LEGACY)
    assert (fixed["track_id"] >= 0).any()
    assert (legacy["track_id"] == -1).all()


def test_build_tracks_writes_columns_boxes_and_provenance(tmp_path):
    det_dir = tmp_path / "clip"
    det_dir.mkdir()
    rows = [_det(f, SWIMMER) for f in range(4)]
    write_chunk(det_dir, 0, _dets(rows), _frames(4))
    meta = {
        "clip_id": "clip",
        "video": {"fps": 30.0, "width": 1280, "height": 720, "frame_count": 4},
        "settings": {"model": "yolo11n-pose.pt"},
        "chunks_done": 1,
        "frames_processed": 4,
        "complete": True,
    }
    (det_dir / "meta.json").write_text(json.dumps(meta))

    out = tmp_path / "keypoints" / "clip.parquet"
    df = build_tracks(det_dir, out)

    back = pd.read_parquet(out)
    assert list(back.columns) == COLUMNS
    assert len(back) == len(df) == 4
    assert np.allclose(back[["bx1", "by1", "bx2", "by2"]].iloc[-1], SWIMMER)
    prov = read_provenance(out)
    assert prov["track_settings"]["merge"] == "ios" and prov["clip_id"] == "clip"
