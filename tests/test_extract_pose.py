"""
Tests for extract_raw() in preprocess/extract_pose.py, with a fake YOLO and a fake video.

extract_raw is Colab-GPU code: it needs real video and weights, so it never runs
locally, and its worst failure is silent — output that looks valid but is empty
or incomplete. 10 clips from the first extraction came back as empty keypoint
files, and nothing on disk could say whether that meant "no swimmers" or "the
run broke". These tests pin the guarantees that make that distinguishable:
a frame row for every frame read (even with zero detections), a completion flag,
and resume without gaps or duplicates.

Uses pytest's built-in monkeypatch/tmp_path fixtures.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from yolo_fakes import Result as _Result

from hydro_knight.preprocess import extract_pose
from hydro_knight.preprocess.extract_pose import (
    DET_COLUMNS,
    FRAME_COLUMNS,
    extract_raw,
    load_detections,
)


def _install_fakes(monkeypatch, n_frames: int, people_per_frame=None, crash_at=None):
    """Fake cv2.VideoCapture (n_frames 720p frames) and a YOLO that finds people_per_frame[i]
    swimmers in the whole-frame pass of frame i (tiles find nothing). crash_at raises on that frame.
    Returns a dict counting YOLO constructions.
    """
    people = people_per_frame or [1] * n_frames
    state = {"frame": -1, "models": 0}

    class _FakeCapture:
        def __init__(self, path):
            self.i = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                extract_pose.cv2.CAP_PROP_FRAME_WIDTH: 1280,
                extract_pose.cv2.CAP_PROP_FRAME_HEIGHT: 720,
                extract_pose.cv2.CAP_PROP_FPS: 60.0,
                extract_pose.cv2.CAP_PROP_FRAME_COUNT: n_frames,
            }[prop]

        def _next(self):
            if self.i >= n_frames:
                return False, None
            state["frame"] = self.i
            img = np.full((720, 1280, 3), self.i % 255, np.uint8)
            self.i += 1
            return True, img

        def read(self):
            return self._next()

        def grab(self):
            return self._next()[0]

        def retrieve(self):
            return True, np.full((720, 1280, 3), (self.i - 1) % 255, np.uint8)

        def release(self):
            pass

    class _FakeYOLO:
        def __init__(self, name):
            state["models"] += 1

        def __call__(self, source, **kw):
            if isinstance(source, list):  # tiles: nothing
                return [_Result(np.empty((0, 4)), []) for _ in source]
            f = state["frame"]
            if crash_at is not None and f == crash_at:
                raise RuntimeError("simulated Colab disconnect")
            n = people[f]
            boxes = [[100.0 * k, 100.0, 100.0 * k + 40, 200.0] for k in range(n)]
            return [_Result(np.array(boxes).reshape(n, 4), [0.9] * n)]

    monkeypatch.setattr(extract_pose.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(extract_pose, "YOLO", _FakeYOLO)
    monkeypatch.setattr(extract_pose, "_versions", lambda: {})
    return state


def test_every_frame_gets_a_row_even_with_no_detections(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 5, people_per_frame=[2, 0, 1, 0, 3])

    meta = extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", chunk_frames=2)

    meta, dets, frames = load_detections(tmp_path / "clip")
    assert meta["complete"] and meta["frames_processed"] == 5
    assert frames["frame"].tolist() == [0, 1, 2, 3, 4]
    assert frames["n_dets"].tolist() == [2, 0, 1, 0, 3]
    assert len(dets) == 6 and list(dets.columns) == DET_COLUMNS
    assert list(frames.columns) == FRAME_COLUMNS
    assert meta["chunks_done"] == 3  # chunks of 2, 2, 1


def test_detections_keep_crop_ids_and_storage_dtypes(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 1, people_per_frame=[1])
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip")

    raw = pd.read_parquet(tmp_path / "clip" / "det-00000.parquet")
    assert raw["crop"].tolist() == [-1]  # found by the whole-frame pass
    assert raw["bx1"].dtype == np.float32 and raw["x0"].dtype == np.float16
    assert raw[["bx1", "by1", "bx2", "by2"]].iloc[0].tolist() == [0, 100, 40, 200]


def test_scene_diff_is_nan_on_first_frame_then_measured(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 3)
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip")
    _, _, frames = load_detections(tmp_path / "clip")
    assert np.isnan(frames["scene_diff"].iloc[0])
    assert frames["scene_diff"].iloc[1:].notna().all()


def test_crash_then_rerun_resumes_without_gaps_or_duplicates(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 7, crash_at=5)
    with pytest.raises(RuntimeError):
        extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", chunk_frames=2)
    meta = json.loads((tmp_path / "clip" / "meta.json").read_text())
    assert (
        not meta["complete"] and meta["frames_processed"] == 4
    )  # frame 4 was in the lost chunk

    _install_fakes(monkeypatch, 7)
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", chunk_frames=2)

    meta, dets, frames = load_detections(tmp_path / "clip")
    assert meta["complete"]
    assert frames["frame"].tolist() == list(range(7))
    assert sorted(dets["frame"].tolist()) == list(range(7))
    assert (
        frames["scene_diff"].iloc[1:].notna().all()
    )  # resume recovered the previous frame


def test_complete_folder_is_skipped_without_loading_the_model(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 3)
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip")

    state = _install_fakes(monkeypatch, 3)
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip")
    assert state["models"] == 0


def test_different_settings_refuse_to_mix_into_an_existing_folder(
    monkeypatch, tmp_path
):
    _install_fakes(monkeypatch, 3)
    extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", conf=0.25)
    with pytest.raises(ValueError, match="different settings"):
        extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", conf=0.10)


def test_incomplete_folder_is_rejected_by_default(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, 4, crash_at=3)
    with pytest.raises(RuntimeError):
        extract_raw(tmp_path / "clip.mp4", tmp_path / "clip", chunk_frames=2)
    with pytest.raises(RuntimeError, match="not complete"):
        load_detections(tmp_path / "clip")
