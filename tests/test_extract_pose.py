"""
Tests for extract() in preprocess/extract_pose.py, using a fake YOLO stream.

extract() is Colab-GPU code: it needs real video and real weights, so it never
runs locally and its failure mode is the worst kind — a valid-looking Parquet
with zero rows. That is indistinguishable from "the footage had no swimmers",
which is a real condition in this dataset (10 clips already extracted empty).

The specific trap: model.track(..., stream=True) returns a *generator*, which
yields each frame once and remembers nothing. Anything that walks it before the
extraction loop (a progress-bar pre-pass, a frame count, a peek at the first
result) silently drains it, and the real loop then iterates an empty stream and
writes nothing. No exception, no warning.

These tests stand in a fake YOLO whose track() returns a genuine generator, so
that one-shot behavior is reproduced rather than imitated. Row counts are the
assertion that matters: a drained stream shows up as 0 rows and nothing else.

Uses pytest's built-in monkeypatch/tmp_path fixtures — requested by parameter
name, so no import is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro_knight.preprocess import extract_pose
from hydro_knight.preprocess.extract_pose import COLUMNS, extract


class _FakeTensor:
    """
    Stands in for the torch tensors YOLO hangs off a Results object. extract()
    reaches through .int()/.cpu() before .tolist()/.numpy(), so those are no-ops
    that return self.
    """

    def __init__(self, values):
        self._values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)

    def numpy(self):
        return np.asarray(self._values, dtype=float)


class _FakeBoxes:
    def __init__(self, n_people: int):
        self.id = _FakeTensor(list(range(1, n_people + 1)))
        self.conf = _FakeTensor([0.9] * n_people)
        self._n = n_people

    def __len__(self):
        return self._n


class _FakeKeypoints:
    def __init__(self, n_people: int, seed: int):
        rng = np.random.RandomState(seed)
        self.data = _FakeTensor(rng.rand(n_people, 17, 3))


class _FakeResult:
    """One frame's worth of detections, shaped like an Ultralytics Results."""

    def __init__(self, n_people: int, seed: int):
        self.boxes = _FakeBoxes(n_people)
        self.keypoints = _FakeKeypoints(n_people, seed)


def _install_fake_yolo(monkeypatch, people_per_frame: list[int]) -> None:
    """
    Replace YOLO and cv2.VideoCapture inside extract_pose so extract() runs with
    no weights and no video. track() hands back a real generator expression, so
    the stream is genuinely one-shot exactly like stream=True.
    """

    class _FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name

        def track(self, **kwargs):
            return (_FakeResult(n, seed=i) for i, n in enumerate(people_per_frame))

    class _FakeCapture:
        def __init__(self, path):
            pass

        def get(self, prop):
            return float(len(people_per_frame))

        def release(self):
            pass

    monkeypatch.setattr(extract_pose, "YOLO", _FakeYOLO)
    monkeypatch.setattr(extract_pose.cv2, "VideoCapture", _FakeCapture)


def test_extract_writes_one_row_per_detection(monkeypatch, tmp_path):
    # The regression guard. 5 frames x 2 swimmers = 10 rows. If anything walks
    # the results generator before the extraction loop, this drops to 0 — which
    # is precisely the bug that a progress-bar pre-pass introduced.
    _install_fake_yolo(monkeypatch, [2] * 5)
    out = tmp_path / "clip.parquet"

    n_rows = extract(tmp_path / "clip.mp4", out)

    assert n_rows == 10, "0 rows means the result stream was drained before the loop"
    df = pd.read_parquet(out)
    assert len(df) == 10
    assert list(df.columns) == COLUMNS  # 54 cols: 3 meta + 17*(x,y,c)


def test_empty_frames_are_skipped_but_still_advance_the_frame_index(
    monkeypatch, tmp_path
):
    # frame_idx comes from enumerate over *every* result, so a frame with no
    # swimmers writes no rows yet still consumes an index. Frame numbers must
    # stay aligned to the video's timeline — event windows are in seconds, and
    # renumbering would silently shift every label.
    _install_fake_yolo(monkeypatch, [2, 0, 1, 0, 3])
    out = tmp_path / "clip.parquet"

    n_rows = extract(tmp_path / "clip.mp4", out)

    assert n_rows == 6  # 2 + 0 + 1 + 0 + 3
    df = pd.read_parquet(out)
    assert sorted(df["frame"].unique().tolist()) == [0, 2, 4]


def test_max_frames_stops_early(monkeypatch, tmp_path):
    # max_frames is a debugging convenience, never used for real runs (events
    # often start >25s in). Pin it anyway so the break stays an early exit and
    # does not become an off-by-one.
    _install_fake_yolo(monkeypatch, [2] * 10)
    out = tmp_path / "clip.parquet"

    n_rows = extract(tmp_path / "clip.mp4", out, max_frames=3)

    assert n_rows == 6  # frames 0,1,2 only
    df = pd.read_parquet(out)
    assert df["frame"].max() == 2


def test_track_ids_and_confidences_are_carried_through(monkeypatch, tmp_path):
    # track_id is the spine of everything downstream — windows.py groups by it,
    # and box_conf carries the submersion signal. Pin that both survive the trip
    # from the Results object into the Parquet.
    _install_fake_yolo(monkeypatch, [3])
    out = tmp_path / "clip.parquet"

    extract(tmp_path / "clip.mp4", out)

    df = pd.read_parquet(out)
    assert sorted(df["track_id"].tolist()) == [1, 2, 3]
    assert np.allclose(df["box_conf"], 0.9)
