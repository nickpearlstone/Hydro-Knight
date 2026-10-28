"""
Tests for `hydroknight heads frames`: the head validation frames must never share a clip
with the benchmark (test) frames, and must land on the drowning onset in the local file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hydro_knight.cli.head_frames import pick_val_frames
from hydro_knight.ingest import download
from hydro_knight.ingest.manifest import Label


@dataclass
class Rec:
    clip_id: str
    label: Label = Label.DISTRESS
    notes: str = ""
    events: list = field(default_factory=lambda: [{"start": 12.0}])
    start_sec: float = 0.0

    def to_file_time(self, t):
        return t - self.start_sec


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_LOCAL", tmp_path)
    for c in ("a", "b", "bench", "held", "calm", "noonset", "cut"):
        (tmp_path / f"{c}.mp4").write_bytes(b"")
    return tmp_path


RECORDS = [
    Rec("a"),
    Rec("b"),
    Rec("bench"),
    Rec("held", notes="[HOLD reef]"),
    Rec("calm", label=Label.NORMAL, events=[]),
    Rec("noonset", events=[{"start": None}]),
    Rec("novideo"),
    Rec("cut", start_sec=10.0),  # section download: file frame 0 = source 10 s
]


def test_only_eligible_clips_and_never_a_benchmark_clip(raw):
    frames = pick_val_frames(RECORDS, {"bench"}, n=99, seed=0, fps_of=lambda c: 30.0)
    assert [f["clip_id"] for f in frames] == ["a", "b", "cut"]
    assert all(f["water"] == [] and f["deck"] == [] for f in frames)


def test_onset_converted_from_source_time_to_file_frame(raw):
    frames = pick_val_frames(RECORDS, set(), n=99, seed=0, fps_of=lambda c: 25.0)
    by = {f["clip_id"]: f["frame"] for f in frames}
    assert by["a"] == 300  # 12 s x 25 fps
    assert by["cut"] == 50  # (12 - 10) s x 25 fps


def test_same_seed_same_pick(raw):
    one = pick_val_frames(RECORDS, set(), n=2, seed=3, fps_of=lambda c: 30.0)
    two = pick_val_frames(RECORDS, set(), n=2, seed=3, fps_of=lambda c: 30.0)
    assert one == two and len(one) == 2
