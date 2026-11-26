from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import asdict

import pytest

from hydro_knight.ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Manifest,
    Setting,
    TimeOfDay,
    Weather,
    make_clip_id,
)


def _record(
    url: str = "https://example.com/v", start: float = 0.0, end: float = 10.0
) -> ClipRecord:
    return ClipRecord(
        clip_id=make_clip_id(url, start, end),
        source_url=url,
        platform="youtube",
        start_sec=start,
        end_sec=end,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.NORMAL,
    )


def test_clip_id_is_deterministic():
    a = make_clip_id("https://example.com/v", 0.0, 10.0)
    b = make_clip_id("https://example.com/v", 0.0, 10.0)
    assert a == b


def test_clip_id_changes_with_inputs():
    base = make_clip_id("https://example.com/v", 0.0, 10.0)
    assert make_clip_id("https://example.com/OTHER", 0.0, 10.0) != base
    assert make_clip_id("https://example.com/v", 1.0, 10.0) != base
    assert make_clip_id("https://example.com/v", 0.0, 99.0) != base


def test_clip_id_is_12_hex_chars():
    cid = make_clip_id("https://example.com/v", 0.0, 10.0)
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)


def test_append_then_dedup(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()

    assert m.append(rec) is True
    assert m.append(rec) is False

    assert len(m.load()) == 1


def test_load_roundtrips_enums_and_events(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    rec.events = [{"start": 3.0, "end": 5.0, "label": "distress"}]
    m.append(rec)

    (loaded,) = m.load()
    assert loaded.camera_view is CameraView.ELEVATED
    assert loaded.label is Label.NORMAL
    assert loaded.events == [{"start": 3.0, "end": 5.0, "label": "distress"}]


def test_fps_roundtrips(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    rec.fps = 59.94
    m.append(rec)

    (loaded,) = m.load()
    assert loaded.fps == 59.94
    assert isinstance(loaded.fps, float)


def test_manifest_without_fps_field_still_loads(tmp_path):
    # Rows without fps load as 0.0 (unknown).
    path = tmp_path / "manifest.jsonl"
    row = json.loads(json.dumps(asdict(_record())))
    del row["fps"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    (loaded,) = Manifest(path).load()
    assert loaded.fps == 0.0


def test_trims_roundtrip_and_old_rows_load_untrimmed(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    trimmed = _record()
    trimmed.trim_start, trimmed.trim_end = 2.5, 8.0
    m.append(trimmed)
    old = json.loads(json.dumps(asdict(_record("https://example.com/old"))))
    del old["trim_start"], old["trim_end"]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")

    new, legacy = m.load()
    assert (new.trim_start, new.trim_end) == (2.5, 8.0)
    assert (legacy.trim_start, legacy.trim_end) == (None, None)


def test_file_time_conversion_uses_the_download_window():
    whole = _record(start=0.0, end=10.0)
    section = _record(start=8820.0, end=19620.0)
    assert whole.to_file_time(12.0) == 12.0
    assert section.file_offset == 8820.0
    assert section.to_file_time(9420.0) == 600.0
    assert section.to_source_time(600.0) == 9420.0


def test_load_missing_file_returns_empty(tmp_path):
    m = Manifest(tmp_path / "does_not_exist.jsonl")
    assert m.load() == []


def test_concurrent_writers_on_one_clip_both_survive(tmp_path):
    # Writers must hold the lock across read-modify-write.
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    m.append(rec)
    started = threading.Event()

    def slow_labeler_save(r):
        started.set()
        time.sleep(0.3)
        return dataclasses.replace(r, notes="labeled")

    t = threading.Thread(target=m.modify, args=(rec.clip_id, slow_labeler_save))
    t.start()
    started.wait()
    m.modify(rec.clip_id, lambda r: dataclasses.replace(r, fps=59.94))
    t.join()

    (out,) = m.load()
    assert out.notes == "labeled" and out.fps == 59.94


def test_failed_change_leaves_file_untouched_and_no_temp_files(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    m.append(rec)
    before = m.path.read_text()

    def bad(_r):
        raise ValueError("bad label")

    with pytest.raises(ValueError):
        m.modify(rec.clip_id, bad)
    assert m.modify("no-such-clip", lambda r: r) is None
    assert m.path.read_text() == before
    assert not list(tmp_path.glob("*.tmp"))
