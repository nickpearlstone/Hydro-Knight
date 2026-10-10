"""
Tests for the manifest layer: deterministic clip IDs and append-safe dedup.

These pin down the two properties the whole reproducibility story depends on:
1. The same clip always hashes to the same ID (so re-running collection is idempotent).
2. Appending the same clip twice never duplicates it in the manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict

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
    """Build a minimal valid ClipRecord with a hashed clip_id."""
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
    # Same inputs -> identical ID, every time. This is what makes re-collecting
    # the same search results idempotent instead of duplicating rows.
    a = make_clip_id("https://example.com/v", 0.0, 10.0)
    b = make_clip_id("https://example.com/v", 0.0, 10.0)
    assert a == b


def test_clip_id_changes_with_inputs():
    base = make_clip_id("https://example.com/v", 0.0, 10.0)
    # Any of the three identifying fields changing must change the ID.
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

    # First append writes the row and reports success.
    assert m.append(rec) is True
    # Second append of the same clip_id is a no-op and reports it.
    assert m.append(rec) is False

    # Only one row should exist on disk.
    assert len(m.load()) == 1


def test_load_roundtrips_enums_and_events(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    rec.events = [{"start": 3.0, "end": 5.0, "label": "distress"}]
    m.append(rec)

    (loaded,) = m.load()
    # Enum-typed fields must come back as Enums, not bare strings.
    assert loaded.camera_view is CameraView.ELEVATED
    assert loaded.label is Label.NORMAL
    # Event windows survive the JSON round-trip intact.
    assert loaded.events == [{"start": 3.0, "end": 5.0, "label": "distress"}]


def test_fps_roundtrips(tmp_path):
    # fps drives every frame<->time conversion downstream, so it has to survive
    # the JSON trip as a number, not a string.
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record()
    rec.fps = 59.94
    m.append(rec)

    (loaded,) = m.load()
    assert loaded.fps == 59.94
    assert isinstance(loaded.fps, float)


def test_manifest_without_fps_field_still_loads(tmp_path):
    # Rows written before fps existed must keep loading — the whole committed
    # manifest predates the field. Missing fps reads as 0.0, meaning "unknown",
    # which callers treat as "fall back and warn", never as a real frame rate.
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
    del old["trim_start"], old["trim_end"]  # rows written before trims existed
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")

    new, legacy = m.load()
    assert (new.trim_start, new.trim_end) == (2.5, 8.0)
    assert (legacy.trim_start, legacy.trim_end) == (None, None)


def test_file_time_conversion_uses_the_download_window():
    whole = _record(start=0.0, end=10.0)
    section = _record(start=8820.0, end=19620.0)  # a 3 h slice of a livestream
    assert whole.to_file_time(12.0) == 12.0  # whole download: one clock
    assert section.file_offset == 8820.0
    assert section.to_file_time(9420.0) == 600.0  # 10 min into the file
    assert section.to_source_time(600.0) == 9420.0


def test_load_missing_file_returns_empty(tmp_path):
    # Callers shouldn't have to check for the file first.
    m = Manifest(tmp_path / "does_not_exist.jsonl")
    assert m.load() == []
