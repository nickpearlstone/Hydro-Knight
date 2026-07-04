"""
Tests for the manifest layer: deterministic clip IDs and append-safe dedup.

These pin down the two properties the whole reproducibility story depends on:
1. The same clip always hashes to the same ID (so re-running collection is idempotent).
2. Appending the same clip twice never duplicates it in the manifest.
"""

from __future__ import annotations

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


def test_load_missing_file_returns_empty(tmp_path):
    # Callers shouldn't have to check for the file first.
    m = Manifest(tmp_path / "does_not_exist.jsonl")
    assert m.load() == []
