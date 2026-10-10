"""
Tests for the download command (ingest/download.py), without touching the network.

Pins the clock contract every label depends on: a section download is cut EXACTLY at
start_sec (so file time = source time - start_sec), whole downloads stay a plain copy,
and trims never change what gets downloaded.
"""

from __future__ import annotations

import subprocess

import pytest

from hydro_knight.ingest import download
from hydro_knight.ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Setting,
    TimeOfDay,
    Weather,
)


def _record(start=0.0, end=-1.0, trim_start=None, trim_end=None):
    return ClipRecord(
        clip_id="abc",
        source_url="https://example.com/v",
        platform="youtube",
        start_sec=start,
        end_sec=end,
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.DISTRESS,
        trim_start=trim_start,
        trim_end=trim_end,
    )


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """Capture the yt-dlp command instead of running it."""
    monkeypatch.setattr(download, "RAW_LOCAL", tmp_path)
    calls = []

    def fake_run(cmd, **_):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def test_section_download_cuts_exactly_at_start(ran):
    assert download.download_clip(_record(start=8820.0, end=19620.0))
    (cmd,) = ran
    assert "*8820.0-19620.0" in cmd
    assert "--force-keyframes-at-cuts" in cmd


@pytest.mark.parametrize(
    "rec",
    [
        _record(),  # whole video
        _record(end=38.0),  # whole video with a known length
        _record(trim_start=8.24, trim_end=25.78),  # trims must not shrink the download
    ],
)
def test_whole_downloads_are_never_sectioned(ran, rec):
    assert download.download_clip(rec)
    (cmd,) = ran
    assert "--download-sections" not in cmd
    assert "--force-keyframes-at-cuts" not in cmd
