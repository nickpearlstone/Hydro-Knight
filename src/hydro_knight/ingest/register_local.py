"""
Register a locally-recorded video file (e.g. a screen-recording of a live
pool cam) into the manifest.

Use this for footage that can't be downloaded from a URL — live webcams you
screen-record, self-recorded clips, etc. The file is copied into raw_local/
under a generated clip_id (the same naming the downloader uses) and a
ClipRecord is written so it flows into pose extraction like any other clip.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from .download import RAW_LOCAL, done_marker, local_path
from .manifest import (
    CameraView,
    ClipRecord,
    Label,
    Manifest,
    Setting,
    TimeOfDay,
    Weather,
    make_clip_id,
)


def _video_duration_sec(path: Path) -> float:
    """Read a video's duration in seconds via OpenCV (frame count / fps).

    Returns 0.0 if OpenCV can't read the metadata (some screen-recorders write odd
    headers), which the manifest treats as "unknown length" (end_sec = -1).
    Args:
        path: video file to inspect.
    Returns:
        Duration in seconds, or 0.0 if unreadable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return (frames / fps) if fps > 0 else 0.0


def register_local(
    manifest_path: Path,
    video_path: Path,
    source_url: str,
    camera_view: CameraView = CameraView.UNKNOWN,
    setting: Setting = Setting.UNKNOWN,
    time_of_day: TimeOfDay = TimeOfDay.UNKNOWN,
    weather: Weather = Weather.UNKNOWN,
    label: Label = Label.UNLABELED,
    notes: str = "",
    move: bool = False,
) -> str | None:
    """Copy a local video into raw_local/ and register it in the manifest.

    source_url doubles as provenance and the clip_id seed, so re-registering the same
    source+length is de-duplicated.
    Args:
        manifest_path: JSONL manifest to append to.
        video_path: the recorded file already on disk.
        source_url: provenance URL (e.g. the cam page); also seeds the clip_id.
        camera_view, setting, time_of_day, weather, label, notes: annotation metadata to
            store now if known, else left UNKNOWN for the annotation pass.
        move: move the file instead of copying (saves disk for large recordings).
    Returns:
        The clip_id on success, or None if it was a duplicate or the file was missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return None

    duration = _video_duration_sec(video_path)
    start_sec = 0.0
    end_sec = duration if duration > 0 else -1.0

    # Derive a stable id from provenance + length so the same recording isn't
    # registered twice. The local file is then named <clip_id>.mp4, matching
    # the downloader's convention so the annotator/extractor find it the same way.
    clip_id = make_clip_id(source_url, start_sec, end_sec)

    record = ClipRecord(
        clip_id=clip_id,
        source_url=source_url,
        platform="local",
        start_sec=start_sec,
        end_sec=end_sec,
        camera_view=camera_view,
        setting=setting,
        time_of_day=time_of_day,
        weather=weather,
        label=label,
        notes=notes,
    )

    manifest = Manifest(manifest_path)
    if not manifest.append(record):
        print(f"Already registered (duplicate clip_id {clip_id}).")
        return None

    # Place the file under raw_local/<clip_id>.mp4 and mark it complete so the
    # downloader won't try to fetch it and the annotator sees it as available.
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    dest = local_path(record)
    if move:
        shutil.move(str(video_path), dest)
    else:
        shutil.copy2(video_path, dest)
    done_marker(record).touch()

    mins = duration / 60 if duration > 0 else 0
    print(f"Registered local clip {clip_id} ({mins:.1f} min) -> {dest}")
    return clip_id


# --- CLI entry point ---------------------------------------------------------
# Example:
#   python -m hydro_knight.ingest.register_local ~/Desktop/ami_pool.mp4
# Edit the metadata below to match the recording.

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m hydro_knight.ingest.register_local <video_file>")
        raise SystemExit(1)

    register_local(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
        video_path=Path(sys.argv[1]),
        source_url="https://www.2fla.com/ami-pool-cam",  # AMI Pool Cam, White Sands Beach Resort
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.UNLABELED,  # leave for your annotation pass
        notes="AMI Pool Cam (White Sands Beach Resort) screen recording",
    )
