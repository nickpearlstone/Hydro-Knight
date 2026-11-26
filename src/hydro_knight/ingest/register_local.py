"""Register a locally-recorded video file (e.g. a screen-recording of a live
pool cam) into the manifest.
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
    """Copy a local video into raw_local/ and register it in the manifest."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return None

    duration = _video_duration_sec(video_path)
    start_sec = 0.0
    end_sec = duration if duration > 0 else -1.0

    # CAUTION[register-local-clip-id]: clip_id must stay stable per recording; file is named <clip_id>.mp4.
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m hydro_knight.ingest.register_local <video_file>")
        raise SystemExit(1)

    register_local(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
        video_path=Path(sys.argv[1]),
        source_url="https://www.2fla.com/ami-pool-cam",
        camera_view=CameraView.ELEVATED,
        setting=Setting.OUTDOOR,
        time_of_day=TimeOfDay.DAY,
        weather=Weather.CLEAR,
        label=Label.UNLABELED,
        notes="AMI Pool Cam (White Sands Beach Resort) screen recording",
    )
