"""
Which fps to use for a clip's frame <-> seconds conversions.

Clicks, events and trims are in seconds; tracks are in frames. The tracker and the
evaluation must use the same fps for each clip, so evaluation reads the one the
tracker recorded in the keypoint file before anything else.
"""

from __future__ import annotations

from pathlib import Path

from ..preprocess.build_tracks import read_provenance


def clip_fps(
    parquet: Path, videos_dir: Path | None, default: float, record=None
) -> tuple[float, bool]:
    """Per-clip fps, from the same source the tracker used whenever possible.

    Order: the video fps recorded in the keypoint file by build_tracks (exactly what
    tracking used), then the manifest, then the video file, then `default`. Falling
    through to `default` makes every frame<->time conversion a guess, so the caller warns.
    Args:
        parquet: the clip's keypoint Parquet (its stem is the clip id).
        videos_dir: directory holding <clip_id>.mp4, or None.
        default: last-resort fps if nothing else is known.
        record: the clip's ClipRecord, when the manifest was loaded.
    Returns:
        (fps, guessed): guessed is True only when `default` was used.
    """
    prov = read_provenance(parquet)
    fps = (prov or {}).get("video", {}).get("fps")
    if fps and fps > 0:
        return float(fps), False
    if record is not None and record.fps > 0:
        return float(record.fps), False
    video = videos_dir / f"{parquet.stem}.mp4" if videos_dir else None
    if video is not None and video.exists():
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps), False
    return default, True
