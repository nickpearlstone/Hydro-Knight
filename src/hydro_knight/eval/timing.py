"""Which fps to use for a clip's frame <-> seconds conversions."""

from __future__ import annotations

from pathlib import Path

from ..preprocess.build_tracks import read_provenance


def clip_fps(
    parquet: Path, videos_dir: Path | None, default: float, record=None
) -> tuple[float, bool]:
    """Per-clip fps, from the same source the tracker used whenever possible."""
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
