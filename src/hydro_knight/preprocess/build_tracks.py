"""
Offline merge + tracking (the CPU step): raw detections -> per-swimmer keypoint Parquet.

Reads a clip folder written by `extract_pose.extract_raw`, merges tile-overlap
duplicates frame by frame, runs ByteTrack over the merged boxes, and writes the
keypoint table everything downstream consumes (`features/`, `eval/`). Because
the raw detections are kept, this step can be rerun with different settings in
minutes, with no GPU and no video.

Tracking is geometry only: ByteTrack predicts where each track's box should be
(Kalman filter), matches confident detections to predictions by overlap, then
gives weaker detections a second chance against unmatched tracks. It never
looks at pixels.

Two presets:
- `TrackSettings()` — current defaults, which fix three problems in the original
  pipeline: half-body seam duplicates survived the merge; a lost track was
  deleted after a fixed 30 frames (0.5 s at 60 fps) regardless of fps; and new
  tracks could only start from detections >= 0.6, which made the configured
  activation threshold of 0.3 dead.
- `LEGACY` — reproduces the original behavior, for before/after comparison.

Output columns: COLUMNS = frame, track_id, box_conf, 17 x (x, y, c), bx1..by2.
track_id -1 marks a detection the tracker has not confirmed (or matched to nothing).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import supervision as sv
from trackers import ByteTrackTracker

from .extract_pose import BOX_COLUMNS, KP_COLUMNS, load_detections
from .tiled_pose import merge_detections

COLUMNS = ["frame", "track_id", "box_conf"] + KP_COLUMNS + BOX_COLUMNS


@dataclass(frozen=True)
class TrackSettings:
    """Merge + tracker parameters; stored in the output file's metadata."""

    merge: str = "ios"  # "ios" (cross-crop IoU or IoS) or "iou" (original)
    iou_thresh: float = 0.5
    ios_thresh: float = 0.8
    # None = fixed 30 frames at any fps (original).
    lost_track_seconds: float | None = 1.0
    # First-round matching and the bar to start a track. 0.5 kept nearly all the long tracks
    # 0.4 found on a real rescue clip, with a third of the short fragments.
    high_conf_threshold: float = 0.5
    activation_threshold: float = 0.5
    min_consecutive_frames: int = 2
    min_iou: float = 0.1


LEGACY = TrackSettings(
    merge="iou",
    lost_track_seconds=None,
    high_conf_threshold=0.6,
    activation_threshold=0.3,
)


def make_tracker(fps: float, settings: TrackSettings) -> ByteTrackTracker:
    """ByteTrack configured from settings, with the lost-track buffer converted to frames at `fps`.

    Args:
        fps: the clip's frame rate.
        settings: merge/tracker parameters.
    Returns:
        A fresh tracker whose ids start at 0.
    """
    if settings.lost_track_seconds is None:
        buffer_frames, rate = (
            30,
            30.0,
        )  # trackers scales buffer by rate/30 -> exactly 30
    else:
        buffer_frames, rate = max(1, round(settings.lost_track_seconds * fps)), 30.0
    tracker = ByteTrackTracker(
        lost_track_buffer=buffer_frames,
        frame_rate=rate,
        track_activation_threshold=settings.activation_threshold,
        minimum_consecutive_frames=settings.min_consecutive_frames,
        minimum_iou_threshold=settings.min_iou,
        high_conf_det_threshold=settings.high_conf_threshold,
    )
    tracker.reset()  # the id counter is class-level; start every clip at 0
    return tracker


def track_detections(
    dets: pd.DataFrame, frames: pd.DataFrame, fps: float, settings: TrackSettings
) -> pd.DataFrame:
    """Merge and track raw detections frame by frame.

    Every processed frame is fed to the tracker, including frames with no detections, so a
    track's lost-time counts real frames rather than only frames where something was seen.
    Args:
        dets: raw detections (extract_pose.DET_COLUMNS).
        frames: every processed frame (extract_pose.FRAME_COLUMNS).
        fps: the clip's frame rate.
        settings: merge/tracker parameters.
    Returns:
        DataFrame with COLUMNS, one row per merged detection the tracker returned.
    """
    tracker = make_tracker(fps, settings)
    dets = dets.sort_values("frame", kind="stable", ignore_index=True)
    frame_col = dets["frame"].to_numpy()
    boxes_all = dets[BOX_COLUMNS].to_numpy(np.float32)
    conf_all = dets["conf"].to_numpy(np.float32)
    crop_all = dets["crop"].to_numpy()
    kp_all = dets[KP_COLUMNS].to_numpy(np.float32)

    bounds = np.searchsorted(frame_col, frames["frame"].to_numpy(), side="left")
    ends = np.searchsorted(frame_col, frames["frame"].to_numpy(), side="right")
    out = []
    for f, lo, hi in zip(frames["frame"].to_numpy(), bounds, ends, strict=True):
        if hi > lo:
            idx = lo + merge_detections(
                boxes_all[lo:hi],
                conf_all[lo:hi],
                crop_all[lo:hi],
                mode=settings.merge,
                iou_thresh=settings.iou_thresh,
                ios_thresh=settings.ios_thresh,
            )
            detections = sv.Detections(
                xyxy=boxes_all[idx].astype(float),
                confidence=conf_all[idx].astype(float),
                class_id=np.zeros(len(idx), dtype=int),
            )
            # The tracker reorders its output; this column rides along to find each row again.
            detections.data["row"] = idx
        else:
            detections = sv.Detections.empty()
        tracked = tracker.update(detections)
        if len(tracked) == 0:
            continue
        rows = tracked.data["row"].astype(int)
        block = np.column_stack(
            [
                np.full(len(rows), f),
                tracked.tracker_id,
                conf_all[rows],
                kp_all[rows],
                boxes_all[rows],
            ]
        )
        out.append(block)

    if not out:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(np.concatenate(out), columns=COLUMNS)
    floats = {c: "float32" for c in COLUMNS[2:]}
    return df.astype({"frame": "int64", "track_id": "int64"} | floats)


def build_tracks(
    det_dir: Path, out_path: Path, settings: TrackSettings | None = None
) -> pd.DataFrame:
    """Build one clip's keypoint Parquet from its raw-detection folder.

    The settings and source extraction facts are written into the Parquet's schema metadata
    under the key "hydro_knight", so every keypoint file records how it was made.
    Args:
        det_dir: clip folder written by extract_raw (must be complete).
        out_path: destination .parquet (parent dirs created).
        settings: merge/tracker parameters; None = TrackSettings() defaults.
    Returns:
        The tracked DataFrame that was written.
    """
    settings = settings or TrackSettings()
    meta, dets, frames = load_detections(det_dir)
    df = track_detections(dets, frames, meta["video"]["fps"], settings)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    provenance = {
        "track_settings": asdict(settings),
        "clip_id": meta["clip_id"],
        "video": meta["video"],
        "extraction_settings": meta["settings"],
        "frames_processed": meta["frames_processed"],
    }
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[b"hydro_knight"] = json.dumps(provenance).encode()
    pq.write_table(table.replace_schema_metadata(schema_meta), out_path)
    return df


def read_provenance(path: Path) -> dict | None:
    """The "hydro_knight" metadata written by build_tracks, or None for older keypoint files."""
    raw = (pq.read_schema(path).metadata or {}).get(b"hydro_knight")
    return json.loads(raw) if raw else None
