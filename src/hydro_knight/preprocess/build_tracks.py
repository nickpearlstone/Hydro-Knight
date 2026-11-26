"""Offline merge + ByteTrack over raw detections -> per-swimmer keypoint Parquet (CPU step)."""

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

    merge: str = "ios"
    iou_thresh: float = 0.5
    ios_thresh: float = 0.8
    lost_track_seconds: float | None = 1.0
    high_conf_threshold: float = 0.5
    activation_threshold: float = 0.5
    min_consecutive_frames: int = 2
    min_iou: float = 0.1
    same_crop_conf: float | None = None


LEGACY = TrackSettings(
    merge="iou",
    lost_track_seconds=None,
    high_conf_threshold=0.6,
    activation_threshold=0.3,
)


def make_tracker(fps: float, settings: TrackSettings) -> ByteTrackTracker:
    """ByteTrack with the lost-track buffer converted to frames at `fps`."""
    if settings.lost_track_seconds is None:
        # CAUTION[build-tracks-rate-30]: rate must stay 30.0 so the buffer is exactly 30 frames.
        buffer_frames, rate = (
            30,
            30.0,
        )
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
    # CAUTION[build-tracks-tracker-reset]: reset required; id counter is class-level.
    tracker.reset()
    return tracker


def track_detections(
    dets: pd.DataFrame, frames: pd.DataFrame, fps: float, settings: TrackSettings
) -> pd.DataFrame:
    """Merge and track raw detections frame by frame; empty frames must be fed too."""
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
                same_crop_below=settings.same_crop_conf,
            )
            detections = sv.Detections(
                xyxy=boxes_all[idx].astype(float),
                confidence=conf_all[idx].astype(float),
                class_id=np.zeros(len(idx), dtype=int),
            )
            # CAUTION[build-tracks-row-column]: "row" column required; tracker reorders output.
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


def tracking_fps(meta: dict) -> float:
    """Rate of the frames the tracker sees: video fps divided by the extraction frame step."""
    return meta["video"]["fps"] / meta["settings"].get("frame_step", 1)


def build_tracks(
    det_dir: Path, out_path: Path, settings: TrackSettings | None = None
) -> pd.DataFrame:
    """Build one clip's keypoint Parquet from its raw-detection folder; settings go in the schema metadata."""
    settings = settings or TrackSettings()
    meta, dets, frames = load_detections(det_dir)
    df = track_detections(dets, frames, tracking_fps(meta), settings)

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


def provenance_frame_step(provenance: dict) -> int:
    """Extraction frame step recorded in a keypoint file's provenance (1 = every frame)."""
    return (provenance.get("extraction_settings") or {}).get("frame_step", 1)
