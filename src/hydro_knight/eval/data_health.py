"""
Data-side health views — the checks that expose representation gaps.

The first training run failed for data reasons, not model reasons: the
feature layer silently deleted the submersion frames. These functions make
that class of failure *visible before training*:

- pose coverage inside event windows: "of the seconds we labeled distress,
  how many produced a usable pose at all?" Low coverage = the event never
  reaches the model (the exact 0.539 mechanism).
- track statistics: fragmentation and gaps, which drive both windowing yield
  and Plan A's re-association design.
- dataset census: what the manifest actually contains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The four reference joints normalize.py gates on (COCO indices).
REF_CONF_COLS = ["c5", "c6", "c11", "c12"]  # L/R shoulder, L/R hip


def usable_pose_mask(df: pd.DataFrame, min_ref_conf: float = 0.3) -> np.ndarray:
    """Per-row bool: would normalize.py keep this pose (all 4 reference joints >= min_ref_conf)?

    Mirrors the normalize.py gate WITHOUT importing it, so this stays a pure data view; update
    it here too if that gate changes, to keep the before/after coverage comparison honest.
    Args:
        df: keypoint rows with reference-joint confidence columns c5, c6, c11, c12.
        min_ref_conf: minimum confidence required of each reference joint.
    Returns:
        (len(df),) bool array — True where the pose would be kept.
    """
    conf = df[REF_CONF_COLS].to_numpy(dtype=float)
    return conf.min(axis=1) >= min_ref_conf


def frame_coverage_in_events(
    df: pd.DataFrame,
    events: list[dict],
    fps: float,
    min_ref_conf: float = 0.3,
) -> list[dict]:
    """Per event: fraction of its frames with >= 1 usable pose ("did the filter eat the drowning?").

    Coverage 0.35 means 65% of the annotated span produced no usable pose — invisible to any
    pose-based model. (v1 is any-swimmer coverage; victim-track coverage needs track-scoped labels.)
    Args:
        df: keypoint-Parquet DataFrame for the clip.
        events: list of {"start","end","label"} windows in seconds.
        fps: clip frame rate, for the seconds<->frame conversion.
        min_ref_conf: reference-joint confidence gate for "usable".
    Returns:
        One dict per event: event_start, event_end, event_label, frames_in_event,
        frames_covered, coverage.
    """
    usable_frames = set(df.loc[usable_pose_mask(df, min_ref_conf), "frame"].unique())
    out = []
    for ev in events:
        f0, f1 = int(ev["start"] * fps), int(np.ceil(ev["end"] * fps))
        span = list(range(f0, max(f1, f0 + 1)))
        covered = sum(f in usable_frames for f in span)
        out.append(
            {
                "event_start": ev["start"],
                "event_end": ev["end"],
                "event_label": ev.get("label", "distress"),
                "frames_in_event": len(span),
                "frames_covered": covered,
                "coverage": covered / len(span),
            }
        )
    return out


def track_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-track continuity: frame count, span, and gaps (breaks in consecutive frame numbers).

    A gap is submersion, occlusion, or detector flicker — exactly what Plan A's re-association
    timer must reason about, so it's surfaced here.
    Args:
        df: keypoint-Parquet DataFrame (needs track_id, frame).
    Returns:
        DataFrame (one row per track) with track_id, n_frames, first_frame, last_frame,
        span_frames, n_gaps, longest_gap; sorted by n_frames descending.
    """
    rows = []
    for tid, g in df.groupby("track_id"):
        frames = np.sort(g["frame"].unique())
        diffs = np.diff(frames)
        gaps = diffs[diffs > 1]
        rows.append(
            {
                "track_id": int(tid),
                "n_frames": len(frames),
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "span_frames": int(frames[-1] - frames[0] + 1),
                "n_gaps": int(len(gaps)),
                "longest_gap": int(gaps.max()) if len(gaps) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("n_frames", ascending=False)


def dataset_census(records) -> dict:
    """Manifest at a glance: label counts, event counts/durations, HOLD count.

    Args:
        records: an iterable of ClipRecord objects.
    Returns:
        Dict with n_clips, label_counts, n_events, event_label_counts,
        event_duration_mean_s, event_duration_median_s, clips_with_hold.
    """
    label_counts: dict[str, int] = {}
    event_durs, event_labels = [], {}
    for r in records:
        label_counts[r.label.value] = label_counts.get(r.label.value, 0) + 1
        for ev in r.events:
            event_durs.append(ev["end"] - ev["start"])
            lab = ev.get("label", "distress")
            event_labels[lab] = event_labels.get(lab, 0) + 1
    return {
        "n_clips": len(records),
        "label_counts": label_counts,
        "n_events": len(event_durs),
        "event_label_counts": event_labels,
        "event_duration_mean_s": float(np.mean(event_durs)) if event_durs else 0.0,
        "event_duration_median_s": float(np.median(event_durs)) if event_durs else 0.0,
        "clips_with_hold": sum("[HOLD" in (r.notes or "") for r in records),
    }
