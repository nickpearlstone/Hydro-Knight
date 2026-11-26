"""Data-side health views that expose representation gaps before training."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Keep in sync with normalize.py's reference joints (COCO indices).
REF_CONF_COLS = ["c5", "c6", "c11", "c12"]


def usable_pose_mask(df: pd.DataFrame, min_ref_conf: float = 0.3) -> np.ndarray:
    """Per-row bool: would normalize.py keep this pose? Keep in sync with normalize.py's gate."""
    conf = df[REF_CONF_COLS].to_numpy(dtype=float)
    return conf.min(axis=1) >= min_ref_conf


def frame_coverage_in_events(
    df: pd.DataFrame,
    events: list[dict],
    fps: float,
    min_ref_conf: float = 0.3,
) -> list[dict]:
    """Per event: fraction of its frames (events in seconds) with >= 1 usable pose."""
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
    """Per-track continuity: frame count, span, and gaps in consecutive frame numbers."""
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
    """Manifest at a glance: label counts, event counts/durations, HOLD count."""
    label_counts: dict[str, int] = {}
    event_durs, event_labels, n_events = [], {}, 0
    for r in records:
        label_counts[r.label.value] = label_counts.get(r.label.value, 0) + 1
        for ev in r.events:
            n_events += 1
            lab = ev.get("label", "distress")
            event_labels[lab] = event_labels.get(lab, 0) + 1
            if ev.get("start") is None or ev.get("end") is None:
                continue
            event_durs.append(ev["end"] - ev["start"])
    return {
        "n_clips": len(records),
        "label_counts": label_counts,
        "n_events": n_events,
        "event_label_counts": event_labels,
        "event_duration_mean_s": float(np.mean(event_durs)) if event_durs else 0.0,
        "event_duration_median_s": float(np.median(event_durs)) if event_durs else 0.0,
        "clips_with_hold": sum("[HOLD" in (r.notes or "") for r in records),
    }
