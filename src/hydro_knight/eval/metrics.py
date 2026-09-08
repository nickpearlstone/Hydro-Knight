"""
Detector-agnostic evaluation metrics.

The core design decision: every detector — the TCN autoencoder today, the
Plan A state machine later — reduces its output to the same neutral format,
a table of *detections*:

    track_id | frame | score | span
    ---------|-------|-------|-----
    17       | 800   | 0.42  | 32     <- a TCN window's anomaly score
    17       | 990   | 1.0   | 1      <- a state-machine ALARM

Everything downstream (per-event recall, latency, false-alarm rate, curves)
consumes only this table plus the manifest's annotated event windows. That is
what lets two completely different detectors land on the same report and be
compared directly.

Timeline convention: `frame / fps` = seconds in the clip, matching how event
windows are annotated. A detection covers [frame/fps, (frame+span)/fps).

Labeling caveat (known, tracked in PLAN): events are *time*-scoped, so during
a rescue the lifeguard's windows also fall inside the event span and count as
distress. Window-level ROC/PR therefore inherit that contamination; the
per-event metrics are less affected (any overlapping detection counts as the
same single catch). Track-scoped (victim) labeling is the future fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

DET_COLUMNS = ["track_id", "frame", "score", "span"]


@dataclass
class ClipEval:
    """Everything the harness needs to know about one clip."""

    clip_id: str
    detections: pd.DataFrame  # columns DET_COLUMNS
    events: list[dict] = field(default_factory=list)  # {"start","end","label"} sec
    fps: float = 30.0
    duration_s: float = 0.0


def detections_from_windows(
    info: list[tuple[int, int]], errors: np.ndarray, span: int = 32
) -> pd.DataFrame:
    """Adapter: turn TCN window output into the neutral detections table.

    Args:
        info: make_windows' list of (track_id, start_frame), one per window.
        errors: matching per-window reconstruction errors.
        span: frames each window covers.
    Returns:
        DataFrame with columns track_id, frame, score, span (one row per window;
        empty with those columns if info is empty).
    """
    if len(info) == 0:
        return pd.DataFrame(columns=DET_COLUMNS)
    tids = np.array([t for t, _ in info], dtype=int)
    frames = np.array([f for _, f in info], dtype=int)
    return pd.DataFrame(
        {
            "track_id": tids,
            "frame": frames,
            "score": np.asarray(errors, dtype=float),
            "span": int(span),
        }
    )


def _det_intervals(det: pd.DataFrame, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Each detection's [start_s, end_s) on the clip timeline."""
    start = det["frame"].to_numpy(dtype=float) / fps
    end = (det["frame"].to_numpy(dtype=float) + det["span"].to_numpy(float)) / fps
    return start, end


def label_detections(clip: ClipEval) -> np.ndarray:
    """Boolean mask: True where a detection's time span overlaps ANY event window.

    Uses the time-overlap rule from the Colab eval, kept identical for comparability.
    Args:
        clip: the ClipEval whose detections and events are compared.
    Returns:
        (len(detections),) bool array; a length-0 array if there are no detections.
    """
    det = clip.detections
    if len(det) == 0:
        return np.zeros(0, dtype=bool)
    start, end = _det_intervals(det, clip.fps)
    mask = np.zeros(len(det), dtype=bool)
    for ev in clip.events:
        mask |= (start < ev["end"]) & (end > ev["start"])
    return mask


def split_scores(clips: list[ClipEval]) -> tuple[np.ndarray, np.ndarray]:
    """Split all detection scores across clips into (normal, distress) by event overlap.

    Rows flagged `trained_on` (the --train-fresh holdout) are excluded: scoring the model on
    windows it trained on would flatter the normal distribution and contaminate ROC/PR/percentiles.
    Args:
        clips: the per-clip evaluation bundles.
    Returns:
        (normal_scores, distress_scores): 1-D float arrays, each empty if that class has none.
    """
    normal, distress = [], []
    for c in clips:
        det = c.detections
        if len(det) == 0:
            continue
        m = label_detections(c)
        s = det["score"].to_numpy(dtype=float)
        if "trained_on" in det.columns:
            keep = ~det["trained_on"].to_numpy(dtype=bool)
            m, s = m[keep], s[keep]
        normal.append(s[~m])
        distress.append(s[m])
    cat = lambda parts: np.concatenate(parts) if parts else np.array([])  # noqa: E731
    return cat(normal), cat(distress)


def event_catches(clip: ClipEval, threshold: float) -> list[dict]:
    """Per-event verdicts at a given alarm threshold: caught? and latency-to-first-catch.

    Caught if any detection scoring >= threshold overlaps the event; latency is that first
    detection's start minus event onset, clipped at 0 (overlap before onset isn't a prediction).
    Args:
        clip: the ClipEval to score.
        threshold: alarm threshold applied to detection scores.
    Returns:
        One dict per event: clip_id, event_label, event_start, event_end, caught (bool),
        latency_s (seconds, NaN if uncaught).
    """
    det = clip.detections
    out = []
    if len(det) == 0:
        return [
            {
                "clip_id": clip.clip_id,
                "event_label": ev.get("label", "distress"),
                "event_start": ev["start"],
                "event_end": ev["end"],
                "caught": False,
                "latency_s": np.nan,
            }
            for ev in clip.events
        ]
    start, end = _det_intervals(det, clip.fps)
    score = det["score"].to_numpy(dtype=float)
    for ev in clip.events:
        hit = (score >= threshold) & (start < ev["end"]) & (end > ev["start"])
        caught = bool(hit.any())
        latency = float(max(start[hit].min() - ev["start"], 0.0)) if caught else np.nan
        out.append(
            {
                "clip_id": clip.clip_id,
                "event_label": ev.get("label", "distress"),
                "event_start": ev["start"],
                "event_end": ev["end"],
                "caught": caught,
                "latency_s": latency,
            }
        )
    return out


def _episodes(times_s: np.ndarray, merge_gap_s: float) -> list[tuple[float, float]]:
    """
    Merge a sorted list of alarm timestamps into episodes.

    Thousands of overlapping above-threshold windows are really ONE alarm as a
    lifeguard experiences it. Consecutive firings closer than `merge_gap_s`
    belong to the same episode; a bigger silence starts a new one.
    """
    if len(times_s) == 0:
        return []
    times_s = np.sort(times_s)
    episodes = []
    t0 = prev = times_s[0]
    for t in times_s[1:]:
        if t - prev > merge_gap_s:
            episodes.append((float(t0), float(prev)))
            t0 = t
        prev = t
    episodes.append((float(t0), float(prev)))
    return episodes


def false_alarm_episodes(
    clip: ClipEval, threshold: float, merge_gap_s: float = 2.0
) -> int:
    """Count alarm episodes that overlap no event window (= false alarms).

    Args:
        clip: the ClipEval to score.
        threshold: alarm threshold applied to detection scores.
        merge_gap_s: firings closer than this merge into one episode.
    Returns:
        Number of false-alarm episodes on this clip.
    """
    det = clip.detections
    if len(det) == 0:
        return 0
    start, _ = _det_intervals(det, clip.fps)
    fired = start[det["score"].to_numpy(dtype=float) >= threshold]
    count = 0
    for t0, t1 in _episodes(fired, merge_gap_s):
        overlaps = any(
            t0 < ev["end"] and t1 + merge_gap_s > ev["start"] for ev in clip.events
        )
        if not overlaps:
            count += 1
    return count


def normal_hours(clips: list[ClipEval]) -> float:
    """Total footage hours OUTSIDE event windows across clips (the false-alarm exposure).

    Args:
        clips: the per-clip evaluation bundles.
    Returns:
        Hours of non-event footage (float).
    """
    total = 0.0
    for c in clips:
        ev_time = sum(max(ev["end"] - ev["start"], 0.0) for ev in c.events)
        total += max(c.duration_s - ev_time, 0.0)
    return total / 3600.0


def sweep_recall_fa(
    clips: list[ClipEval],
    thresholds: np.ndarray | None = None,
    merge_gap_s: float = 2.0,
) -> pd.DataFrame:
    """Operating-point curve: event recall vs. false alarms/hour, swept over thresholds.

    This is the plot the alarm threshold gets chosen from — recall bought at the price of
    alarm fatigue.
    Args:
        clips: the per-clip evaluation bundles.
        thresholds: thresholds to sweep; default = 61 score quantiles.
        merge_gap_s: firing-merge gap for false-alarm counting.
    Returns:
        DataFrame with threshold, recall, events_caught, false_alarms, fa_per_hour per row.
    """
    all_scores = np.concatenate(
        [
            c.detections["score"].to_numpy(dtype=float)
            for c in clips
            if len(c.detections)
        ]
    )
    n_events = sum(len(c.events) for c in clips)
    if thresholds is None:
        qs = np.linspace(0.0, 1.0, 61)
        thresholds = np.unique(np.quantile(all_scores, qs))
    hours = normal_hours(clips)
    rows = []
    for thr in thresholds:
        catches = [e for c in clips for e in event_catches(c, thr)]
        caught = sum(e["caught"] for e in catches)
        fa = sum(false_alarm_episodes(c, thr, merge_gap_s) for c in clips)
        rows.append(
            {
                "threshold": float(thr),
                "recall": caught / n_events if n_events else np.nan,
                "events_caught": caught,
                "false_alarms": fa,
                "fa_per_hour": fa / hours if hours > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def roc_pr(err_normal: np.ndarray, err_distress: np.ndarray) -> dict | None:
    """Window-level ROC and PR curves + AUCs from per-class error arrays.

    Args:
        err_normal: reconstruction errors of normal windows.
        err_distress: reconstruction errors of distress windows.
    Returns:
        Dict with fpr, tpr, roc_auc, precision, recall, pr_auc; None if either class is empty.
    """
    if len(err_normal) == 0 or len(err_distress) == 0:
        return None
    y = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_distress))])
    s = np.concatenate([err_normal, err_distress])
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(auc(fpr, tpr)),
        "precision": prec,
        "recall": rec,
        "pr_auc": float(auc(rec, prec)),
    }


def percentile_table(err_normal: np.ndarray, err_distress: np.ndarray) -> pd.DataFrame:
    """Side-by-side error percentiles (p10/50/90/99) per class — the 0.539 storyteller.

    Args:
        err_normal: reconstruction errors of normal windows.
        err_distress: reconstruction errors of distress windows.
    Returns:
        DataFrame with one row per non-empty class (class, n, p10, p50, p90, p99).
    """
    pcts = [10, 50, 90, 99]
    rows = []
    for name, e in [("normal", err_normal), ("distress", err_distress)]:
        if len(e) == 0:
            continue
        p = np.percentile(e, pcts)
        rows.append(
            {
                "class": name,
                "n": len(e),
                **{f"p{q}": v for q, v in zip(pcts, p, strict=True)},
            }
        )
    return pd.DataFrame(rows)
