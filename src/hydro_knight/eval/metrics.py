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

Scoring is about the victim, never just the time window. Each event lists the
victim's track ids ("victim_tracks", from matching the labeler's clicks to tracks):
  - catch:       an alarm on a victim track between onset and guard contact
  - false alarm: any other alarm episode (another swimmer, the guard, or the
                 victim before onset), counted per track
  - excluded:    the victim's own alarms after contact (the rescue itself)
A clip whose events have no victim_tracks yet can't be scored; is_scorable()
says which, and the report lists the clips it skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

DET_COLUMNS = ["track_id", "frame", "score", "span"]


def resolve_events(events: list[dict], clip_end_s: float) -> list[dict]:
    """Events ready for scoring: unlabeled onsets dropped, open saves closed at clip end.

    The labeler stores None for an onset not yet marked and for a save that happens
    after the clip ends ("saved_off_camera"); scoring needs both ends as numbers.
    Args:
        events: manifest events (source seconds).
        clip_end_s: end of the clip in the same timeline.
    Returns:
        Copies of the events with a start, each with a numeric "end".
    """
    return [
        {**ev, "end": clip_end_s if ev.get("end") is None else ev["end"]}
        for ev in events
        if ev.get("start") is not None
    ]


@dataclass
class ClipEval:
    """Everything the harness needs to know about one clip."""

    clip_id: str
    detections: pd.DataFrame  # columns DET_COLUMNS
    # {"start","end","label"} sec, plus "contact" and "victim_tracks" once victims are matched
    events: list[dict] = field(default_factory=list)
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


def is_scorable(clip: ClipEval) -> bool:
    """True when every event in the clip has its victim's tracks matched."""
    return all("victim_tracks" in ev for ev in clip.events)


def _victim_spans(clip: ClipEval) -> list[tuple[set[int], float, float, float]]:
    """(victim track ids, onset, end of catch window, event end) per event, in seconds.

    The catch window closes at guard contact (after that the guard has found the victim);
    without a contact time it runs to the event end.
    Raises:
        ValueError: an event has no victim_tracks (run victim matching first).
    """
    spans = []
    for ev in clip.events:
        if "victim_tracks" not in ev:
            raise ValueError(f"{clip.clip_id}: event has no victim_tracks")
        contact = ev.get("contact")
        catch_end = ev["end"] if contact is None else contact
        spans.append((set(ev["victim_tracks"]), ev["start"], catch_end, ev["end"]))
    return spans


def classify_detections(clip: ClipEval) -> np.ndarray:
    """Per detection: 1 = victim in distress, 0 = normal, -1 = excluded (victim being rescued).

    Victim distress means a victim track overlapping onset..contact; the victim's windows
    from contact to the event end are the rescue and belong to neither class.
    Args:
        clip: a scorable ClipEval.
    Returns:
        (len(detections),) int8 array.
    """
    det = clip.detections
    out = np.zeros(len(det), dtype=np.int8)
    if len(det) == 0:
        return out
    start, end = _det_intervals(det, clip.fps)
    tids = det["track_id"].to_numpy()
    for victims, onset, catch_end, ev_end in _victim_spans(clip):
        on_victim = np.isin(tids, list(victims))
        rescue = on_victim & (start < ev_end) & (end > catch_end)
        out[rescue & (out == 0)] = -1
        out[on_victim & (start < catch_end) & (end > onset)] = 1
    return out


def label_detections(clip: ClipEval) -> np.ndarray:
    """Boolean mask: True where a detection is the victim in distress (see classify_detections)."""
    return classify_detections(clip) == 1


def split_scores(clips: list[ClipEval]) -> tuple[np.ndarray, np.ndarray]:
    """Split all detection scores across clips into (normal, distress) by the victim rule.

    The victim's rescue windows (after guard contact) are left out of both classes. Rows flagged `trained_on` (the --train-fresh holdout) are excluded: scoring the model on
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
        cls = classify_detections(c)
        s = det["score"].to_numpy(dtype=float)
        if "trained_on" in det.columns:
            keep = ~det["trained_on"].to_numpy(dtype=bool)
            cls, s = cls[keep], s[keep]
        normal.append(s[cls == 0])
        distress.append(s[cls == 1])
    cat = lambda parts: np.concatenate(parts) if parts else np.array([])  # noqa: E731
    return cat(normal), cat(distress)


def event_catches(clip: ClipEval, threshold: float) -> list[dict]:
    """Per-event verdicts at a given alarm threshold: caught? and latency-to-first-catch.

    Caught if a detection on a victim track scoring >= threshold overlaps onset..guard
    contact. Latency is that first detection's start minus onset, clipped at 0.
    Args:
        clip: a scorable ClipEval.
        threshold: alarm threshold applied to detection scores.
    Returns:
        One dict per event: clip_id, event_label, event_start, catch_end, caught (bool),
        latency_s (seconds, NaN if uncaught).
    """
    det = clip.detections
    if len(det):
        start, end = _det_intervals(det, clip.fps)
        fired = det["score"].to_numpy(dtype=float) >= threshold
        tids = det["track_id"].to_numpy()
    out = []
    for ev, (victims, onset, catch_end, _) in zip(
        clip.events, _victim_spans(clip), strict=True
    ):
        caught, latency = False, np.nan
        if len(det):
            hit = fired & np.isin(tids, list(victims))
            hit &= (start < catch_end) & (end > onset)
            caught = bool(hit.any())
            if caught:
                latency = float(max(start[hit].min() - onset, 0.0))
        out.append(
            {
                "clip_id": clip.clip_id,
                "event_label": ev.get("label", "distress"),
                "event_start": onset,
                "catch_end": catch_end,
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


def _false_firings(clip: ClipEval, threshold: float):
    """(start times, track ids) of every above-threshold detection that isn't the victim
    between onset and the event end."""
    det = clip.detections
    start, end = _det_intervals(det, clip.fps)
    false = det["score"].to_numpy(dtype=float) >= threshold
    tids = det["track_id"].to_numpy()
    for victims, onset, _, ev_end in _victim_spans(clip):
        false &= ~(np.isin(tids, list(victims)) & (start < ev_end) & (end > onset))
    return start[false], tids[false]


def false_alarm_episodes(
    clip: ClipEval, threshold: float, merge_gap_s: float = 2.0
) -> int:
    """Count false-alarm episodes: alarm runs on any track other than the victim in distress.

    Episodes are counted per track, so twenty swimmers firing at once are twenty false
    alarms, never one. The victim's alarms from onset to the event end (distress, then
    rescue) are not false alarms; the victim's alarms before onset are.

    This count can fall as the threshold drops: a flickering alarm (several episodes)
    becomes one continuous alarm. falsely_flagged_tracks never falls.
    Args:
        clip: a scorable ClipEval.
        threshold: alarm threshold applied to detection scores.
        merge_gap_s: one track's firings closer than this merge into one episode.
    Returns:
        Number of false-alarm episodes on this clip.
    """
    if len(clip.detections) == 0:
        return 0
    times, tids = _false_firings(clip, threshold)
    return sum(len(_episodes(times[tids == t], merge_gap_s)) for t in np.unique(tids))


def falsely_flagged_tracks(clip: ClipEval, threshold: float) -> int:
    """How many swimmer tracks raise at least one false alarm (rises as the threshold drops)."""
    if len(clip.detections) == 0:
        return 0
    return len(np.unique(_false_firings(clip, threshold)[1]))


def footage_hours(clips: list[ClipEval]) -> float:
    """Total hours of footage scored: any swimmer can raise a false alarm at any time.

    Args:
        clips: the per-clip evaluation bundles.
    Returns:
        Hours (float).
    """
    return sum(c.duration_s for c in clips) / 3600.0


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
        DataFrame with threshold, recall, events_caught, false_alarms, fa_per_hour and
        flagged_tracks (swimmers falsely flagged at least once) per row.
    Raises:
        ValueError: a clip isn't scorable (filter with is_scorable first).
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
    hours = footage_hours(clips)
    rows = []
    for thr in thresholds:
        catches = [e for c in clips for e in event_catches(c, thr)]
        caught = sum(e["caught"] for e in catches)
        fa = sum(false_alarm_episodes(c, thr, merge_gap_s) for c in clips)
        flagged = sum(falsely_flagged_tracks(c, thr) for c in clips)
        rows.append(
            {
                "threshold": float(thr),
                "recall": caught / n_events if n_events else np.nan,
                "events_caught": caught,
                "false_alarms": fa,
                "fa_per_hour": fa / hours if hours > 0 else np.nan,
                "flagged_tracks": flagged,
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
