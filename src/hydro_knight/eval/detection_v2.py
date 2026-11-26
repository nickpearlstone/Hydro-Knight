"""Detection metrics for the v2 model: recall, time to alarm, false alarms per hour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.track_series import TARGET_FPS
from ..features.windows_v2 import WindowSet
from .metrics import (
    DET_COLUMNS,
    ClipEval,
    event_catches,
    false_alarm_episodes,
    footage_hours,
    sweep_recall_fa,
)


def alarm_table(ws: WindowSet, scores: np.ndarray) -> pd.DataFrame:
    """One alarm row per window: clip_id, track_id, t (file seconds), score."""
    if not len(ws):
        return pd.DataFrame(columns=["clip_id", "track_id", "t", "score"])
    s_idx, a = ws.index[:, 0], ws.index[:, 1]
    t0 = np.array([ts.t0 for ts in ws.series])[s_idx]
    return pd.DataFrame(
        {
            "clip_id": np.array([ts.clip_id for ts in ws.series], dtype=object)[s_idx],
            "track_id": np.array([ts.track_id for ts in ws.series])[s_idx],
            "t": t0 + (a + ws.window + ws.context) / TARGET_FPS,
            "score": np.asarray(scores, dtype=float),
        }
    )


def clip_evals(alarms: pd.DataFrame, records, rep) -> list[ClipEval]:
    """ClipEvals for every held-out clip, ready for metrics.sweep_recall_fa."""
    by_id = {r.clip_id: r for r in records}
    out = []
    for cid in rep.used_test:
        r = by_id[cid]
        ev = r.events[0]
        onset, contact = r.to_file_time(ev["start"]), r.to_file_time(ev["contact"])
        start = 0.0 if r.trim_start is None else r.to_file_time(r.trim_start)
        a = alarms[(alarms["clip_id"] == cid) & (alarms["t"] <= contact)]
        det = pd.DataFrame(
            {
                "track_id": a["track_id"].to_numpy(int),
                "frame": np.rint(a["t"].to_numpy() * TARGET_FPS).astype(int),
                "score": a["score"].to_numpy(float),
                "span": 1,
            },
            columns=DET_COLUMNS,
        )
        events = [
            {
                "start": onset,
                "contact": contact,
                "end": contact,
                "label": "distress",
                "victim_tracks": list(rep.victim_tracks.get(cid, [])),
            }
        ]
        out.append(
            ClipEval(cid, det, events, fps=TARGET_FPS, duration_s=contact - start)
        )
    return out


def operating_points(
    clips: list[ClipEval], fa_targets=(30, 60, 120, 240), n_thresholds: int = 200
) -> dict:
    """Best recall at or under each false-alarm budget, plus the full sweep.

    Thresholds are picked on the clips they are scored on: an upper bound (see crossfit_points).
    """
    scores = np.concatenate([c.detections["score"].to_numpy(float) for c in clips])
    thr = np.unique(np.quantile(scores, np.linspace(0, 1, n_thresholds)))
    sweep = sweep_recall_fa(clips, thr)
    n_events = sum(len(c.events) for c in clips)
    points = []
    for target in fa_targets:
        ok = sweep[sweep["fa_per_hour"] <= target]
        if ok.empty:
            points.append({"fa_target": target, "recall": 0.0})
            continue
        best = ok.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
        catches = [e for c in clips for e in event_catches(c, best["threshold"])]
        lat = [e["latency_s"] for e in catches if e["caught"]]
        points.append(
            {
                "fa_target": target,
                "threshold": float(best["threshold"]),
                "recall": float(best["recall"]),
                "fa_per_hour": float(best["fa_per_hour"]),
                "caught": int(best["events_caught"]),
                "events": n_events,
                "median_latency_s": float(np.median(lat)) if lat else None,
            }
        )
    return {
        "sweep": sweep,
        "points": points,
        "events": n_events,
        "hours": float(sum(c.duration_s for c in clips) / 3600),
    }


def crossfit_points(
    clips: list[ClipEval],
    folds: dict[str, int],
    fa_targets=(30, 60, 120, 240),
    n_thresholds: int = 200,
) -> list[dict]:
    """operating_points without hindsight: each fold's threshold is picked on the other folds only."""
    by_fold: dict[int, list[ClipEval]] = {}
    for c in clips:
        by_fold.setdefault(folds[c.clip_id], []).append(c)
    totals = {t: {"caught": 0, "fa": 0, "lat": [], "thr": {}} for t in fa_targets}
    for f, test in sorted(by_fold.items()):
        train = [c for c in clips if folds[c.clip_id] != f]
        picked = operating_points(train, fa_targets, n_thresholds)["points"]
        for pt in picked:
            tot = totals[pt["fa_target"]]
            thr = pt.get("threshold", np.inf)
            tot["thr"][str(f)] = float(thr)
            catches = [e for c in test for e in event_catches(c, thr)]
            tot["caught"] += sum(e["caught"] for e in catches)
            tot["lat"] += [e["latency_s"] for e in catches if e["caught"]]
            tot["fa"] += sum(false_alarm_episodes(c, thr) for c in test)
    n_events = sum(len(c.events) for c in clips)
    hours = footage_hours(clips)
    return [
        {
            "fa_target": t,
            "recall": tot["caught"] / n_events if n_events else float("nan"),
            "fa_per_hour": tot["fa"] / hours if hours else float("nan"),
            "caught": tot["caught"],
            "events": n_events,
            "median_latency_s": float(np.median(tot["lat"])) if tot["lat"] else None,
            "threshold_by_fold": tot["thr"],
        }
        for t, tot in totals.items()
    ]
