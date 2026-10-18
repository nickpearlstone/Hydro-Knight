"""
Two-tier evaluation protocol: a direction check that works today, and the full protocol for later.

Input is one neutral score table from any model (columns `clip_id, track_id, t, score`,
where `t` is the file-time second at which a live system could have raised that score),
plus the truth per rescue event (`EventTruth`). Nothing here knows how scores were made.

Shared rules
- A catch is a victim-track score at or above the threshold inside the catch window:
  from `onset_lo - onset_tol_s` to guard contact. `onset_lo` is the earlier of our onset
  and an independent labeller's onset (when one exists), so the onset is a range, not a
  point, and an alarm a few seconds before the labelled onset still counts.
- Rescue clips are scored from the footage start (trim) to guard contact; after contact
  the rescue's commotion is excluded. Victim scores before the catch window are ordinary
  false alarms.
- False alarms are episodes per track (a run of firings with gaps under `merge_gap_s`), so
  twenty swimmers firing at once are twenty alarms.
- Every headline number comes with a random baseline (same table, scores replaced by
  uniform noise) and a bootstrap interval that resamples whole clips.

Tier 1 ("are we heading the right way?") needs only the labelled rescue clips. Its
negatives are the other swimmers in the same clip, so it asks how the victim ranks among
the people a lifeguard is actually watching, never pooled across clips (pooled AUC rewards
telling clips apart). It also reports catches against false alarms counted on the rescue
footage itself, clearly a proxy: crowded, short, and right next to a rescue.

Tier 2 (the real protocol) counts false alarms only on normal footage (`NormalSpan`), gives
exact Poisson (Garwood) intervals on the alarm rate, reports alerts per pool per day next to
alerts per hour, and marks a budget as verified only when the interval's upper end is under
it. With zero alarms in `h` hours the best provable rate is about 3.0 / h, so with a quarter
hour of normal footage no budget under ~12 per hour can be verified.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2
from sklearn.metrics import roc_auc_score

from .metrics import _episodes

SCORE_COLUMNS = ["clip_id", "track_id", "t", "score"]


@dataclass(frozen=True)
class EventTruth:
    """One labelled rescue: times in file seconds."""

    clip_id: str
    start: float  # footage start (trim start, else 0)
    onset: float  # our onset
    contact: float  # guard contact: scoring stops here
    victim_tracks: frozenset[int]
    onset_alt: float | None = None  # independent labeller's onset, if any
    silent: bool = False  # no visible struggle before going under
    # tracks scored as nothing: their rows are dropped before scoring (no catch, no false
    # alarm, not in the AUC, not a rival in the rank), e.g. unconfirmed victim fragments
    ignore_tracks: frozenset[int] = frozenset()

    @property
    def onset_lo(self) -> float:
        return self.onset if self.onset_alt is None else min(self.onset, self.onset_alt)

    @property
    def onset_hi(self) -> float:
        return self.onset if self.onset_alt is None else max(self.onset, self.onset_alt)


@dataclass(frozen=True)
class NormalSpan:
    """A stretch of footage with no incident: any score here that fires is a false alarm."""

    clip_id: str
    start: float
    end: float


@dataclass
class ProtocolConfig:
    onset_tol_s: float = 5.0
    merge_gap_s: float = 2.0
    open_hours_per_day: float = 10.0
    top_k: tuple[int, ...] = (1, 3, 5)
    rank_window_s: float = (
        2.0  # a track's score at time t = its max over (t - window, t]
    )
    step_s: float = 0.5  # time step of the rank timeline
    budgets_per_hour: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 20.0)
    n_thresholds: int = 200
    n_boot: int = 1000
    n_random: int = 20
    conf: float = 0.95
    seed: int = 0


# ---------------------------------------------------------------- truth from the manifest


def truths_from_records(
    records: Iterable,
    clip_ids: Iterable[str],
    victim_tracks: dict[str, Sequence[int]],
    onset_alt: dict[str, float] | None = None,
    silent: Iterable[str] = (),
    ignore_tracks: dict[str, Sequence[int]] | None = None,
) -> dict[str, EventTruth]:
    """EventTruth per clip from manifest records (first event of each clip).

    Args:
        records: manifest ClipRecords.
        clip_ids: the clips to evaluate (e.g. a run's held-out set).
        victim_tracks: clip_id -> matched victim track ids (eval.victims).
        onset_alt: clip_id -> an independent onset in SOURCE seconds (converted here).
        silent: clip ids whose drowning shows no visible struggle.
        ignore_tracks: clip_id -> track ids whose rows are dropped before scoring.
    Returns:
        clip_id -> EventTruth; clips without onset or contact are left out.
    """
    by_id = {r.clip_id: r for r in records}
    onset_alt = onset_alt or {}
    ignore_tracks = ignore_tracks or {}
    silent = set(silent)
    out = {}
    for cid in clip_ids:
        r = by_id[cid]
        ev = r.events[0] if r.events else {}
        if ev.get("start") is None or ev.get("contact") is None:
            continue
        alt = onset_alt.get(cid)
        out[cid] = EventTruth(
            clip_id=cid,
            start=0.0 if r.trim_start is None else r.to_file_time(r.trim_start),
            onset=r.to_file_time(ev["start"]),
            contact=r.to_file_time(ev["contact"]),
            victim_tracks=frozenset(int(t) for t in victim_tracks.get(cid, [])),
            onset_alt=None if alt is None else r.to_file_time(alt),
            silent=cid in silent,
            ignore_tracks=frozenset(int(t) for t in ignore_tracks.get(cid, [])),
        )
    return out


def pre_onset_spans(
    truths: dict[str, EventTruth], onset_tol_s: float = 5.0
) -> list[NormalSpan]:
    """Pseudo-normal footage: each rescue clip from its start to the catch window (a stopgap, label it so)."""
    return [
        NormalSpan(t.clip_id, t.start, t.onset_lo - onset_tol_s)
        for t in truths.values()
        if t.onset_lo - onset_tol_s > t.start
    ]


# ---------------------------------------------------------------- statistics


def garwood(k: int, hours: float, conf: float = 0.95) -> tuple[float, float, float]:
    """Exact Poisson rate and two-sided interval, per hour.

    Returns:
        (rate, lo, hi) alarms per hour; (nan, 0, inf) when there is no footage.
    """
    if hours <= 0:
        return math.nan, 0.0, math.inf
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return k / hours, lo / hours, hi / hours


def zero_alarm_bound(hours: float, conf: float = 0.95) -> float:
    """Lowest alarm rate (per hour) that `hours` of alarm-free footage can prove one-sided (~3.0 / h)."""
    return math.inf if hours <= 0 else -math.log(1 - conf) / hours


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Exact binomial proportion and interval: (p, lo, hi); (nan, 0, 1) when n == 0."""
    if n == 0:
        return math.nan, 0.0, 1.0
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return k / n, float(lo), float(hi)


def bootstrap_clips(
    clip_ids: Sequence[str],
    stat: Callable[[list[str]], float],
    n_boot: int = 1000,
    seed: int = 0,
    conf: float = 0.95,
) -> dict:
    """Point value and percentile interval of `stat`, resampling whole clips with replacement.

    Args:
        clip_ids: the clips the statistic is computed over.
        stat: list of clip ids (with repeats) -> float (nan allowed).
    Returns:
        {"value", "lo", "hi"}.
    """
    ids = list(clip_ids)
    value = stat(ids)
    if not ids:
        return {"value": value, "lo": math.nan, "hi": math.nan}
    rng = np.random.default_rng(seed)
    draws = np.array(
        [stat(list(rng.choice(ids, len(ids)))) for _ in range(n_boot)], float
    )
    a = (1 - conf) / 2 * 100
    lo, hi = (
        np.nanpercentile(draws, [a, 100 - a])
        if np.isfinite(draws).any()
        else (math.nan, math.nan)
    )
    return {"value": value, "lo": float(lo), "hi": float(hi)}


def paired_difference(
    clip_ids: Sequence[str],
    stat_a: Callable[[list[str]], float],
    stat_b: Callable[[list[str]], float],
    n_boot: int = 1000,
    seed: int = 0,
    conf: float = 0.95,
) -> dict:
    """B minus A on the same resampled clips: is model B really better than model A?

    Returns:
        {"diff", "lo", "hi", "p_b_better"}: p_b_better = share of resamples where B > A.
    """
    ids = list(clip_ids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = list(rng.choice(ids, len(ids)))
        diffs.append(stat_b(s) - stat_a(s))
    diffs = np.array(diffs, float)
    a = (1 - conf) / 2 * 100
    lo, hi = np.nanpercentile(diffs, [a, 100 - a])
    return {
        "diff": stat_b(ids) - stat_a(ids),
        "lo": float(lo),
        "hi": float(hi),
        "p_b_better": float(np.nanmean(diffs > 0)),
    }


def randomize(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same rows, scores replaced by uniform noise: the baseline every metric is compared against."""
    return table.assign(score=rng.random(len(table)))


# ---------------------------------------------------------------- per-clip primitives


def _by_clip(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(c): g.sort_values("t") for c, g in table.groupby("clip_id", sort=False)}


def _drop_ignored(
    groups: dict[str, pd.DataFrame], truths: dict[str, EventTruth]
) -> dict[str, pd.DataFrame]:
    """Remove each clip's ignore_tracks rows, so those tracks count as nothing at all."""
    out = dict(groups)
    for cid, t in truths.items():
        if t.ignore_tracks and cid in out:
            g = out[cid]
            out[cid] = g[~g["track_id"].isin(t.ignore_tracks)]
    return out


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCORE_COLUMNS)


def first_catch(
    rows: pd.DataFrame, truth: EventTruth, thr: float, onset_tol_s: float
) -> float | None:
    """Seconds from our onset to the first victim score >= thr in the catch window (negative = early); None = missed."""
    if rows.empty:
        return None
    t = rows["t"].to_numpy(float)
    hit = (
        rows["track_id"].isin(truth.victim_tracks).to_numpy()
        & (rows["score"].to_numpy(float) >= thr)
        & (t >= truth.onset_lo - onset_tol_s)
        & (t <= truth.contact)
    )
    return float(t[hit].min() - truth.onset) if hit.any() else None


def false_alarms(
    rows: pd.DataFrame,
    thr: float,
    *,
    truth: EventTruth | None = None,
    span: NormalSpan | None = None,
    onset_tol_s: float = 5.0,
    merge_gap_s: float = 2.0,
) -> int:
    """False-alarm episodes per track in a rescue clip (start..contact) or a normal span."""
    if rows.empty:
        return 0
    t = rows["t"].to_numpy(float)
    fired = rows["score"].to_numpy(float) >= thr
    tids = rows["track_id"].to_numpy()
    if truth is not None:
        keep = (t >= truth.start) & (t <= truth.contact)
        in_window = np.isin(tids, list(truth.victim_tracks)) & (
            t >= truth.onset_lo - onset_tol_s
        )
        keep &= ~in_window
    else:
        keep = (t >= span.start) & (t <= span.end)
    m = fired & keep
    return sum(
        len(_episodes(t[m & (tids == k)], merge_gap_s)) for k in np.unique(tids[m])
    )


def victim_rank_timeline(
    rows: pd.DataFrame, truth: EventTruth, cfg: ProtocolConfig
) -> pd.DataFrame:
    """The victim's rank among the swimmers in view, every `step_s` across the catch window.

    A track's score at time t is its max over (t - rank_window_s, t]; rank 1 = the most
    suspicious track in the clip at that moment.
    Returns:
        DataFrame t, victim_score (nan = victim not scored then), rank (nan likewise),
        n_tracks (tracks scored in that slice).
    """
    t0, t1 = truth.onset_lo - cfg.onset_tol_s, truth.contact
    times = np.arange(t0, t1 + 1e-9, cfg.step_s)
    tt = rows["t"].to_numpy(float)
    tids = rows["track_id"].to_numpy()
    sc = rows["score"].to_numpy(float)
    vic = np.isin(tids, list(truth.victim_tracks))
    out = []
    for t in times:
        m = (tt > t - cfg.rank_window_s) & (tt <= t)
        if not m.any():
            out.append((t, math.nan, math.nan, 0))
            continue
        best = pd.Series(sc[m]).groupby(tids[m]).max()
        vm = vic[m]
        if not vm.any():
            out.append((t, math.nan, math.nan, len(best)))
            continue
        v = sc[m][vm].max()
        others = best[~best.index.isin(list(truth.victim_tracks))]
        out.append((t, float(v), float(1 + (others > v).sum()), len(best)))
    return pd.DataFrame(out, columns=["t", "victim_score", "rank", "n_tracks"])


def within_clip_auc(rows: pd.DataFrame, truth: EventTruth, onset_tol_s: float) -> float:
    """Chance a victim score in the catch window beats another swimmer's score in the same clip; nan if a side is empty."""
    t = rows["t"].to_numpy(float)
    vic = rows["track_id"].isin(truth.victim_tracks).to_numpy()
    in_clip = (t >= truth.start) & (t <= truth.contact)
    pos = vic & (t >= truth.onset_lo - onset_tol_s) & in_clip
    neg = ~vic & in_clip
    if not pos.any() or not neg.any():
        return math.nan
    s = rows["score"].to_numpy(float)
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    return float(roc_auc_score(y, np.r_[s[pos], s[neg]]))


# ---------------------------------------------------------------- tier 1


def _tier1_per_clip(
    groups: dict[str, pd.DataFrame], truths: dict[str, EventTruth], cfg: ProtocolConfig
) -> pd.DataFrame:
    rows = []
    for cid, truth in truths.items():
        g = groups.get(cid, _empty())
        tl = victim_rank_timeline(g, truth, cfg)
        seen = tl["rank"].notna()
        rec = {
            "clip_id": cid,
            "coverage": float(seen.mean()) if len(tl) else 0.0,
            "median_tracks": float(tl["n_tracks"].median()) if len(tl) else 0.0,
            "best_rank": float(tl["rank"].min()) if seen.any() else math.nan,
            "within_clip_auc": within_clip_auc(g, truth, cfg.onset_tol_s),
        }
        for k in cfg.top_k:
            top = (tl["rank"] <= k).to_numpy()
            rec[f"time_in_top{k}"] = float(top.mean()) if len(tl) else 0.0
            rec[f"first_top{k}_s"] = (
                float(tl["t"].to_numpy()[top][0] - truth.onset)
                if top.any()
                else math.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _tier1_summary(per_clip: pd.DataFrame, cfg: ProtocolConfig) -> dict:
    """Clip-averaged headline numbers with clip-bootstrap intervals."""
    pc = per_clip.set_index("clip_id")
    ids = list(pc.index)

    def mean_of(col):
        return lambda s: (
            float(np.nanmean(pc.loc[s, col].to_numpy(float))) if len(s) else math.nan
        )

    def share_reaching(k):
        return lambda s: (
            float(np.mean(np.isfinite(pc.loc[s, f"first_top{k}_s"].to_numpy(float))))
            if len(s)
            else math.nan
        )

    out = {
        "clips": len(ids),
        "coverage": bootstrap_clips(
            ids, mean_of("coverage"), cfg.n_boot, cfg.seed, cfg.conf
        ),
        "within_clip_auc": bootstrap_clips(
            ids, mean_of("within_clip_auc"), cfg.n_boot, cfg.seed, cfg.conf
        ),
    }
    for k in cfg.top_k:
        out[f"time_in_top{k}"] = bootstrap_clips(
            ids, mean_of(f"time_in_top{k}"), cfg.n_boot, cfg.seed, cfg.conf
        )
        out[f"reach_top{k}"] = bootstrap_clips(
            ids, share_reaching(k), cfg.n_boot, cfg.seed, cfg.conf
        )
        lat = pc[f"first_top{k}_s"].dropna()
        out[f"median_first_top{k}_s"] = float(lat.median()) if len(lat) else math.nan
    return out


def _thresholds(scores: np.ndarray, n: int) -> np.ndarray:
    """Score quantiles plus +inf (fire nothing), so a zero-alarm operating point always exists."""
    q = (
        np.unique(np.quantile(scores, np.linspace(0, 1, n)))
        if len(scores)
        else np.array([])
    )
    return np.r_[q, np.inf]


def proxy_operating_points(
    groups: dict[str, pd.DataFrame], truths: dict[str, EventTruth], cfg: ProtocolConfig
) -> dict:
    """Catches vs false alarms counted on the rescue footage itself (start..contact): a proxy budget curve."""
    hours = sum(t.contact - t.start for t in truths.values()) / 3600
    scores = (
        np.concatenate([g["score"].to_numpy(float) for g in groups.values()])
        if groups
        else np.array([])
    )
    sweep = []
    for thr in _thresholds(scores, cfg.n_thresholds):
        caught = [
            first_catch(groups.get(c, _empty()), t, thr, cfg.onset_tol_s)
            for c, t in truths.items()
        ]
        fa = sum(
            false_alarms(
                groups.get(c, _empty()),
                thr,
                truth=t,
                onset_tol_s=cfg.onset_tol_s,
                merge_gap_s=cfg.merge_gap_s,
            )
            for c, t in truths.items()
        )
        sweep.append(
            {
                "threshold": float(thr),
                "caught": sum(c is not None for c in caught),
                "false_alarms": fa,
                "latencies": [c for c in caught if c is not None],
            }
        )
    points = []
    for b in cfg.budgets_per_hour:
        ok = [s for s in sweep if hours > 0 and s["false_alarms"] / hours <= b]
        if not ok:
            points.append({"budget_per_hour": b, "caught": 0, "events": len(truths)})
            continue
        best = max(ok, key=lambda s: (s["caught"], s["threshold"]))
        rate, lo, hi = garwood(best["false_alarms"], hours, cfg.conf)
        rec, rlo, rhi = clopper_pearson(best["caught"], len(truths), cfg.conf)
        points.append(
            {
                "budget_per_hour": b,
                "threshold": best["threshold"],
                "caught": best["caught"],
                "events": len(truths),
                "recall": rec,
                "recall_lo": rlo,
                "recall_hi": rhi,
                "fa_per_hour": rate,
                "fa_lo": lo,
                "fa_hi": hi,
                "median_latency_s": float(np.median(best["latencies"]))
                if best["latencies"]
                else math.nan,
            }
        )
    return {"hours": hours, "points": points}


def tier1(
    table: pd.DataFrame,
    truths: dict[str, EventTruth],
    cfg: ProtocolConfig | None = None,
) -> dict:
    """Direction check from rescue clips alone: victim rank among swimmers in view, coverage, proxy budget curve.

    Args:
        table: score table (SCORE_COLUMNS).
        truths: clip_id -> EventTruth for the evaluated rescue clips.
    Returns:
        {"per_clip": DataFrame, "summary": {...}, "random": {...} (same keys, mean over
        n_random noise tables), "proxy": proxy_operating_points, "proxy_random": [...]}.
    """
    cfg = cfg or ProtocolConfig()
    groups = _drop_ignored(_by_clip(table), truths)
    per_clip = _tier1_per_clip(groups, truths, cfg)
    rng = np.random.default_rng(cfg.seed)
    rand_runs, proxy_rand = [], []
    quick = ProtocolConfig(**{**cfg.__dict__, "n_boot": 1})
    for _ in range(cfg.n_random):
        rg = _drop_ignored(_by_clip(randomize(table, rng)), truths)
        rand_runs.append(_tier1_summary(_tier1_per_clip(rg, truths, cfg), quick))
        proxy_rand.append(proxy_operating_points(rg, truths, quick)["points"])
    keys = (
        [k for k in rand_runs[0] if isinstance(rand_runs[0][k], dict)]
        if rand_runs
        else []
    )
    random = {k: float(np.nanmean([r[k]["value"] for r in rand_runs])) for k in keys}
    return {
        "per_clip": per_clip,
        "summary": _tier1_summary(per_clip, cfg),
        "random": random,
        "proxy": proxy_operating_points(groups, truths, cfg),
        "proxy_random_caught": {
            p["budget_per_hour"]: float(
                np.mean([run[i].get("caught", 0) for run in proxy_rand])
            )
            for i, p in enumerate(proxy_rand[0])
        }
        if proxy_rand
        else {},
    }


# ---------------------------------------------------------------- tier 2


def tier2(
    table: pd.DataFrame,
    truths: dict[str, EventTruth],
    normal: list[NormalSpan],
    cfg: ProtocolConfig | None = None,
) -> dict:
    """Full protocol: false alarms on normal footage only, recall at fixed budgets, latency curve, readiness.

    Args:
        table: score table covering the rescue clips AND the normal spans' clips.
        truths: clip_id -> EventTruth for rescue clips.
        normal: incident-free footage spans; false alarms are counted only here.
    Returns:
        {"normal_hours", "min_verifiable_per_hour", "points": [...], "amoc": DataFrame,
        "silent": {...} or None}. A point is "verified" when the Garwood upper bound on
        its alarm rate is within the budget.
    """
    cfg = cfg or ProtocolConfig()
    groups = _drop_ignored(_by_clip(table), truths)
    hours = sum(max(s.end - s.start, 0) for s in normal) / 3600

    def span_scores(sp: NormalSpan) -> np.ndarray:
        g = groups.get(sp.clip_id, _empty())
        return g["score"].to_numpy(float)[
            (g["t"] >= sp.start).to_numpy() & (g["t"] <= sp.end).to_numpy()
        ]

    normal_scores = (
        np.concatenate([span_scores(sp) for sp in normal]) if normal else np.array([])
    )
    all_scores = table["score"].to_numpy(float)
    thrs = _thresholds(np.r_[normal_scores, all_scores], cfg.n_thresholds)
    rows = []
    for thr in thrs:
        fa = sum(
            false_alarms(
                groups.get(s.clip_id, _empty()),
                thr,
                span=s,
                merge_gap_s=cfg.merge_gap_s,
            )
            for s in normal
        )
        lats = {
            c: first_catch(groups.get(c, _empty()), t, thr, cfg.onset_tol_s)
            for c, t in truths.items()
        }
        rows.append({"threshold": float(thr), "false_alarms": fa, "lats": lats})
    points = []
    for b in cfg.budgets_per_hour:
        ok = [r for r in rows if hours > 0 and r["false_alarms"] / hours <= b]
        if not ok:
            points.append(
                {
                    "budget_per_hour": b,
                    "budget_per_day": b * cfg.open_hours_per_day,
                    "caught": 0,
                    "events": len(truths),
                    "verified": False,
                }
            )
            continue
        best = max(
            ok,
            key=lambda r: (
                sum(v is not None for v in r["lats"].values()),
                r["threshold"],
            ),
        )
        caught = [c for c, v in best["lats"].items() if v is not None]
        rate, lo, hi = garwood(best["false_alarms"], hours, cfg.conf)
        rec, rlo, rhi = clopper_pearson(len(caught), len(truths), cfg.conf)
        lat = [best["lats"][c] for c in caught]
        lat_lo = [
            best["lats"][c] + truths[c].onset - truths[c].onset_lo for c in caught
        ]
        points.append(
            {
                "budget_per_hour": b,
                "budget_per_day": b * cfg.open_hours_per_day,
                "threshold": best["threshold"],
                "caught": len(caught),
                "events": len(truths),
                "recall": rec,
                "recall_lo": rlo,
                "recall_hi": rhi,
                "fa_per_hour": rate,
                "fa_lo": lo,
                "fa_hi": hi,
                "alerts_per_day": rate * cfg.open_hours_per_day,
                "verified": bool(hi <= b),
                "median_latency_s": float(np.median(lat)) if lat else math.nan,
                "median_latency_from_earliest_onset_s": float(np.median(lat_lo))
                if lat_lo
                else math.nan,
            }
        )
    amoc = pd.DataFrame(
        [
            {
                "threshold": r["threshold"],
                "fa_per_hour": r["false_alarms"] / hours if hours > 0 else math.nan,
                # misses count as the whole window: a lifeguard learns at contact anyway
                "mean_delay_s": float(
                    np.mean(
                        [
                            (
                                v
                                if v is not None
                                else truths[c].contact - truths[c].onset
                            )
                            for c, v in r["lats"].items()
                        ]
                    )
                )
                if truths
                else math.nan,
            }
            for r in rows
        ]
    )
    silent_ids = [c for c, t in truths.items() if t.silent]
    silent = None
    if silent_ids:
        silent = {
            "events": len(silent_ids),
            "caught_at_budget": {
                p["budget_per_hour"]: sum(
                    1
                    for c in silent_ids
                    if p.get("threshold") is not None
                    and first_catch(
                        groups.get(c, _empty()),
                        truths[c],
                        p["threshold"],
                        cfg.onset_tol_s,
                    )
                    is not None
                )
                for p in points
            },
        }
    return {
        "normal_hours": hours,
        "min_verifiable_per_hour": zero_alarm_bound(hours, cfg.conf),
        "points": points,
        "amoc": amoc,
        "silent": silent,
    }


# ---------------------------------------------------------------- report


def _s(x: float, fmt: str = "{:.1f} s") -> str:
    return fmt.format(x) if np.isfinite(x) else "-"


def _ci(d: dict, pct: bool = False) -> str:
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    if not np.isfinite(d.get("value", math.nan)):
        return "n/a"
    return f"{f(d['value'])} [{f(d['lo'])}, {f(d['hi'])}]"


def format_variants(variants: dict[str, dict], cfg: ProtocolConfig) -> str:
    """One row per victim definition (name -> tier1 result): how much the result depends on that choice."""
    k = 3 if 3 in cfg.top_k else cfg.top_k[0]
    lines = [
        f"| Victim definition | Clips | Visible | Same-clip AUC | Time in top {k} | Reach top {k} | Reach top {k}, random |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, t1 in variants.items():
        s, r = t1["summary"], t1["random"]
        lines.append(
            f"| {name} | {s['clips']} | {_ci(s['coverage'], True)} | {_ci(s['within_clip_auc'])} "
            f"| {_ci(s[f'time_in_top{k}'], True)} | {_ci(s[f'reach_top{k}'], True)} | {100 * r[f'reach_top{k}']:.0f}% |"
        )
    return "\n".join(lines)


def format_report(
    name: str, t1: dict | None, t2: dict | None, cfg: ProtocolConfig
) -> str:
    """Markdown summary of tier 1 and tier 2 results (brackets = 95% clip-bootstrap or exact interval)."""
    lines = [f"# Evaluation protocol: {name}", ""]
    if t1:
        s, r = t1["summary"], t1["random"]
        lines += [
            "## Tier 1: direction check (rescue clips only)",
            "",
            f"{s['clips']} rescue clips; catch window = earliest onset - {cfg.onset_tol_s:g} s to guard contact.",
            "",
            "| Measure | Model | Random |",
            "|---|---|---|",
            f"| Victim visible to the model (share of catch window) | {_ci(s['coverage'], True)} | (not score-based) |",
            f"| Victim vs other swimmers, same clip (AUC) | {_ci(s['within_clip_auc'])} | {r['within_clip_auc']:.2f} |",
        ]
        for k in cfg.top_k:
            lines.append(
                f"| Share of catch window victim is top {k} | {_ci(s[f'time_in_top{k}'], True)} | {100 * r[f'time_in_top{k}']:.0f}% |"
            )
        for k in cfg.top_k:
            lines.append(
                f"| Clips where victim reaches top {k} before contact | {_ci(s[f'reach_top{k}'], True)} | {100 * r[f'reach_top{k}']:.0f}% |"
            )
        lines += [
            "",
            f"Median time from onset to first top-3: {_s(s.get('median_first_top3_s', math.nan))}.",
            "",
        ]
        p = t1["proxy"]
        lines += [
            f"**Proxy budget curve** (false alarms counted on {p['hours'] * 60:.0f} min of rescue footage, crowded and "
            "next to a rescue; not normal footage):",
            "",
            "| Budget / h | Caught (95%) | Random caught | Alarms / h (95%) | Median delay |",
            "|---|---|---|---|---|",
        ]
        for pt in p["points"]:
            if "recall" not in pt:
                lines.append(
                    f"| {pt['budget_per_hour']:g} | 0/{pt['events']} | - | - | - |"
                )
                continue
            lines.append(
                f"| {pt['budget_per_hour']:g} | {pt['caught']}/{pt['events']} ({100 * pt['recall_lo']:.0f}-{100 * pt['recall_hi']:.0f}%) "
                f"| {t1['proxy_random_caught'].get(pt['budget_per_hour'], math.nan):.1f} "
                f"| {pt['fa_per_hour']:.1f} ({pt['fa_lo']:.1f}-{pt['fa_hi']:.1f}) | {_s(pt['median_latency_s'])} |"
            )
        lines.append("")
    if t2:
        lines += [
            "## Tier 2: full protocol (false alarms on normal footage)",
            "",
            f"Normal footage: {t2['normal_hours']:.2f} h. Lowest alarm rate this footage can prove: "
            f"{t2['min_verifiable_per_hour']:.1f} per hour ({t2['min_verifiable_per_hour'] * cfg.open_hours_per_day:.0f} "
            f"per pool per {cfg.open_hours_per_day:g}-hour day).",
            "",
            "| Budget / h | Per day | Caught (95%) | Alarms / h (95%) | Verified | Median delay |",
            "|---|---|---|---|---|---|",
        ]
        for pt in t2["points"]:
            if "recall" not in pt:
                lines.append(
                    f"| {pt['budget_per_hour']:g} | {pt['budget_per_day']:g} | 0/{pt['events']} | - | no | - |"
                )
                continue
            lines.append(
                f"| {pt['budget_per_hour']:g} | {pt['budget_per_day']:g} | {pt['caught']}/{pt['events']} "
                f"({100 * pt['recall_lo']:.0f}-{100 * pt['recall_hi']:.0f}%) | {pt['fa_per_hour']:.2f} "
                f"({pt['fa_lo']:.2f}-{pt['fa_hi']:.2f}) | {'yes' if pt['verified'] else 'no'} | {_s(pt['median_latency_s'])} |"
            )
        if t2["silent"]:
            lines += [
                "",
                f"Silent submersions: {t2['silent']['events']} events, caught per budget: {t2['silent']['caught_at_budget']}",
            ]
        lines.append("")
    return "\n".join(lines)
