"""
Figures attached to MLflow runs (one function per graph, each returns a matplotlib Figure
or None when there is nothing to draw). Nothing here imports MLflow, so the graphs are
tested without the optional `tracking` extra, and the same figures can be saved to disk.

Figures are built with matplotlib.figure.Figure directly, never pyplot, so no global
figure registry exists to leak memory across a long run; tracking.log_figure saves them.

Chart conventions follow eval/report.py: recessive grey axes, the model in blue, the two
baselines in grey (random dashed, track-shuffled dotted) and always direct-labelled.
Seeds and models take categorical colours in a fixed order.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import protocol as P
from .report import GRID, INK, MUTED, NORMAL, SURFACE, _style

# Categorical slots in fixed order (validated palette, light mode): seeds, models.
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
ONSET = "#e34948"  # onset marker, same red as "distress" in report.py
CONTACT = "#0b0b0b"


def _fig(w: float, h: float, nrows: int = 1, ncols: int = 1, **kw):
    fig = Figure(figsize=(w, h), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over n points (shorter at the start), same length as y."""
    if n <= 1 or len(y) < 2:
        return y
    c = np.cumsum(np.r_[0.0, y])
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) / (idx - lo)


def _spread(ys, gap: float = 0.07) -> list[float]:
    """Label heights near ys, pushed apart so no two are closer than `gap` (data units)."""
    ys = [0.0 if not np.isfinite(y) else float(y) for y in ys]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:], strict=False):
        out[b] = max(out[b], out[a] + gap)
    return out


def loss_curves(histories: dict[str, dict[int, list[float]]]) -> Figure | None:
    """Training loss per step, one panel per model size, one line per seed.

    Args:
        histories: size -> seed -> per-step training loss.
    Returns:
        Figure, or None if no history has a finite value (e.g. re-scored checkpoints).
    """
    sizes = [
        s
        for s, by_seed in histories.items()
        if any(
            np.isfinite(h).any() for h in map(np.asarray, by_seed.values()) if len(h)
        )
    ]
    if not sizes:
        return None
    seeds = sorted({seed for s in sizes for seed in histories[s]})
    fig, axes = _fig(4.2 * len(sizes), 3.2, 1, len(sizes), sharey=True)
    for ax, size in zip(axes[0], sizes, strict=True):
        for seed, hist in sorted(histories[size].items()):
            y = np.asarray(hist, float)
            if not len(y):
                continue
            color = SERIES[seeds.index(seed) % len(SERIES)]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=color, linewidth=0.6, alpha=0.25)
            ax.plot(
                x,
                _smooth(y, max(1, len(y) // 40)),
                color=color,
                linewidth=2,
                label=f"seed {seed}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title(f"{size}: training loss", loc="left", fontsize=10)
    axes[0][0].set_ylabel("masked reconstruction error (log)")
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK)
    return fig


def victim_timelines(
    table: pd.DataFrame,
    truths: dict[str, P.EventTruth],
    cfg: P.ProtocolConfig,
    per_clip: pd.DataFrame | None = None,
    ncols: int = 3,
) -> Figure | None:
    """The victim's rank among swimmers in view across each test clip's catch window.

    One panel per clip: x is seconds from our onset, y the victim's outright rank (1 at
    the top), the grey line the number of swimmers scored at that moment (the worst
    possible rank), the band the top-3 zone. Red line = onset (shaded between ours and the
    independent labeller's when both exist), black line = guard contact. Gaps in the blue
    line are moments the model had no score for the victim.

    Args:
        table: score table (P.SCORE_COLUMNS) for the test clips.
        truths: clip_id -> EventTruth.
        per_clip: tier-1 per-clip table; adds within-clip AUC to each panel title.
    Returns:
        Figure, or None with no clips.
    """
    if not truths:
        return None
    groups = P._drop_ignored(P._by_clip(table), truths)
    auc = (
        per_clip.set_index("clip_id")["within_clip_auc"].to_dict()
        if per_clip is not None and len(per_clip)
        else {}
    )
    cids = sorted(truths)
    ncols = min(ncols, len(cids))
    nrows = math.ceil(len(cids) / ncols)
    fig, axes = _fig(3.8 * ncols, 2.8 * nrows + 0.9, nrows, ncols)
    for ax in axes.flat[len(cids) :]:
        ax.set_visible(False)
    for k, (ax, cid) in enumerate(zip(axes.flat, cids, strict=False)):
        tr = truths[cid]
        tl = P.victim_rank_timeline(groups.get(cid, P._empty()), tr, cfg)
        x = tl["t"].to_numpy(float) - tr.onset
        n = tl["n_tracks"].to_numpy(float)
        rank = tl["rank"].to_numpy(float)
        ax.axhspan(0.5, 3.5, color=GRID, alpha=0.8, linewidth=0, label="top 3")
        ax.plot(
            x,
            np.where(n > 0, n, np.nan),
            color=MUTED,
            linewidth=1,
            label="swimmers scored (= worst possible rank)",
        )
        ax.plot(x, rank, color=NORMAL, linewidth=2, label="victim's rank")
        if tr.onset_alt is not None:
            ax.axvspan(
                tr.onset_lo - tr.onset,
                tr.onset_hi - tr.onset,
                color=ONSET,
                alpha=0.12,
                linewidth=0,
                label="onset range (two labellers)",
            )
        ax.axvline(0, color=ONSET, linewidth=1.5, label="drowning onset")
        ax.axvline(
            tr.contact - tr.onset, color=CONTACT, linewidth=1.5, label="guard contact"
        )
        top = max(3.5, float(np.nanmax(n)) + 1 if np.isfinite(n).any() else 3.5)
        ax.set_ylim(top, 0.5)  # rank 1 at the top
        for xv, txt, c, ha in (
            (0, " onset", ONSET, "left"),
            (tr.contact - tr.onset, "contact ", CONTACT, "right"),
        ):
            ax.text(
                xv,
                1.0,
                txt,
                color=c,
                fontsize=7,
                ha=ha,
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )
        if not np.isfinite(rank).any():
            ax.text(
                0.5,
                0.5,
                "victim never scored\nin this window",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=8,
                transform=ax.transAxes,
            )
        a = auc.get(cid)
        auc_txt = (
            f"within-clip AUC {a:.2f}"
            if a is not None and np.isfinite(a)
            else "AUC n/a"
        )
        ax.set_title(f"clip {cid[:8]}   {auc_txt}", loc="left", fontsize=9, pad=12)
        ax.tick_params(labelsize=8)
        if k % ncols == 0:
            ax.set_ylabel("victim rank (1 = top)", fontsize=8)
        if k + ncols >= len(cids):  # lowest visible panel in its column
            ax.set_xlabel("seconds from onset", fontsize=8)
    handles, labels = {}, []
    for ax in axes.flat[: len(cids)]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    labels = list(handles)
    fig.legend(
        [handles[lab] for lab in labels],
        labels,
        loc="outside lower center",
        ncols=min(len(labels), 6),
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    fig.suptitle(
        "Where the victim ranks among swimmers in view, from onset to guard contact\n"
        "one panel per held-out clip; higher on the chart = the model finds them"
        " more unusual than the others",
        color=INK,
        fontsize=10,
    )
    return fig


def catch_vs_false_alarms(
    proxy: dict, random_caught: dict, shuffled_caught: dict
) -> Figure | None:
    """Rescues caught at each false-alarm budget for the model and both baselines on one axis.

    Budgets are false alarms per hour counted on the rescue footage itself (the tier-1
    proxy). The model's band is its 95% Clopper-Pearson interval on the share caught.

    Args:
        proxy: tier1(...)["proxy"] ({"hours", "points"}).
        random_caught, shuffled_caught: budget -> mean rescues caught by the baseline.
    Returns:
        Figure, or None with no operating points.
    """
    pts = proxy.get("points", [])
    if not pts:
        return None
    n = max(p.get("events", 0) for p in pts) or 1
    b = np.array([p["budget_per_hour"] for p in pts], float)
    model = np.array([p.get("caught", 0) for p in pts], float) / n
    fig, axes = _fig(5.6, 3.6)
    ax = axes[0][0]
    if all("recall_lo" in p for p in pts):
        ax.fill_between(
            b,
            [p["recall_lo"] for p in pts],
            [p["recall_hi"] for p in pts],
            color=NORMAL,
            alpha=0.15,
            linewidth=0,
            label="model, 95% range (few rescues, so wide)",
        )
        ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK)
    lines = [("model", model, NORMAL, "-", 2.0)]
    for label, d, style in (
        ("random scores", random_caught, "--"),
        ("track-shuffled", shuffled_caught, ":"),
    ):
        if d:
            y = np.array([d.get(x, d.get(float(x), np.nan)) for x in b], float) / n
            lines.append((label, y, MUTED, style, 1.6))
    for _label, y, color, style, lw in lines:
        ax.plot(b, y, color=color, linestyle=style, linewidth=lw, marker="o", ms=4)
    for (label, *_), y_lab in zip(
        lines, _spread([ln[1][-1] for ln in lines]), strict=True
    ):
        ax.annotate(
            label,
            (b[-1], y_lab),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
            annotation_clip=False,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-alarm budget (per hour of rescue footage, log)")
    ax.set_ylabel(f"share of {n} rescues caught")
    ax.set_title(
        "Rescues caught before guard contact vs false alarms allowed\n"
        f"higher = better; {proxy.get('hours', 0):.2f} h of rescue footage,"
        " baselines are averages over reshuffles",
        loc="left",
        fontsize=9,
    )
    return fig


def per_clip_results(per_clip: pd.DataFrame) -> Figure | None:
    """Per test clip: within-clip AUC, time in the top 3, and victim detected vs scored.

    Clips are sorted by AUC so the three panels line up row by row. AUC 0.5 and the
    share-of-time axes all run 0 to 1.

    Args:
        per_clip: tier-1 per-clip table (clip_id, within_clip_auc, time_in_top3,
            detected, scored).
    Returns:
        Figure, or None with no clips.
    """
    if per_clip is None or not len(per_clip):
        return None
    pc = per_clip.sort_values("within_clip_auc", na_position="first").reset_index(
        drop=True
    )
    y = np.arange(len(pc))
    labels = [c[:8] for c in pc["clip_id"]]
    fig, axes = _fig(12, 0.32 * len(pc) + 1.8, 1, 3, sharey=True)
    a1, a2, a3 = axes[0]
    auc = pc["within_clip_auc"].to_numpy(float)
    a1.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    a1.hlines(y, 0.5, auc, color=NORMAL, linewidth=2)
    a1.plot(auc, y, "o", color=NORMAL, ms=6)
    a1.text(0.51, -0.6, "chance", color=MUTED, fontsize=7, va="top")
    a1.set_xlim(0, 1)
    a1.set_title(
        "within-clip AUC\n0.5 = chance, 1 = victim always above others",
        loc="left",
        fontsize=9,
    )
    a1.set_yticks(y, labels, fontsize=8)
    a1.set_ylabel("held-out clip (sorted by AUC)", fontsize=8)
    if "time_in_top3" in pc:
        a2.barh(y, pc["time_in_top3"].to_numpy(float), color=NORMAL, height=0.6)
    a2.set_xlim(0, 1)
    a2.set_title(
        "share of onset-to-contact time\nthe victim is in the top 3",
        loc="left",
        fontsize=9,
    )
    h = 0.38
    a3.barh(
        y + h / 2,
        pc["detected"].to_numpy(float),
        height=h,
        color=SERIES[0],
        label="detected (input tracks)",
    )
    a3.barh(
        y - h / 2,
        pc["scored"].to_numpy(float),
        height=h,
        color=SERIES[1],
        label="scored by this model",
    )
    a3.set_xlim(0, 1)
    a3.set_title(
        "share of time the victim is tracked\n(detected) vs given a score (scored)",
        loc="left",
        fontsize=9,
    )
    a3.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
    for ax in axes[0]:
        ax.set_xlabel("share" if ax is not a1 else "AUC")
    return fig


def cross_run_comparison(
    runs: pd.DataFrame,
    metric: str = "headline.within_clip_auc",
    baselines: tuple[str, ...] = (
        "headline.random.within_clip_auc",
        "headline.shuffled.within_clip_auc",
    ),
    group: str = "run_folder",
) -> Figure | None:
    """One metric across many evaluated runs: a dot per seed, a bar per run folder, baselines as ticks.

    Args:
        runs: one row per seed-level run, columns `group`, `metric` and the baseline
            columns (e.g. flattened from mlflow.search_runs).
    Returns:
        Figure, or None if no row has the metric.
    """
    if metric not in runs or not runs[metric].notna().any():
        return None
    df = runs.dropna(subset=[metric])
    order = df.groupby(group)[metric].mean().sort_values().index.tolist()
    fig, axes = _fig(7, 0.45 * len(order) + 1.9)
    ax = axes[0][0]
    names = ("random-score baseline (mean)", "track-shuffled baseline (mean)")
    for i, name in enumerate(order):
        g = df[df[group] == name]
        ax.barh(
            i,
            g[metric].mean(),
            color=NORMAL,
            alpha=0.25,
            height=0.6,
            label="model, mean over seeds" if i == 0 else None,
        )
        ax.plot(
            g[metric],
            np.full(len(g), i),
            "o",
            color=NORMAL,
            ms=5,
            label="model, one seed" if i == 0 else None,
        )
        for b, mk, lab in zip(baselines, ("|", "x"), names, strict=False):
            if b in g and g[b].notna().any():
                ax.plot(
                    g[b].mean(),
                    i,
                    mk,
                    color=MUTED,
                    ms=9,
                    mew=1.5,
                    label=lab if i == 0 else None,
                )
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_ylabel("run folder", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"{metric}  (0.5 = chance, 1 = victim always above others)")
    fig.legend(
        frameon=False, fontsize=7, loc="outside lower center", ncols=2, labelcolor=INK
    )
    ax.set_title(
        "The same measure across every evaluated run, best at the top",
        loc="left",
        fontsize=10,
    )
    return fig


def pose_benchmark(rows: pd.DataFrame) -> Figure | None:
    """Pose `run`: recall vs false positives, one point per model, whole frame vs tiled.

    Args:
        rows: benchmark summary (model, mode, recall, false_pos, ms_per_frame).
    Returns:
        Figure, or None when empty.
    """
    if rows is None or not len(rows):
        return None
    models = list(dict.fromkeys(rows["model"]))
    fig, axes = _fig(6, 4)
    ax = axes[0][0]
    for m in models:
        g = rows[rows["model"] == m]
        color = SERIES[models.index(m) % len(SERIES)]
        for mode, mk in (("whole", "s"), ("tiled", "o")):
            r = g[g["mode"] == mode]
            if len(r):
                ax.plot(r["false_pos"], r["recall"], mk, color=color, ms=8)
        if len(g) > 1:
            ax.plot(g["false_pos"], g["recall"], color=color, linewidth=1, alpha=0.5)
        last = g.iloc[-1]
        ax.annotate(
            m,
            (last["false_pos"], last["recall"]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.set_xlabel("false positives (boxes on no person)")
    ax.set_ylabel("recall (swimmers in water found)")
    ax.set_ylim(0, max(0.05, float(rows["recall"].max()) * 1.15))
    ax.set_title("Pose backends: square = whole frame, circle = tiled", loc="left")
    return fig


def pose_sweep(rows: pd.DataFrame) -> Figure | None:
    """Pose `sweep`: recall vs false positives as the confidence floor moves, one line per model.

    Args:
        rows: sweep table (model, threshold, recall, false_pos).
    Returns:
        Figure, or None when empty.
    """
    if rows is None or not len(rows):
        return None
    models = list(dict.fromkeys(rows["model"]))
    fig, axes = _fig(6, 4)
    ax = axes[0][0]
    for m in models:
        g = rows[rows["model"] == m].sort_values("threshold")
        color = SERIES[models.index(m) % len(SERIES)]
        ax.plot(g["false_pos"], g["recall"], color=color, linewidth=2, marker="o", ms=3)
        for _, r in g.iloc[:: max(1, len(g) // 4)].iterrows():
            ax.annotate(
                f"{r['threshold']:.2f}",
                (r["false_pos"], r["recall"]),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=7,
                color=MUTED,
            )
        first = g.iloc[0]
        ax.annotate(
            m,
            (first["false_pos"], first["recall"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("false positives (boxes on no person)")
    ax.set_ylabel("recall (swimmers in water found)")
    ax.set_title("Tiled recall vs false positives across confidence floors", loc="left")
    return fig
