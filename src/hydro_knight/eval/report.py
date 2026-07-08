"""
Single-run evaluation report: PNGs + a markdown summary in one folder.

Each run writes into its own directory (e.g. runs/2026-07-08_tcn_baseline/),
so comparing two runs = opening two folders side by side. The report never
touches footage — every figure is derived from keypoints/scores only, so
there is nothing to pixelate.

Chart conventions (kept deliberately boring and consistent):
- normal = blue (#2a78d6), distress = red (#e34948) — same two colors everywhere
- caught = green (#0ca30c) / missed = red (#d03b3b), always with a text label,
  never color alone
- recall-first: accuracy appears nowhere in this report by design
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we only ever write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import (
    ClipEval,
    event_catches,
    percentile_table,
    roc_pr,
    split_scores,
    sweep_recall_fa,
)

# --- palette (validated light-mode set) --------------------------------------
NORMAL = "#2a78d6"  # blue
DISTRESS = "#e34948"  # red
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style(ax):
    """House style: recessive chrome so the data is the loudest thing."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, out_dir: Path, name: str) -> str:
    fig.patch.set_facecolor(SURFACE)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def plot_loss(history: list[float], out_dir: Path) -> str | None:
    """Training curve — the 'did it even converge?' view we trained blind without."""
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, len(history) + 1), history, color=NORMAL, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE")
    ax.set_title("Training loss")
    _style(ax)
    return _save(fig, out_dir, "01_loss.png")


def plot_error_overlap(
    err_n: np.ndarray, err_d: np.ndarray, out_dir: Path
) -> str | None:
    """
    The 0.539 storyteller: both error distributions on one axis. If these two
    shapes sit on top of each other, no threshold can separate them.
    """
    if len(err_n) == 0 or len(err_d) == 0:
        return None
    hi = float(np.percentile(np.concatenate([err_n, err_d]), 99.5))
    bins = np.linspace(0, hi, 60)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(
        err_n,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=NORMAL,
        edgecolor=NORMAL,
        linewidth=1.5,
        label="normal",
    )
    ax.hist(
        err_d,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=DISTRESS,
        edgecolor=DISTRESS,
        linewidth=1.5,
        label="distress",
    )
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("density")
    ax.set_title("Anomaly-score overlap (normal vs distress windows)")
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax)
    return _save(fig, out_dir, "02_error_overlap.png")


def plot_roc_pr(res: dict, out_dir: Path) -> str | None:
    """Window-level ROC + PR, side by side (each its own axes)."""
    if res is None:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(res["fpr"], res["tpr"], color=NORMAL, linewidth=2)
    a1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate")
    a1.set_title(f"ROC (AUC = {res['roc_auc']:.3f})")
    a2.plot(res["recall"], res["precision"], color=NORMAL, linewidth=2)
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a2.set_title(f"PR (AUC = {res['pr_auc']:.3f})")
    for a in (a1, a2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        _style(a)
    fig.tight_layout()
    return _save(fig, out_dir, "03_roc_pr.png")


def plot_recall_fa(sweep: pd.DataFrame, out_dir: Path) -> str | None:
    """
    The operating-point curve: event recall vs false alarms per hour. The
    project's safety philosophy in one picture — buy recall, pay in alarms.
    """
    s = sweep.dropna(subset=["fa_per_hour"]).sort_values("fa_per_hour")
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(s["fa_per_hour"], s["recall"], color=NORMAL, linewidth=2)
    ax.set_xlabel("false alarms per hour of normal footage")
    ax.set_ylabel("event recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Recall vs alarm cost (threshold sweep)")
    _style(ax)
    return _save(fig, out_dir, "04_recall_vs_fa.png")


def plot_latency(latencies_s: list[float], out_dir: Path) -> str | None:
    """Of the caught events: how early did we fire? Late catches are hollow."""
    lat = [x for x in latencies_s if np.isfinite(x)]
    if not lat:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(lat, bins=20, color=NORMAL, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("seconds after event onset")
    ax.set_ylabel("events")
    ax.set_title("Detection latency (caught events)")
    _style(ax)
    return _save(fig, out_dir, "05_latency.png")


def plot_event_board(catches: list[dict], out_dir: Path) -> str | None:
    """
    Per-event catch/miss board. One row per annotated event: green bar whose
    length is the latency for a catch, red full-row marker for a miss. Labels
    carry the verdict too (✓/✗) so color never stands alone.
    """
    if not catches:
        return None
    n = len(catches)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * n)))
    max_lat = max(
        [c["latency_s"] for c in catches if np.isfinite(c.get("latency_s", np.nan))],
        default=1.0,
    )
    for i, c in enumerate(catches):
        y = n - 1 - i
        if c["caught"]:
            ax.barh(y, max(c["latency_s"], 0.05), height=0.62, color=GOOD)
            mark, color = "✓", GOOD
        else:
            ax.barh(y, max_lat * 1.05, height=0.62, color=CRITICAL, alpha=0.25)
            mark, color = "✗", CRITICAL
        ax.text(
            -0.3,
            y,
            f"{mark} {c['clip_id'][:8]}",
            ha="right",
            va="center",
            fontsize=7.5,
            color=color,
            family="monospace",
        )
    ax.set_yticks([])
    ax.set_xlabel("latency after onset (s)   [red rows = missed]")
    ax.set_title("Per-event catches at operating threshold")
    ax.set_xlim(left=0)
    _style(ax)
    return _save(fig, out_dir, "06_event_board.png")


def plot_coverage(coverage_rows: list[dict], out_dir: Path) -> str | None:
    """Pose coverage inside each event window — the representation-gap detector."""
    if not coverage_rows:
        return None
    cov = [r["coverage"] for r in coverage_rows]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(
        cov, bins=np.linspace(0, 1, 21), color=DISTRESS, edgecolor=SURFACE, linewidth=1
    )
    ax.set_xlabel("fraction of event frames with a usable pose")
    ax.set_ylabel("events")
    ax.set_title("Pose coverage inside distress events")
    _style(ax)
    return _save(fig, out_dir, "07_event_coverage.png")


def plot_track_lengths(stats: pd.DataFrame, out_dir: Path) -> str | None:
    """Track-length distribution — fragmentation at a glance."""
    if stats.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(stats["n_frames"], bins=30, color=NORMAL, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("frames per track")
    ax.set_ylabel("tracks")
    ax.set_title("Track lengths (fragmentation view)")
    _style(ax)
    return _save(fig, out_dir, "08_track_lengths.png")


def _md_table(df: pd.DataFrame) -> str:
    """Tiny DataFrame -> markdown table (avoids the `tabulate` dependency)."""

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    cols = list(df.columns)
    rows = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    rows += [
        "| " + " | ".join(fmt(v) for v in rec) + " |"
        for rec in df.itertuples(index=False)
    ]
    return "\n".join(rows)


def generate_report(
    out_dir: Path,
    clips: list[ClipEval],
    threshold: float,
    loss_history: list[float] | None = None,
    coverage_rows: list[dict] | None = None,
    tracks: pd.DataFrame | None = None,
    census: dict | None = None,
    run_name: str = "run",
) -> Path:
    """
    Orchestrate: compute every metric from the neutral detections, write all
    figures + stats.parquet + summary.md into `out_dir`. Returns summary path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    err_n, err_d = split_scores(clips)
    curves = roc_pr(err_n, err_d)
    sweep = sweep_recall_fa(clips) if any(c.events for c in clips) else pd.DataFrame()
    catches = [e for c in clips for e in event_catches(c, threshold)]
    pct = percentile_table(err_n, err_d)

    figures = [
        plot_loss(loss_history or [], out_dir),
        plot_error_overlap(err_n, err_d, out_dir),
        plot_roc_pr(curves, out_dir) if curves else None,
        plot_recall_fa(sweep, out_dir) if not sweep.empty else None,
        plot_latency([c["latency_s"] for c in catches if c["caught"]], out_dir),
        plot_event_board(catches, out_dir),
        plot_coverage(coverage_rows or [], out_dir),
        plot_track_lengths(tracks if tracks is not None else pd.DataFrame(), out_dir),
    ]

    if not sweep.empty:
        sweep.to_parquet(out_dir / "sweep.parquet")
    if catches:
        pd.DataFrame(catches).to_parquet(out_dir / "event_catches.parquet")

    caught = sum(c["caught"] for c in catches)
    lines = [
        f"# Eval report — {run_name}",
        "",
        f"- clips evaluated: **{len(clips)}**  |  events: **{len(catches)}**",
        f"- operating threshold: **{threshold:.4f}**",
    ]
    if catches:
        lines.append(
            f"- **event recall: {caught}/{len(catches)}** ({caught / len(catches):.0%})"
        )
    if curves:
        lines.append(
            f"- window-level ROC-AUC: **{curves['roc_auc']:.3f}**"
            f"  |  PR-AUC: **{curves['pr_auc']:.3f}**"
        )
    if not pct.empty:
        lines += ["", "## Error percentiles", "", _md_table(pct)]
    if census:
        lines += [
            "",
            "## Dataset census",
            "",
            "```json",
            json.dumps(census, indent=2),
            "```",
        ]
    lines += ["", "## Figures", ""]
    lines += [f"![{f}]({f})" for f in figures if f]

    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary
