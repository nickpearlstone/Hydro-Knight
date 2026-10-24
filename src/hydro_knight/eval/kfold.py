"""
Grouped K-fold: pool every fold model's held-out scores into one table, on one scale.

Each clip gets one fold (split.assign_folds). Fold f's model trains on the other folds and
scores only fold f, so stacking the K held-out tables tests every clip exactly once, each
by a model that never trained on it.

Within one clip every swimmer is scored by the same model, so within-clip measures (same-
clip AUC, time in top k) need nothing more. Measures that apply one alarm threshold across
clips do: the K models' reconstruction errors sit at different levels. So each model's
scores are divided by the median score of its own training windows. That reference comes
from training folds only, and dividing by a positive number never reorders scores within a
clip. The raw error is kept next to it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..features.dataset_v2 import DatasetReport
from ..features.windows_v2 import WindowSet


def training_sample(ws: WindowSet, n: int = 20000) -> WindowSet:
    """At most n evenly spaced windows of a training set (enough for a stable median)."""
    rows = np.unique(np.linspace(0, len(ws) - 1, min(n, len(ws))).astype(np.int64))
    return WindowSet(ws.series, ws.index[rows], ws.window, ws.context)


def scale_reference(train_scores: np.ndarray) -> float:
    """Median score of a fold model's own training windows: its common-scale unit.

    Raises:
        ValueError: no positive training scores to scale by.
    """
    s = np.asarray(train_scores, float)
    s = s[np.isfinite(s)]
    ref = float(np.median(s)) if len(s) else float("nan")
    if not ref > 0:
        raise ValueError(f"cannot scale by training median {ref}")
    return ref


def pool(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-fold held-out tables (each with a clip_id column) into one, adding a fold column.

    Raises:
        ValueError: a clip appears in more than one fold's table.
    """
    seen: dict[str, int] = {}
    for f, t in tables.items():
        for c in pd.unique(t["clip_id"]):
            if c in seen:
                raise ValueError(f"clip {c} held out by folds {seen[c]} and {f}")
            seen[c] = f
    return pd.concat(
        [t.assign(fold=f) for f, t in sorted(tables.items())], ignore_index=True
    )


def pooled_report(base: DatasetReport, folds: dict[str, int]) -> DatasetReport:
    """The loaded report with every clip in `folds` as tested (in its key order) and the fold map.

    used_train stays empty: each clip trained K-1 of the K models.
    """
    return replace(base, used_train=[], used_test=list(folds), folds=dict(folds))
