"""Grouped K-fold: pool every fold model's held-out scores into one table, on one scale."""

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
    """Median score of a fold model's own training windows: its common-scale unit."""
    s = np.asarray(train_scores, float)
    s = s[np.isfinite(s)]
    ref = float(np.median(s)) if len(s) else float("nan")
    if not ref > 0:
        raise ValueError(f"cannot scale by training median {ref}")
    return ref


def pool(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-fold held-out tables (each with a clip_id column) into one, adding a fold column."""
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
    """The loaded report with every clip in `folds` as tested (in its key order) and the fold map."""
    return replace(base, used_train=[], used_test=list(folds), folds=dict(folds))
