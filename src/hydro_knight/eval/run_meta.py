"""Run metadata for experiment tracking: git state, data fingerprints, standard tags."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path("data/manifests/pool_footage.jsonl")
LABEL_FILES = (
    Path("data/annotations/label_reviews.json"),
    Path("data/annotations/labeled_by_claude.json"),
)


def file_sha256(path: Path, n: int = 12) -> str:
    """First n hex digits of a file's SHA-256, or "missing"."""
    path = Path(path)
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def data_version(manifest: Path = MANIFEST, extra=LABEL_FILES) -> str:
    """One short hash over the manifest and the label-review files; changes when any label does."""
    h = hashlib.sha256()
    for p in (Path(manifest), *map(Path, extra)):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()[:12]


def git_commit() -> str:
    """Short HEAD commit, "-dirty" suffix if uncommitted changes, "unknown" outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def standard_tags(command: str, manifest: Path = MANIFEST, **extra) -> dict[str, str]:
    """The tag set every run of `command` starts with; `extra` adds or overrides (None values dropped)."""
    tags = {
        "command": command,
        "data_version": data_version(manifest),
        "manifest_sha256": file_sha256(manifest),
        "git_commit": git_commit(),
    }
    tags.update({k: str(v) for k, v in extra.items() if v is not None})
    return tags


def numeric_items(row: dict, prefix: str = "") -> dict[str, float]:
    """The finite int/float values of a dict (bools and None skipped), keys optionally prefixed."""
    return {
        f"{prefix}{k}": float(v)
        for k, v in row.items()
        if isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        and np.isfinite(v)
    }


def mean_sd_metrics(rows: list[dict], prefix: str = "") -> dict[str, float]:
    """For every numeric key shared by the rows: its mean and standard deviation across rows."""
    vals: dict[str, list[float]] = {}
    for r in rows:
        for k, v in numeric_items(r).items():
            vals.setdefault(k, []).append(v)
    out = {}
    for k, v in vals.items():
        out[f"{prefix}{k}.mean"] = float(np.mean(v))
        out[f"{prefix}{k}.sd"] = float(np.std(v))
    return out


def split_table(clip_ids, window_clips=None) -> pd.DataFrame:
    """One row per clip of a train or test split: clip_id and its window count."""
    counts = pd.Series(window_clips).value_counts() if window_clips is not None else {}
    rows = [{"clip_id": c, "n_windows": int(counts.get(c, 0))} for c in clip_ids]
    return pd.DataFrame(rows, columns=["clip_id", "n_windows"]).sort_values(
        "clip_id", ignore_index=True
    )


def flatten_search_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """mlflow.search_runs output with "metrics." stripped and one run_folder column."""
    runs = runs.copy()
    runs.columns = [c.removeprefix("metrics.") for c in runs.columns]
    empty = pd.Series(None, index=runs.index, dtype=object)
    tag = runs.get("tags.run_folder", empty)
    param = runs.get("params.run_folder", empty).map(
        lambda p: Path(p).name if isinstance(p, str) and p else None
    )
    runs["run_folder"] = tag.where(tag.notna(), param)
    return runs
