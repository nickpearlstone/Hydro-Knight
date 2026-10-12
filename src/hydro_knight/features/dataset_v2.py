"""
Assemble v2 training and test window sets from the keypoint files and the manifest.

Only clips that are active (no "[HOLD") and fully victim-labeled are used, because
the autoencoder must learn normal swimming only: each labeled clip's victim tracks
(matched from the clicks) are removed, and every other swimmer is normal. Steps
outside the clip's trim are dropped. Clips are split whole into train and test, so
test footage is never seen in training. require_labels=False exists only for smoke
tests of the code path, before labels exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.split import split_clips
from ..eval.victims import is_labeled, match_victim_tracks
from ..preprocess.build_tracks import read_provenance
from .track_series import TrackSeries, track_series, trim_series
from .windows_v2 import WindowSet, make_window_set


@dataclass
class DatasetReport:
    """Which clips went where, and what was removed."""

    used_train: list[str] = field(default_factory=list)
    used_test: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    victim_tracks: dict[str, list[int]] = field(default_factory=dict)
    unmatched_clicks: dict[str, int] = field(default_factory=dict)


def clip_series(
    path: Path, record, joint_velocity: bool = False, exclude_tracks=frozenset()
) -> list[TrackSeries]:
    """A clip's track series, victim tracks excluded and trimmed to the labeled trim."""
    video = read_provenance(path)["video"]
    df = pd.read_parquet(path)
    df = df[~df["track_id"].isin(exclude_tracks)]
    t0 = 0.0 if record.trim_start is None else record.to_file_time(record.trim_start)
    t1 = np.inf if record.trim_end is None else record.to_file_time(record.trim_end)
    out = []
    for ts in track_series(
        df, video["fps"], video["height"], record.clip_id, joint_velocity
    ):
        kept = trim_series(ts, t0, t1)
        if kept is not None:
            out.append(kept)
    return out


def build_datasets(
    keypoints_dir,
    records,
    *,
    test_frac: float = 0.2,
    seed: int = 0,
    window: int = 90,
    stride_s: float = 0.25,
    joint_velocity: bool = False,
    require_labels: bool = True,
    max_clips: int | None = None,
) -> tuple[WindowSet, WindowSet, DatasetReport]:
    """Train and test window sets of non-victim swimmers, split by whole clips.

    Raises:
        ValueError: no usable clip (label clips first, or pass require_labels=False).
    """
    rep = DatasetReport()
    usable = []
    for r in records:
        path = Path(keypoints_dir) / f"{r.clip_id}.parquet"
        if "[HOLD" in (r.notes or ""):
            rep.held.append(r.clip_id)
        elif not path.exists():
            rep.missing.append(r.clip_id)
        elif require_labels and not is_labeled(r):
            rep.unlabeled.append(r.clip_id)
        else:
            usable.append((r, path))
    if max_clips is not None:
        usable = usable[:max_clips]
    if not usable:
        raise ValueError(
            "no usable clips: label clips first (require_labels=False for a smoke test)"
        )

    train_ids, _ = split_clips([r.clip_id for r, _ in usable], test_frac, seed)
    parts = {"train": [], "test": []}
    for r, path in usable:
        exclude = frozenset()
        if is_labeled(r):
            df = pd.read_parquet(
                path, columns=["frame", "track_id", "bx1", "by1", "bx2", "by2"]
            )
            match = match_victim_tracks(df, r, read_provenance(path)["video"]["fps"])
            exclude = frozenset(match.tracks)
            rep.victim_tracks[r.clip_id] = sorted(match.tracks)
            rep.unmatched_clicks[r.clip_id] = len(match.unmatched)
        series = clip_series(path, r, joint_velocity, exclude)
        side = "train" if r.clip_id in train_ids else "test"
        parts[side].append(make_window_set(series, window, stride_s))
        (rep.used_train if side == "train" else rep.used_test).append(r.clip_id)

    empty = make_window_set([], window, stride_s)
    train = WindowSet.concat(parts["train"]) if parts["train"] else empty
    test = WindowSet.concat(parts["test"]) if parts["test"] else empty
    return train, test, rep
