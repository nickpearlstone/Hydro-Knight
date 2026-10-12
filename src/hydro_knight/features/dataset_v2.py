"""
Assemble v2 training and test window sets from the keypoint files and the manifest.

Only clips that are active (no "[HOLD") and fully victim-labeled are used, because
the autoencoder must learn normal swimming only: each labeled clip's victim tracks
(matched from the clicks, plus fragments that follow the victim's path between clicks)
are removed, and every other swimmer is normal. Steps outside the clip's trim are
dropped. Clips listed in `exclude_clips` (labels a review marked wrong) are skipped.
For evaluation, the removed victim tracks of test clips are kept as their own window
set (`victim_windows`). Clips are split whole into train and test, so
test footage is never seen in training. require_labels=False exists only for smoke
tests of the code path, before labels exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.split import split_clips
from ..eval.victims import is_labeled, match_victim_tracks, near_path_tracks
from ..preprocess.build_tracks import read_provenance
from .track_series import TARGET_FPS, TrackSeries, track_series, trim_series
from .windows_v2 import WindowSet, make_window_set


@dataclass
class DatasetReport:
    """Which clips went where, and what was removed."""

    used_train: list[str] = field(default_factory=list)
    used_test: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # labels a review marked wrong
    victim_tracks: dict[str, list[int]] = field(default_factory=dict)
    unmatched_clicks: dict[str, int] = field(default_factory=dict)
    path_fragments: dict[str, list[int]] = field(default_factory=dict)


def clip_series(
    path: Path,
    record,
    joint_velocity: bool = False,
    exclude_tracks=frozenset(),
    rates: str = "raw",
    only_tracks=None,
) -> list[TrackSeries]:
    """A clip's track series trimmed to the labeled trim.

    Args:
        exclude_tracks: track ids to leave out (the victim, for normal data).
        only_tracks: if given, keep only these track ids (the victim, for scoring).
    """
    video = read_provenance(path)["video"]
    df = pd.read_parquet(path)
    df = df[~df["track_id"].isin(exclude_tracks)]
    if only_tracks is not None:
        df = df[df["track_id"].isin(only_tracks)]
    t0 = 0.0 if record.trim_start is None else record.to_file_time(record.trim_start)
    t1 = np.inf if record.trim_end is None else record.to_file_time(record.trim_end)
    out = []
    for ts in track_series(
        df, video["fps"], video["height"], record.clip_id, joint_velocity, rates
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
    rates: str = "raw",
    exclude_clips=frozenset(),
    path_fragments: bool = True,
    min_track_s: float | None = None,
) -> tuple[WindowSet, WindowSet, DatasetReport]:
    """Train and test window sets of non-victim swimmers, split by whole clips.

    Args:
        rates: box rate handling (track_series.RATE_MODES).
        exclude_clips: clip ids to skip (labels a review marked wrong).
        path_fragments: also remove tracks that follow the victim's path between clicks.
        min_track_s: also score tracks shorter than a window (windows_v2.make_window_set).

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
        elif r.clip_id in exclude_clips:
            rep.rejected.append(r.clip_id)
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
            fps = read_provenance(path)["video"]["fps"]
            match = match_victim_tracks(df, r, fps)
            frags = (
                near_path_tracks(df, r, fps, match.tracks) if path_fragments else set()
            )
            exclude = frozenset(match.tracks | frags)
            rep.victim_tracks[r.clip_id] = sorted(exclude)
            rep.path_fragments[r.clip_id] = sorted(frags)
            rep.unmatched_clicks[r.clip_id] = len(match.unmatched)
        series = clip_series(path, r, joint_velocity, exclude, rates)
        side = "train" if r.clip_id in train_ids else "test"
        parts[side].append(make_window_set(series, window, stride_s, min_track_s))
        (rep.used_train if side == "train" else rep.used_test).append(r.clip_id)

    empty = make_window_set([], window, stride_s)
    train = WindowSet.concat(parts["train"]) if parts["train"] else empty
    test = WindowSet.concat(parts["test"]) if parts["test"] else empty
    return train, test, rep


def victim_windows(
    keypoints_dir,
    records,
    rep: DatasetReport,
    *,
    window: int = 90,
    stride_s: float = 0.25,
    joint_velocity: bool = False,
    rates: str = "raw",
    min_track_s: float | None = None,
) -> tuple[WindowSet, np.ndarray]:
    """Windows of the test clips' victim tracks that overlap onset -> guard contact.

    Uses the victim tracks build_datasets removed (rep.victim_tracks). A victim track
    shorter than one window gives no windows, so a clip can have none.
    Returns:
        (windows, keep): the victim tracks' full window set and a bool mask of the
        windows whose scored span overlaps [onset, contact] (file seconds).
    """
    by_id = {r.clip_id: r for r in records}
    sets, keep = [], []
    for cid in rep.used_test:
        r, tracks = by_id[cid], rep.victim_tracks.get(cid, [])
        if not tracks:
            continue
        ev = r.events[0]
        t_on, t_ct = r.to_file_time(ev["start"]), r.to_file_time(ev["contact"])
        path = Path(keypoints_dir) / f"{cid}.parquet"
        series = clip_series(path, r, joint_velocity, rates=rates, only_tracks=tracks)
        ws = make_window_set(series, window, stride_s, min_track_s)
        span = window / TARGET_FPS
        starts = np.array([ws.info(i)[2] for i in range(len(ws))])
        sets.append(ws)
        keep.append((starts < t_ct) & (starts + span > t_on))
    if not sets:
        return make_window_set([], window, stride_s), np.zeros(0, bool)
    return WindowSet.concat(sets), np.concatenate(keep)
