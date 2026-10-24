"""Load the labelled wave-pool rescue clips one at a time, with victim tracks and zone.

All times are file seconds (ClipRecord.to_file_time applied). The Mac has 8 GB, so
`iter_clips` yields one clip at a time; keep only small results between clips.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..annotate import zones as zones_mod
from ..eval.timing import clip_fps
from ..eval.victims import is_labeled, match_victim_tracks, near_path_tracks
from ..ingest.manifest import Manifest

MANIFEST = Path("data/manifests/pool_footage.jsonl")
KP_DIR = Path("data/keypoints_tiled")
VIDEOS = Path("raw_local")
BOX = ["bx1", "by1", "bx2", "by2"]
HEAD_KP = [f"{a}{i}" for i in range(5) for a in "xyc"]  # nose, eyes, ears


@dataclass
class Clip:
    clip_id: str
    fps: float
    df: (
        pd.DataFrame
    )  # frame, track_id (-1 = unconfirmed), box_conf, face keypoints, box
    start: float  # footage start (trim start, else 0)
    end: float  # footage end (trim end, else last frame)
    onset: float
    contact: float
    victim_click_tracks: frozenset[int]
    victim_path_tracks: frozenset[int]  # fragments along the click path
    zone: dict | None

    @property
    def victim_tracks(self) -> frozenset[int]:
        return self.victim_click_tracks | self.victim_path_tracks


def labelled_records(manifest: Path = MANIFEST, kp_dir: Path = KP_DIR):
    """(record, parquet) for every active, fully labelled clip with a keypoint file."""
    for r in Manifest(manifest).load():
        p = kp_dir / f"{r.clip_id}.parquet"
        if "[HOLD" not in (r.notes or "") and is_labeled(r) and p.exists():
            yield r, p


def load_clip(record, parquet: Path, zones: dict) -> Clip:
    fps, _ = clip_fps(parquet, VIDEOS, 30.0, record)
    df = pd.read_parquet(
        parquet, columns=["frame", "track_id", "box_conf", *HEAD_KP, *BOX]
    )
    for c in df.columns:
        if c not in ("frame", "track_id"):
            df[c] = df[c].astype(np.float32)
    ev = record.events[0]
    m = match_victim_tracks(df, record, fps)
    path = near_path_tracks(df, record, fps, exclude=m.tracks)
    last = df["frame"].max() / fps
    return Clip(
        clip_id=record.clip_id,
        fps=fps,
        df=df,
        start=0.0
        if record.trim_start is None
        else record.to_file_time(record.trim_start),
        end=last
        if record.trim_end is None
        else min(record.to_file_time(record.trim_end), last),
        onset=record.to_file_time(ev["start"]),
        contact=record.to_file_time(ev["contact"]),
        victim_click_tracks=frozenset(int(t) for t in m.tracks),
        victim_path_tracks=frozenset(int(t) for t in path),
        zone=zones.get(record.clip_id),
    )


def iter_clips(only: set[str] | None = None) -> Iterator[Clip]:
    zones = zones_mod.load()
    for r, p in labelled_records():
        if only is None or r.clip_id in only:
            yield load_clip(r, p, zones)


def head_points(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Head position: mean of confident face keypoints, else a point near the box top."""
    xs = df[[f"x{i}" for i in range(5)]].to_numpy(np.float64)
    ys = df[[f"y{i}" for i in range(5)]].to_numpy(np.float64)
    ok = df[[f"c{i}" for i in range(5)]].to_numpy(np.float64) > 0.3
    n = ok.sum(1)
    bx1, by1, bx2, by2 = (df[c].to_numpy(np.float64) for c in BOX)
    hx = np.where(n > 0, (xs * ok).sum(1) / np.maximum(n, 1), (bx1 + bx2) / 2)
    hy = np.where(n > 0, (ys * ok).sum(1) / np.maximum(n, 1), by1 + 0.12 * (by2 - by1))
    return hx, hy
