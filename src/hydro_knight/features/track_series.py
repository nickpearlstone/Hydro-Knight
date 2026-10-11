"""
Input representation v2: each swimmer track as a fixed-rate series of features.

Every confirmed track is resampled to TARGET_FPS (nearest source frame in time), so
one step means the same time on every camera. A step whose source frame has no
detection for the track is kept as a missing step: confidences, the detected flag,
rates and velocities are 0 and coordinates are held from the last detection, so a
dropout reads as confidence falling to zero. Pose is normalized by the bounding box
(center, height), which stays sane however unreliable the joints are, so no
confidence threshold drops anything.

Feature layout per step (58):
  0-33   pose: 17 joints x (x, y), relative to the box center, divided by box height
  34-50  keypoint confidences
  51     box confidence
  52     detected flag (1 seen this step, 0 missing)
  53     box aspect (width / height)
  54     box area change rate, per second (d ln area / dt)
  55     box size (box height / frame height)
  56-57  body velocity (box center), per second, in box heights
  58-91  optional joint velocities, per second, in box heights (joint_velocity=True)
Rates and velocities are measured between consecutive detected steps and stored on
the later one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_FPS = 30.0
N_POSE = 34
N_BASE = 58
DETECTED = 52
_XY = [f"{a}{i}" for i in range(17) for a in ("x", "y")]
_CONF = [f"c{i}" for i in range(17)]
_BOX = ["bx1", "by1", "bx2", "by2"]


def zero_when_missing(n_feat: int) -> np.ndarray:
    """Feature indices that are 0 on a missing step (confidences, flag, rates, velocities)."""
    return np.array([*range(34, 53), 54, 56, 57, *range(N_BASE, n_feat)])


@dataclass
class TrackSeries:
    """One track resampled to TARGET_FPS; step k is at file time t0 + k / TARGET_FPS."""

    clip_id: str
    track_id: int
    t0: float
    feats: np.ndarray  # (T, F) float32
    detected: np.ndarray  # (T,) bool

    def __len__(self) -> int:
        return len(self.detected)

    def times(self) -> np.ndarray:
        """(T,) file seconds of each step."""
        return self.t0 + np.arange(len(self)) / TARGET_FPS


def resample_track(
    g: pd.DataFrame,
    fps_src: float,
    frame_h: float,
    clip_id: str = "",
    joint_velocity: bool = False,
) -> TrackSeries:
    """Resample one track's keypoint rows to TARGET_FPS and compute its features.

    Args:
        g: keypoint rows of a single track (frame, track_id, box_conf, x/y/c 0-16, bx1..by2).
        fps_src: the clip's source frame rate (from the keypoint file's provenance).
        frame_h: video frame height in pixels.
        clip_id: stored on the series for bookkeeping.
        joint_velocity: append the 34 joint velocities (feature count 92).
    Returns:
        TrackSeries spanning the track's first to last detection.
    """
    g = g.sort_values("frame").drop_duplicates("frame").set_index("frame")
    t_start = g.index[0] / fps_src
    n = int(np.floor((g.index[-1] - g.index[0]) / fps_src * TARGET_FPS + 1e-9)) + 1
    times = t_start + np.arange(n) / TARGET_FPS
    r = g.reindex(np.rint(times * fps_src).astype(np.int64))
    det = r["box_conf"].notna().to_numpy()
    box = r[_BOX].ffill().to_numpy(np.float64)
    xy = r[_XY].ffill().to_numpy(np.float64).reshape(n, 17, 2)
    conf = r[_CONF].fillna(0.0).to_numpy(np.float64)
    bconf = r["box_conf"].fillna(0.0).to_numpy(np.float64)

    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    w = np.maximum(box[:, 2] - box[:, 0], 1e-6)
    h = np.maximum(box[:, 3] - box[:, 1], 1e-6)
    center = np.stack([cx, cy], axis=1)[:, None, :]
    pose = ((xy - center) / h[:, None, None]).reshape(n, N_POSE)
    log_area = np.log(w * h)

    area_rate, vel = np.zeros(n), np.zeros((n, 2))
    jvel = np.zeros((n, N_POSE))
    di = np.flatnonzero(det)
    if len(di) > 1:
        a, b = di[:-1], di[1:]
        dt = times[b] - times[a]
        area_rate[b] = (log_area[b] - log_area[a]) / dt
        vel[b, 0] = (cx[b] - cx[a]) / h[b] / dt
        vel[b, 1] = (cy[b] - cy[a]) / h[b] / dt
        jvel[b] = (pose[b] - pose[a]) / dt[:, None]

    cols = [
        pose,
        conf,
        bconf[:, None],
        det[:, None].astype(np.float64),
        (w / h)[:, None],
        area_rate[:, None],
        (h / frame_h)[:, None],
        vel,
    ]
    if joint_velocity:
        cols.append(jvel)
    track_id = int(g["track_id"].iloc[0])
    feats = np.hstack(cols).astype(np.float32)
    return TrackSeries(clip_id, track_id, float(t_start), feats, det)


def track_series(
    df: pd.DataFrame,
    fps_src: float,
    frame_h: float,
    clip_id: str = "",
    joint_velocity: bool = False,
) -> list[TrackSeries]:
    """All confirmed tracks (track_id >= 0) of a clip as TrackSeries, by track id."""
    return [
        resample_track(g, fps_src, frame_h, clip_id, joint_velocity)
        for tid, g in df[df["track_id"] >= 0].groupby("track_id")
    ]


def trim_series(ts: TrackSeries, t0: float, t1: float) -> TrackSeries | None:
    """Keep only steps with file time in [t0, t1]; None if no step is left.

    Args:
        ts: the series to trim.
        t0, t1: file seconds (use 0 and inf for an untrimmed side).
    """
    keep = np.flatnonzero((ts.times() >= t0 - 1e-9) & (ts.times() <= t1 + 1e-9))
    if len(keep) == 0:
        return None
    a, b = keep[0], keep[-1] + 1
    return TrackSeries(
        ts.clip_id,
        ts.track_id,
        ts.t0 + a / TARGET_FPS,
        ts.feats[a:b],
        ts.detected[a:b],
    )
