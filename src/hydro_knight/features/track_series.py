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
Rates and velocities are measured between consecutive detected steps, over the time
between the source frames they sampled, and stored on
the later one.

Box rates (54, 56-57) are mostly tracker box jitter at the frame-to-frame scale (p99
about 7 box heights/s, max over 100). `rates` picks how they enter the model: "raw",
"capped" (clipped to +-RATE_CAP per second: ~3 box heights/s is a fast swim, and a real
sink shrinks the box over a quarter second or more), or "off" (set to 0, so the
feature count and every index stay the same).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_FPS = 30.0
N_POSE = 34
N_BASE = 58
DETECTED = 52
RATE_COLS = [54, 56, 57]  # box area change rate, body velocity x / y
RATE_CAP = 3.0  # per second, for rates="capped"
RATE_MODES = ("raw", "capped", "off")
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
    rates: str = "raw",
    frame_step: int = 1,
) -> TrackSeries:
    """Resample one track's keypoint rows to TARGET_FPS and compute its features.

    Args:
        g: keypoint rows of a single track (frame, track_id, box_conf, x/y/c 0-16, bx1..by2).
        fps_src: the clip's source frame rate (from the keypoint file's provenance).
        frame_h: video frame height in pixels.
        clip_id: stored on the series for bookkeeping.
        joint_velocity: append the 34 joint velocities (feature count 92).
        rates: box rate handling, one of RATE_MODES (see module docstring).
        frame_step: extraction frame step; only multiples of it were ever processed.
    Returns:
        TrackSeries spanning the track's first to last detection.
    """
    rate_mode = rates  # `rates` is reused below for the computed rate matrix
    if rate_mode not in RATE_MODES:
        raise ValueError(f"rates must be one of {RATE_MODES}, got {rate_mode!r}")
    g = g.sort_values("frame").drop_duplicates("frame").set_index("frame")
    t_start = g.index[0] / fps_src
    n = int(np.floor((g.index[-1] - g.index[0]) / fps_src * TARGET_FPS + 1e-9)) + 1
    times = t_start + np.arange(n) / TARGET_FPS
    # Nearest processed source frame per step. At 59.94 fps with frame_step 2 the plain
    # nearest frame drifts onto odd frames, which were never processed and would read
    # as missing.
    src = (np.rint(times * fps_src / frame_step) * frame_step).astype(np.int64)
    r = g.reindex(src)
    det = r["box_conf"].notna().to_numpy()
    box = r[_BOX].ffill().to_numpy(np.float64)
    xy = r[_XY].ffill().to_numpy(np.float64).reshape(n, 17, 2)
    conf = r[_CONF].fillna(0.0).to_numpy(np.float64)
    bconf = r["box_conf"].fillna(0.0).to_numpy(np.float64)

    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    w = np.maximum(box[:, 2] - box[:, 0], 1.0)  # 1 px floor keeps degenerate boxes sane
    h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    center = np.stack([cx, cy], axis=1)[:, None, :]
    pose = ((xy - center) / h[:, None, None]).reshape(n, N_POSE)
    log_area = np.log(w * h)

    area_rate, vel = np.zeros(n), np.zeros((n, 2))
    jvel = np.zeros((n, N_POSE))
    di = np.flatnonzero(det)
    if len(di) > 1:
        a, b = di[:-1], di[1:]
        # Time between the source frames actually sampled, not between 30 fps steps:
        # at 24 or 59.94 fps the two differ, which would distort every rate.
        dt = (src[b] - src[a]) / fps_src
        same = dt == 0  # upsampling: two steps read one source frame
        dt = np.where(same, 1.0, dt)
        rates = np.column_stack(
            [
                (log_area[b] - log_area[a]) / dt,
                (cx[b] - cx[a]) / h[b] / dt,
                (cy[b] - cy[a]) / h[b] / dt,
                (pose[b] - pose[a]) / dt[:, None],
            ]
        )
        for k in np.flatnonzero(same):  # a repeated frame carries the previous rate
            rates[k] = rates[k - 1] if k > 0 else 0.0
        area_rate[b], vel[b, 0], vel[b, 1] = rates[:, 0], rates[:, 1], rates[:, 2]
        jvel[b] = rates[:, 3:]
    if rate_mode == "capped":
        area_rate, vel = (
            np.clip(area_rate, -RATE_CAP, RATE_CAP),
            np.clip(vel, -RATE_CAP, RATE_CAP),
        )
    elif rate_mode == "off":
        area_rate, vel = np.zeros_like(area_rate), np.zeros_like(vel)

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
    rates: str = "raw",
    frame_step: int = 1,
) -> list[TrackSeries]:
    """All confirmed tracks (track_id >= 0) of a clip as TrackSeries, by track id."""
    return [
        resample_track(g, fps_src, frame_h, clip_id, joint_velocity, rates, frame_step)
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
