"""Window set v2: fixed 3-second windows over track series, with real context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track_series import N_BASE, TARGET_FPS, TrackSeries, zero_when_missing

WINDOW = 90  # 3 s at 30 fps
STRIDE_S = 0.25


def dilations_for(window: int) -> list[int]:
    """Encoder dilations: doubling from 1, then a remainder, so the reach fits the window."""
    half = (window - 1) // 2
    out, d = [], 1
    while sum(out) + d <= half:
        out.append(d)
        d *= 2
    if sum(out) < half:
        out.append(half - sum(out))
    return out


def context_for(window: int) -> int:
    """Context steps needed on each side of a window (= how far the encoder reaches)."""
    return sum(dilations_for(window))


def _padded_slice(ts: TrackSeries, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = max(lo, 0), min(hi, len(ts))
    x, m = ts.feats[a:b], ts.detected[a:b]
    before, after = a - lo, hi - b
    if before or after:
        x = np.pad(x, ((before, after), (0, 0)), mode="edge")
        m = np.pad(m, (before, after), constant_values=False)
        pad = np.zeros(len(m), bool)
        pad[:before] = True
        pad[len(pad) - after :] = True
        x[np.ix_(pad, zero_when_missing(x.shape[1]))] = 0.0
    return x, m


@dataclass
class WindowSet:
    """Index of scored windows over a list of track series; gathers batches on demand."""

    series: list[TrackSeries]
    index: np.ndarray  # series position, first scored step
    window: int
    context: int

    def __len__(self) -> int:
        return len(self.index)

    @property
    def n_feat(self) -> int:
        return self.series[0].feats.shape[1] if self.series else N_BASE

    def info(self, i: int) -> tuple[str, int, float]:
        """(clip_id, track_id, file seconds of the first scored step) of window i."""
        s, a = self.index[i]
        ts = self.series[s]
        return ts.clip_id, ts.track_id, ts.t0 + a / TARGET_FPS

    def gather(self, rows) -> tuple[np.ndarray, np.ndarray]:
        """Windows with context for the given row numbers."""
        span = self.window + 2 * self.context
        x = np.empty((len(rows), span, self.n_feat), np.float32)
        m = np.empty((len(rows), span), bool)
        for k, i in enumerate(rows):
            s, a = self.index[i]
            lo = a - self.context
            x[k], m[k] = _padded_slice(self.series[s], lo, lo + span)
        return x, m

    @staticmethod
    def concat(sets: list[WindowSet]) -> WindowSet:
        """One set over all the given sets' series (all must share window and context)."""
        series, index, offset = [], [], 0
        for ws in sets:
            series += ws.series
            index.append(ws.index + np.array([offset, 0]))
            offset += len(ws.series)
        idx = np.concatenate(index) if index else np.empty((0, 2), np.int64)
        return WindowSet(series, idx, sets[0].window, sets[0].context)


def make_window_set(
    series: list[TrackSeries],
    window: int = WINDOW,
    stride_s: float = STRIDE_S,
    min_track_s: float | None = None,
) -> WindowSet:
    """Index every scored window (one every stride_s) that fits inside its track."""
    step = stride_s * TARGET_FPS
    min_steps = None if min_track_s is None else max(1, round(min_track_s * TARGET_FPS))
    index = []
    for si, ts in enumerate(series):
        last = len(ts) - window
        if last < 0:
            if min_steps is not None and len(ts) >= min_steps:
                index.append((si, -((window - len(ts)) // 2)))
            continue
        starts = np.unique(np.rint(np.arange(0, last + 1e-9, step)).astype(np.int64))
        index += [(si, int(a)) for a in starts]
    idx = np.array(index, np.int64).reshape(-1, 2)
    return WindowSet(series, idx, window, context_for(window))
