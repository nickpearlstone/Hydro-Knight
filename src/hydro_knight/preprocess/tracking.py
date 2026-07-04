"""
A minimal IoU tracker for assigning persistent IDs to per-frame detections.

The SAHI detector gives us boxes per frame with no identity. This links them
across frames: a detection that overlaps an existing track (high IoU) inherits
that track's id; unmatched detections start new tracks; tracks that go unseen
for too long are dropped.

Deliberately simple (greedy IoU matching). It's the "make it work" version —
ByteTrack/BoT-SORT are the robust upgrade, especially for re-identifying a
swimmer after a submersion gap (the signal we ultimately care about).
"""

from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """
    Greedy IoU tracker.

    iou_thresh : minimum overlap to consider a detection the same swimmer.
    max_age    : frames a track may go unmatched before it's dropped (lets a
                 track survive a brief miss — e.g. one bad frame — without
                 losing its id).
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}  # id -> {"box": xyxy, "age": int}
        self._next_id = 1

    def update(self, boxes: list[np.ndarray]) -> list[int]:
        """
        Match this frame's boxes to existing tracks; return an id per box
        (same order as the input list).
        """
        ids: list[int | None] = [None] * len(boxes)

        # Score every (track, detection) pair, then assign greedily from the
        # highest-IoU pair down, so the most confident matches win first and
        # each track/detection is used at most once.
        pairs = []
        for tid, tr in self.tracks.items():
            for di, box in enumerate(boxes):
                iou = _iou(tr["box"], box)
                if iou >= self.iou_thresh:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)

        used_tracks, used_dets = set(), set()
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            ids[di] = tid
            self.tracks[tid]["box"] = boxes[di]
            self.tracks[tid]["age"] = 0
            used_tracks.add(tid)
            used_dets.add(di)

        # Unmatched detections -> brand-new tracks.
        for di, box in enumerate(boxes):
            if ids[di] is None:
                ids[di] = self._next_id
                self.tracks[self._next_id] = {"box": box, "age": 0}
                self._next_id += 1

        # Age unmatched tracks; drop the ones missing too long.
        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        return ids  # type: ignore[return-value]
