"""
SAHI-style tiled pose detection.

Distant swimmers vanish because the whole-frame downscale shrinks them below
detectability. Fix: slice the frame into overlapping tiles, run YOLO-pose on
each tile (where a far swimmer is now a large fraction of the tile), map the
detections back to full-frame coordinates, and merge duplicates from tile
overlaps with non-max-suppression.

Detection only (no tracking) — tracking over merged detections is a later step.
"""

from __future__ import annotations

import numpy as np


def _tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """
    Start coordinates of tiles along one axis (width or height).

    Step is the tile size minus the overlap, so neighbouring tiles share a
    margin (a swimmer on a seam still appears whole in one of them). We also
    force the last tile to touch the far edge so nothing past the final step
    is missed.
    """
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, length - tile + 1, step))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two xyxy boxes (overlap fraction, 0–1)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(dets: list, iou_thresh: float) -> list:
    """
    Greedy non-max-suppression: keep the highest-confidence detections, drop
    any that overlap an already-kept one by more than iou_thresh. This removes
    the duplicate of a swimmer seen in two overlapping tiles.

    dets: list of (box_xyxy, conf, keypoints). Returns the surviving subset.
    """
    kept: list = []
    for box, conf, kpts in sorted(dets, key=lambda d: d[1], reverse=True):
        if all(_iou(box, kb) < iou_thresh for kb, _, _ in kept):
            kept.append((box, conf, kpts))
    return kept


def detect_tiled(
    model,
    frame,
    tile: int = 480,
    overlap: float = 0.25,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou_merge: float = 0.5,
    include_full: bool = True,
) -> list:
    """
    Run YOLO-pose over a tiled grid and return merged full-frame detections.

    Each detection is (box_xyxy, conf, keypoints[17,3]) in FULL-frame pixels.
    include_full also runs one whole-frame pass to catch large/close swimmers
    a tile might cut in half.

    KEY: imgsz must exceed `tile` for SAHI to help — that upscales each tile so
    distant swimmers reach the size YOLO was trained to detect. With imgsz==tile
    there's no zoom and tiling only adds cost (and can lose detections to merge
    seams). Here tile=480 run at imgsz=1280 = ~2.7x upscale. On our footage this
    took recall from ~13 to ~50 swimmers/frame on a crowded clip.
    """
    H, W = frame.shape[:2]

    # Build the list of (offset_x, offset_y, sub-image) crops to run.
    crops = []
    for oy in _tile_origins(H, tile, overlap):
        for ox in _tile_origins(W, tile, overlap):
            x2, y2 = min(ox + tile, W), min(oy + tile, H)
            crops.append((ox, oy, frame[oy:y2, ox:x2]))
    if include_full:
        crops.append((0, 0, frame))  # offset 0 — already full-frame coords

    dets = []
    for ox, oy, sub in crops:
        r = model(sub, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
        for i in range(len(boxes)):
            box = boxes[i].copy()
            box[[0, 2]] += ox  # shift x by tile offset
            box[[1, 3]] += oy  # shift y by tile offset
            k = kpts[i].copy()
            k[:, 0] += ox  # shift keypoint x
            k[:, 1] += oy  # shift keypoint y
            dets.append((box, float(confs[i]), k))

    return _nms_merge(dets, iou_merge)
