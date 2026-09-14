"""
SAHI-style tiled pose detection: the per-frame pieces used by extraction and tracking.

Distant swimmers vanish because the whole-frame downscale shrinks them below
detectability. Fix: slice the frame into overlapping tiles, run YOLO-pose on
each tile (where a far swimmer is now a large fraction of the tile), map the
detections back to full-frame coordinates, and merge the duplicates that the
tile overlaps create.

The two halves live apart on purpose. `detect_crops` is the expensive GPU step
and returns every raw detection tagged with the crop it came from; extraction
stores those as-is. `merge_detections` is cheap and runs later, offline, so the
merge rule can change without re-running YOLO.

Crop ids: tiles are numbered 0..n-1 in `tile_grid` order (row by row), and the
whole-frame pass is `FULL_FRAME` (-1).
"""

from __future__ import annotations

import numpy as np

FULL_FRAME = -1


def _tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Tile start offsets along one axis, overlapping and pinned to the far edge.

    Step = tile*(1-overlap) so neighbours share a margin (a swimmer on a seam stays whole in one
    tile); the last origin is forced to length-tile so nothing past the final step is missed.
    Args:
        length: axis length in pixels (width or height).
        tile: tile size in pixels.
        overlap: fractional overlap between neighbouring tiles.
    Returns:
        List of start offsets; [0] when the axis fits within one tile.
    """
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, length - tile + 1, step))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def tile_grid(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int]]:
    """Top-left (ox, oy) of every tile, row by row; list position is the tile's crop id.

    Args:
        width: frame width in pixels.
        height: frame height in pixels.
        tile: tile size in pixels.
        overlap: fractional overlap between neighbouring tiles.
    Returns:
        List of (ox, oy) pixel offsets.
    """
    return [
        (ox, oy)
        for oy in _tile_origins(height, tile, overlap)
        for ox in _tile_origins(width, tile, overlap)
    ]


def _unpack(result, ox: int, oy: int):
    """One Ultralytics Results -> (boxes (n,4), confs (n,), kpts (n,17,3)) shifted by (ox, oy)."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.cpu().numpy().astype(np.float32)
    kpts = result.keypoints.data.cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] += ox
    boxes[:, [1, 3]] += oy
    kpts[:, :, 0] += ox
    kpts[:, :, 1] += oy
    return boxes, confs, kpts


def detect_crops(
    model,
    frame: np.ndarray,
    origins: list[tuple[int, int]],
    tile: int = 480,
    imgsz: int = 1280,
    conf: float = 0.25,
    device=None,
    include_full: bool = True,
):
    """Run YOLO-pose on every tile (one batched call) plus the whole frame; no merging.

    KEY: imgsz must exceed `tile` for tiling to help — each tile is upscaled so distant swimmers
    reach the size YOLO was trained on (imgsz == tile means no zoom, pure cost). YOLO returns boxes
    in the crop's own pixels, so mapping back to the full frame is a pure offset.
    Args:
        model: a YOLO-pose model (callable on a list of images).
        frame: full BGR frame, (H, W, 3).
        origins: tile offsets from `tile_grid`.
        tile: tile size in pixels.
        imgsz: inference resolution per crop.
        conf: detection confidence floor.
        device: torch device passed to YOLO, or None for Ultralytics' default.
        include_full: also run one whole-frame pass (catches large, close swimmers a tile cuts).
    Returns:
        (boxes, confs, kpts, crops): (N,4) float32 xyxy and (N,17,3) float32 keypoints in
        full-frame pixels, (N,) float32 confidences, (N,) int8 crop ids. N may be 0.
    """
    jobs = []
    if origins:
        tiles = [frame[oy : oy + tile, ox : ox + tile] for ox, oy in origins]
        results = model(tiles, imgsz=imgsz, conf=conf, verbose=False, device=device)
        jobs += [
            (r, ox, oy, i)
            for i, (r, (ox, oy)) in enumerate(zip(results, origins, strict=True))
        ]
    if include_full:
        r = model(frame, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        jobs.append((r, 0, 0, FULL_FRAME))

    parts = []
    for r, ox, oy, crop in jobs:
        unpacked = _unpack(r, ox, oy)
        if unpacked is not None:
            parts.append((*unpacked, np.full(len(unpacked[0]), crop, dtype=np.int8)))
    if not parts:
        return (
            np.empty((0, 4), np.float32),
            np.empty(0, np.float32),
            np.empty((0, 17, 3), np.float32),
            np.empty(0, np.int8),
        )
    return tuple(np.concatenate(col) for col in zip(*parts, strict=True))


def _overlaps(box: np.ndarray, boxes: np.ndarray):
    """IoU and IoS (intersection over the smaller box) of one xyxy box against many."""
    ix = np.clip(
        np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None
    )
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    smaller = np.minimum(area, areas)
    ios = np.divide(inter, smaller, out=np.zeros_like(inter), where=smaller > 0)
    return iou, ios


def merge_detections(
    boxes: np.ndarray,
    confs: np.ndarray,
    crops: np.ndarray,
    mode: str = "ios",
    iou_thresh: float = 0.5,
    ios_thresh: float = 0.8,
) -> np.ndarray:
    """Greedy merge of tile-overlap duplicates; returns the indices of detections to keep.

    Highest confidence first. mode="ios" (default) drops a detection from a *different* crop that
    overlaps a kept one by IoU >= iou_thresh or IoS >= ios_thresh — IoS catches the half-body box a
    tile seam produces, which sits inside the full box but has low IoU. Same-crop pairs are left
    alone: YOLO already separated them, so they are usually two real, adjacent people.
    mode="iou" reproduces the original merge (IoU only, any crop) for comparison.
    Args:
        boxes: (N,4) xyxy boxes in full-frame pixels.
        confs: (N,) detection confidences.
        crops: (N,) crop ids.
        mode: "ios" or "iou".
        iou_thresh: IoU at or above which a duplicate is dropped.
        ios_thresh: IoS at or above which a duplicate is dropped (mode="ios" only).
    Returns:
        (K,) int array of kept indices into the inputs, highest confidence first.
    Raises:
        ValueError: unknown mode.
    """
    if mode not in ("ios", "iou"):
        raise ValueError(f"unknown merge mode {mode!r} (expected 'ios' or 'iou')")
    order = np.argsort(-confs, kind="stable")
    alive = np.ones(len(boxes), dtype=bool)
    kept = []
    for i in order:
        if not alive[i]:
            continue
        kept.append(i)
        alive[i] = False
        rest = np.flatnonzero(alive)
        if len(rest) == 0:
            break
        iou, ios = _overlaps(boxes[i], boxes[rest])
        if mode == "iou":
            dup = iou >= iou_thresh
        else:
            dup = (crops[rest] != crops[i]) & (
                (iou >= iou_thresh) | (ios >= ios_thresh)
            )
        alive[rest[dup]] = False
    return np.asarray(kept, dtype=int)
