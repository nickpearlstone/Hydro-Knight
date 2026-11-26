"""SAHI-style tiled pose detection; crop ids are tile_grid order (row by row), whole-frame is FULL_FRAME (-1)."""

from __future__ import annotations

import numpy as np

FULL_FRAME = -1


def _tile_origins(length: int, tile: int, overlap: float) -> list[int]:
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
    """Top-left (ox, oy) pixel offset of every tile, row by row; list position is the crop id."""
    return [
        (ox, oy)
        for oy in _tile_origins(height, tile, overlap)
        for ox in _tile_origins(width, tile, overlap)
    ]


def _unpack(result, ox: int, oy: int):
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
    """Run YOLO-pose on every tile plus the whole frame, no merging; boxes/keypoints returned in full-frame pixels.

    `imgsz` must exceed `tile` or tiling gives no zoom.
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
    same_crop_below: float | None = None,
) -> np.ndarray:
    """Greedy merge of tile-overlap duplicates, highest confidence first; returns kept indices.

    mode="ios" drops cross-crop duplicates by IoU or IoS; "iou" is the original rule.
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
            other = crops[rest] != crops[i]
            if same_crop_below is not None:
                other |= confs[rest] < same_crop_below
            dup = other & ((iou >= iou_thresh) | (ios >= ios_thresh))
        alive[rest[dup]] = False
    return np.asarray(kept, dtype=int)
