"""
Tests for preprocess/tiled_pose.py: tile layout, crop detection offsets, and the merge.

The merge tests pin the seam bug found on a real rescue frame: a swimmer cut by
a tile edge produces a small partial box inside the full-body box from another
crop. Their IoU stays under 0.5, so the original IoU-only merge kept both and the
tracker could spawn a phantom swimmer. IoS (overlap / smaller area) is ~1.0 for
that pair.
"""

from __future__ import annotations

import numpy as np
from yolo_fakes import Result as _Result

from hydro_knight.preprocess.tiled_pose import (
    FULL_FRAME,
    detect_crops,
    merge_detections,
    tile_grid,
)


def test_tile_grid_on_720p_pins_last_tiles_to_the_edges():
    # 1280 wide: range(0, 801, 360) = 0,360,720 then 800 is appended.
    # 720 tall: range(0, 241, 360) = 0 then 240 is appended. Row by row.
    grid = tile_grid(1280, 720, tile=480, overlap=0.25)
    xs, ys = [0, 360, 720, 800], [0, 240]
    assert grid == [(x, y) for y in ys for x in xs]


def test_tile_grid_frame_smaller_than_tile_is_one_tile():
    assert tile_grid(400, 300, tile=480, overlap=0.25) == [(0, 0)]


# --- merge -------------------------------------------------------------------

FULL = [770.0, 440.0, 790.0, 490.0]  # 20 x 50 full body
HALF = [770.0, 440.0, 790.0, 462.0]  # top 20 x 22 of the same body; IoU 0.44, IoS 1.0


def _merge(boxes, confs, crops, **kw):
    return merge_detections(
        np.array(boxes, np.float32), np.array(confs, np.float32), np.array(crops), **kw
    ).tolist()


def test_ios_merge_drops_half_body_duplicate_from_another_crop():
    assert _merge([FULL, HALF], [0.81, 0.55], [1, 2]) == [0]


def test_iou_mode_reproduces_the_original_seam_bug():
    assert _merge([FULL, HALF], [0.81, 0.55], [1, 2], mode="iou") == [0, 1]


def test_ios_merge_keeps_overlapping_people_from_the_same_crop():
    # YOLO already separated these two within one tile; merging them would delete a real person.
    assert sorted(_merge([FULL, HALF], [0.81, 0.55], [3, 3])) == [0, 1]


def test_merge_keeps_the_most_confident_duplicate():
    shifted = [771.0, 441.0, 791.0, 491.0]
    assert _merge([FULL, shifted], [0.4, 0.9], [0, FULL_FRAME]) == [1]


def test_merge_keeps_separate_people():
    far = [100.0, 100.0, 120.0, 150.0]
    assert sorted(_merge([FULL, far], [0.5, 0.6], [0, 1])) == [0, 1]


def test_merge_empty_input():
    assert _merge(np.empty((0, 4)), [], []) == []


# --- detect_crops -----------------------------------------------------------


class _FakeModel:
    """Returns one detection at crop-local (10, 20, 30, 40) per image."""

    def __init__(self):
        self.calls = []

    def __call__(self, source, **kw):
        images = source if isinstance(source, list) else [source]
        self.calls.append(len(images))
        return [_Result([[10, 20, 30, 40]], [0.9]) for _ in images]


def test_detect_crops_batches_tiles_and_shifts_to_full_frame():
    frame = np.zeros((720, 1280, 3), np.uint8)
    origins = tile_grid(1280, 720, 480, 0.25)
    model = _FakeModel()

    boxes, confs, kpts, crops = detect_crops(model, frame, origins, tile=480)

    assert model.calls == [
        len(origins),
        1,
    ]  # one batched call for tiles, one whole-frame call
    assert len(boxes) == len(origins) + 1
    assert crops.tolist() == list(range(len(origins))) + [FULL_FRAME]
    for (ox, oy), box, kp in zip(origins, boxes, kpts, strict=False):
        assert box.tolist() == [10 + ox, 20 + oy, 30 + ox, 40 + oy]
        assert np.allclose(kp[:, 0], 5 + ox) and np.allclose(kp[:, 1], 5 + oy)
    assert boxes[-1].tolist() == [10, 20, 30, 40]  # whole frame: no offset


def test_detect_crops_no_detections():
    class _Empty(_FakeModel):
        def __call__(self, source, **kw):
            images = source if isinstance(source, list) else [source]
            return [_Result(np.empty((0, 4)), []) for _ in images]

    frame = np.zeros((720, 1280, 3), np.uint8)
    boxes, confs, kpts, crops = detect_crops(
        _Empty(), frame, tile_grid(1280, 720, 480, 0.25)
    )
    assert boxes.shape == (0, 4) and kpts.shape == (0, 17, 3) and len(crops) == 0
