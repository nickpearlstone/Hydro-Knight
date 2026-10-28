"""Appearance crops, per-track caching, the stacked probe's fold hygiene, whole-frame input size."""

import numpy as np
import pandas as pd

from hydro_knight.stack import appearance, whole_frame


def test_crop_is_centred_below_the_head():
    img = np.zeros((400, 400, 3), np.uint8)
    img[195:205, 195:205] = 255  # a dot at (200, 200)
    # head at (200, 170), 40 px tall: crop centre = 170 + 0.75 * 40 = 200
    c = appearance.crop(img, 200, 170, 40)
    assert c.shape == (appearance.CROP_PX, appearance.CROP_PX, 3)
    ys, xs = np.nonzero(c[..., 0] > 128)
    mid = appearance.CROP_PX / 2
    assert abs(ys.mean() - mid) < 3 and abs(xs.mean() - mid) < 3


def test_crop_off_the_edge_keeps_its_size():
    img = np.full((100, 100, 3), 7, np.uint8)
    assert appearance.crop(img, 2, 2, 30).shape == (224, 224, 3)


def _rows():
    return pd.DataFrame(
        {
            "clip_id": ["a"] * 6 + ["b"] * 2,
            "src_track": [1, 1, 1, 1, 2, 2, 1, 1],
            "t": [0.0, 0.5, 1.0, 2.0, 0.5, 1.5, 0.0, 0.5],
        }
    )


def test_cache_embeds_first_row_per_track_bucket_and_carries_it_forward():
    r = _rows()
    pick = appearance.cache_rows(r, 2.0)
    assert pick.tolist() == [True, False, False, True, True, False, True, False]
    s = np.where(pick, np.arange(len(r), dtype=float), np.nan)
    assert appearance.carry_forward(r, s, 2.0).tolist() == [0, 0, 0, 3, 4, 4, 6, 6]
    assert appearance.cache_rows(r, 0).all()


def test_stacked_feature_never_scores_a_row_with_a_probe_that_saw_its_fold():
    rng = np.random.default_rng(0)
    n, folds = 600, 3
    fold = np.repeat(np.arange(folds), n // folds)
    victim = rng.random(n) < 0.2
    X = rng.normal(size=(n, 80)).astype(np.float32)
    X[victim, 0] += 3  # a real, shared signal
    labelled = np.ones(n, bool)
    a = appearance.stacked_feature(X, victim, labelled, fold, 0)
    # flipping fold 1's labels must not change fold 1's own scores (inner out-of-fold)
    flipped = victim.copy()
    flipped[fold == 1] = ~flipped[fold == 1]
    b = appearance.stacked_feature(X, flipped, labelled, fold, 0)
    assert np.allclose(a[fold == 1], b[fold == 1])
    assert not np.allclose(a[fold == 2], b[fold == 2])  # fold 2's probe did see fold 1
    assert not np.isnan(a).any()


def test_unembedded_rows_stay_nan():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 70)).astype(np.float32)
    X[::7] = np.nan
    victim = rng.random(300) < 0.3
    fold = np.repeat([0, 1, 2], 100)
    a = appearance.stacked_feature(X, victim, np.ones(300, bool), fold, 2)
    assert np.isnan(a[::7]).all() and not np.isnan(np.delete(a, np.s_[::7])).any()


def test_whole_frame_input_keeps_aspect_in_multiples_of_32():
    assert whole_frame.input_size(1920, 1080, 640) == (640, 352)
    assert whole_frame.input_size(1280, 720, 1280) == (1280, 704)
