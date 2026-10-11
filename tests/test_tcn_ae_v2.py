"""TCN autoencoder v2: shapes per size, masked loss, it can fit, scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydro_knight.features.track_series import TrackSeries
from hydro_knight.features.windows_v2 import make_window_set
from hydro_knight.models.tcn_ae_v2 import (
    SIZES,
    TCNAutoencoderV2,
    masked_errors,
    n_params,
    score_v2,
    train_v2,
)


@pytest.mark.parametrize("size", list(SIZES))
def test_shapes_for_each_size(size):
    width, code = SIZES[size]
    model = TCNAutoencoderV2(58, width, code, window=90)
    assert model.context == 44
    x = torch.randn(2, 58, 90 + 2 * model.context)
    assert model.encode(x).shape == (2, code, 12)
    assert model(x).shape == (2, 58, 90)


def test_sizes_grow():
    p = [
        n_params(TCNAutoencoderV2(58, *SIZES[s])) for s in ("small", "medium", "large")
    ]
    assert p[0] < p[1] < p[2]


def test_missing_step_coordinates_are_not_graded_by_default():
    target = torch.zeros(1, 58, 4)
    pred = target.clone()
    pred[0, :34, 2] = 5.0  # wrong coordinates on step 2 ...
    det = torch.tensor([[True, True, False, True]])  # ... which is a missing step
    assert masked_errors(pred, target, det).item() == 0.0
    assert masked_errors(pred, target, det, coord_weight=1.0).item() > 0.0
    pred[0, 40, 2] = 5.0  # a confidence on the missing step is always graded
    assert masked_errors(pred, target, det).item() > 0.0


def _tiny_set(n_tracks=4, n=140):
    """Smooth, compressible tracks (a few sine rhythms): random noise can't be squeezed
    through a small code, so it would make the fit test meaningless."""
    t = np.arange(n)[:, None]
    f = np.arange(58)[None, :]
    series = []
    for k in range(n_tracks):
        feats = np.sin(2 * np.pi * t / (20 + 5 * k) + 0.3 * f).astype(np.float32)
        feats[:, 52] = 1.0
        series.append(TrackSeries("c", k, 0.0, feats, np.ones(n, bool)))
    return make_window_set(series)


def test_small_model_can_fit_a_tiny_set_and_scores_every_window():
    ws = _tiny_set()
    model, scaler = train_v2(ws, size="small", steps=300, batch=16, lr=3e-3, seed=0)
    h = model.loss_history_
    assert len(h) == 300 and h[-1] < 0.5 * h[0]
    scores = score_v2(model, ws, scaler)
    assert (
        scores.shape == (len(ws),) and np.isfinite(scores).all() and (scores >= 0).all()
    )
