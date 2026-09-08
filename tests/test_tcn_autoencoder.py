"""
Tests for the Rung-3 TCN autoencoder (models/tcn_autoencoder.py).

train_tcn is called only from scripts (eval_report.py, rung3_demo.py), so nothing
in the suite exercised it until now — a bare NameError in the epoch loop could sit
in the file with all tests green. These pin the training contract the rest of the
pipeline leans on:

  - train_tcn actually runs, and returns a usable (model, scaler),
  - model.loss_history_ holds one finite entry per epoch — eval/report.py plots it
    and eval/tracking.py logs it to MLflow, so its length is a real contract,
  - the loss genuinely falls, proving the optimizer is attached to the graph,
  - reconstruction_error yields one finite, non-negative score per window,
  - the net adapts to the feature width it is handed: 34-dim (pose only) and
    70-dim (pose + velocity) both train, which the velocity-off A/B depends on,
  - a fixed seed reproduces a run exactly, so A/B numbers in MLflow mean something,
  - the Rung-3 premise itself: fit on normal motion, and out-of-distribution
    windows score higher.

Note: training prints a tqdm bar to stderr. pytest captures it, so it only shows
on failure or under `-s`.
"""

from __future__ import annotations

import numpy as np

from hydro_knight.models.tcn_autoencoder import reconstruction_error, train_tcn

WINDOW = 32  # production window length; the encoder halves time twice, so keep /4
N_FEAT = 70  # pose (34) + per-keypoint velocity (34) + centroid velocity (2)


def _normal_windows(n: int, n_feat: int = N_FEAT, seed: int = 0) -> np.ndarray:
    """
    Smooth, low-amplitude sinusoidal motion standing in for an ordinary swimmer.
    Each window gets its own random phase per feature, so the set has variety but
    stays inside one tight, learnable distribution.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 2.0 * np.pi, WINDOW)
    out = np.empty((n, WINDOW, n_feat), dtype=np.float32)
    for i in range(n):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_feat)
        out[i] = 0.5 * np.sin(t[:, None] + phase[None, :])
    return out


def _anomalous_windows(n: int, n_feat: int = N_FEAT, seed: int = 99) -> np.ndarray:
    """High-amplitude noise — deliberately far outside the smooth training set."""
    rng = np.random.RandomState(seed)
    return (5.0 * rng.randn(n, WINDOW, n_feat)).astype(np.float32)


def test_train_tcn_runs_and_returns_model_and_scaler():
    # The smoke test the suite was missing. A typo in the epoch loop (`tdqm`) or
    # any import-time break fails here, instead of at the top of a real GPU run.
    model, scaler = train_tcn(_normal_windows(12), epochs=3)
    mean, std = scaler
    assert mean.shape == (N_FEAT,)
    assert std.shape == (N_FEAT,)
    assert np.all(std > 0)  # the +1e-6 floor keeps the divide safe on dead features
    assert hasattr(model, "loss_history_")


def test_loss_history_has_one_finite_entry_per_epoch():
    # report.py plots this curve and tracking.py logs it per-step to MLflow, so
    # "one entry per epoch" is a contract, not an implementation detail.
    model, _ = train_tcn(_normal_windows(12), epochs=7)
    assert len(model.loss_history_) == 7
    assert np.all(np.isfinite(model.loss_history_))


def test_loss_decreases_so_the_optimizer_is_wired():
    # Guards the classic silent failure: a detached graph or a dropped opt.step()
    # still produces a plausible-looking loss history while learning nothing.
    model, _ = train_tcn(_normal_windows(16), epochs=60)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_reconstruction_error_is_one_finite_score_per_window():
    windows = _normal_windows(16)
    model, scaler = train_tcn(windows, epochs=5)
    err = reconstruction_error(model, windows, scaler)
    assert err.shape == (16,)
    assert np.all(np.isfinite(err))
    assert np.all(err >= 0.0)  # it is a mean of squares


def test_trains_at_both_feature_widths():
    # n_feat is inferred from windows.shape[-1]. The velocity-off baseline feeds
    # 34-dim pose-only windows and must train exactly like the 70-dim run it is
    # being compared against, or the A/B is measuring plumbing instead of features.
    for n_feat in (34, 70):
        windows = _normal_windows(12, n_feat=n_feat)
        model, scaler = train_tcn(windows, epochs=3)
        err = reconstruction_error(model, windows, scaler)
        assert err.shape == (12,), f"n_feat={n_feat}"
        assert np.all(np.isfinite(err)), f"n_feat={n_feat}"


def test_same_seed_reproduces_the_run():
    # An A/B in MLflow only means something if a run is repeatable; otherwise the
    # gap between two runs is just seed noise.
    windows = _normal_windows(12)
    a, _ = train_tcn(windows, epochs=5, seed=0)
    b, _ = train_tcn(windows, epochs=5, seed=0)
    assert a.loss_history_ == b.loss_history_


def test_different_seed_changes_the_run():
    # The flip side: the seed must actually reach weight init, or `seed` is a lie.
    windows = _normal_windows(12)
    a, _ = train_tcn(windows, epochs=5, seed=0)
    b, _ = train_tcn(windows, epochs=5, seed=1)
    assert a.loss_history_ != b.loss_history_


def test_anomalous_windows_score_higher_than_normal():
    # The Rung-3 premise in miniature: fit only on normal motion, and windows from
    # outside that distribution reconstruct worse. Medians, so no single outlier
    # decides the result.
    model, scaler = train_tcn(_normal_windows(32), epochs=80)
    normal = reconstruction_error(model, _normal_windows(16, seed=7), scaler)
    anomalous = reconstruction_error(model, _anomalous_windows(16), scaler)
    assert np.median(anomalous) > np.median(normal)
