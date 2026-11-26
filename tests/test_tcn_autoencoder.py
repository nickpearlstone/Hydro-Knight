from __future__ import annotations

import numpy as np

from hydro_knight.models.tcn_autoencoder import reconstruction_error, train_tcn

WINDOW = 32  # keep divisible by 4
N_FEAT = 70


def _normal_windows(n: int, n_feat: int = N_FEAT, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 2.0 * np.pi, WINDOW)
    out = np.empty((n, WINDOW, n_feat), dtype=np.float32)
    for i in range(n):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_feat)
        out[i] = 0.5 * np.sin(t[:, None] + phase[None, :])
    return out


def _anomalous_windows(n: int, n_feat: int = N_FEAT, seed: int = 99) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (5.0 * rng.randn(n, WINDOW, n_feat)).astype(np.float32)


def test_train_tcn_runs_and_returns_model_and_scaler():
    model, scaler = train_tcn(_normal_windows(12), epochs=3)
    mean, std = scaler
    assert mean.shape == (N_FEAT,)
    assert std.shape == (N_FEAT,)
    assert np.all(std > 0)
    assert hasattr(model, "loss_history_")


def test_loss_history_has_one_finite_entry_per_epoch():
    model, _ = train_tcn(_normal_windows(12), epochs=7)
    assert len(model.loss_history_) == 7
    assert np.all(np.isfinite(model.loss_history_))


def test_loss_decreases_so_the_optimizer_is_wired():
    model, _ = train_tcn(_normal_windows(16), epochs=60)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_reconstruction_error_is_one_finite_score_per_window():
    windows = _normal_windows(16)
    model, scaler = train_tcn(windows, epochs=5)
    err = reconstruction_error(model, windows, scaler)
    assert err.shape == (16,)
    assert np.all(np.isfinite(err))
    assert np.all(err >= 0.0)


def test_trains_at_both_feature_widths():
    for n_feat in (34, 70):
        windows = _normal_windows(12, n_feat=n_feat)
        model, scaler = train_tcn(windows, epochs=3)
        err = reconstruction_error(model, windows, scaler)
        assert err.shape == (12,), f"n_feat={n_feat}"
        assert np.all(np.isfinite(err)), f"n_feat={n_feat}"


def test_same_seed_reproduces_the_run():
    windows = _normal_windows(12)
    a, _ = train_tcn(windows, epochs=5, seed=0)
    b, _ = train_tcn(windows, epochs=5, seed=0)
    assert a.loss_history_ == b.loss_history_


def test_different_seed_changes_the_run():
    windows = _normal_windows(12)
    a, _ = train_tcn(windows, epochs=5, seed=0)
    b, _ = train_tcn(windows, epochs=5, seed=1)
    assert a.loss_history_ != b.loss_history_


def test_anomalous_windows_score_higher_than_normal():
    model, scaler = train_tcn(_normal_windows(32), epochs=80)
    normal = reconstruction_error(model, _normal_windows(16, seed=7), scaler)
    anomalous = reconstruction_error(model, _anomalous_windows(16), scaler)
    assert np.median(anomalous) > np.median(normal)
