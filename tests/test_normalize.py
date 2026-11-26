from __future__ import annotations

import numpy as np

from hydro_knight.features.normalize import (
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    normalize_pose,
)


def _good_pose() -> np.ndarray:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 1.0
    kpts[L_HIP] = [90.0, 100.0, 1.0]
    kpts[R_HIP] = [110.0, 100.0, 1.0]
    kpts[L_SHOULDER] = [90.0, 60.0, 1.0]
    kpts[R_SHOULDER] = [110.0, 60.0, 1.0]
    return kpts


def test_good_pose_returns_34_vector():
    v = normalize_pose(_good_pose())
    assert v is not None
    assert v.shape == (34,)
    assert v.dtype == np.float32


def test_hip_center_maps_to_origin():
    v = normalize_pose(_good_pose()).reshape(17, 2)
    hip_center = (v[L_HIP] + v[R_HIP]) / 2.0
    assert np.allclose(hip_center, [0.0, 0.0], atol=1e-5)


def test_translation_invariance():
    base = _good_pose()
    shifted = base.copy()
    shifted[:, :2] += np.array([37.0, -12.0])
    assert np.allclose(normalize_pose(base), normalize_pose(shifted), atol=1e-5)


def test_scale_invariance():
    base = _good_pose()
    bigger = base.copy()
    bigger[:, :2] *= 2.0
    assert np.allclose(normalize_pose(base), normalize_pose(bigger), atol=1e-5)


def test_low_reference_confidence_drops_pose():
    # one weak reference joint discards the pose
    pose = _good_pose()
    pose[R_HIP, 2] = 0.1
    assert normalize_pose(pose, min_ref_conf=0.3) is None


def test_degenerate_torso_drops_pose():
    pose = _good_pose()
    pose[L_SHOULDER, :2] = pose[L_HIP, :2]
    pose[R_SHOULDER, :2] = pose[R_HIP, :2]
    assert normalize_pose(pose) is None
