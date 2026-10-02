"""Rotation helpers used for camera aiming, pin offsets, and EE velocity. Runs without Isaac Sim."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from isaac.isaac_env import _matrix_from_quat, _quat_from_matrix, _quat_mul, _rotvec_between  # noqa: E402
from mujuco.cloth_params import CAMERA_POS, CAMERA_TARGET, camera_axes  # noqa: E402


def random_rotation(rng):
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    return _matrix_from_quat(q)


def test_matrix_quat_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = random_rotation(rng)
        assert np.allclose(_matrix_from_quat(_quat_from_matrix(R)), R, atol=1e-6)


def test_camera_pose_roundtrip():
    right, up = camera_axes(CAMERA_POS, CAMERA_TARGET)
    forward = np.cross(up, right)
    R = np.column_stack([forward, -right, up])
    R2 = _matrix_from_quat(_quat_from_matrix(R))
    assert np.allclose(R2, R, atol=1e-6)
    assert np.allclose(R2 @ np.array([1.0, 0.0, 0.0]), forward, atol=1e-6)


def test_quat_mul_matches_matrix_product():
    rng = np.random.default_rng(1)
    for _ in range(50):
        Ra = random_rotation(rng)
        Rb = random_rotation(rng)
        q = _quat_mul(_quat_from_matrix(Ra), _quat_from_matrix(Rb))
        assert np.allclose(_matrix_from_quat(q), Ra @ Rb, atol=1e-6)


def test_rotvec_between_recovers_small_rotation():
    angle = 0.2
    axis = np.array([0.0, 0.0, 1.0])
    q_from = np.array([1.0, 0.0, 0.0, 0.0])
    q_to = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
    assert np.allclose(_rotvec_between(q_from, q_to), angle * axis, atol=1e-6)
