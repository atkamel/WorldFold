"""Rotation helpers used for camera and light aiming, pin offsets, and the EE quaternion. Runs without Isaac Sim."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest  # noqa: E402

from isaac.isaac_env import LIGHTS, _matrix_from_quat, _quat_facing, _quat_from_matrix  # noqa: E402
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


def test_quat_sign_matches_mujoco():
    # proprio carries the EE quaternion, so Isaac must pick the same one of q / -q as mju_mat2Quat
    mujoco = pytest.importorskip("mujoco")
    rng = np.random.default_rng(1)
    for _ in range(200):
        R = random_rotation(rng)
        expected = np.zeros(4)
        mujoco.mju_mat2Quat(expected, R.ravel())
        assert np.allclose(_quat_from_matrix(R), expected, atol=1e-9)


def test_lights_face_mujoco_directions():
    for direction, _ in LIGHTS:
        d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        R = _matrix_from_quat(_quat_facing(direction))
        assert np.allclose(R @ np.array([0.0, 0.0, -1.0]), d, atol=1e-9)
