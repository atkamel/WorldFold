"""Roy's grasp port without Isaac: flat pads meet at the closed limit, the roy layout, the bench's knob parser."""

import numpy as np
import pytest

from isaac.isaac_env import BOX_PAD, PROFILES, TABLE_TOP_Z, box_pad_poses


def _rigid(seed):
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(A) < 0:
        A[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = A, rng.normal(scale=0.05, size=3)
    return T


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pad_faces_meet_flush_at_the_closed_limit(seed):
    T0 = _rigid(seed)
    fixed, moving, size = box_pad_poses(T0)
    a = np.radians(BOX_PAD["q_closed_deg"])
    Rz = np.eye(4)
    Rz[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    moving_in_gripper = T0 @ Rz @ moving
    t = size[0] / 2
    fixed_face = fixed[:3, 3] + fixed[:3, :3] @ [t, 0, 0]
    moving_face = moving_in_gripper[:3, 3] + moving_in_gripper[:3, :3] @ [-t, 0, 0]
    assert np.allclose(fixed_face, moving_face, atol=1e-9)
    assert np.allclose(fixed_face, [BOX_PAD["x_face"], 0.0, BOX_PAD["z_center"]])
    assert np.allclose(moving_in_gripper[:3, :3], np.eye(3), atol=1e-9)      # faces parallel when closed


def test_roy_layout():
    prof = PROFILES["roy"]
    left, right = (prof["arm_bases"][p][0] for p in ("left_", "right_"))
    assert left[2] - TABLE_TOP_Z == pytest.approx(0.0125)                    # ~1 cm above the table, as on LeHome
    assert right[0] - left[0] == pytest.approx(0.46)
    assert prof["cloth_center"][1] - left[1] == pytest.approx(0.2657)        # the cloth ~26 cm out
    assert prof["friction_grasp"]["pads"] == "box" and prof["friction_grasp"]["gripper_mode"] == "position"


def test_bench_knobs_take_bare_words():
    from isaac.grasp_bench import parse_kv
    assert parse_kv("pads=box,gripper_closed=-0.17,gripper_drive.stiffness=50") == \
        {"pads": "box", "gripper_closed": -0.17, "gripper_drive": {"stiffness": 50}}
