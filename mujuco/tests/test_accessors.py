"""ClothFoldEnv accessors that task wrappers rely on instead of MuJoCo internals."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim_main import ClothFoldEnv  # noqa: E402


def test_accessors_match_mujoco_state():
    env = ClothFoldEnv(observation_mode="state", action_mode="joint_delta")
    env.reset(seed=0)
    corners = env.corner_positions()
    assert corners.shape == (4, 3)
    assert np.allclose(corners, env.data.xpos[env._corner_ids])
    left = env.gripper_position("left_")
    assert left.shape == (3,)
    assert np.allclose(left, env.data.site_xpos[env._site_id["left_"]])
    # returned arrays are copies: mutating them must not touch the sim
    corners[0, 0] += 1.0
    assert not np.allclose(corners, env.corner_positions())
