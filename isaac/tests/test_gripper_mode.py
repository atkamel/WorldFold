"""IsaacClothFoldEnv's gripper modes without Isaac: binary hysteresis unchanged, "position" maps the command onto the
jaw target and reads a stalled squeeze as a grasp."""

import numpy as np
import pytest

from isaac.isaac_env import GRIPPER_OPEN, JAW_CLOSED_BELOW, IsaacClothFoldEnv

LOW = -0.174533


class _Env(IsaacClothFoldEnv):
    """Just the state set_gripper / grasp_active read; jaw angle and speed are set by the test."""

    def __init__(self, knobs):
        self.grasp_mode, self.grasp_knobs = "friction", dict(knobs)
        self._closed_q, self._dof_limits = 0.05, np.array([[-2.0, 2.0]] * 5 + [[LOW, 1.745]])
        self._joint_targets = {"left_": np.zeros(6)}
        self._gripper_closed = {"left_": False}
        self.grasp_corners, self.weld_mask = {"left_": (0,)}, {}
        self.jaw, self.jaw_speed = 1.0, 0.0

    def joint_positions(self, prefix):
        return np.r_[np.zeros(5), self.jaw]

    def joint_velocities(self, prefix):
        return np.r_[np.zeros(5), self.jaw_speed]

    def gripper_position(self, prefix):
        return np.zeros(3)

    def cloth_positions(self):
        return np.zeros((1, 3))       # the corner sits in the jaws


def test_binary_unchanged():
    env = _Env({})
    env.set_gripper("left_", -1.0)
    assert env._joint_targets["left_"][5] == 0.05 and env._gripper_closed["left_"]
    env.set_gripper("left_", 0.0)                       # hold
    assert env._gripper_closed["left_"]
    env.jaw = 0.05
    assert env.grasp_active("left_")
    env.set_gripper("left_", 1.0)
    assert env._joint_targets["left_"][5] == GRIPPER_OPEN and not env.grasp_active("left_")


@pytest.mark.parametrize("command, target", [(-1.0, LOW), (1.0, GRIPPER_OPEN), (0.0, (LOW + GRIPPER_OPEN) / 2),
                                             (-3.0, LOW)])
def test_position_maps_linearly(command, target):
    env = _Env({"gripper_mode": "position"})
    env.set_gripper("left_", command)
    assert env._joint_targets["left_"][5] == pytest.approx(target)


def test_position_rate_limit_both_ways():
    env = _Env({"gripper_mode": "position", "jaw_open_rate": 0.1})
    env._joint_targets["left_"][5] = 0.5
    env.set_gripper("left_", -1.0)
    assert env._joint_targets["left_"][5] == pytest.approx(0.4)
    env.set_gripper("left_", 1.0)
    assert env._joint_targets["left_"][5] == pytest.approx(0.5)


def test_position_grasp_reads_a_stalled_squeeze():
    env = _Env({"gripper_mode": "position"})
    env.set_gripper("left_", -1.0)                      # target at the limit, well past contact
    env.jaw, env.jaw_speed = 0.01, 0.0                  # stopped on the cloth, short of the target
    assert env.grasp_active("left_")
    env.jaw_speed = 1.0                                 # still closing
    assert not env.grasp_active("left_")
    env.jaw, env.jaw_speed = 0.5, 0.0                   # held wide open by something
    assert not env.grasp_active("left_")
    env.set_gripper("left_", 0.0)                       # a mid target is open
    assert env._joint_targets["left_"][5] > JAW_CLOSED_BELOW and not env.grasp_active("left_")
