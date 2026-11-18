"""collision.closest on the so101_nexus model: clear poses read clear, the release against the base reads as overlap."""

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus")
sys.path.insert(0, str(Path(__file__).parent))

import collision  # noqa: E402
import fold_traj  # noqa: E402

ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# fingers straight down 8 cm in front of the base origin: where the first fold on arm 2 struck its base
AT_THE_BASE = [0.0, math.radians(-81), math.radians(80), math.radians(95), 0.0, -0.17]


def test_the_zero_pose_and_the_stand_pose_are_clear():
    stand = [float(angle) for angle in fold_traj.STAND] + [fold_traj.OPEN]
    for row in (ZERO, stand):
        distance, step, pair = collision.closest([row])
        assert distance >= collision.CLEARANCE


def test_the_gripper_against_the_base_is_an_overlap():
    distance, step, pair = collision.closest([AT_THE_BASE])
    assert distance < 0
    assert "gripper" in pair


def test_closest_names_the_worst_row():
    distance, step, pair = collision.closest([ZERO, ZERO, AT_THE_BASE, ZERO])
    assert step == 2
    assert distance < 0
