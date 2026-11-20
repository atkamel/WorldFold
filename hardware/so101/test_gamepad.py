"""Gamepad routing and clearance checks with fake servo transport."""

from dataclasses import replace
import importlib
import sys
import threading

import numpy as np
import pytest

from hardware.controller.gip import ControllerState
from hardware.controller.inputs import _XInputGamepad, xinput_state

pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus")

import kinematics
from pair_clearance import PairClearance
from test_ui_lend import make_fake_sdk


class SimulatedArm:
    def __init__(self, mapping, angles):
        self.name = mapping.name
        self.lock = threading.Lock()
        self.limits = mapping.limits
        self.present = {i: round(mapping.to_ticks(i, angle)) for i, angle in enumerate(angles, 1)}
        self.command = dict(self.present)
        self.target = dict(self.present)
        self.torque = True
        self.pending_torque = None
        self.speed = 50
        self.writes = []

    def read_present(self):
        self.present = {m: round(v) for m, v in self.command.items()}
        return True

    def send_goals(self, goals):
        self.writes.append(dict(goals))
        return True


def work_pose(x, y, z):
    model = kinematics.Arm()
    angles, error, _ = model.solve(np.array([x, y, z]), np.radians([0, 40, -30, 80, 0]))
    assert error < 0.001
    return [*angles, 0.4]


def test_pair_geometry_checks_both_arms_at_independent_progress():
    guard = PairClearance(0.30)
    left, right = work_pose(.25, .08, .10), work_pose(.25, -.08, .10)
    left_inward, right_inward = work_pose(.25, -.15, .10), work_pose(.25, .15, .10)
    assert guard.obstruction([left], [right]) is None
    assert guard.obstruction([left_inward], [right]) is None
    assert guard.obstruction([left], [right_inward]) is None
    assert guard.obstruction([left_inward, left], [right, right_inward]) == "between arms"
    assert guard.obstruction([work_pose(.24, 0, -.02)], [right]) == "table"


def test_xinput_preserves_stick_signs_and_converts_triggers():
    state = xinput_state(_XInputGamepad(0x2300, 255, 0, -32768, 2000, 32767, -1200))
    assert state == ControllerState(buttons=("lb", "rb", "b"), left_trigger=1023,
                                    left_x=-32768, left_y=2000, right_x=32767, right_y=-1200)


@pytest.fixture
def mode(monkeypatch):
    monkeypatch.setitem(sys.modules, "scservo_sdk", make_fake_sdk({}))
    gamepad = importlib.import_module("gamepad")
    maps = [gamepad.replay.Follower(name) for name in ("arm2", "arm1")]
    left = SimulatedArm(maps[0], work_pose(.25, .08, .10))
    right = SimulatedArm(maps[1], work_pose(.25, -.08, .10))
    return gamepad.Gamepad(left, right, .30)


def test_two_arm_jog_grip_latch_and_deadman(mode):
    neutral = ControllerState(left_x=600, right_y=5500)
    rb = replace(neutral, buttons=("rb",))
    now = 10.0

    def send(state, age=0):
        nonlocal now
        now += .02
        mode.tick(state, now - age, now)

    for _ in range(30):
        send(neutral)
    send(rb)
    left, right = mode.arms
    assert not left.writes and not right.writes
    origin = [mode.kinematics.tip(mode.pose(m, arm.command)[:5])[0] for m, arm in zip(mode.maps, mode.arms)]
    send(replace(rb, left_y=32767, right_x=32767))
    assert len(left.writes) == len(right.writes) == 1
    end = [mode.kinematics.tip(mode.pose(m, arm.command)[:5])[0] for m, arm in zip(mode.maps, mode.arms)]
    assert end[0][0] > origin[0][0]
    assert end[1][1] < origin[1][1]
    assert [a.command[5] for a in mode.arms] == [a.writes[0][5] for a in mode.arms]

    before_jaws = left.command[6], right.command[6]
    send(replace(rb, buttons=("lb", "rb"), left_y=32767, right_y=32767, left_trigger=900, right_trigger=900))
    elevated = [mode.kinematics.tip(mode.pose(m, arm.command)[:5])[0] for m, arm in zip(mode.maps, mode.arms)]
    assert all(current[2] > previous[2] for current, previous in zip(elevated, end))
    first_jaws = left.command[6], right.command[6]
    send(rb)
    assert all(abs(arm.command[6] - initial) > abs(first - initial)
               for arm, initial, first in zip(mode.arms, before_jaws, first_jaws))
    send(neutral)
    counts = len(left.writes), len(right.writes)
    send(replace(neutral, left_y=32767, right_y=32767))
    send(replace(rb, left_y=32767, right_y=32767))
    assert (len(left.writes), len(right.writes)) == counts
    send(neutral)
    send(rb)
    send(replace(rb, left_y=32767, right_y=32767), age=1)
    assert (len(left.writes), len(right.writes)) == counts
    assert "timed out" in mode.status


def test_clearance_refusal_sends_neither_goal(mode, monkeypatch):
    neutral = ControllerState()
    rb = replace(neutral, buttons=("rb",))
    for now in (0, 0.4, 0.42):
        mode.tick(neutral, now, now)
    mode.tick(rb, .44, .44)
    monkeypatch.setattr(mode.guard, "obstruction", lambda *samples: "between arms")
    mode.tick(replace(rb, left_y=32767, right_y=32767), .46, .46)
    assert all(not arm.writes for arm in mode.arms)
    assert "between arms" in mode.status
