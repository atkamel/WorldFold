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
        self.stuck = False

    def read_present(self):
        if not self.stuck:
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


def test_pair_geometry_checks_neighbouring_parts_and_lets_a_close_arm_back_away():
    guard = PairClearance(0.30)
    folded = np.radians([0, -104, 90, -96, 0, 0])
    deeper = np.radians([0, -104, 97, -96, 0, 0])
    opened = np.radians([0, -104, 60, -96, 0, 0])
    assert guard.obstruction([folded], [folded]) == "within an arm"
    floors = guard.floors([folded], [folded])
    assert guard.obstruction([folded], [folded], floors) is None
    assert guard.obstruction([opened], [folded], floors) is None
    assert guard.obstruction([deeper], [folded], floors) == "within an arm"


def test_xinput_preserves_stick_signs_and_converts_triggers():
    state = xinput_state(_XInputGamepad(0x2300, 255, 0, -32768, 2000, 32767, -1200))
    assert state == ControllerState(buttons=("lb", "rb", "b"), left_trigger=1023,
                                    left_x=-32768, left_y=2000, right_x=32767, right_y=-1200)
    clicks = xinput_state(_XInputGamepad(0x10C0, 0, 0, 0, 0, 0, 0))
    assert set(clicks.buttons) == {"a", "left_stick", "right_stick"}


@pytest.fixture
def mode(monkeypatch):
    monkeypatch.setitem(sys.modules, "scservo_sdk", make_fake_sdk({}))
    gamepad = importlib.import_module("gamepad")
    maps = [gamepad.replay.Follower(name) for name in ("arm2", "arm1")]
    left = SimulatedArm(maps[0], work_pose(.25, .08, .10))
    right = SimulatedArm(maps[1], work_pose(.25, -.08, .10))
    return gamepad.Gamepad(left, right, .30)


class Driver:
    """Feeds controller states to the mode on a 20 ms clock."""

    def __init__(self, mode):
        self.mode = mode
        self.now = 10.0

    def send(self, state, age=0.0, gap=0.02):
        self.now += gap
        self.mode.tick(state, self.now - age, self.now)

    def enable(self):
        for _ in range(30):
            self.send(ControllerState())
        self.send(ControllerState(buttons=("a",)))
        self.send(ControllerState())

    def tips(self):
        points = []
        for mapping, arm in zip(self.mode.maps, self.mode.arms):
            points.append(self.mode.kinematics.tip(self.mode.pose(mapping, arm.command)[:5])[0])
        return points

    def jaws(self):
        angles = []
        for mapping, arm in zip(self.mode.maps, self.mode.arms):
            angles.append(mapping.to_rad(6, arm.command[6]))
        return angles

    def write_counts(self):
        return [len(arm.writes) for arm in self.mode.arms]


def test_a_enables_each_stick_drives_its_own_arm_and_b_stops(mode):
    driver = Driver(mode)
    pushed = ControllerState(left_y=32767, right_x=32767)
    for _ in range(30):
        driver.send(ControllerState())
    driver.send(pushed)
    assert driver.write_counts() == [0, 0]

    driver.send(ControllerState())
    driver.send(ControllerState(buttons=("a",)))
    driver.send(ControllerState())
    origin = driver.tips()
    driver.send(pushed)
    assert driver.write_counts() == [1, 1]
    end = driver.tips()
    assert end[0][0] > origin[0][0]
    assert end[1][1] < origin[1][1]

    driver.send(replace(pushed, buttons=("b",)))
    driver.send(pushed)
    assert driver.write_counts() == [1, 1]


def test_a_does_not_enable_while_a_stick_is_pushed(mode):
    driver = Driver(mode)
    for _ in range(30):
        driver.send(ControllerState())
    driver.send(ControllerState(buttons=("a",), left_y=32767))
    driver.send(ControllerState(left_y=32767))
    assert driver.write_counts() == [0, 0]


def test_stick_click_switches_only_that_arm_to_height(mode):
    driver = Driver(mode)
    driver.enable()
    driver.send(ControllerState(buttons=("left_stick",)))
    driver.send(ControllerState())
    origin = driver.tips()
    for _ in range(5):
        driver.send(ControllerState(left_y=32767, right_y=32767))
    end = driver.tips()
    assert end[0][2] - origin[0][2] > 0.002
    assert abs(end[0][0] - origin[0][0]) < 0.001
    assert end[1][0] - origin[1][0] > 0.002
    assert abs(end[1][2] - origin[1][2]) < 0.001


def test_bumper_opens_and_trigger_closes_one_jaw_while_held(mode):
    driver = Driver(mode)
    driver.enable()
    start = driver.jaws()
    for _ in range(5):
        driver.send(ControllerState(buttons=("lb",)))
    opened = driver.jaws()
    assert opened[0] > start[0]
    assert opened[1] == start[1]

    driver.send(ControllerState())
    counts = driver.write_counts()
    driver.send(ControllerState())
    assert driver.write_counts() == counts

    for _ in range(3):
        driver.send(ControllerState(right_trigger=900))
    closed = driver.jaws()
    assert closed[1] < start[1]
    assert closed[0] == opened[0]


def test_stale_input_stops_and_needs_a_again(mode):
    driver = Driver(mode)
    driver.enable()
    pushed = ControllerState(left_y=32767)
    driver.send(pushed, age=1.0)
    assert driver.write_counts() == [0, 0]
    assert "timed out" in mode.status
    for _ in range(30):
        driver.send(ControllerState())
    driver.send(pushed)
    assert driver.write_counts() == [0, 0]


def test_clearance_refusal_sends_neither_goal(mode, monkeypatch):
    driver = Driver(mode)
    driver.enable()
    monkeypatch.setattr(mode.guard, "obstruction", lambda *samples: "between arms")
    driver.send(ControllerState(left_y=32767, right_y=32767))
    assert driver.write_counts() == [0, 0]
    assert "between arms" in mode.status


def test_far_apart_parts_are_skipped_without_changing_any_distance():
    guard = PairClearance(0.30)
    poses = [
        (work_pose(.25, .08, .10), work_pose(.25, -.08, .10)),
        (work_pose(.25, -.15, .10), work_pose(.25, .15, .10)),
        (np.radians([0, -104, 90, -96, 0, 0]), work_pose(.24, 0, .01)),
    ]
    for left, right in poses:
        guard.skip_far = True
        quick = guard.distances(left, right)
        guard.skip_far = False
        full = guard.distances(left, right)
        assert np.allclose(quick, full, atol=1e-9)


def test_a_blocked_arm_holds_while_the_other_keeps_moving(mode):
    driver = Driver(mode)
    driver.enable()
    driver.send(ControllerState(buttons=("left_stick",)))
    driver.send(ControllerState())
    for _ in range(100):
        driver.send(ControllerState(left_y=-32767))
    assert "table" in mode.status
    assert driver.tips()[0][2] > 0

    before = driver.tips()
    counts = driver.write_counts()
    for _ in range(5):
        driver.send(ControllerState(left_y=-32767, right_y=32767))
    after = driver.tips()
    assert driver.write_counts()[0] == counts[0]
    assert after[1][0] - before[1][0] > 0.002


def test_an_arm_out_of_reach_holds_while_the_other_keeps_moving(mode):
    driver = Driver(mode)
    driver.enable()
    for _ in range(200):
        driver.send(ControllerState(left_y=32767))
    counts = driver.write_counts()
    before = driver.tips()
    for _ in range(5):
        driver.send(ControllerState(left_y=32767, right_y=32767))
    after = driver.tips()
    assert driver.write_counts()[0] == counts[0]
    assert after[1][0] - before[1][0] > 0.002


def test_a_lagging_arm_waits_and_the_drive_stays_on(mode):
    driver = Driver(mode)
    driver.enable()
    left = mode.arms[0]
    left.stuck = True
    left.present[2] = left.command[2] + 100
    for _ in range(5):
        driver.send(ControllerState(left_y=32767, right_y=32767))
    assert driver.write_counts()[0] == 0
    assert driver.write_counts()[1] == 5
    assert mode.enabled


def test_a_late_step_covers_the_time_that_passed(mode):
    driver = Driver(mode)
    driver.enable()
    pushed = ControllerState(right_y=32767)
    start = driver.tips()[1][0]
    driver.send(pushed)
    usual = driver.tips()[1][0] - start
    driver.send(pushed, gap=0.04)
    late = driver.tips()[1][0] - start - usual
    assert 1.6 < late / usual < 2.4


def test_sideways_on_a_stick_turns_the_base_and_keeps_the_reach(mode):
    driver = Driver(mode)
    driver.enable()
    axis = mode.pan_axis
    before = driver.tips()[1]
    for _ in range(10):
        driver.send(ControllerState(right_x=32767))
    after = driver.tips()[1]
    assert after[1] < before[1] - 0.005
    assert abs(np.linalg.norm(after[:2] - axis) - np.linalg.norm(before[:2] - axis)) < 0.001
    assert abs(after[2] - before[2]) < 0.001
