"""Gamepad routing and clearance checks with fake servo transport."""

from dataclasses import replace
import importlib
import io
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
        self.missed_reads = 0
        self.failed_writes = 0

    def read_present(self):
        if self.missed_reads > 0:
            self.missed_reads -= 1
            return False
        if not self.stuck:
            self.present = {m: round(v) for m, v in self.command.items()}
        return True

    def send_goals(self, goals):
        if self.failed_writes > 0:
            self.failed_writes -= 1
            return False
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
    clicks = xinput_state(_XInputGamepad(0x90C0, 0, 0, 0, 0, 0, 0))
    assert set(clicks.buttons) == {"a", "y", "left_stick", "right_stick"}


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


def finger_directions(mode):
    found = []
    for mapping, arm in zip(mode.maps, mode.arms):
        found.append(mode.kinematics.tip(mode.pose(mapping, arm.command)[:5])[1])
    return found


def test_a_bumper_tilts_that_wrist_up_and_a_trigger_tilts_it_down_while_held(mode):
    driver = Driver(mode)
    driver.enable()
    start = finger_directions(mode)
    wrists = [arm.command[4] for arm in mode.arms]
    wrist_angle = mode.pose(mode.maps[0], mode.arms[0].command)[3]
    for _ in range(5):
        driver.send(ControllerState(buttons=("lb",)))
    up = finger_directions(mode)
    up_turn = mode.pose(mode.maps[0], mode.arms[0].command)[3] - wrist_angle
    assert up[0][2] > start[0][2] + 0.01
    assert mode.arms[0].command[4] != wrists[0]
    assert mode.arms[1].command[4] == wrists[1]
    for m in (1, 2, 3, 5, 6):
        assert mode.arms[0].command[m] == mode.arms[0].present[m]

    driver.send(ControllerState())
    counts = driver.write_counts()
    driver.send(ControllerState())
    assert driver.write_counts() == counts

    before = mode.pose(mode.maps[1], mode.arms[1].command)[3]
    for _ in range(5):
        driver.send(ControllerState(right_trigger=900))
    right_turned = mode.pose(mode.maps[1], mode.arms[1].command)[3] - before
    assert abs(right_turned) > 0.05
    assert (right_turned > 0) != (up_turn > 0)
    assert finger_directions(mode)[0][2] == up[0][2]


def test_a_claw_toggles_open_and_closed_from_the_panel_while_driving(mode):
    driver = Driver(mode)
    right = mode.arms[1]
    assert not mode.toggle_claw(right.name)
    driver.enable()
    assert not mode.toggle_claw("no such arm")
    start = driver.jaws()

    assert mode.toggle_claw(right.name)
    for _ in range(150):
        driver.send(ControllerState())
    opened = driver.jaws()
    assert opened[1] > start[1] + 0.5
    assert opened[0] == start[0]
    counts = driver.write_counts()
    driver.send(ControllerState())
    assert driver.write_counts() == counts

    assert mode.toggle_claw(right.name)
    for _ in range(150):
        driver.send(ControllerState())
    assert driver.jaws()[1] < start[1]


def test_a_faster_arm_may_trail_its_command_further_before_it_waits(mode):
    gamepad = sys.modules["gamepad"]
    arm = mode.arms[0]
    arm.speed = 50
    slow = mode.lag_limit(arm)
    arm.speed = 180
    fast = mode.lag_limit(arm)
    assert slow >= gamepad.MAX_LAG
    assert fast > slow


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
    left.present[2] = left.command[2] + 300
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


def calibrated(mode):
    driver = Driver(mode)
    for _ in range(30):
        driver.send(ControllerState())
    return driver


def test_y_while_paused_glides_both_arms_to_their_idle_poses(mode):
    driver = calibrated(mode)
    driver.send(ControllerState(buttons=("y",)))
    for _ in range(600):
        driver.send(ControllerState())
    for arm, idle in zip(mode.arms, mode.idle):
        for m in idle:
            assert abs(arm.command[m] - idle[m]) < 1
    assert not mode.homing
    counts = driver.write_counts()
    driver.send(ControllerState())
    assert driver.write_counts() == counts


def test_y_while_driving_does_nothing(mode):
    driver = Driver(mode)
    driver.enable()
    driver.send(ControllerState(buttons=("y",)))
    driver.send(ControllerState())
    assert driver.write_counts() == [0, 0]
    assert mode.enabled


def test_b_stops_the_glide_to_idle(mode):
    driver = calibrated(mode)
    driver.send(ControllerState(buttons=("y",)))
    for _ in range(5):
        driver.send(ControllerState())
    assert driver.write_counts()[0] > 0
    driver.send(ControllerState(buttons=("b",)))
    counts = driver.write_counts()
    for _ in range(5):
        driver.send(ControllerState())
    assert driver.write_counts() == counts


def test_a_glide_to_idle_that_is_blocked_stops_and_says_why(mode, monkeypatch):
    driver = calibrated(mode)
    monkeypatch.setattr(mode.guard, "obstruction", lambda *samples: "table")
    driver.send(ControllerState(buttons=("y",)))
    driver.send(ControllerState())
    assert driver.write_counts() == [0, 0]
    assert not mode.homing
    assert "table" in mode.status


def test_a_saved_idle_pose_outside_the_limits_is_held_to_them(mode):
    for arm, idle in zip(mode.arms, mode.idle):
        for m in idle:
            low, high = arm.limits[m]
            assert low <= idle[m] <= high


def test_y_with_a_stick_pushed_does_not_start_the_glide(mode):
    driver = calibrated(mode)
    driver.send(ControllerState(buttons=("y",), left_y=32767))
    driver.send(ControllerState(left_y=32767))
    assert driver.write_counts() == [0, 0]
    assert not mode.homing


def test_a_glide_step_is_not_sent_once_the_controller_input_is_stale(mode, monkeypatch):
    driver = calibrated(mode)
    gamepad = sys.modules["gamepad"]
    clock = [100.0]

    def slow_check(*samples):
        clock[0] += 1.0
        return None

    monkeypatch.setattr(gamepad.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mode.guard, "obstruction", slow_check)
    driver.send(ControllerState(buttons=("y",)))
    assert driver.write_counts() == [0, 0]
    assert not mode.homing


def test_a_dragged_point_brings_that_arm_s_fingertip_to_it_and_leaves_the_other_arm(mode):
    driver = Driver(mode)
    driver.enable()
    right = mode.arms[1]
    point = driver.tips()[1] + np.array([0.03, 0.0, 0.02])
    assert mode.set_tip(right.name, list(point))
    assert mode.tips_state()[right.name] is not None
    for _ in range(100):
        driver.send(ControllerState())
    assert np.linalg.norm(driver.tips()[1] - point) < 0.003
    assert driver.write_counts()[0] == 0
    assert mode.tips_state()[right.name] is None


def test_a_dragged_point_is_refused_while_driving_is_off(mode):
    driver = calibrated(mode)
    right = mode.arms[1]
    point = driver.tips()[1] + np.array([0.03, 0.0, 0.0])
    assert not mode.set_tip(right.name, list(point))
    assert not mode.set_tip("no such arm", list(point))
    for _ in range(5):
        driver.send(ControllerState())
    assert driver.write_counts() == [0, 0]


def test_a_stick_takes_over_from_a_dragged_point_and_b_drops_it(mode):
    driver = Driver(mode)
    driver.enable()
    right = mode.arms[1]
    point = driver.tips()[1] + np.array([0.08, 0.0, 0.0])
    assert mode.set_tip(right.name, list(point))
    driver.send(ControllerState())
    driver.send(ControllerState(right_x=32767))
    assert mode.tips_state()[right.name] is None

    assert mode.set_tip(right.name, list(point))
    driver.send(ControllerState(buttons=("b",)))
    assert mode.tips_state()[right.name] is None


def test_a_point_dragged_past_the_arm_s_reach_snaps_to_the_nearest_place_it_can_go(mode):
    driver = Driver(mode)
    driver.enable()
    right = mode.arms[1]
    start = driver.tips()[1]

    near = start + np.array([0.02, 0.0, 0.01])
    assert mode.set_tip(right.name, list(near))
    assert np.linalg.norm(np.array(mode.tips_state()[right.name]) - near) < 0.001

    far = start + np.array([1.0, 0.0, 0.0])
    assert mode.set_tip(right.name, list(far))
    snapped = np.array(mode.tips_state()[right.name])
    assert snapped[0] > start[0] + 0.02
    assert np.linalg.norm(snapped) < 0.6
    seed = mode.pose(mode.maps[1], right.command)[:5]
    _, error, _ = mode.kinematics.solve(snapped, seed, iters=200, keep_direction=True)
    assert error < 0.002

    under = start + np.array([0.0, 0.0, -0.5])
    assert mode.set_tip(right.name, list(under))
    assert mode.tips_state()[right.name][2] > 0


def test_a_missed_encoder_read_holds_one_step_and_driving_stays_on(mode):
    driver = Driver(mode)
    driver.enable()
    pushed = ControllerState(left_y=32767, right_y=32767)
    mode.arms[0].missed_reads = 1
    driver.send(pushed)
    assert driver.write_counts() == [0, 0]
    assert mode.enabled
    driver.send(pushed)
    assert driver.write_counts() == [1, 1]


def test_a_failed_write_stops_the_drive(mode):
    driver = Driver(mode)
    driver.enable()
    pushed = ControllerState(right_y=32767)
    mode.arms[1].failed_writes = 1
    driver.send(pushed)
    assert driver.write_counts() == [0, 0]
    assert not mode.enabled


def test_a_servo_link_that_stays_down_stops_the_drive(mode):
    driver = Driver(mode)
    driver.enable()
    mode.arms[0].missed_reads = 1000
    for _ in range(40):
        driver.send(ControllerState(left_y=32767))
    assert not mode.enabled
    assert driver.write_counts() == [0, 0]


def test_what_the_loop_does_is_written_down(mode):
    mode.log = io.StringIO()
    driver = Driver(mode)
    driver.enable()
    mode.record(driver.now)
    mode.arms[0].missed_reads = 2
    for _ in range(5):
        driver.send(ControllerState(left_y=32767))
    mode.record(driver.now + 5.0)
    written = mode.log.getvalue()
    assert "Driving" in written
    assert "missed reads 2" in written


def idle_mode(mode):
    """The same mode with both simulated arms at their saved idle poses."""
    for arm, idle in zip(mode.arms, mode.idle):
        arm.present = {m: round(v) for m, v in idle.items()}
        arm.command = dict(arm.present)
        arm.target = dict(arm.present)
    return mode


@pytest.mark.parametrize("pushed", [
    ControllerState(left_y=32767, right_y=32767),
    ControllerState(left_x=32767, right_x=-32767),
    ControllerState(left_x=-32767, right_x=32767),
])
def test_from_the_folded_pose_the_sticks_are_not_refused_by_the_arm_s_own_parts(mode, pushed):
    driver = Driver(idle_mode(mode))
    driver.enable()
    refused = 0
    for _ in range(100):
        driver.send(pushed)
        if "blocked" in mode.status:
            refused += 1
    assert refused == 0
    assert driver.write_counts()[0] > 50
    assert driver.write_counts()[1] > 50


def test_a_forearm_deep_in_the_shoulder_housing_is_still_refused(mode):
    idle = [mode.pose(mapping, ticks) for mapping, ticks in zip(mode.maps, mode.idle)]
    deep = np.radians([0, 20, 91, -96, 0, 0])
    floors = mode.guard.floors([idle[0]], [idle[1]])
    assert mode.guard.obstruction([idle[0]], [idle[1]], floors) is None
    assert mode.guard.obstruction([deep], [idle[1]], floors) == "within an arm"


def test_the_stick_readings_are_reported(mode):
    driver = Driver(mode)
    driver.enable()
    driver.send(ControllerState(left_x=32767, right_y=-32767))
    shown = mode.sticks_state()
    assert shown["left"][0] > 0.9
    assert abs(shown["left"][1]) < 0.01
    assert shown["right"][1] < -0.9
    assert shown["raw"]["left"][0] == 32767
