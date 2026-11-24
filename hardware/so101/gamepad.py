"""An exclusive panel mode that drives both followers from one controller."""

import json
import threading
import time

import numpy as np

from hardware.controller.controller import ControllerError
from hardware.controller.inputs import controller_samples

import arms
import kinematics
import replay
from pair_clearance import PairClearance

PERIOD = 0.02
STALE_AFTER = 0.30
CALIBRATE_FOR = 0.35
CENTER_RANGE = 8500
CENTER_MOTION = 3000
DEAD_ZONE = 5000
TRIGGER_PRESS = 550
TRIGGER_RELEASE = 350
REACH_SPEED = 0.10  # m/s the fingertip moves out, in, up or down with a stick fully over
PAN_SPEED = 45  # deg/s the base turns with a stick fully over
MAX_JOINT_SPEED = 60  # deg/s
JAW_SPEED = 60  # deg/s while a claw button is held
HOME_SPEED = 30  # deg/s of the fastest joint on the glide to the idle poses
MAX_STEP_TIME = 0.06  # s of motion one step may cover when the loop runs late
ARRIVED = 0.002  # m from a dragged point at which the fingertip counts as there
TIP_FLOOR = 0.01  # m above the model's table: the lowest a dragged point may sit
SNAP_ITERS = 60  # solver steps spent finding the nearest reachable place for a dragged point
MAX_LAG = 100  # encoder ticks an arm joint may trail its command before that arm waits; the jaw may be gripping
MAX_FAILS = 15  # servo reads or writes missed in a row before driving is switched off (0.3 s)
LOG_EVERY = 2.0  # s between the counts written to the log
JOINTS = tuple(replay.MOTOR_IDS)


def stick(x, y, center):
    delta = np.array([x, y]) - center
    scale = np.where(delta < 0, 32768 + center, 32767 - center)
    point = delta / scale
    length = np.linalg.norm(point)
    dead_zone = DEAD_ZONE / 32768
    if length <= dead_zone:
        return np.zeros(2)
    return point * min(1.0, (length - dead_zone) / (1.0 - dead_zone)) / length


class Gamepad:
    def __init__(self, left, right, spacing):
        self.arms = (left, right)
        self.maps = tuple(replay.Follower(arm.name) for arm in self.arms)
        self.kinematics = kinematics.Arm()
        self.guard = PairClearance(spacing)
        self.shutdown = threading.Event()
        self.sample = (None, 0.0, "Waiting for the controller")
        self.status = "Waiting for the controller"
        self.center = None
        self.center_candidate = None
        self.enabled = False
        self.a_was_down = False
        self.click_was_down = [False, False]
        self.height = [False, False]
        self.last_tick = None
        self.loop_ms = 0.0
        self.homing = False
        self.y_was_down = False
        self.tips = [None, None]  # per arm: a fingertip point dragged in the panel, base frame, m
        self.fails = 0  # servo reads or writes missed in a row
        self.log = None  # a text file the panel opens: what the loop did, for reading after a run
        self.logged_status = None
        self.logged_at = None
        self.missed_reads = 0
        self.failed_writes = 0
        self.most_lag = 0
        self.idle = []
        idle_poses = []
        for arm, mapping in zip(self.arms, self.maps):
            saved = json.loads((arms.folder(arm.name) / "idle_pose.json").read_text())
            ticks = {}
            for m in JOINTS:
                low, high = arm.limits[m]
                value = float(saved[str(m)])
                if not np.isfinite(value):
                    raise ValueError(f"{arm.name}: the idle pose of motor {m} is not a number")
                ticks[m] = max(low, min(high, value))
            self.idle.append(ticks)
            idle_poses.append(self.pose(mapping, ticks))
        # The real arms rest in their idle poses, which the model draws partly inside themselves.
        self.guard.rest(idle_poses[0], idle_poses[1])
        self.idle_floors = self.guard.floors([idle_poses[0]], [idle_poses[1]])
        self.shown_sticks = {"left": [0.0, 0.0], "right": [0.0, 0.0], "raw": {"left": [0, 0], "right": [0, 0]}}
        # The base turns about a vertical line through the first joint.
        self.pan_axis = self.kinematics.links[0][0][:2].copy()
        if self.facing(0.01)[1] > self.facing(0.0)[1]:
            self.pan_left = 1.0
        else:
            self.pan_left = -1.0
        self.jaw_open = []
        self.jaw_closed = []
        for arm, mapping in zip(self.arms, self.maps):
            low, high = arm.limits[replay.GRIPPER_ID]
            if mapping.to_rad(replay.GRIPPER_ID, high) > mapping.to_rad(replay.GRIPPER_ID, low):
                self.jaw_open.append(high)
                self.jaw_closed.append(low)
            else:
                self.jaw_open.append(low)
                self.jaw_closed.append(high)

    def input_loop(self):
        while not self.shutdown.is_set():
            try:
                for state in controller_samples(self.shutdown, PERIOD):
                    self.sample = (state, time.monotonic(), "Waiting for the controller")
            except (ControllerError, OSError) as exc:
                self.sample = (None, 0.0, str(exc))
            self.shutdown.wait(0.25)

    def stop_locked(self, status):
        self.status = status
        self.enabled = False
        self.homing = False
        self.height = [False, False]
        self.tips = [None, None]
        for arm in self.arms:
            arm.target = dict(arm.command)

    def stop(self):
        left, right = self.arms
        with left.lock, right.lock:
            self.stop_locked("Stopped. Press A to drive again")

    def reachable_point(self, index, point):
        """The place nearest `point` (the arm's base frame, m) that the fingertip can be brought to from where the
        arm is, with its joints inside their recorded limits and the point above the table. None when the solver
        gives no answer."""
        arm = self.arms[index]
        mapping = self.maps[index]
        with arm.lock:
            command = dict(arm.command)
        wanted = np.array(point, dtype=float)
        if wanted[2] < TIP_FLOOR:
            wanted[2] = TIP_FLOOR
        seed = self.pose(mapping, command)[:5]
        solved, error, _ = self.kinematics.solve(wanted, seed, iters=SNAP_ITERS, keep_direction=True)
        if not np.isfinite(solved).all() or not np.isfinite(error):
            return None
        for i in range(5):
            m = JOINTS[i]
            first = mapping.to_rad(m, arm.limits[m][0])
            second = mapping.to_rad(m, arm.limits[m][1])
            # A joint resting past a recorded limit may stay where it is.
            low = min(first, second, seed[i])
            high = max(first, second, seed[i])
            solved[i] = max(low, min(high, solved[i]))
        nearest = self.kinematics.tip(solved)[0]
        if np.linalg.norm(nearest - wanted) < ARRIVED:
            return wanted
        return nearest

    def set_tip(self, name, point):
        """Take a fingertip point dragged in the panel for the arm called `name` (its base frame, metres), moved
        to the nearest place the arm can reach. False when driving is off or the arm is not one of the two."""
        index = None
        for i in range(2):
            if self.arms[i].name == name:
                index = i
        if index is None or not self.enabled:
            return False
        # Worked out before the locks are taken, so the control loop is not held up.
        nearest = self.reachable_point(index, point)
        if nearest is None:
            return False
        left, right = self.arms
        with left.lock, right.lock:
            if not self.enabled:
                return False
            self.tips[index] = nearest
            return True

    def sticks_state(self):
        """The stick readings for the panel: what the loop acts on (-1 to 1, after the centre and the dead zone)
        and the raw values from the controller."""
        return self.shown_sticks

    def tips_state(self):
        """{arm name: the dragged point it is going to, or None}, for the panel's marker."""
        shown = {}
        for i in range(2):
            if self.tips[i] is None:
                shown[self.arms[i].name] = None
            else:
                shown[self.arms[i].name] = [float(v) for v in self.tips[i]]
        return shown

    def calibrate(self, axes, released, now):
        if not released or np.max(np.abs(axes)) > CENTER_RANGE:
            self.center_candidate = None
            return
        if self.center_candidate is None or np.max(np.abs(axes - self.center_candidate[0])) > CENTER_MOTION:
            self.center_candidate = (axes, now)
        if now - self.center_candidate[1] >= CALIBRATE_FOR:
            self.center = self.center_candidate[0]

    def pose(self, mapping, values):
        return np.array([mapping.to_rad(m, values[m]) for m in JOINTS])

    def facing(self, pan):
        """The level direction the arm reaches in with its base turned to `pan`: from the turning axis to the
        fingertip of a pose stretched out in front. The folded arm's own fingertip can sit behind the axis."""
        stretched = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])
        stretched[0] = pan
        outward = self.kinematics.tip(stretched)[0][:2] - self.pan_axis
        return outward / np.linalg.norm(outward)

    def stick_step(self, arm, mapping, xy, height, dt):
        """(fingertip step (3,) m, base turn rad) a stick asks for in one step. Stick x turns the base. Stick y
        moves the fingertip out and in the way the arm faces, or up and down in height mode."""
        step = np.zeros(3)
        distance = xy[1] * REACH_SPEED * dt
        if height:
            step[2] = distance
        else:
            pan = mapping.to_rad(JOINTS[0], arm.command[JOINTS[0]])
            step[:2] = self.facing(pan) * distance
        # On the gamepad positive x is right.
        turn = -self.pan_left * xy[0] * np.radians(PAN_SPEED) * dt
        return step, turn

    def tip_step(self, arm, mapping, point, dt):
        """(fingertip step (3,) m toward the dragged `point`, whether the fingertip is there)."""
        angles = self.pose(mapping, arm.command)
        tip = self.kinematics.tip(angles[:5])[0]
        apart = point - tip
        distance = np.linalg.norm(apart)
        if distance < ARRIVED:
            return np.zeros(3), True
        return apart * min(1.0, REACH_SPEED * dt / distance), False

    def proposed(self, arm, mapping, step, turn, jaw_goal, dt):
        """(the next commanded ticks for one arm, whether the fingertip step could be solved). The fingertip moves
        by `step` with the fingers keeping their lean, and the base turns by `turn`. `jaw_goal` is the jaw limit a
        held claw button asks for, or None to leave the jaw."""
        current = np.array([arm.command[m] for m in JOINTS], dtype=float)
        goal = current.copy()
        reachable = True
        if np.any(step) or turn != 0:
            angles = self.pose(mapping, arm.command)
            wanted = angles[:5].copy()
            if np.any(step):
                tip = self.kinematics.tip(wanted)[0]
                solved, error, _ = self.kinematics.solve(tip + step, wanted, iters=8, keep_direction=True)
                if np.isfinite(solved).all() and np.isfinite(error) and error <= 0.0002:
                    wanted = solved
                else:
                    reachable = False
            wanted[0] = wanted[0] + turn
            for i in range(5):
                goal[i] = mapping.to_ticks(JOINTS[i], wanted[i])
        if jaw_goal is not None:
            goal[-1] = jaw_goal

        limits = np.array([arm.limits[m] for m in JOINTS])
        # The saved idle pose can start slightly past a soft limit. Permit movement back into range.
        goal = np.clip(goal, np.minimum(limits[:, 0], current), np.maximum(limits[:, 1], current))
        change = goal - current
        # Every arm joint is slowed by the same share, so the fingertip keeps its path.
        arm_step = min(arm.speed, MAX_JOINT_SPEED) * replay.STEPS_PER_DEGREE * dt
        largest = np.max(np.abs(change[:5]))
        if largest > arm_step:
            change[:5] = change[:5] * arm_step / largest
        jaw_step = min(arm.speed, JAW_SPEED) * replay.STEPS_PER_DEGREE * dt
        change[-1] = np.clip(change[-1], -jaw_step, jaw_step)
        return dict(zip(JOINTS, current + change)), reachable

    def clear(self, move, held, ahead, floors):
        """None when the arms chosen in `move` (left, right) may take their step with the other held, else what
        is in the way. The held poses against each other set the floors, so only pairings with a pose ahead are
        tested."""
        right_poses = held[1]
        if move[1]:
            right_poses = np.concatenate([held[1], ahead[1]])
        if move[0]:
            obstruction = self.guard.obstruction(ahead[0], right_poses, floors)
            if obstruction is not None:
                return obstruction
        if move[1]:
            return self.guard.obstruction(held[0], ahead[1], floors)
        return None

    def choose(self, goals, to_idle):
        """((left moves, right moves), what blocked the first choice) for the proposed ticks `goals`. Both arms
        together first; if that is refused, whichever arm can move alone still moves. (None, what) when neither
        may move. `to_idle` lets each pair come as close as it is at the idle poses."""
        moving = [goals[0] != self.arms[0].command, goals[1] != self.arms[1].command]
        held = []
        ahead = []
        for arm, mapping, goal in zip(self.arms, self.maps, goals):
            measured = self.pose(mapping, arm.present)
            commanded = self.pose(mapping, {m: round(v) for m, v in arm.command.items()})
            proposed = self.pose(mapping, {m: round(v) for m, v in goal.items()})
            held.append(np.unique([measured, commanded], axis=0))
            ahead.append(np.unique([proposed, (measured + proposed) / 2], axis=0))
        floors = self.guard.floors(held[0], held[1])
        if to_idle and floors is not None and self.idle_floors is not None:
            # Only an arm against itself: the table and the other arm keep their full clearance.
            for i in range(len(floors)):
                if self.guard.pairs[i][3] == "within an arm":
                    floors[i] = min(floors[i], self.idle_floors[i])
        options = []
        if moving[0] and moving[1]:
            options.append((True, True))
        if moving[0]:
            options.append((True, False))
        if moving[1]:
            options.append((False, True))
        blocked = None
        for option in options:
            obstruction = self.clear(option, held, ahead, floors)
            if obstruction is None:
                return option, blocked
            if blocked is None:
                blocked = obstruction
        return None, blocked

    def link_failed(self, what):
        """Count one missed read or write. The step is skipped and the next one tries again; driving is switched
        off only when the link stays down."""
        self.fails += 1
        if self.fails >= MAX_FAILS:
            self.stop_locked(f"The servo link dropped out ({what}). Check the cables, then press A")
        else:
            self.status = f"Waiting for the servos ({what})"

    def send(self, goals, chosen):
        """Send the chosen arms their goals. False (and stopped) when a write fails."""
        for i in range(2):
            arm = self.arms[i]
            goal = goals[i]
            if chosen[i] and goal != arm.command:
                # A failed write can leave this physical goal uncertain. Preserve the prior software target.
                if not arm.send_goals(goal):
                    self.failed_writes += 1
                    self.stop_locked(f"Could not command {arm.name}. Press A to drive again")
                    return False
                arm.command = dict(goal)
                arm.target = dict(goal)
        self.fails = 0
        return True

    def record(self, now):
        """Write a line to the log when the status changes, and the counts every LOG_EVERY seconds."""
        if self.log is None:
            return
        status = self.status.split(" Loop ")[0]
        if status != self.logged_status:
            self.log.write(f"{now:.2f} {status}\n")
            self.logged_status = status
        if self.logged_at is None:
            self.logged_at = now
        if now - self.logged_at >= LOG_EVERY:
            self.log.write(f"{now:.2f} loop {self.loop_ms:.0f} ms, missed reads {self.missed_reads}, "
                           f"failed writes {self.failed_writes}, most lag {self.most_lag} steps\n")
            self.logged_at = now
            self.missed_reads = 0
            self.failed_writes = 0
            self.most_lag = 0

    def lagging(self, arm):
        behind = False
        for m in JOINTS[:-1]:
            lag = abs(arm.present[m] - arm.command[m])
            if lag > self.most_lag:
                self.most_lag = round(lag)
            if lag > MAX_LAG:
                behind = True
        return behind

    def too_late(self, now, captured, started):
        """Whether the controller sample has gone stale while this step was worked out."""
        return now - captured + time.monotonic() - started > STALE_AFTER

    def home_step(self, dt, now, captured, started):
        """One step of both arms straight toward their idle poses, every joint of an arm arriving together."""
        goals = []
        for i in range(2):
            arm = self.arms[i]
            if self.lagging(arm):
                goals.append(dict(arm.command))
                continue
            current = np.array([arm.command[m] for m in JOINTS], dtype=float)
            wanted = np.array([self.idle[i][m] for m in JOINTS], dtype=float)
            change = wanted - current
            step = min(arm.speed, HOME_SPEED) * replay.STEPS_PER_DEGREE * dt
            largest = np.max(np.abs(change))
            if largest > step:
                goals.append(dict(zip(JOINTS, current + change * step / largest)))
            else:
                goals.append(dict(zip(JOINTS, wanted)))
        at_idle = True
        for i in range(2):
            for m in JOINTS:
                if abs(self.arms[i].command[m] - self.idle[i][m]) > 0.5:
                    at_idle = False
        if at_idle:
            self.stop_locked("Both arms are at their idle poses. Press A to drive")
            return
        if goals[0] == self.arms[0].command and goals[1] == self.arms[1].command:
            self.status = "Going to the idle poses: an arm is catching up. B stops"
            return
        chosen, blocked = self.choose(goals, True)
        if chosen is None:
            self.stop_locked(f"Stopped on the way to idle: clearance {blocked}. Move the arm clear, then press Y")
            return
        if self.too_late(now, captured, started):
            self.center = self.center_candidate = None
            self.stop_locked("Controller input timed out")
            return
        if self.shutdown.is_set():
            return
        if self.send(goals, chosen):
            self.status = "Going to the idle poses. B stops"

    def run(self):
        threading.Thread(target=self.input_loop, daemon=True).start()
        while not self.shutdown.is_set():
            started = time.monotonic()
            state, captured, message = self.sample
            try:
                self.tick(state, captured, started, message)
                self.loop_ms = 0.9 * self.loop_ms + 0.1 * 1000 * (time.monotonic() - started)
                self.record(started)
            except (OSError, ValueError, np.linalg.LinAlgError) as exc:
                self.shutdown.set()
                self.stop()
                self.status = f"Gamepad control stopped: {exc}"
                return
            self.shutdown.wait(max(0, PERIOD - (time.monotonic() - started)))

    def tick(self, state, captured, now, message="Waiting for the controller"):
        started = time.monotonic()
        dt = PERIOD
        if self.last_tick is not None:
            dt = min(max(now - self.last_tick, 0.0), MAX_STEP_TIME)
        self.last_tick = now
        left, right = self.arms
        with left.lock, right.lock:
            # Refresh both encoder snapshots before solving. Torque requests still use the panel's arm API.
            reads = [arm.read_present() for arm in self.arms]
            torque_changed = False
            for arm, fresh in zip(self.arms, reads):
                if arm.pending_torque is not None and (fresh or not arm.pending_torque):
                    arm.apply_torque(arm.pending_torque)
                    arm.pending_torque = None
                    torque_changed = True
                if fresh and not arm.torque:
                    arm.command = dict(arm.present)
                    arm.target = dict(arm.present)
            if self.shutdown.is_set():
                self.stop_locked("Stopped")
                return
            if state is None or now - captured > STALE_AFTER:
                self.center = self.center_candidate = None
                self.stop_locked(message if state is None else "Controller input timed out")
                return
            buttons = state.buttons
            triggers = (state.left_trigger, state.right_trigger)
            bumpers = ("lb" in buttons, "rb" in buttons)
            clicks = ("left_stick" in buttons, "right_stick" in buttons)
            a_down = "a" in buttons
            a_pressed = a_down and not self.a_was_down
            self.a_was_down = a_down
            y_down = "y" in buttons
            y_pressed = y_down and not self.y_was_down
            self.y_was_down = y_down
            clicked = [False, False]
            for i in range(2):
                clicked[i] = clicks[i] and not self.click_was_down[i]
                self.click_was_down[i] = clicks[i]
            axes = np.array([state.left_x, state.left_y, state.right_x, state.right_y])
            used = [np.zeros(2), np.zeros(2)]
            if self.center is not None:
                used = [stick(*axes[:2], self.center[:2]), stick(*axes[2:], self.center[2:])]
            self.shown_sticks = {
                "left": [round(float(used[0][0]), 2), round(float(used[0][1]), 2)],
                "right": [round(float(used[1][0]), 2), round(float(used[1][1]), 2)],
                "raw": {"left": [int(axes[0]), int(axes[1])], "right": [int(axes[2]), int(axes[3])]},
            }
            claws_released = not any(bumpers) and max(triggers) < TRIGGER_RELEASE
            if self.center is None:
                self.calibrate(axes, claws_released and not a_down, now)
                if self.center is None:
                    self.stop_locked("Center both sticks and release every button")
                else:
                    self.stop_locked("Press A to drive both arms")
                return
            if "b" in buttons:
                self.stop_locked("Stopped. Press A to drive again")
                return
            if not all(arm.torque for arm in self.arms) or torque_changed:
                self.stop_locked("Turn on torque for both arms, then press A")
                return
            if not all(reads):
                # A missed read is routine on the serial link: skip this step and keep driving.
                self.missed_reads += 1
                self.link_failed("an encoder read was missed")
                return
            sticks = (stick(*axes[:2], self.center[:2]), stick(*axes[2:], self.center[2:]))
            sticks_centered = not np.any(sticks[0]) and not np.any(sticks[1])
            if a_pressed:
                self.homing = False
                if self.enabled:
                    self.stop_locked("Holding. Press A to drive again")
                    return
                if not sticks_centered or not claws_released:
                    self.stop_locked("Center both sticks and release the claw buttons, then press A")
                    return
                self.enabled = True
            if not self.enabled:
                if y_pressed:
                    if sticks_centered and claws_released:
                        self.homing = True
                    else:
                        self.stop_locked("Center both sticks and release the claw buttons, then press Y")
                        return
                if self.homing:
                    self.home_step(dt, now, captured, started)
                    return
                self.stop_locked(self.status)
                return
            jaw_goals = [None, None]
            for i in range(2):
                if clicked[i]:
                    self.height[i] = not self.height[i]
                opening = bumpers[i]
                closing = triggers[i] >= TRIGGER_PRESS
                if opening and not closing:
                    jaw_goals[i] = self.jaw_open[i]
                if closing and not opening:
                    jaw_goals[i] = self.jaw_closed[i]
            modes = []
            for i in range(2):
                if self.height[i]:
                    modes.append("up and down")
                else:
                    modes.append("over the table")
            driving = f"Driving. Left stick: {modes[0]}. Right stick: {modes[1]}."
            names = ("Left", "Right")
            notes = []
            goals = []
            for i in range(2):
                arm = self.arms[i]
                if self.lagging(arm):
                    goals.append(dict(arm.command))
                    notes.append(f"{names[i]} arm is catching up.")
                    continue
                step = np.zeros(3)
                turn = 0.0
                if np.any(sticks[i]):
                    # A stick takes over from a point dragged in the panel.
                    self.tips[i] = None
                    step, turn = self.stick_step(arm, self.maps[i], sticks[i], self.height[i], dt)
                elif self.tips[i] is not None:
                    step, arrived = self.tip_step(arm, self.maps[i], self.tips[i], dt)
                    if arrived:
                        self.tips[i] = None
                    else:
                        notes.append(f"{names[i]} arm: going to the dragged point.")
                goal, reachable = self.proposed(arm, self.maps[i], step, turn, jaw_goals[i], dt)
                if not reachable:
                    notes.append(f"{names[i]} arm: out of reach.")
                goals.append(goal)
            moving = [goals[0] != self.arms[0].command, goals[1] != self.arms[1].command]
            if not any(moving):
                self.status = " ".join([driving] + notes)
                return
            chosen, blocked = self.choose(goals, False)
            if chosen is None:
                self.status = f"Movement blocked: clearance {blocked}"
                return
            for i in range(2):
                if moving[i] and not chosen[i]:
                    notes.append(f"{names[i]} arm blocked: clearance {blocked}.")
            if self.too_late(now, captured, started):
                self.center = self.center_candidate = None
                self.stop_locked("Controller input timed out")
                return
            if self.shutdown.is_set():
                return
            if not self.send(goals, chosen):
                return
            self.status = " ".join([driving] + notes + [f"Loop {self.loop_ms:.0f} ms"])
