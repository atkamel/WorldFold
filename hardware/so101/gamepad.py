"""An exclusive panel mode that drives both followers from one controller."""

import threading
import time

import numpy as np

from hardware.controller.controller import ControllerError
from hardware.controller.inputs import controller_samples

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
MAX_STEP_TIME = 0.06  # s of motion one step may cover when the loop runs late
MAX_LAG = 50  # encoder ticks; applies to arm joints, since a closed jaw may be gripping something
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
        self.height = [False, False]
        for arm in self.arms:
            arm.target = dict(arm.command)

    def stop(self):
        left, right = self.arms
        with left.lock, right.lock:
            self.stop_locked("Stopped. Press A to drive again")

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

    def proposed(self, arm, mapping, xy, height, jaw_goal, dt):
        """(the next commanded ticks for one arm, whether the reach step could be solved). Stick x turns the base.
        Stick y moves the fingertip out and in along the arm, or up and down in height mode, with the fingers
        keeping their lean. `jaw_goal` is the jaw limit a held claw button asks for, or None to leave the jaw."""
        current = np.array([arm.command[m] for m in JOINTS], dtype=float)
        goal = current.copy()
        reachable = True
        if np.any(xy):
            angles = self.pose(mapping, arm.command)
            wanted = angles[:5].copy()
            if xy[1] != 0:
                tip = self.kinematics.tip(wanted)[0]
                step = np.zeros(3)
                distance = xy[1] * REACH_SPEED * dt
                if height:
                    step[2] = distance
                else:
                    step[:2] = self.facing(wanted[0]) * distance
                solved, error, _ = self.kinematics.solve(tip + step, wanted, iters=8, keep_direction=True)
                if np.isfinite(solved).all() and np.isfinite(error) and error <= 0.0002:
                    wanted = solved
                else:
                    reachable = False
            if xy[0] != 0:
                # On the gamepad positive x is right.
                wanted[0] = wanted[0] - self.pan_left * xy[0] * np.radians(PAN_SPEED) * dt
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

    def run(self):
        threading.Thread(target=self.input_loop, daemon=True).start()
        while not self.shutdown.is_set():
            started = time.monotonic()
            state, captured, message = self.sample
            try:
                self.tick(state, captured, started, message)
                self.loop_ms = 0.9 * self.loop_ms + 0.1 * 1000 * (time.monotonic() - started)
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
            clicked = [False, False]
            for i in range(2):
                clicked[i] = clicks[i] and not self.click_was_down[i]
                self.click_was_down[i] = clicks[i]
            axes = np.array([state.left_x, state.left_y, state.right_x, state.right_y])
            claws_released = not any(bumpers) and max(triggers) < TRIGGER_RELEASE
            if self.center is None:
                self.calibrate(axes, claws_released and not a_down, now)
                if self.center is None:
                    self.stop_locked("Center both sticks and release every button")
                else:
                    self.stop_locked("Press A to drive both arms")
                return
            if not all(reads) or not all(arm.torque for arm in self.arms) or torque_changed:
                self.stop_locked("Turn on torque for both arms and check the servo links")
                return
            if "b" in buttons:
                self.stop_locked("Stopped. Press A to drive again")
                return
            sticks = (stick(*axes[:2], self.center[:2]), stick(*axes[2:], self.center[2:]))
            sticks_centered = not np.any(sticks[0]) and not np.any(sticks[1])
            if a_pressed:
                if self.enabled:
                    self.stop_locked("Holding. Press A to drive again")
                    return
                if not sticks_centered or not claws_released:
                    self.stop_locked("Center both sticks and release the claw buttons, then press A")
                    return
                self.enabled = True
            if not self.enabled:
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
                lagging = False
                for m in JOINTS[:-1]:
                    if abs(arm.present[m] - arm.command[m]) > MAX_LAG:
                        lagging = True
                if lagging:
                    goals.append(dict(arm.command))
                    notes.append(f"{names[i]} arm is catching up.")
                    continue
                goal, reachable = self.proposed(arm, self.maps[i], sticks[i], self.height[i], jaw_goals[i], dt)
                if not reachable:
                    notes.append(f"{names[i]} arm: out of reach.")
                goals.append(goal)
            moving = [goals[0] != self.arms[0].command, goals[1] != self.arms[1].command]
            if not any(moving):
                self.status = " ".join([driving] + notes)
                return
            held = []
            ahead = []
            for arm, mapping, goal in zip(self.arms, self.maps, goals):
                measured = self.pose(mapping, arm.present)
                commanded = self.pose(mapping, {m: round(v) for m, v in arm.command.items()})
                proposed = self.pose(mapping, {m: round(v) for m, v in goal.items()})
                held.append(np.unique([measured, commanded], axis=0))
                ahead.append(np.unique([proposed, (measured + proposed) / 2], axis=0))
            floors = self.guard.floors(held[0], held[1])
            # Both arms together first. If that is refused, whichever arm can move alone still moves.
            options = []
            if moving[0] and moving[1]:
                options.append((True, True))
            if moving[0]:
                options.append((True, False))
            if moving[1]:
                options.append((False, True))
            chosen = None
            blocked = None
            for option in options:
                obstruction = self.clear(option, held, ahead, floors)
                if obstruction is None:
                    chosen = option
                    break
                if blocked is None:
                    blocked = obstruction
            if chosen is None:
                self.status = f"Movement blocked: clearance {blocked}"
                return
            for i in range(2):
                if moving[i] and not chosen[i]:
                    notes.append(f"{names[i]} arm blocked: clearance {blocked}.")
            if now - captured + time.monotonic() - started > STALE_AFTER:
                self.center = self.center_candidate = None
                self.stop_locked("Controller input timed out")
                return
            if self.shutdown.is_set():
                return
            for i in range(2):
                arm = self.arms[i]
                goal = goals[i]
                if chosen[i] and goal != arm.command:
                    # A failed write can leave this physical goal uncertain. Preserve the prior software target.
                    if not arm.send_goals(goal):
                        self.stop_locked(f"Could not command {arm.name}. Press A to drive again")
                        return
                    arm.command = dict(goal)
                    arm.target = dict(goal)
            self.status = " ".join([driving] + notes + [f"Loop {self.loop_ms:.0f} ms"])
