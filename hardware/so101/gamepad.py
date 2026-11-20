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
TIP_SPEED = 0.035  # m/s
MAX_JOINT_SPEED = 25  # deg/s
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


def cartesian_step(x, y, height_mode):
    # Model axes are forward, left, up. On the gamepad positive x is right.
    return TIP_SPEED * PERIOD * np.array([0 if height_mode else y, -x, y if height_mode else 0])


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
        self.ready = False
        self.active = False
        self.pressed = [False, False]
        self.jaw = [None, None]

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
        self.active = self.ready = False
        self.jaw = [None, None]
        for arm in self.arms:
            arm.target = dict(arm.command)

    def stop(self):
        left, right = self.arms
        with left.lock, right.lock:
            self.stop_locked("Stopped. Release RB before enabling motion again")

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

    def proposed(self, arm, mapping, index, xy, height):
        current = np.array([arm.command[m] for m in JOINTS])
        goal = current.copy()
        if np.any(xy):
            angles = self.pose(mapping, arm.command)
            tip = self.kinematics.tip(angles[:5])[0]
            target = tip + cartesian_step(*xy, height)
            wanted, error, _ = self.kinematics.solve(target, angles[:5], iters=8, keep_direction=True)
            if not np.isfinite(wanted).all() or not np.isfinite(error) or error > 0.0002:
                return None
            goal[:5] = [mapping.to_ticks(m, a) for m, a in zip(JOINTS, wanted)]
        if self.jaw[index] is not None:
            goal[-1] = self.jaw[index]

        limits = np.array([arm.limits[m] for m in JOINTS])
        # The saved idle pose can start slightly past a soft limit. Permit movement back into range.
        goal = np.clip(goal, np.minimum(limits[:, 0], current), np.maximum(limits[:, 1], current))
        maximum = min(arm.speed, MAX_JOINT_SPEED) * replay.STEPS_PER_DEGREE * PERIOD
        return dict(zip(JOINTS, current + np.clip(goal - current, -maximum, maximum)))

    def run(self):
        threading.Thread(target=self.input_loop, daemon=True).start()
        while not self.shutdown.is_set():
            started = time.monotonic()
            state, captured, message = self.sample
            try:
                self.tick(state, captured, started, message)
            except (OSError, ValueError, np.linalg.LinAlgError) as exc:
                self.shutdown.set()
                self.stop()
                self.status = f"Gamepad control stopped: {exc}"
                return
            self.shutdown.wait(max(0, PERIOD - (time.monotonic() - started)))

    def tick(self, state, captured, now, message="Waiting for the controller"):
        started = time.monotonic()
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
            rb = "rb" in state.buttons
            triggers = (state.left_trigger, state.right_trigger)
            axes = np.array([state.left_x, state.left_y, state.right_x, state.right_y])
            released = not rb and max(triggers) < TRIGGER_RELEASE
            if self.center is None:
                self.calibrate(axes, released, now)
                self.stop_locked("Center both sticks and release RB and the triggers")
                return
            if not all(reads) or not all(arm.torque for arm in self.arms) or torque_changed:
                self.stop_locked("Turn on torque for both arms and check the servo links")
                return
            if "b" in state.buttons:
                self.stop_locked("Stopped. Release RB before enabling motion again")
                return
            sticks = (stick(*axes[:2], self.center[:2]), stick(*axes[2:], self.center[2:]))
            if not rb:
                self.stop_locked("Hold RB to drive both arms")
                self.ready = released and not any(np.any(xy) for xy in sticks)
                self.pressed = [False, False]
                return
            if not self.ready:
                self.status = "Center both sticks and release RB before enabling motion"
                return
            if not self.active and (max(triggers) >= TRIGGER_RELEASE or any(np.any(xy) for xy in sticks)):
                self.stop_locked("Enable with the sticks centered and triggers released")
                return
            self.active = True
            for i, (arm, value) in enumerate(zip(self.arms, triggers)):
                if value <= TRIGGER_RELEASE:
                    self.pressed[i] = False
                elif value >= TRIGGER_PRESS and not self.pressed[i]:
                    self.pressed[i] = True
                    lo, hi = arm.limits[replay.GRIPPER_ID]
                    previous = self.jaw[i] if self.jaw[i] is not None else arm.command[replay.GRIPPER_ID]
                    self.jaw[i] = hi if previous < (lo + hi) / 2 else lo
            if any(abs(arm.present[m] - arm.command[m]) > MAX_LAG for arm in self.arms for m in JOINTS[:-1]):
                self.stop_locked("A follower is lagging. Release RB before retrying")
                return

            goals = [self.proposed(arm, mapping, i, xy, "lb" in state.buttons)
                     for i, (arm, mapping, xy) in enumerate(zip(self.arms, self.maps, sticks))]
            if any(goal is None for goal in goals):
                self.status = "Gripper target out of reach"
                return
            if all(goal == arm.command for goal, arm in zip(goals, self.arms)):
                self.status = "Driving both arms"
                return
            samples = []
            for arm, mapping, goal in zip(self.arms, self.maps, goals):
                measured = self.pose(mapping, arm.present)
                commanded = self.pose(mapping, {m: round(v) for m, v in arm.command.items()})
                proposed = self.pose(mapping, {m: round(v) for m, v in goal.items()})
                samples.append(np.unique([measured, commanded, proposed, (measured + proposed) / 2], axis=0))
            obstruction = self.guard.obstruction(*samples)
            if obstruction is not None:
                self.status = f"Movement blocked: clearance {obstruction}"
                return
            if now - captured + time.monotonic() - started > STALE_AFTER:
                self.center = self.center_candidate = None
                self.stop_locked("Controller input timed out")
                return
            if self.shutdown.is_set():
                return
            for arm, goal in zip(self.arms, goals):
                if goal != arm.command:
                    # A failed write can leave this physical goal uncertain. Preserve the prior software target.
                    if not arm.send_goals(goal):
                        self.stop_locked(f"Could not command {arm.name}. Release RB before retrying")
                        return
                    arm.command = dict(goal)
                    arm.target = dict(goal)
            self.status = "Driving both arms"
