"""The intervention button of dagger_run.py: press the Xbox button on the controller and the fold waits where it
is, with the arms handed to the gamepad teleop (gamepad.py) exactly as the panel's gamepad mode drives them. Press
the Xbox button again and the fold goes on with each fingertip as far off the plan as you left it. That offset is
kept (nudges.json in the store), so the next fold starts with it, and the corrected rows are what the network is
taught.

While the fold waits the controls are gamepad.py's own, whatever its layout is, and so are its checks (both arms
against each other, the table and themselves, a lagging arm, a lost controller). This file only lends
gamepad.Gamepad the two arms the fold holds, reads the Xbox button, and presses gamepad.py's Drive for you: the
panel's Drive button belongs to the panel's own gamepad, which has lent the arms to the fold. Driving starts once
the sticks have rested a moment with every button released, as gamepad.py asks.

    python hardware/so101/dagger_nudge.py      # no arms: prints the Xbox button as it is read, to check it first

On Windows the Xbox button is read through XInput's unlisted call (number 100 of xinput1_4.dll); the listed one,
which hardware/controller reads, leaves that button out. Windows also opens the Game Bar on that button unless
"Open Xbox Game Bar using this button on a controller" is switched off in Settings, Gaming. A controller that
stops answering for LOST seconds while the fold waits lets the fold go on.

What is kept of an intervention is where each fingertip ended against the plan's row the fold waited on: an offset
(x forward, y left, z up) m in the arm's own base frame, like fold_camera.py's --overshoot and --pinch-height, no
axis past MAX_NUDGE. A fingertip left further than that comes back to MAX_NUDGE when the fold goes on. Each arm has
two offsets: "pinch", from the first row until the corner is 2 cm into the lift (the plan's "check" mark), and
"carry", from there to the end. An intervention before the lift moves both, so a corrected pinch carries to the
place; one after it moves the carry alone.

A corrected row is the plan's row with the fingertip moved by the offset (the planner's own solver, started from
the plan's row, the jaw as planned). A row the solver cannot reach, that leaves a joint's limits by more than
replay.py allows, or that comes nearer the arm itself than replay.py allows, stays the plan's row; an offset that
fails on the row the fold waited on is not kept. The last row is always the plan's.
"""

import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import arms
import replay
from kinematics import Arm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # hardware.controller, for gamepad.py

MAX_NUDGE = 0.03        # m either way, each axis: fold_camera.MAX_OVERSHOOT
TOUCHED = 0.0005        # m: a fingertip left nearer than this to where it was has not been moved
LOST = 2.0              # s without the controller after which a waiting fold goes on
REACH = 1e-3            # m the solver may miss a corrected fingertip by
SOLVE_STEPS = 40
PHASES = ("pinch", "carry")
MOTOR_IDS = replay.MOTOR_IDS


class Lent:
    """What gamepad.Gamepad asks of an arm (ui.Arm), on a replay.Follower the fold already holds."""

    def __init__(self, follower):
        self.follower, self.name, self.limits = follower, follower.name, follower.limits
        self.lock, self.lent, self.torque, self.pending_torque = threading.Lock(), False, True, None
        self.speed = replay.MAX_DEG_S       # deg/s: the fold's own rate cap
        self.present = self.command = self.target = None

    def read_present(self):
        try:
            self.present = self.follower.read_present()
        except OSError:
            return False
        return True

    def send_goals(self, goals):
        try:
            self.follower.send_goals(goals)
        except OSError:
            return False
        return True

    def apply_torque(self, on):
        pass


class Teleop:
    """gamepad.Gamepad driving the fold's two arms while the fold waits."""

    def __init__(self, followers):
        import gamepad
        if set(followers) != {"left", "right"}:
            raise ValueError("it drives two arms and the fold has one")
        spacing = arms.CONFIG.get("spacing_cm")
        if spacing is None:
            raise ValueError("arms.json has no spacing_cm")
        self.hands = {side: Lent(followers[side]) for side in ("left", "right")}
        for hand in self.hands.values():    # Gamepad reads limits and poses at once: give it somewhere to start
            hand.present = hand.command = hand.target = {m: sum(hand.limits[m]) / 2 for m in MOTOR_IDS}
        self.pad = gamepad.Gamepad(self.hands["left"], self.hands["right"], float(spacing) / 100)
        threading.Thread(target=self.pad.input_loop, daemon=True).start()

    def take(self, command):
        """Start from what the fold sent last ({side: {motor: ticks}}), not driving until A is pressed."""
        for side, hand in self.hands.items():
            hand.command, hand.target, hand.present = dict(command[side]), dict(command[side]), dict(command[side])
        self.pad.enabled, self.pad.last_tick = False, None

    def tick(self):
        """One step of gamepad.py's loop, with its Drive pressed whenever it is not driving. Returns its status
        line."""
        if not self.pad.enabled:
            self.pad.press_drive()
        state, captured, message = self.pad.sample
        self.pad.tick(state, captured, time.monotonic(), message)
        return self.pad.status

    def command(self, side):
        return dict(self.hands[side].command)

    def xbox_button(self):
        """Whether the Xbox button is down, from gamepad.py's own reading; None with no fresh reading."""
        state, captured, _ = self.pad.sample
        if state is None or time.monotonic() - captured > 0.3:
            return None
        return bool(state.guide)

    def close(self):
        self.pad.shutdown.set()


class _XInputPad(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8),
                ("rt", ctypes.c_uint8), ("lx", ctypes.c_int16), ("ly", ctypes.c_int16), ("rx", ctypes.c_int16),
                ("ry", ctypes.c_int16), ("reserved", ctypes.c_uint32)]


def windows_xbox_button():
    """A function giving whether the Xbox button is down (bit 0x0400 of XInput's call number 100), None while no
    controller answers."""
    read = ctypes.WinDLL("xinput1_4.dll")[100]
    read.argtypes, read.restype = (ctypes.c_uint32, ctypes.POINTER(_XInputPad)), ctypes.c_uint32
    pad = _XInputPad()
    return lambda: bool(pad.buttons & 0x0400) if read(0, ctypes.byref(pad)) == 0 else None


class Button:
    """The Xbox button, watched on its own thread: each press turns `waiting` over. `down` is a function giving
    whether it is down now, or None with no controller."""

    def __init__(self, down, period=0.02):
        self.down, self.period, self.stopped = down, period, threading.Event()
        self.waiting, self.was_down, self.seen = False, False, time.monotonic()
        threading.Thread(target=self.watch, daemon=True).start()

    def look(self):
        """Read the button once."""
        now = self.down()
        if now is not None:
            if now and not self.was_down:
                self.waiting = not self.waiting
            self.was_down, self.seen = now, time.monotonic()

    def watch(self):
        while not self.stopped.is_set():
            try:
                self.look()
            except Exception as err:        # no driver for the controller: the fold runs without the button
                print(f"The Xbox button cannot be read ({err}).")
                return
            self.stopped.wait(self.period)

    def read(self):
        """Whether the fold is to wait."""
        if self.waiting and time.monotonic() - self.seen > LOST:
            self.waiting = False
            print("The controller stopped answering: the fold goes on.")
        return self.waiting

    def close(self):
        self.stopped.set()


def controls(followers):
    """(Teleop, Button) for a fold's arms. ValueError says why there is no intervention button."""
    try:
        teleop = Teleop(followers)
    except (OSError, KeyError, ImportError) as err:
        raise ValueError(str(err)) from err
    return teleop, Button(windows_xbox_button() if sys.platform == "win32" else teleop.xbox_button)


class Nudges:
    """The stored offsets, and one fold's rows with them applied."""

    def __init__(self, path):
        self.path = Path(path)
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.offsets = {side: {phase: np.clip(np.array(saved.get(side, {}).get(phase, [0.0, 0.0, 0.0]), float),
                                              -MAX_NUDGE, MAX_NUDGE) for phase in PHASES}
                        for side in ("left", "right")}
        self.arm, self.changed, self.kept_plan = Arm(), False, 0

    def describe(self):
        parts = [f"{side} {phase} ({', '.join(f'{v * 100:+.1f}' for v in offset)}) cm"
                 for side, phases in self.offsets.items() for phase, offset in phases.items() if np.any(offset)]
        return "; ".join(parts) if parts else "none"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({side: {phase: offset.tolist() for phase, offset in phases.items()}
                                         for side, phases in self.offsets.items()}))

    def begin(self, followers, ticks, check, collision):
        """Take a fold's plan (`ticks` {side: rows of 6 ticks}) and the row of its "check" mark, and work out
        every row with the stored offsets applied."""
        self.followers, self.plan, self.check, self.collision = followers, ticks, check, collision
        self.final = len(next(iter(ticks.values()))) - 1
        self.rows = {side: [dict(zip(MOTOR_IDS, row)) for row in ticks[side]] for side in followers}
        for side in followers:
            self.rebuild(side, 0)

    def phase(self, k):
        return "pinch" if k <= self.check else "carry"

    def corrected(self, side, k, offset):
        """Row k of `side`'s plan with the fingertip moved by `offset`, as {motor: ticks}; None when the row cannot
        be had (see the module's docstring)."""
        follower = self.followers[side]
        plan = np.array([follower.to_rad(m, t) for m, t in zip(MOTOR_IDS, self.plan[side][k])])
        q, miss, _ = self.arm.solve(self.arm.tip(plan[:5])[0] + offset, plan[:5], SOLVE_STEPS)
        if miss > REACH:
            return None
        pose = [*map(float, q), float(plan[5])]
        row, stats = replay.convert(follower, [pose])
        if any(worst > replay.CLAMP_REFUSE_DEG for _, worst in stats.values()):
            return None
        try:
            if self.collision.closest([pose])[0] < self.collision.CLEARANCE:
                return None
        except ValueError:
            return None
        return dict(zip(MOTOR_IDS, row[0]))

    def rebuild(self, side, start):
        """Work out `side`'s rows from `start` on with the offsets as they are now. The last row stays the plan's."""
        for k in range(start, self.final):
            offset = self.offsets[side][self.phase(k)]
            row = self.corrected(side, k, offset) if np.any(offset) else None
            if row is None and np.any(offset):
                self.kept_plan += 1
            self.rows[side][k] = row if row is not None else dict(zip(MOTOR_IDS, self.plan[side][k]))

    def take(self, side, k, command):
        """After an intervention on row k that left `side` at `command` ({motor: ticks}): keep the fingertip's
        offset from the plan's row. Returns "kept", "not moved", or why it is not kept."""
        follower = self.followers[side]
        at = np.array([follower.to_rad(m, command[m]) for m in MOTOR_IDS[:5]])
        plan = np.array([follower.to_rad(m, t) for m, t in zip(MOTOR_IDS[:5], self.plan[side][k])])
        phase = self.phase(k)
        wanted = np.clip(self.arm.tip(at)[0] - self.arm.tip(plan)[0], -MAX_NUDGE, MAX_NUDGE)
        taken = wanted - self.offsets[side][phase]
        if np.linalg.norm(taken) < TOUCHED:
            return "not moved"
        if self.corrected(side, k, wanted) is None:
            return "not kept: the plan's row cannot be had with the fingertip there"
        self.offsets[side][phase] = wanted
        if phase == "pinch":            # a corrected pinch carries to the place
            self.offsets[side]["carry"] = np.clip(self.offsets[side]["carry"] + taken, -MAX_NUDGE, MAX_NUDGE)
        self.changed = True
        return "kept"


def main():
    """Print the Xbox button as it is read, for 30 s: check it before a fold. No arm is touched."""
    if sys.platform == "win32":
        down = windows_xbox_button()
    else:
        from hardware.controller.inputs import controller_samples
        seen, stopped = [None], threading.Event()

        def listen():
            for state in controller_samples(stopped):
                seen[0] = None if state is None else bool(state.guide)
        threading.Thread(target=listen, daemon=True).start()
        down = lambda: seen[0]    # noqa: E731
    button = Button(down)
    print("Press the Xbox button a few times: 'waiting' should turn over on each press. Ctrl-C ends.")
    for _ in range(150):
        print(f"button {'no controller' if down() is None else 'down' if down() else 'up'}   waiting {button.read()}")
        time.sleep(0.2)
    button.close()


if __name__ == "__main__":
    main()
