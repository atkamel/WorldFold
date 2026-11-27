"""The intervention button of dagger_run.py: press the Xbox button on the controller and the fold waits where it
is; move a fingertip with the sticks; press it again and the fold goes on with the fingertip that much off the plan.
The offset is kept (nudges.json in the store), so the next fold starts with it, and the corrected rows are what the
network is taught.

    Xbox button     the fold waits on its current row; pressed again, it goes on
    left stick      the left arm's fingertip: forward and back, left and right      (right stick: the right arm)
    LB / LT         the left arm's fingertip up / down                              (RB / RT: the right arm)

    python hardware/so101/dagger_nudge.py      # no arms: prints what the controller is read as, to check it first

The sticks and bumpers do nothing while the fold is not waiting. A controller that stops answering for LOST
seconds while the fold waits lets the fold go on with the offsets as they are.

On Windows the Xbox button is read through XInput's unlisted call (number 100 of xinput1_4.dll); the listed one
leaves that button out. Windows also opens the Game Bar on that button unless "Open Xbox Game Bar using this button
on a controller" is switched off in Settings, Gaming.

An offset is (x forward, y left, z up) m in the arm's own base frame, like fold_camera.py's --overshoot and
--pinch-height, and no axis goes past MAX_NUDGE. Each arm has two: "pinch", from the first row until the corner is
2 cm into the lift (the plan's "check" mark), and "carry", from there to the end. A nudge made before the lift
moves both, so a corrected pinch carries to the place; a nudge made after it moves the carry alone.

A corrected row is the plan's row with the fingertip moved by the offset (the planner's own solver, started from
the plan's row, the jaw as planned). A row the solver cannot reach, that leaves a joint's limits by more than
replay.py allows, or that comes nearer the arm itself than replay.py allows, stays the plan's row. A nudge that would
make the row the fold is waiting on fail one of those is not taken. The last row is always the plan's.
"""

import ctypes
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import replay
from kinematics import Arm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))        # hardware.controller

MAX_NUDGE = 0.03        # m either way, each axis: fold_camera.MAX_OVERSHOOT
SPEED = 0.02            # m/s of the fingertip with a stick pushed all the way
LIFT = 0.01             # m/s up with a bumper held, down with a trigger pulled all the way
DEAD_ZONE = 9000        # raw stick units from its rest that count as rest (gamepad.DEAD_ZONE)
TRIGGER_START = 0.1     # share of its travel under which a trigger is at rest
STALE = 0.3             # s after which a controller sample is no longer acted on
LOST = 2.0              # s without the controller after which a waiting fold goes on
REACH = 1e-3            # m the solver may miss a corrected fingertip by
SOLVE_STEPS = 40
PHASES = ("pinch", "carry")
MOTOR_IDS = replay.MOTOR_IDS


class _XInputPad(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("buttons", ctypes.c_uint16), ("lt", ctypes.c_uint8),
                ("rt", ctypes.c_uint8), ("lx", ctypes.c_int16), ("ly", ctypes.c_int16), ("rx", ctypes.c_int16),
                ("ry", ctypes.c_int16), ("reserved", ctypes.c_uint32)]


def windows_samples(stopped, period=0.02):
    """(xbox button, lb, rb, left trigger 0-1, right trigger 0-1, the four stick axes) from XInput's call number
    100, which reports the Xbox button (bit 0x0400); None while no controller answers."""
    read = ctypes.WinDLL("xinput1_4.dll")[100]
    read.argtypes, read.restype = (ctypes.c_uint32, ctypes.POINTER(_XInputPad)), ctypes.c_uint32
    pad = _XInputPad()
    while not stopped.is_set():
        if read(0, ctypes.byref(pad)) == 0:
            yield (bool(pad.buttons & 0x0400), bool(pad.buttons & 0x0100), bool(pad.buttons & 0x0200),
                   pad.lt / 255, pad.rt / 255, (pad.lx, pad.ly, pad.rx, pad.ry))
        else:
            yield None
        stopped.wait(period)


def usb_samples(stopped):
    """The same from hardware/controller's reader (macOS: raw USB)."""
    from hardware.controller.inputs import controller_samples
    for state in controller_samples(stopped):
        yield None if state is None else (
            bool(state.guide), "lb" in state.buttons, "rb" in state.buttons, state.left_trigger / 1023,
            state.right_trigger / 1023, (state.left_x, state.left_y, state.right_x, state.right_y))


class Pad:
    """The controller, read on its own thread. Each press of the Xbox button turns `waiting` over."""

    def __init__(self, samples=None):
        self.sample, self.centre, self.stopped = (None, 0.0), None, threading.Event()
        self.waiting, self.was_down = False, False
        self.samples = samples or (windows_samples if sys.platform == "win32" else usb_samples)
        threading.Thread(target=self.listen, daemon=True).start()

    def listen(self):
        try:
            for sample in self.samples(self.stopped):
                if sample is not None:
                    if sample[0] and not self.was_down:
                        self.waiting = not self.waiting
                    self.was_down = sample[0]
                    self.sample = (sample, time.monotonic())
        except Exception as err:        # no controller, no driver: the fold runs without the button
            self.sample = (None, 0.0)
            print(f"No controller for the intervention button ({err}).")

    def close(self):
        self.stopped.set()

    def stick(self, x, y, centre):
        point = (np.array([x, y], float) - centre) / 32768
        length, rest = float(np.linalg.norm(point)), DEAD_ZONE / 32768
        if length <= rest:
            return np.zeros(2)
        return point * min(1.0, (length - rest) / (1.0 - rest)) / length

    def read(self):
        """(whether the fold is to wait, {side: the fingertip's rate (x, y, z), each -1 to 1} or None when the
        controller has not answered for STALE seconds)."""
        sample, seen = self.sample
        if sample is None or time.monotonic() - seen > STALE:
            if self.waiting and time.monotonic() - seen > LOST:
                self.waiting = False
                print("The controller stopped answering: the fold goes on.")
            return self.waiting, None
        _, lb, rb, lt, rt, axes = sample
        axes = np.array(axes, float)
        if self.centre is None:
            self.centre = axes          # where the sticks rest
        rates = {}
        for side, (x, y), up, pulled in (("left", self.stick(*axes[:2], self.centre[:2]), lb, lt),
                                        ("right", self.stick(*axes[2:], self.centre[2:]), rb, rt)):
            down = max(0.0, (pulled - TRIGGER_START) / (1.0 - TRIGGER_START))
            rates[side] = np.array([y, -x, (1.0 if up else 0.0) - down])    # stick up is forward, stick right is -y
        return self.waiting, rates


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

    def step(self, side, k, move):
        """Move `side`'s offset for row k's part of the fold by `move` (m), if row k can be had with it. Returns
        that row ({motor: ticks}) or None when the nudge is not taken."""
        phase = self.phase(k)
        wanted = np.clip(self.offsets[side][phase] + move, -MAX_NUDGE, MAX_NUDGE)
        taken = wanted - self.offsets[side][phase]
        row = self.corrected(side, k, wanted) if np.any(taken) else None
        if row is None:
            return None
        self.offsets[side][phase] = wanted
        if phase == "pinch":            # a corrected pinch carries to the place
            self.offsets[side]["carry"] = np.clip(self.offsets[side]["carry"] + taken, -MAX_NUDGE, MAX_NUDGE)
        self.changed = True
        return row


def main():
    """Print what the controller is read as, for 30 s: check the Xbox button and the sticks before a fold."""
    pad = Pad()
    print("Press the Xbox button, push the sticks, hold the bumpers and triggers. Ctrl-C ends.")
    for _ in range(150):
        waiting, rates = pad.read()
        print(f"waiting {waiting}  " + ("no controller" if rates is None else "  ".join(
            f"{side} forward {r[0]:+.2f} left {r[1]:+.2f} up {r[2]:+.2f}" for side, r in rates.items())))
        time.sleep(0.2)
    pad.close()


if __name__ == "__main__":
    main()
