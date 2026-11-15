"""Open-loop replay of a simulator trajectory on one or two real SO-101 followers.

    .venv-hw/bin/python replay.py traj.json --left arm2 --right arm1 [--dry-run]
        [--speed 0.5] [--only left|right] [--no-gripper] [--allow-clamp] [--log run.json] [--grasp-check]
    .venv-hw/bin/python replay.py --check arm2     # confirm each joint's sign, one joint at a time
    .venv-hw/bin/python replay.py --read arm2      # no motion: print the joint angles the map gives, live

The trajectory (exported from the simulator) holds joint targets in radians in the SO-101
URDF convention. Per arm they become servo ticks, are clamped to <name>/limits.json,
and are sent at dt / speed per step. No feedback: the arm is told where to go and nothing
checks that it got there, except the measured final pose printed at the end, and:
  --log FILE     reads every arm each step and writes the measured angles (model radians) beside the targets
  --grasp-check  for a trajectory with "marks" (fold_traj.py): compares each jaw 2 cm into the lift with its reading
                 shut on nothing; a jaw no further open than --grasp-margin deg has no cloth, and both arms go back
                 down, open, and pinch again, up to --retries times, then carry on regardless

Map (an unverified hypothesis, see docs): q_urdf_rad = sign * (encoder - zero) * 2*pi/4096,
zero = 2047 and sign = +1 for motors 1-5 (set_limits.py puts the middle of each swept range at 2047, and the URDF's
arm joints are symmetric about 0). The gripper's URDF range is -10 to 100 deg, so its zero is 10 deg above the
closed end of its limits, taken to be the lower tick. <name>/urdf.json overrides per motor:
    {"zero": {"1": 2047, ...}, "sign": {"1": 1, "2": -1, ...}}
twin.json is not used. Use --check to confirm the signs before trusting a replay.

Ctrl+C or a bus error stops sending and leaves torque on, so the arms hold where they are.
"""

import argparse
import json
import math
import sys
import time

import scservo_sdk as scs

import arms

BAUDRATE = 1_000_000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 6
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STEPS_PER_DEGREE = 4096 / 360
RAD_PER_STEP = 2 * math.pi / 4096
TICK_SECONDS = 0.02  # ramp / check update period (50 Hz, as ui.py)
RAMP_DEG_S = 20  # slowest allowed approach to the first target
MAX_DEG_S = 60  # per-joint ceiling while playing
CLAMP_REFUSE_DEG = 2
GRASP_RAD = 0.2  # gripper target below this = the grasp
CHECK_DEG = 8
SETTLE_SECONDS = 0.5  # hold before reading the final pose

# STS3215 control table addresses (same as ui.py)
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56
GRIPPER_TORQUE_LIMIT = 500  # 50% of max, as ui.py


class Follower:
    """One arm: its tick<->radian map, its limits, and (after connect) its bus."""

    def __init__(self, name):
        self.name = name
        folder = arms.folder(name)
        limits = json.loads((folder / "limits.json").read_text())  # missing file = error
        self.limits = {m: (limits[str(m)]["true"]["min"], limits[str(m)]["true"]["max"]) for m in MOTOR_IDS}
        urdf = json.loads((folder / "urdf.json").read_text()) if (folder / "urdf.json").exists() else {}
        self.overridden = bool(urdf)
        default = {m: 2047 for m in MOTOR_IDS} | {GRIPPER_ID: self.limits[GRIPPER_ID][0] + 10 * STEPS_PER_DEGREE}
        self.zero = {m: urdf.get("zero", {}).get(str(m), default[m]) for m in MOTOR_IDS}
        self.sign = {m: urdf.get("sign", {}).get(str(m), 1) for m in MOTOR_IDS}

    def describe(self):
        kind = "from urdf.json" if self.overridden else "DEFAULT, unverified on hardware"
        return (f"{self.name}: zero {[self.zero[m] for m in MOTOR_IDS]} "
                f"sign {[self.sign[m] for m in MOTOR_IDS]} ({kind})")

    def to_ticks(self, m, rad):
        return self.zero[m] + self.sign[m] * rad / RAD_PER_STEP

    def to_rad(self, m, ticks):
        return self.sign[m] * (ticks - self.zero[m]) * RAD_PER_STEP

    # --- bus access, copied from ui.Arm (read_present, send_goals, apply_torque) ---

    def connect(self):
        self.port = scs.PortHandler(arms.port(self.name))
        self.packet = scs.PacketHandler(0)  # protocol 0 = Feetech STS series
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            raise OSError(f"Could not open {arms.port(self.name)} ({self.name} arm)")
        self.reader = scs.GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, 2)
        for m in MOTOR_IDS:
            self.reader.addParam(m)

    def read_present(self):
        for _ in range(10):
            self.port.ser.reset_input_buffer()
            try:
                ok = self.reader.txRxPacket() == scs.COMM_SUCCESS
            except IndexError:  # SDK crashes on a truncated reply
                ok = False
            if ok and all(self.reader.isAvailable(m, ADDR_PRESENT_POSITION, 2) for m in MOTOR_IDS):
                return {m: self.reader.getData(m, ADDR_PRESENT_POSITION, 2) for m in MOTOR_IDS}
            time.sleep(0.02)
        raise OSError(f"Could not read all motors of the {self.name} arm")

    def send_goals(self, goals):
        """Every listed motor's goal ({motor id: ticks}) in ONE packet."""
        sync = scs.GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, 2)
        for m, goal in goals.items():
            goal = max(0, min(4095, round(goal)))
            sync.addParam(m, [scs.SCS_LOBYTE(goal), scs.SCS_HIBYTE(goal)])
        if sync.txPacket() != scs.COMM_SUCCESS:
            raise OSError(f"Write to the {self.name} arm failed")

    def torque_on_holding(self, present):
        """Goal = where each joint is NOW, then torque on, so it holds still."""
        self.send_goals(present)
        self.packet.write2ByteTxRx(self.port, GRIPPER_ID, ADDR_TORQUE_LIMIT, GRIPPER_TORQUE_LIMIT)
        for m in MOTOR_IDS:
            self.packet.write1ByteTxRx(self.port, m, ADDR_TORQUE_ENABLE, 1)


# --- trajectory file ---

def matrix(value, rows, cols, label):
    ok = (isinstance(value, list) and (rows is None or len(value) == rows) and len(value) > 0
          and all(isinstance(r, list) and len(r) == cols for r in value))
    if not ok:
        raise ValueError(f"{label}: expected {rows if rows is not None else 'T'} rows of {cols} numbers")
    for r in value:
        for x in r:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f"{label}: non-finite or non-numeric value {x!r}")
    return value


def validate(traj):
    """Shapes, finite values, both arms the same T. Raises ValueError; touches no bus."""
    if not isinstance(traj, dict):
        raise ValueError("trajectory must be a JSON object")
    dt = traj.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be a positive number, got {dt!r}")
    if traj.get("joints") != JOINTS:
        raise ValueError(f"joints must be {JOINTS}, got {traj.get('joints')!r}")
    steps = set()
    for side in ("left", "right"):
        arm = (traj.get("arms") or {}).get(side)
        if not isinstance(arm, dict):
            raise ValueError(f"arms.{side} missing")
        target = matrix(arm.get("target"), None, 6, f"{side}.target")
        matrix(arm.get("q"), len(target) + 1, 6, f"{side}.q")
        if "tip" in arm:
            matrix(arm["tip"], len(target) + 1, 3, f"{side}.tip")
        steps.add(len(target))
    if len(steps) != 1:
        raise ValueError(f"left and right have different step counts {sorted(steps)}")


def convert(follower, target):
    """Radian targets -> clamped ticks per step. Also returns per-joint (count, max deg) clamped."""
    out, stats = [], {m: [0, 0.0] for m in MOTOR_IDS}
    for row in target:
        ticks = []
        for m, rad in zip(MOTOR_IDS, row):
            t = follower.to_ticks(m, rad)
            lo, hi = follower.limits[m]
            c = max(lo, min(hi, t))
            if c != t:
                stats[m][0] += 1
                stats[m][1] = max(stats[m][1], abs(t - c) / STEPS_PER_DEGREE)
            ticks.append(c)
        out.append(ticks)
    return out, stats


def print_clamps(side, follower, stats):
    for m, name in zip(MOTOR_IDS, JOINTS):
        count, worst = stats[m]
        print(f"  {side} {name}: {count} steps clamped, largest {worst:.2f} deg")


def print_tips(side, arm):
    tip = arm.get("tip")
    if not tip:
        return

    def fmt(p):
        return f"right {p[0] * 100:.1f}, forward {p[1] * 100:.1f}, up {p[2] * 100:.1f} cm"

    print(f"  {side} tip at start: {fmt(tip[0])}")
    grasp = next((k for k, row in enumerate(arm["target"]) if row[5] < GRASP_RAD), None)
    if grasp is None:
        print(f"  {side} gripper never drops below {GRASP_RAD} rad: no grasp")
    else:
        print(f"  {side} tip at the grasp (step {grasp}): {fmt(tip[grasp])}")


def print_dry_run(side, arm):
    rows = arm["target"]
    for j, name in enumerate(JOINTS):
        col = [math.degrees(r[j]) for r in rows]
        step = max((abs(b - a) for a, b in zip(col, col[1:])), default=0.0)
        print(f"  {side} {name}: {min(col):.1f} to {max(col):.1f} deg, max step change {step:.2f} deg")


# --- motion ---

def sleep_until(deadline):
    wait = deadline - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def ramp_points(start, end, speed_deg_s=RAMP_DEG_S, tick=TICK_SECONDS):
    """Poses from start to end ({motor id: ticks}), no joint faster than speed_deg_s."""
    longest = max(abs(end[m] - start[m]) for m in end) / STEPS_PER_DEGREE
    n = math.ceil(longest / (speed_deg_s * tick) - 1e-9)
    return [{m: start[m] + (end[m] - start[m]) * i / n for m in end} for i in range(1, n + 1)]


def run_poses(followers, poses):
    """Send each pose ({name: {motor id: ticks}}) in step, one per tick, on a monotonic clock."""
    t0 = time.monotonic()
    for i, pose in enumerate(poses):
        sleep_until(t0 + i * TICK_SECONDS)
        for name, goals in pose.items():
            followers[name].send_goals(goals)


def play(followers, ticks, dt, speed, motors, progress, steps=None, log=None):
    """Send the rows `steps` of ticks (default: all, in order), every arm each tick. Returns how many times the rate
    cap engaged. With `log` (a list), every arm is read each tick: (row, {name: {motor id: ticks}}) is appended."""
    tick = dt / speed
    cap = MAX_DEG_S * tick * speed * STEPS_PER_DEGREE  # ticks per tick
    steps = range(len(next(iter(ticks.values())))) if steps is None else steps
    command = {name: {m: ticks[name][steps[0]][m - 1] for m in motors} for name in followers}
    engaged = 0
    t0 = time.monotonic()
    for i, k in enumerate(steps):
        sleep_until(t0 + i * tick)
        progress["step"] = k
        if log is not None:
            log.append((k, {name: f.read_present() for name, f in followers.items()}))
        for name, f in followers.items():
            for m in motors:
                delta = ticks[name][k][m - 1] - command[name][m]
                if abs(delta) > cap:
                    engaged += 1
                    delta = math.copysign(cap, delta)
                command[name][m] += delta
            f.send_goals(command[name])
    return engaged


def jaws(followers):
    """Each arm's jaw angle (model degrees) after a short settle."""
    time.sleep(SETTLE_SECONDS)
    return {side: math.degrees(f.to_rad(GRIPPER_ID, f.read_present()[GRIPPER_ID])) for side, f in followers.items()}


def play_with_grasp_check(followers, marks, T, run, margin, retries):
    """The whole trajectory through run(steps), stopping at marks["check"] to compare each jaw with its reading at
    marks["empty"] (shut on nothing). A missed grasp plays the rows back to marks["hover"] and forward again."""
    if not marks:
        sys.exit("--grasp-check needs a trajectory with marks (fold_traj.py)")
    empty_at, hover, check_at = marks["empty"], marks["hover"], marks["check"]
    engaged = run(range(empty_at + 1))
    empty = jaws(followers)
    engaged += run(range(empty_at + 1, check_at + 1))
    for attempt in range(retries + 1):
        lifted = jaws(followers)
        held = {side: lifted[side] - empty[side] > margin for side in followers}
        print("  ".join(f"{side} jaw {lifted[side]:.1f} deg lifted, {empty[side]:.1f} empty: "
                        f"{'holding' if held[side] else 'MISSED'}" for side in followers))
        if all(held.values()) or attempt == retries:
            break
        print(f"Pinching again (retry {attempt + 1} of {retries})")
        engaged += run(range(check_at, hover - 1, -1))
        engaged += run(range(hover, check_at + 1))
    return engaged + run(range(check_at + 1, T))


def check_joints(follower, present, motors, progress):
    """For joints 1-5: wait for Enter, +8 deg in the URDF-positive direction over 1 s, hold 1 s, back."""
    n = round(1 / TICK_SECONDS)
    for m in motors:
        progress["step"] = f"check, before {JOINTS[m - 1]}"
        input(f"\n{JOINTS[m - 1]} (motor {m}): Enter to move it +{CHECK_DEG} deg (URDF positive) ")
        lo, hi = follower.limits[m]
        goal = max(lo, min(hi, present[m] + follower.sign[m] * CHECK_DEG * STEPS_PER_DEGREE))
        out = [{m: present[m] + (goal - present[m]) * i / n} for i in range(1, n + 1)]
        back = [{m: goal + (present[m] - goal) * i / n} for i in range(1, n + 1)]
        run_poses({follower.name: follower}, [{follower.name: g} for g in out + [out[-1]] * n + back])


# --- main ---

def speed_arg(text):
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return value


def parse(argv):
    p = argparse.ArgumentParser(description="Replay a simulator trajectory on real SO-101 arms.")
    p.add_argument("trajectory", nargs="?")
    p.add_argument("--left")
    p.add_argument("--right")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--speed", type=speed_arg, default=0.5)
    p.add_argument("--only", choices=["left", "right"])
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--allow-clamp", action="store_true")
    p.add_argument("--log", metavar="FILE")
    p.add_argument("--grasp-check", action="store_true")
    p.add_argument("--grasp-margin", type=float, default=0.5, help="deg the jaw must stay open past its empty reading")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--check", metavar="ARM")
    p.add_argument("--read", metavar="ARM")
    args = p.parse_args(argv)
    for name in (args.left, args.right, args.check, args.read):
        if name is not None and name not in arms.ARMS:
            p.error(f"unknown arm {name!r}; arms.json has {', '.join(arms.ARMS)}")
    if args.check is None and args.read is None:
        if not args.trajectory:
            p.error("a trajectory file is required")
        if not (args.left or args.right):
            p.error("name at least one arm with --left / --right")
        if args.only and not getattr(args, args.only):
            p.error(f"--only {args.only} needs --{args.only} <arm name>")
        if args.left and args.left == args.right:
            p.error("--left and --right name the same arm")
        if args.grasp_check and args.no_gripper:
            p.error("--grasp-check needs the gripper")
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    progress = {"step": "before playing"}
    try:
        if args.read:
            return read(args)
        if args.check:
            return check(args, progress)
        return replay(args, progress)
    except (KeyboardInterrupt, OSError) as err:
        reason = "Ctrl-C" if isinstance(err, KeyboardInterrupt) else f"bus error: {err}"
        print(f"\nStopped ({reason}) at step {progress['step']}. No more targets sent; torque left on, arms hold.")
        return 1


def read(args):
    """Print the arm's joint angles (model degrees, through the map) until Ctrl-C. Sends nothing: torque stays as it
    is, so a limp arm can be posed by hand against known angles (stretched straight forward = all 0)."""
    f = Follower(args.read)
    print("Map in use:", f.describe())
    f.connect()
    print("  ".join(f"{name:>13}" for name in JOINTS))
    try:
        while True:
            present = f.read_present()
            print("  ".join(f"{math.degrees(f.to_rad(m, present[m])):13.1f}" for m in MOTOR_IDS), end="\r", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    return 0


def check(args, progress):
    f = Follower(args.check)
    print("Map in use:", f.describe())
    f.connect()
    present = f.read_present()
    f.torque_on_holding(present)
    check_joints(f, present, MOTOR_IDS[:5], progress)
    print("\nDone. Each joint should have moved the same way as the URDF's positive direction.")
    return 0


def replay(args, progress):
    try:
        traj = json.loads(open(args.trajectory).read())
        validate(traj)
    except (OSError, ValueError) as err:
        sys.exit(f"Bad trajectory {args.trajectory}: {err}")
    print(f"layout: {traj.get('layout')}\nsource: {traj.get('source')}")
    T, dt = len(traj["arms"]["left"]["target"]), traj["dt"]
    print(f"{T} steps, dt {dt} s, speed {args.speed}: {T * dt / args.speed:.1f} s")

    names = {side: name for side, name in (("left", args.left), ("right", args.right))
             if name and args.only in (None, side)}
    followers = {side: Follower(name) for side, name in names.items()}
    ticks, refuse = {}, False
    for side, f in followers.items():
        print("Map in use:", f.describe())
        ticks[side], stats = convert(f, traj["arms"][side]["target"])
        print_clamps(side, f, stats)
        refuse |= any(worst > CLAMP_REFUSE_DEG for _, worst in stats.values())
        print_tips(side, traj["arms"][side])
    if refuse and not args.allow_clamp:
        sys.exit(f"Clamped by more than {CLAMP_REFUSE_DEG} deg: the arm cannot do this trajectory. "
                 "Re-run with --allow-clamp to play the clamped version.")
    if args.dry_run:
        for side in followers:
            print_dry_run(side, traj["arms"][side])
        return 0

    motors = MOTOR_IDS[:5] if args.no_gripper else MOTOR_IDS
    for f in followers.values():
        f.connect()
    present = {side: f.read_present() for side, f in followers.items()}
    for side, f in followers.items():
        f.torque_on_holding(present[side])

    progress["step"] = "ramp to the first target"
    first = {side: {m: ticks[side][0][m - 1] for m in motors} for side in followers}
    steps = [ramp_points({m: present[s][m] for m in motors}, first[s]) for s in followers]
    n = max(len(p) for p in steps)
    poses = [{s: p[min(i, len(p) - 1)] for s, p in zip(followers, steps) if p} for i in range(n)]
    run_poses(followers, poses)
    input("At the first target. Check the arms are clear, then press Enter to play (Ctrl-C stops) ")

    progress["step"] = 0
    log = [] if args.log else None
    run = lambda steps: play(followers, ticks, dt, args.speed, motors, progress, steps, log)   # noqa: E731
    if args.grasp_check:
        engaged = play_with_grasp_check(followers, traj.get("marks"), T, run, args.grasp_margin, args.retries)
    else:
        engaged = run(range(T))
    print(f"Rate cap ({MAX_DEG_S} deg/s) {'engaged ' + str(engaged) + ' times' if engaged else 'never engaged'}.")
    time.sleep(SETTLE_SECONDS)
    if args.log:
        rows = {side: [[k] + [round(f.to_rad(m, present[side][m]), 4) for m in MOTOR_IDS] for k, present in log]
                for side, f in followers.items()}
        with open(args.log, "w") as out:
            json.dump({"trajectory": args.trajectory, "speed": args.speed, "columns": ["row"] + JOINTS,
                       "measured": rows}, out)
        print(f"Measured angles for {len(log)} steps written to {args.log}")
    for side, f in followers.items():
        now, final_q = f.read_present(), traj["arms"][side]["q"][-1]
        print(f"{side} final (deg), measured vs trajectory q:")
        for m, name in zip(MOTOR_IDS, JOINTS):
            got, want = math.degrees(f.to_rad(m, now[m])), math.degrees(final_q[m - 1])
            print(f"  {name}: {got:7.1f} vs {want:7.1f}  (error {got - want:+.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
