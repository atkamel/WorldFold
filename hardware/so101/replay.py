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

An arm with a saved <name>/idle_pose.json (idle.py record) glides to it before the first target and back to it
after the last row, so a run starts and ends in the arm's rest pose.

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
FULL_TORQUE_LIMIT = 1000
MIN_TORQUE_PERCENT = 10  # below this an arm cannot hold itself up


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

    def torque_on_holding(self, present, arm_limit=FULL_TORQUE_LIMIT):
        """Goal = where each joint is NOW, then torque on, so it holds still. `arm_limit` (0 to 1000) caps the torque
        of every joint; the gripper never gets more than its own cap."""
        self.send_goals(present)
        self.set_torque_limits(arm_limit)
        for m in MOTOR_IDS:
            result, _ = self.packet.write1ByteTxRx(self.port, m, ADDR_TORQUE_ENABLE, 1)
            if result != scs.COMM_SUCCESS:
                raise OSError(f"Could not turn on the torque of motor {m} on the {self.name} arm")

    def set_torque_limits(self, arm_limit):
        """Write the torque limit of every motor and read it back. The register keeps its value until the arm is
        powered off, so it is written on every run."""
        for m in MOTOR_IDS:
            limit = arm_limit
            if m == GRIPPER_ID:
                limit = min(GRIPPER_TORQUE_LIMIT, arm_limit)
            result, _ = self.packet.write2ByteTxRx(self.port, m, ADDR_TORQUE_LIMIT, limit)
            held, read_result, _ = self.packet.read2ByteTxRx(self.port, m, ADDR_TORQUE_LIMIT)
            if result != scs.COMM_SUCCESS or read_result != scs.COMM_SUCCESS or held != limit:
                raise OSError(f"Motor {m} on the {self.name} arm did not take the torque limit {limit}")


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


def lowest_points(follower, ticks):
    """[(height above the table in m, step)] for the elbow joint centre, the wrist joint centre and the fingertip:
    the lowest each one gets over the clamped tick rows that are sent, from the arm model."""
    import kinematics  # needs numpy, which playing a trajectory does not

    model = kinematics.Arm()
    lowest = [None, None, None]
    for step, row in enumerate(ticks):
        angles = []
        for m in MOTOR_IDS[:5]:
            angles.append(follower.to_rad(m, row[m - 1]))
        points = model.joint_points(angles)
        for i in range(3):
            height = float(points[i][2])
            if lowest[i] is None or height < lowest[i][0]:
                lowest[i] = (height, step)
    return lowest


def keeps_clear_of_itself(side, follower, ticks):
    """Print the nearest the arm comes to itself over the clamped tick rows that are sent. False if that is under
    the clearance."""
    try:
        import collision  # needs mujoco and so101_nexus, which playing a trajectory does not
    except ImportError as err:
        sys.exit(f"The self-collision check cannot run ({err}). Install mujoco and so101_nexus, or pass "
                 "--no-collision-check to play without it.")
    rows = []
    for row in ticks:
        angles = []
        for m in MOTOR_IDS:
            angles.append(follower.to_rad(m, row[m - 1]))
        rows.append(angles)
    distance, step, parts = collision.closest(rows)
    if distance >= collision.FAR:
        print(f"  {side} no part of the arm comes within {collision.FAR * 100:.0f} cm of another")
    else:
        print(f"  {side} nearest the arm comes to itself: {distance * 1000:.0f} mm ({parts}) at step {step}")
    return distance >= collision.CLEARANCE


def print_lowest(side, follower, ticks):
    names = ["elbow joint centre", "wrist joint centre", "fingertip"]
    lowest = lowest_points(follower, ticks)
    for i in range(3):
        height, step = lowest[i]
        print(f"  {side} lowest {names[i]}: {height * 100:.1f} cm above the table at step {step}")
    print(f"  {side}: the motor bodies reach about 3 cm below their joint centres")


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


def idle_goals(follower, motors):
    """The arm's saved idle pose ({motor id: ticks}, inside its limits), or None if it has none."""
    path = arms.folder(follower.name) / "idle_pose.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    goals = {}
    for m in motors:
        lo, hi = follower.limits[m]
        goals[m] = max(lo, min(hi, saved[str(m)]))
    return goals


def ramp_together(followers, start, goal):
    """Ramp every arm from start[side] to goal[side] in step. An arm whose goal is None stays where it is.
    Returns where each arm ends up."""
    end = {}
    steps = {}
    for side in followers:
        if goal[side] is None:
            end[side] = start[side]
            continue
        end[side] = goal[side]
        points = ramp_points(start[side], goal[side])
        if points:
            steps[side] = points
    longest = 0
    for side in steps:
        longest = max(longest, len(steps[side]))
    poses = []
    for i in range(longest):
        pose = {}
        for side in steps:
            points = steps[side]
            pose[side] = points[min(i, len(points) - 1)]
        poses.append(pose)
    run_poses(followers, poses)
    return end


def run_poses(followers, poses):
    """Send each pose ({name: {motor id: ticks}}) in step, one per tick, on a monotonic clock."""
    t0 = time.monotonic()
    for i, pose in enumerate(poses):
        sleep_until(t0 + i * TICK_SECONDS)
        for name, goals in pose.items():
            followers[name].send_goals(goals)


def play(followers, ticks, dt, speed, motors, progress, steps=None, log=None):
    """Send the rows `steps` of ticks (default: all, in order), every arm each tick. Returns how many times the rate
    cap engaged. With `log` (a list), every arm is read each tick: (row, time.monotonic(), {name: {motor id: ticks}})
    is appended."""
    tick = dt / speed
    cap = MAX_DEG_S * tick * speed * STEPS_PER_DEGREE  # ticks per tick
    sends = max(1, math.ceil(tick / TICK_SECONDS - 1e-9))  # a row is reached in this many equal moves, not one jump
    steps = range(len(next(iter(ticks.values())))) if steps is None else steps
    command = {name: {m: ticks[name][steps[0]][m - 1] for m in motors} for name in followers}
    engaged = 0
    t0 = time.monotonic()
    for i, k in enumerate(steps):
        row_start = t0 + i * tick
        sleep_until(row_start)
        progress["step"] = k
        if log is not None:
            log.append((k, time.monotonic(), {name: f.read_present() for name, f in followers.items()}))
        before = {}
        for name in followers:
            before[name] = dict(command[name])
            for m in motors:
                delta = ticks[name][k][m - 1] - command[name][m]
                if abs(delta) > cap:
                    engaged += 1
                    delta = math.copysign(cap, delta)
                command[name][m] += delta
        for j in range(1, sends + 1):
            sleep_until(row_start + (j - 1) * tick / sends)
            for name, f in followers.items():
                goal = {}
                for m in motors:
                    goal[m] = before[name][m] + (command[name][m] - before[name][m]) * j / sends
                f.send_goals(goal)
    return engaged


def jaws(followers):
    """Each arm's jaw angle (model degrees) after a short settle."""
    time.sleep(SETTLE_SECONDS)
    return {side: math.degrees(f.to_rad(GRIPPER_ID, f.read_present()[GRIPPER_ID])) for side, f in followers.items()}


def play_with_grasp_check(followers, marks, T, run, margin, retries, note=None):
    """Rows 0 to T - 1 through run(steps), stopping at marks["check"] to compare each jaw with its reading at
    marks["empty"] (shut on nothing). A missed grasp plays the rows back to marks["hover"] and forward again.
    note(attempt, empty, lifted, held), if given, gets each check's readings (degrees per arm)."""
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
        if note:
            note(attempt, empty, lifted, held)
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


def torque_arg(text):
    value = int(text)
    if not MIN_TORQUE_PERCENT <= value <= 100:
        raise argparse.ArgumentTypeError(f"must be {MIN_TORQUE_PERCENT} to 100")
    return value


def parse(argv):
    p = argparse.ArgumentParser(description="Replay a simulator trajectory on real SO-101 arms.")
    p.add_argument("trajectory", nargs="?")
    p.add_argument("--left")
    p.add_argument("--right")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--speed", type=speed_arg, default=0.5)
    p.add_argument("--torque-limit", type=torque_arg, default=100, metavar="PERCENT",
                   help="cap every joint's torque, so a collision stalls gently; lifted when the run ends at the idle "
                        "pose, left on after a stop")
    p.add_argument("--only", choices=["left", "right"])
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--allow-clamp", action="store_true")
    p.add_argument("--no-collision-check", action="store_true",
                   help="play even if the arm model passes through itself, or when mujoco is not installed")
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
        if args.torque_limit < 100:
            print(f"A {args.torque_limit}% torque cap was asked for and may be on some or all motors. "
                  "The next replay or a power cycle sets every motor again.")
        return 1


def read(args):
    """Print the arm's joint angles (model degrees, through the map) until Ctrl-C. Sends nothing: torque stays as it
    is, so a limp arm can be posed by hand against known angles (upper arm straight up, forearm and gripper level and
    pointing forward, fingers stacked with the moving one on top = all 0; jaw shut = -10)."""
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
    f.torque_on_holding(present, args.torque_limit * 10)
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
    hits_itself = False
    for side, f in followers.items():
        print("Map in use:", f.describe())
        ticks[side], stats = convert(f, traj["arms"][side]["target"])
        print_clamps(side, f, stats)
        refuse |= any(worst > CLAMP_REFUSE_DEG for _, worst in stats.values())
        print_tips(side, traj["arms"][side])
        if not args.no_collision_check and not keeps_clear_of_itself(side, f, ticks[side]):
            hits_itself = True
    if hits_itself:
        sys.exit("The arm would pass through or too close to itself: not playing this trajectory.")
    if refuse and not args.allow_clamp:
        sys.exit(f"Clamped by more than {CLAMP_REFUSE_DEG} deg: the arm cannot do this trajectory. "
                 "Re-run with --allow-clamp to play the clamped version.")
    if args.dry_run:
        for side in followers:
            print_dry_run(side, traj["arms"][side])
            print_lowest(side, followers[side], ticks[side])
        return 0

    motors = MOTOR_IDS[:5] if args.no_gripper else MOTOR_IDS
    for f in followers.values():
        f.connect()
    present = {side: f.read_present() for side, f in followers.items()}
    for side, f in followers.items():
        f.torque_on_holding(present[side], args.torque_limit * 10)

    at = {side: {m: present[side][m] for m in motors} for side in followers}
    idle = {side: idle_goals(f, motors) for side, f in followers.items()}
    progress["step"] = "ramp to the idle pose"
    at = ramp_together(followers, at, idle)

    progress["step"] = "ramp to the first target"
    first = {side: {m: ticks[side][0][m - 1] for m in motors} for side in followers}
    at = ramp_together(followers, at, first)
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
        rows = {side: [[k] + [round(f.to_rad(m, present[side][m]), 4) for m in MOTOR_IDS] for k, _, present in log]
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

    progress["step"] = "ramp back to the idle pose"
    last = {side: {m: ticks[side][T - 1][m - 1] for m in motors} for side in followers}
    ramp_together(followers, last, idle)
    for side, f in followers.items():
        if idle[side] is not None:
            print(f"{side} ({f.name}) is back at its idle pose, torque on.")
    if args.torque_limit < 100:
        for side, f in followers.items():
            if idle[side] is None:
                print(f"{side} ({f.name}) has no idle pose to rest in: its {args.torque_limit}% torque cap stays on.")
            else:
                f.set_torque_limits(FULL_TORQUE_LIMIT)
                print(f"{side} ({f.name}): torque cap lifted, every joint is back at full torque.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
