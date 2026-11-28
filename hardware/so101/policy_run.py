"""Play the learned half fold (policy_net.py) on two SO-101 followers, closed loop: every control row the network
reads both arms' measured joints and gives their next targets. The cloth's place comes from flags for now; run() is
what a camera front end calls with the points it finds.

    .venv-hw/bin/python hardware/so101/policy_run.py --left arm2 --right arm1 --layout behind --dry-run
    .venv-hw/bin/python hardware/so101/policy_run.py --left arm2 --right arm1 --layout sides --gap 0.16 \
        [--shift-x 0.01] [--shift-y 0] [--turn 3] [--speed 0.5] [--torque-limit 40] [--log run.jsonl]

Before any port opens, the policy is rolled out in the arm model (every joint reaching its target each row) and the
rolled-out rows get the checks replay.py gives a planned trajectory: limit clamping (refused over CLAMP_REFUSE_DEG),
self clearance, the glides from and back to the idle pose, and the rate cap (no joint more than MAX_DEG_S * DT from
one row to the next). The rollout is then held against the planner's fold of the same points (fold_traj.fold): the
fingertip where the jaw shuts and where it opens again must be within --max-error-mm of the planner's. Points the
planner refuses are refused. --dry-run stops there.

On the arms, before each row is sent: every joint of both arms within --track-deg of the row sent before (else an arm
is not where it was told: blocked, stalled or slipping), and the new row finite, inside the limits, within the rate
cap of the row before, clear of itself as a pose, and within --stay-deg of the same row of the rollout that passed
the checks above (a policy that behaves on the arms as it never did in the model is stopped, not followed). A failed check stops the run: nothing more is sent, torque stays
on, the arms hold the last row sent. Ctrl-C and bus errors do the same, as in replay.py. Every row is logged (StepLog
lines: the row sent and the joints read before it).

Needs mujoco and so101_nexus (the planner, the self-clearance checks) besides the servo SDK and numpy.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import arms
import fold_traj
import policy_data
import policy_net
import replay
from kinematics import Arm
from policy_net import DT

MOTOR_IDS = replay.MOTOR_IDS
RATE_DEG = replay.MAX_DEG_S * DT        # most a joint may move from one row to the next (deg)
POLICY = Path(__file__).parent / "policies" / "half_fold_v0.npz"


def rollout(policy, points, tau=0.0, offset=None):
    """The policy played in the arm model: ({side: commanded rows [policy.rows, 6]}, {side: the joints each row's
    observation read [policy.rows, 6]}), rad. Row 0 reads START; the glide then leaves the arm on row 0. With tau = 0
    every joint reaches its target within the row; with tau (s) it follows it as a first-order lag. `offset`
    {side: 6 rad} is added to every reading."""
    sides = list(points)
    reading = {side: policy_net.START for side in sides}
    command, pose, rows, read = None, None, {side: [] for side in sides}, {side: [] for side in sides}
    for k in range(policy.rows):
        command = policy.next_targets(points, reading, k * DT, None if k == 0 else command)
        for side in sides:
            read[side].append(reading[side])
            rows[side].append(command[side])
        if pose is None or tau == 0:
            pose = command
        else:
            pose = {side: command[side] + (pose[side] - command[side]) * math.exp(-DT / tau) for side in sides}
        reading = {side: pose[side] + (0.0 if offset is None else offset[side]) for side in sides}
    return {side: np.array(rows[side]) for side in sides}, {side: np.array(read[side]) for side in sides}


def grasp(rows):
    """(row where the jaw shuts, last row before it opens again): the one run of rows whose jaw is under
    replay.GRASP_RAD. ValueError if the jaw never shuts, shuts twice, or is still shut at the end."""
    shut = np.flatnonzero(np.asarray(rows)[:, 5] < replay.GRASP_RAD)
    if len(shut) == 0:
        raise ValueError("the jaw never shuts")
    if shut[-1] - shut[0] + 1 != len(shut):
        raise ValueError("the jaw shuts more than once")
    if shut[-1] == len(rows) - 1:
        raise ValueError("the jaw never opens again")
    return int(shut[0]), int(shut[-1])


def tip_errors(rows, plan):
    """(m at the pinch, m at the release): how far the fingertip of `rows` is from the planner's (`plan`) where each
    one's jaw shuts and where it opens again. ValueError for rows whose jaw does not shut once and open again."""
    arm = Arm()
    return tuple(float(np.linalg.norm(arm.tip(rows[a][:5])[0] - arm.tip(plan[b][:5])[0]))
                 for a, b in zip(grasp(rows), grasp(plan)))


def pose(row):
    """A row of 6 ticks as {motor id: ticks}."""
    return {m: row[m - 1] for m in MOTOR_IDS}


def angles(follower, tick_rows):
    """Rows of ticks as rows of the 6 joint angles (rad) the map gives."""
    return np.array([[follower.to_rad(m, row[m - 1]) for m in MOTOR_IDS] for row in tick_rows])


def check(rows, plan, followers, max_error_mm):
    """Print replay.py's checks of the rolled-out rows ({side: [R, 6] rad}) and how far they miss the planner's pinch
    and release (`plan` {side: [T, 6] rad}). Returns (the reasons to refuse, none to play; {side: tick rows})."""
    reasons, ticks = [], {}
    for side, f in followers.items():
        print("Map in use:", f.describe())
        if not np.all(np.isfinite(rows[side])):         # convert() would clamp a NaN to a limit without a word
            reasons.append(f"the {side} arm's rows hold a value that is not a number")
            continue
        ticks[side], stats = replay.convert(f, rows[side].tolist())
        replay.print_clamps(side, f, stats)
        if any(worst > replay.CLAMP_REFUSE_DEG for _, worst in stats.values()):
            reasons.append(f"the {side} arm is clamped by more than {replay.CLAMP_REFUSE_DEG} deg")
        if not replay.keeps_clear_of_itself(side, f, ticks[side], True):
            reasons.append(f"the {side} arm would pass through or too close to itself")
        step = np.abs(np.diff(ticks[side], axis=0)).max(axis=0) / replay.STEPS_PER_DEGREE
        print(f"  {side} largest step from one row to the next: {step.max():.2f} deg "
              f"({replay.JOINTS[int(np.argmax(step))]}), cap {RATE_DEG:.2f}")
        if step.max() > RATE_DEG:
            reasons.append(f"the {side} arm's {replay.JOINTS[int(np.argmax(step))]} moves {step.max():.1f} deg in one "
                           f"row, over the {RATE_DEG:.1f} deg cap")
        idle = replay.idle_goals(f, MOTOR_IDS)
        if idle is not None:
            first, last = pose(ticks[side][0]), pose(ticks[side][-1])
            for label, start, goal in (("from the idle pose to the first row", idle, first),
                                       ("from the last row back to the idle pose", last, idle)):
                clear, distance, parts = replay.glide_keeps_clear(f, start, goal, first[replay.GRIPPER_ID])
                print(f"  {side} glide {label}: nearest to itself {distance * 1000:.0f} mm ({parts})")
                if not clear:
                    reasons.append(f"the {side} arm's glide {label} comes within {distance * 1000:.0f} mm of itself")
        try:
            pinch, release = tip_errors(angles(f, ticks[side]), plan[side])
        except ValueError as err:
            reasons.append(f"the {side} arm's rows do not pinch and release once: {err}")
        else:
            print(f"  {side} fingertip against the planner: {pinch * 1000:.1f} mm where the jaw shuts, "
                  f"{release * 1000:.1f} mm where it opens (limit {max_error_mm:g} mm)")
            if max(pinch, release) * 1000 > max_error_mm:
                reasons.append(f"the {side} arm misses the planner's pinch or release by "
                               f"{max(pinch, release) * 1000:.1f} mm")
        replay.print_lowest(side, f, ticks[side])
    return reasons, ticks


def row_fault(follower, target, previous, collision):
    """(why `target` (6 rad) must not be sent after `previous` ({motor id: ticks}), or None; its ticks)."""
    if not np.all(np.isfinite(target)):
        return "a target that is not a number", None
    ticks, stats = replay.convert(follower, [target.tolist()])
    goal = pose(ticks[0])
    for m, name in zip(MOTOR_IDS, replay.JOINTS):
        if stats[m][1] > replay.CLAMP_REFUSE_DEG:
            return f"{name} {stats[m][1]:.1f} deg outside its limits", goal
    for m, name in zip(MOTOR_IDS, replay.JOINTS):
        step = abs(goal[m] - previous[m]) / replay.STEPS_PER_DEGREE
        if step > RATE_DEG:
            return f"{name} would move {step:.1f} deg in one row, over the {RATE_DEG:.1f} deg cap", goal
    distance, _, parts = replay.nearest_to_itself(collision, follower, [goal], goal[replay.GRIPPER_ID])
    if distance < collision.CLEARANCE:
        return f"the pose comes within {distance * 1000:.0f} mm of itself ({parts})", goal
    return None, goal


def closed_loop(followers, policy, points, first, checked, options, progress, log):
    """Rows 1 on, each from the joints read just before it. `checked` {side: rows [policy.rows, 6] rad} is the
    rollout the dry-run checks passed. Returns (0, the last rows sent) when the fold is played, (1, the last rows
    sent) when a check stopped it: then nothing more has been sent."""
    tick = DT / options.speed
    sends = max(1, math.ceil(tick / replay.TICK_SECONDS - 1e-9))   # a row is reached in this many moves, as in play
    collision = replay.load_collision()
    command = {side: dict(first[side]) for side in followers}
    t0 = time.monotonic()
    for k in range(1, policy.rows):
        row_start = t0 + (k - 1) * tick
        replay.sleep_until(row_start)
        progress["step"] = k
        present = {side: f.read_present() for side, f in followers.items()}
        late = time.monotonic() - row_start
        if late > replay.TICK_SECONDS:      # a slow read: the rest of the fold starts later, at its usual pace
            t0 += late
            row_start += late
        measured, fault = {}, None
        for side, f in followers.items():
            measured[side] = angles(f, [[present[side][m] for m in MOTOR_IDS]])[0]
            off = np.degrees(np.abs(measured[side] - angles(f, [[command[side][m] for m in MOTOR_IDS]])[0]))
            if fault is None and off.max() > options.track_deg:
                fault = (f"the {side} arm's {replay.JOINTS[int(np.argmax(off))]} is {off.max():.1f} deg from the row "
                         f"sent before (limit {options.track_deg:g})")
        new = {}
        if fault is None:
            sent = {side: angles(f, [[command[side][m] for m in MOTOR_IDS]])[0] for side, f in followers.items()}
            targets = policy.next_targets(points, measured, k * DT, sent)
            for side, f in followers.items():
                reason, new[side] = row_fault(f, targets[side], command[side], collision)
                away = np.degrees(np.abs(targets[side] - checked[side][k]))
                if reason is None and away.max() > options.stay_deg:
                    reason = (f"{replay.JOINTS[int(np.argmax(away))]} is {away.max():.1f} deg from the row the dry "
                              f"run checked (limit {options.stay_deg:g})")
                if reason is not None and fault is None:
                    fault = f"the {side} arm's next row: {reason}"
        if fault is not None:
            log.step("stopped", k, command, present)
            log.write({"phase": "stopped", "row": k, "reason": fault})
            print(f"\nStopped at row {k}: {fault}. Nothing more sent; torque left on, the arms hold the last row sent.")
            return 1, command
        log.step("fold", k, new, present)
        for j in range(1, sends + 1):
            replay.sleep_until(row_start + (j - 1) * tick / sends)
            for side, f in followers.items():
                f.send_goals({m: command[side][m] + (new[side][m] - command[side][m]) * j / sends for m in MOTOR_IDS})
        command = new
    return 0, command


def open_log(options, followers, points):
    """The run's StepLog in --log's place or a new folder under replay.RUNS_DIR, with what was played beside it."""
    path = Path(options.log) if options.log else replay.new_run_folder() / "steps.jsonl"
    log = replay.StepLog(path, followers, None)
    meta = {"started": datetime.now().isoformat(), "policy": str(options.policy),
            "points": {side: np.asarray(p, float).tolist() for side, p in points.items()},
            "speed": options.speed, "torque_limit": options.torque_limit, "track_deg": options.track_deg,
            "arms": {side: {"name": f.name, "zero": [f.zero[m] for m in MOTOR_IDS],
                            "sign": [f.sign[m] for m in MOTOR_IDS]} for side, f in followers.items()},
            "joints": replay.JOINTS}
    (path.parent / "policy_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"Logging every step to {path}")
    return log


def play(points, followers, policy, options, ticks, checked, progress):
    """Everything that moves the arms: the glides, the closed-loop fold, the glide back."""
    first = {side: pose(ticks[side][0]) for side in followers}
    for f in followers.values():
        f.connect()
    present = {side: f.read_present() for side, f in followers.items()}
    idle = {side: replay.idle_goals(f, MOTOR_IDS) for side, f in followers.items()}
    for side, f in followers.items():
        label, goal = ("to the first row", first[side]) if idle[side] is None else ("to the idle pose", idle[side])
        clear, distance, parts = replay.glide_keeps_clear(f, present[side], goal, present[side][replay.GRIPPER_ID])
        if not clear:
            sys.exit(f"The {side} arm's glide {label} comes within {distance * 1000:.0f} mm of itself ({parts}): not "
                     "moving. Pose the arm by hand nearer its idle pose and run again.")
    log = open_log(options, followers, points)
    try:
        for side, f in followers.items():
            f.torque_on_holding(present[side], options.torque_limit * 10)
        progress["step"] = "glide to the idle pose"
        at = replay.ramp_together(followers, present, idle, log, "glide to the idle pose")
        progress["step"] = "glide to the first row"
        replay.ramp_together(followers, at, first, log, "glide to the first row")
        input("At the first row. Check the arms are clear, then press Enter to fold (Ctrl-C stops) ")
        code, last = closed_loop(followers, policy, points, first, checked, options, progress, log)
        if code:
            return code
        time.sleep(replay.SETTLE_SECONDS)
        for side, f in followers.items():
            now = f.read_present()
            print(f"{side} final (deg): " + "  ".join(f"{name} {math.degrees(f.to_rad(m, now[m])):.1f}"
                                                     for m, name in zip(MOTOR_IDS, replay.JOINTS)))
            if idle[side] is not None:
                clear, distance, parts = replay.glide_keeps_clear(f, last[side], idle[side], last[side][replay.GRIPPER_ID])
                if not clear:
                    print(f"The {side} arm's glide back to the idle pose would come within {distance * 1000:.0f} mm of "
                          f"itself ({parts}): not gliding back. Torque left on, the arms hold the last row.")
                    return 1
        progress["step"] = "glide back to the idle pose"
        replay.ramp_together(followers, last, idle, log, "glide back to the idle pose")
        for side, f in followers.items():
            if idle[side] is not None:
                print(f"{side} ({f.name}) is back at its idle pose, torque on.")
                if options.torque_limit < 100:
                    f.set_torque_limits(replay.FULL_TORQUE_LIMIT)
            elif options.torque_limit < 100:
                print(f"{side} ({f.name}) has no idle pose to rest in: its {options.torque_limit}% torque cap stays on.")
        return 0
    finally:
        log.close()


def run(points, followers, policy, options):
    """The fold for the cloth's `points` ({"left": (pinch, place), "right": ...}, each arm's base frame, m) on
    `followers` ({"left": replay.Follower, "right": ...}) with `policy` (policy_net.Policy). `options` holds
    dry_run, speed, torque_limit, max_error_mm, track_deg, stay_deg, log and policy (the weights' path, for the log). Exits
    (SystemExit) when a check refuses the fold before any port opens; returns 0 when it is played (or, dry, passes
    the checks), 1 when it stopped on the arms."""
    for side in ("left", "right"):
        print(f"{side} arm: pinch {np.round(np.array(points[side][0]) * 100, 1)} cm, place "
              f"{np.round(np.array(points[side][1]) * 100, 1)} cm (x forward, y left of its base)")
    try:
        plan = fold_traj.fold(points)
    except ValueError as err:
        sys.exit(f"The planner refuses these points ({err}): not playing.")
    rows, _ = rollout(policy, points)
    print(f"{policy.rows} rows, {policy.rows * DT / options.speed:.1f} s at speed {options.speed}")
    plan_rows = {side: np.array(plan["arms"][side]["target"]) for side in followers}
    reasons, ticks = check(rows, plan_rows, followers, options.max_error_mm)
    if reasons:
        sys.exit("Not playing: " + "; ".join(reasons) + ".")
    if options.dry_run:
        print("Dry run: every check passed. No port opened.")
        return 0
    progress = {"step": "before playing"}
    try:
        return play(points, followers, policy, options, ticks, rows, progress)
    except (KeyboardInterrupt, OSError) as err:
        reason = "Ctrl-C" if isinstance(err, KeyboardInterrupt) else f"bus error: {err}"
        print(f"\nStopped ({reason}) at row {progress['step']}. No more targets sent; torque left on, arms hold.")
        if options.torque_limit < 100:
            print(f"A {options.torque_limit}% torque cap was asked for and may be on some or all motors. "
                  "The next run or a power cycle sets every motor again.")
        return 1


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--left", required=True, help="the left arm's name in arms.json")
    p.add_argument("--right", required=True)
    p.add_argument("--layout", required=True, choices=policy_data.LAYOUTS,
                   help="behind: side by side behind the cloth's near edge; sides: one at each side edge, facing in")
    p.add_argument("--cloth", type=float, default=policy_data.CLOTH, help="cloth side (m)")
    p.add_argument("--near", type=float, default=policy_data.NEAR, help="behind: near edge in front of the bases (m)")
    p.add_argument("--spacing", type=float, default=policy_data.SPACING, help="behind: between the base origins (m)")
    p.add_argument("--gap", type=float, default=policy_data.GAP, help="sides: base origin outside the side edge (m)")
    p.add_argument("--shift-x", type=float, default=0.0, help="cloth moved right, seen from its near edge (m)")
    p.add_argument("--shift-y", type=float, default=0.0, help="cloth moved toward its far edge (m)")
    p.add_argument("--turn", type=float, default=0.0, help="cloth turned counterclockwise from above (deg)")
    p.add_argument("--policy", default=str(POLICY))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--speed", type=replay.speed_arg, default=0.5)
    p.add_argument("--torque-limit", type=replay.torque_arg, default=100, metavar="PERCENT",
                   help="cap every joint's torque, so a collision stalls gently; lifted when the run ends at the idle "
                        "pose, left on after a stop")
    p.add_argument("--max-error-mm", type=float, default=5.0,
                   help="most the rollout's fingertip may miss the planner's at the pinch and at the release")
    p.add_argument("--track-deg", type=float, default=8.0,
                   help="most a measured joint may be from the row sent before; further stops the run")
    p.add_argument("--stay-deg", type=float, default=6.0,
                   help="most a row sent on the arms may differ from the same row of the checked rollout")
    p.add_argument("--log", metavar="FILE")
    args = p.parse_args(argv)
    for name in (args.left, args.right):
        if name not in arms.ARMS:
            p.error(f"unknown arm {name!r}; arms.json has {', '.join(arms.ARMS)}")
    if args.left == args.right:
        p.error("--left and --right name the same arm")
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    points = policy_data.cloth_points(args.layout, args.cloth, args.near, args.spacing, args.gap,
                                      (args.shift_x, args.shift_y), args.turn)
    followers = {"left": replay.Follower(args.left), "right": replay.Follower(args.right)}
    return run(points, followers, policy_net.Policy(args.policy), args)


if __name__ == "__main__":
    sys.exit(main())
