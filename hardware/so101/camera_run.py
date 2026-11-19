"""Play a camera policy on two SO-101 followers: a network trained by imitation.train on a rig_dataset.py version,
which sees the phone's picture and both arms' joints and gives the next joint targets.

    python camera_run.py --policy ../../outputs/rig/runs/vision_v0/final.pt --dataset ../../outputs/rig/datasets/rig_v0 \
        --left arm2 --right arm1 --torque-limit 50 [--dry-run] [--log run.jsonl]

The dataset version tells it what the policy was trained on: the picture size, the action's unit, how long a row
lasted, how many rows the longest good fold took, and where the good folds began. The arms glide from their idle
poses to that beginning (replay.py's glides and its check on them), the policy then plays that many rows, and the
arms glide back.

A camera policy cannot be rolled out before the arms move, as policy_run.py rolls out the planner's copy: what it
does depends on pictures not taken yet. So everything is checked row by row, before the row is sent:
    every joint of both arms within --track-deg of the row sent before (an arm that is not where it was told)
    a picture newer than the last one the policy saw, at least every --stale seconds
    the new row finite, inside the limits, clear of itself as a pose (policy_run.row_fault)
    no joint more than replay.py's rate cap from the row before: a larger step is cut to the cap, not refused
    the fingertip, wrist and elbow above the table (the arm model; nothing else here knows where the table is)
A failed check stops the run: nothing more is sent, torque stays on, the arms hold the last row. Ctrl-C and bus
errors do the same. Not checked: the other arm, the cloth, and whether the fold is any good.

Every run is recorded in a new folder under runs/ as rig_record.py records a fold: the step log, and each picture the
policy was shown with its depth. rig_dataset.py reads it as a run played by a policy.

Needs torch and this repo's imitation package on the machine at the arms, besides what replay.py needs.
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

import arms
import policy_run
import recorder
import replay
import rig_dataset
from kinematics import Arm
from policy_net import STEP_LIMIT

MOTOR_IDS = replay.MOTOR_IDS
SIDES = rig_dataset.SIDES
FINGERTIP_FLOOR = 0.003     # m above the table the fingertip may come down to (the fold's pinch is at 0.005)
JOINT_FLOOR = 0.045         # m for the wrist and elbow joint centres: their motor bodies reach about 3 cm lower


def load_config(dataset):
    """What camera_run needs of a rig_dataset version: {"size", "action_scale", "start", "rows", "row_seconds"}."""
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    config = manifest.get("config", {})
    missing = [k for k in ("size", "action_scale", "start", "rows", "row_seconds") if k not in config]
    if missing:
        raise ValueError(f"{dataset} does not say {', '.join(missing)}: build it again with this rig_dataset.py")
    return config


def table_fault(arm, target):
    """Why the pose `target` (6 rad) would put the arm into the table, or None."""
    elbow, wrist, tip = arm.joint_points(target[:5])
    for name, point, floor in (("fingertip", tip, FINGERTIP_FLOOR), ("wrist", wrist, JOINT_FLOOR), ("elbow", elbow, JOINT_FLOOR)):
        if point[2] < floor:
            return f"the {name} would be {point[2] * 1000:.0f} mm above the table (floor {floor * 1000:.0f})"
    return None


def start_rows(config, followers):
    """The glide target before the policy plays, as {side: {motor id: ticks}}, checked as a row of a fold is.
    ValueError says why it cannot be used."""
    collision, arm, out = replay.load_collision(), Arm(), {}
    for k, (side, f) in enumerate(followers.items()):
        target = np.array(config["start"][6 * k:6 * k + 6], float)
        ticks, stats = replay.convert(f, [target.tolist()])
        worst = max(w for _, w in stats.values())
        if worst > replay.CLAMP_REFUSE_DEG:
            raise ValueError(f"the {side} arm's start is {worst:.1f} deg outside its limits")
        out[side] = policy_run.pose(ticks[0])
        distance, _, parts = replay.nearest_to_itself(collision, f, [out[side]], out[side][replay.GRIPPER_ID])
        if distance < collision.CLEARANCE:
            raise ValueError(f"the {side} arm's start comes within {distance * 1000:.0f} mm of itself ({parts})")
        reason = table_fault(arm, target)
        if reason:
            raise ValueError(f"the {side} arm's start: {reason}")
    return out


def closed_loop(followers, policy, phone, config, first, options, progress, log, keep=None):
    """The policy's rows, each from the joints read just before it and the phone's latest picture. `keep(frame)`,
    if given, gets every picture the policy is shown. Returns (0, the last rows sent) when all rows were played,
    (1, the last rows sent) when a check stopped it."""
    tick = config["row_seconds"]
    sends = max(1, math.ceil(tick / replay.TICK_SECONDS - 1e-9))
    collision, arm = replay.load_collision(), Arm()
    command = {side: dict(first[side]) for side in followers}
    history, before, chunk, seen, fresh = deque(maxlen=policy.obs_horizon), None, None, None, time.monotonic()
    t0 = time.monotonic()

    def stop(k, fault, present):
        log.step("stopped", k, command, present)
        log.write({"phase": "stopped", "row": k, "reason": fault})
        print(f"\nStopped at row {k}: {fault}. Nothing more sent; torque left on, the arms hold the last row sent.")
        return 1, command

    for k in range(config["rows"]):
        row_start = t0 + k * tick
        replay.sleep_until(row_start)
        progress["step"] = k
        present = {side: f.read_present() for side, f in followers.items()}
        now = time.monotonic()
        measured = {side: policy_run.angles(f, [[present[side][m] for m in MOTOR_IDS]])[0] for side, f in followers.items()}
        sent = {side: policy_run.angles(f, [[command[side][m] for m in MOTOR_IDS]])[0] for side, f in followers.items()}
        for side in followers:
            off = np.degrees(np.abs(measured[side] - sent[side]))
            if off.max() > options.track_deg:
                return stop(k, f"the {side} arm's {replay.JOINTS[int(np.argmax(off))]} is {off.max():.1f} deg from the "
                               f"row sent before (limit {options.track_deg:g})", present)
        row, before = rig_dataset.proprio_row(arm, now, np.concatenate([measured[side] for side in SIDES]), before)
        history.append(row)
        if k % options.replan == 0:
            frame = phone.latest()
            count = getattr(phone, "count", None)       # rgbd.Phone counts the frames it was sent
            if frame is not None and (count is None or count != seen):
                seen, fresh = count, now
            if frame is None or now - fresh > options.stale:
                return stop(k, f"no new picture from the phone for {now - fresh:.1f} s", present)
            if keep is not None:
                keep(frame)
            obs = np.stack([history[0]] * (policy.obs_horizon - len(history)) + list(history))[None]
            chunk = np.asarray(policy.predict(obs, [{"main": rig_dataset.squash(frame.rgb, config["size"])}]))[0]
        action = chunk[k % options.replan] * config["action_scale"]
        new = {}
        for i, (side, f) in enumerate(followers.items()):
            target = np.asarray(action[6 * i:6 * i + 6], float)
            if np.all(np.isfinite(target)):             # a step over the rate cap is cut to it, as replay.py's play does
                target = sent[side] + np.clip(target - sent[side], -STEP_LIMIT, STEP_LIMIT)
            reason, new[side] = policy_run.row_fault(f, target, command[side], collision)
            reason = reason or table_fault(arm, target)
            if reason:
                return stop(k, f"the {side} arm's next row: {reason}", present)
        late = time.monotonic() - row_start
        if late > replay.TICK_SECONDS:      # a slow read or a slow network: the rest starts later, at its usual pace
            t0 += late
            row_start += late
        log.step("fold", k, new, present)
        for j in range(1, sends + 1):
            replay.sleep_until(row_start + (j - 1) * tick / sends)
            for side, f in followers.items():
                f.send_goals({m: command[side][m] + (new[side][m] - command[side][m]) * j / sends for m in MOTOR_IDS})
        command = new
    return 0, command


def open_log(path, options, followers, config):
    """The run's StepLog at `path`, with what was played beside it."""
    log = replay.StepLog(path, followers, None)
    meta = {"started": datetime.now().isoformat(), "policy": str(options.policy), "dataset": str(options.dataset),
            "camera_policy": True, "rows": config["rows"], "row_seconds": config["row_seconds"],
            "torque_limit": options.torque_limit, "track_deg": options.track_deg, "replan": options.replan,
            "arms": {side: {"name": f.name, "zero": [f.zero[m] for m in MOTOR_IDS],
                            "sign": [f.sign[m] for m in MOTOR_IDS]} for side, f in followers.items()},
            "joints": replay.JOINTS}
    (path.parent / "policy_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"Logging every step to {path}")
    return log


def play(followers, policy, phone, config, first, options, progress):
    """Everything that moves the arms: the glides, the policy's rows, the glide back."""
    for f in followers.values():
        f.connect()
    present = {side: f.read_present() for side, f in followers.items()}
    idle = {side: replay.idle_goals(f, MOTOR_IDS) for side, f in followers.items()}
    for side, f in followers.items():
        label, goal = ("to the start", first[side]) if idle[side] is None else ("to the idle pose", idle[side])
        clear, distance, parts = replay.glide_keeps_clear(f, present[side], goal, present[side][replay.GRIPPER_ID])
        if not clear:
            sys.exit(f"The {side} arm's glide {label} comes within {distance * 1000:.0f} mm of itself ({parts}): not "
                     "moving. Pose the arm by hand nearer its idle pose and run again.")
    # recorder.py's folder for the pictures; the step log is replay.py's, written into it (as a panel fold's is)
    run = recorder.Recorder(replay.RUNS_DIR, meta={"task": "half_fold", "command": "camera_run.py"})
    keep = lambda frame: run.frame(time.time(), frame.rgb, frame.depth, frame.K)                    # noqa: E731
    log = open_log(Path(options.log) if options.log else run.path / "steps.jsonl", options, followers, config)
    try:
        for side, f in followers.items():
            f.torque_on_holding(present[side], options.torque_limit * 10)
        progress["step"] = "glide to the idle pose"
        at = replay.ramp_together(followers, present, idle, log, "glide to the idle pose")
        progress["step"] = "glide to the start"
        replay.ramp_together(followers, at, first, log, "glide to the start")
        input("At the start. Check the arms are clear and the phone sees the cloth, then press Enter (Ctrl-C stops) ")
        code, last = closed_loop(followers, policy, phone, config, first, options, progress, log, keep)
        if code:
            return code
        time.sleep(replay.SETTLE_SECONDS)
        for side, f in followers.items():
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
        run.close()
        print(f"Recorded in {run.path}. Say whether it folded: rig_dataset.py label {run.path} good|bad")


def run(policy, config, followers, phone, options):
    """Play `policy` (predict(obs history [1, H, 139], [{"main": uint8 [3, S, S]}]) -> [1, K, 12]) trained on the
    dataset whose `config` this is, on `followers` {"left": replay.Follower, "right": ...}, with `phone`'s
    pictures (latest()). `options`: dry_run, torque_limit, track_deg, stale, replan, log, policy, dataset. Exits
    (SystemExit) when a check refuses before any port opens; returns 0 when every row was played (or, dry, the
    checks passed), 1 when it stopped on the arms."""
    cameras = [(name, int(size)) for name, size in policy.cameras]
    if cameras != [("main", int(config["size"]))]:
        sys.exit(f"The policy looks through {cameras}; this dataset holds the phone's picture as ('main', {config['size']}).")
    if not 1 <= options.replan <= policy.chunk:
        sys.exit(f"--replan must be 1 to {policy.chunk}, the rows the policy gives at a time")
    try:
        first = start_rows(config, followers)
    except ValueError as err:
        sys.exit(f"Not playing: {err}.")
    for side, f in followers.items():
        idle = replay.idle_goals(f, MOTOR_IDS)
        if idle is not None:
            clear, distance, parts = replay.glide_keeps_clear(f, idle, first[side], idle[replay.GRIPPER_ID])
            if not clear:
                sys.exit(f"Not playing: the {side} arm's glide from the idle pose to the start comes within "
                         f"{distance * 1000:.0f} mm of itself ({parts}).")
    print(f"{config['rows']} rows of {config['row_seconds']:.3f} s, {config['rows'] * config['row_seconds']:.1f} s; "
          f"a new picture every {options.replan} rows")
    if options.dry_run:
        print("Dry run: the start and the glides to it pass. No port opened. What the policy does on the arms is "
              "checked row by row as it plays.")
        return 0
    progress = {"step": "before playing"}
    try:
        return play(followers, policy, phone, config, first, options, progress)
    except (KeyboardInterrupt, OSError) as err:
        reason = "Ctrl-C" if isinstance(err, KeyboardInterrupt) else f"bus error: {err}"
        print(f"\nStopped ({reason}) at row {progress['step']}. No more targets sent; torque left on, arms hold.")
        if options.torque_limit < 100:
            print(f"A {options.torque_limit}% torque cap was asked for and may be on some or all motors. "
                  "The next run or a power cycle sets every motor again.")
        return 1


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="final.pt of an imitation.train --policy vision run")
    p.add_argument("--dataset", required=True, help="the rig_dataset.py version it was trained on (ROOT/VERSION)")
    p.add_argument("--left", required=True, help="the left arm's name in arms.json")
    p.add_argument("--right", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--torque-limit", type=replay.torque_arg, default=100, metavar="PERCENT",
                   help="cap every joint's torque, so a collision stalls gently; lifted when the run ends at the idle "
                        "pose, left on after a stop")
    p.add_argument("--track-deg", type=float, default=8.0,
                   help="most a measured joint may be from the row sent before; further stops the run")
    p.add_argument("--stale", type=float, default=1.0, help="s without a new picture that stops the run")
    p.add_argument("--replan", type=int, default=4, help="rows played from each look at the picture")
    p.add_argument("--device", type=int, default=0, help="Record3D device index")
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
    try:
        config = load_config(args.dataset)
    except (OSError, ValueError) as err:
        sys.exit(f"Bad dataset: {err}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from imitation.policies.common import load_policy
    policy = load_policy(args.policy, "cpu")
    followers = {"left": replay.Follower(args.left), "right": replay.Follower(args.right)}
    phone = None
    if not args.dry_run:
        import rgbd
        phone = rgbd.Phone(args.device)
        phone.start()
        phone.wait()
    return run(policy, config, followers, phone, args)


if __name__ == "__main__":
    sys.exit(main())
