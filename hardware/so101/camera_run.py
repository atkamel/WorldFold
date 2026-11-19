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
    a picture the phone sent after the last look, at least every --stale seconds, of the size the policy learned on
    the new row finite, inside the limits, clear of itself as a pose (policy_run.row_fault)
    no joint more than replay.py's rate cap from the row before: a larger step is cut to the cap, not refused
    no part of the arm beyond the shoulder further into the table than the planned pinch presses (the arm model's
        meshes at the pose that will be sent, open jaw included; nothing else here knows where the table is)
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

import mujoco
import numpy as np

import arms
import policy_run
import recorder
import replay
import rig_dataset
from kinematics import Arm
from policy_net import DT, STEP_LIMIT

MOTOR_IDS = replay.MOTOR_IDS
SIDES = rig_dataset.SIDES
TABLE_FLOOR = -0.003        # m: how far below the model's table any part may be asked to go. The planned pinch
                            # comes within 1 mm of it either way; a pose 3 mm into it is not a pinch


def load_config(dataset):
    """What camera_run needs of a rig_dataset version: {"size", "action_scale", "start", "rows", "row_seconds",
    "frame_shape"}. ValueError when the version does not say, or says something a run cannot be built on."""
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    config = manifest.get("config", {})
    missing = [k for k in ("size", "action_scale", "start", "rows", "row_seconds", "frame_shape") if k not in config]
    if missing:
        raise ValueError(f"{dataset} does not say {', '.join(missing)}: build it again with this rig_dataset.py")
    start = np.asarray(config["start"], float)
    if start.shape != (12,) or not np.all(np.isfinite(start)):
        raise ValueError(f"its start is not 12 joint angles: {config['start']}")
    if not isinstance(config["rows"], int) or config["rows"] < 1:
        raise ValueError(f"its rows is {config['rows']!r}")
    if not config["row_seconds"] >= DT:
        raise ValueError(f"a row of {config['row_seconds']!r} s is shorter than the arms' {DT} s: the rate cap is per row")
    if config["action_scale"] != rig_dataset.ACTION_SCALE:
        raise ValueError(f"its actions are in units of {config['action_scale']} rad, this code's in {rig_dataset.ACTION_SCALE}")
    return config


class Body:
    """The arm's solid parts beyond the shoulder (collision.py's model), for how low a pose reaches."""

    def __init__(self):
        self.model, self.data, _ = replay.load_collision().load()
        self.parts, base = [], []
        for g in range(self.model.ngeom):
            if self.model.geom_contype[g] == 0 or self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = self.model.geom_dataid[g]
            first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            name = self.model.body(self.model.geom_bodyid[g]).name
            (base if name == "base" else self.parts).append((g, name, self.model.mesh_vert[first:first + count].copy()))
        self.table = min(self.lowest_of(np.zeros(6), base)[0], 0.0)     # the base stands on the table

    def lowest_of(self, q, parts):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        return min(((vertices @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g])[:, 2].min(), name)
                   for g, name, vertices in parts if name != "shoulder")

    def table_fault(self, q):
        """Why the pose `q` (6 rad, jaw included) would put the arm into the table, or None."""
        z, name = self.lowest_of(q, self.parts)
        if z - self.table < TABLE_FLOOR:
            return f"the {name.replace('_so101_v1', '').replace('_', ' ')} would be {(self.table - z) * 1000:.0f} mm into the table"
        return None


def start_rows(config, followers, body):
    """The glide target before the policy plays, as {side: {motor id: ticks}}, checked as a row of a fold is.
    ValueError says why it cannot be used."""
    collision, out = replay.load_collision(), {}
    for side, f in followers.items():
        k = SIDES.index(side)
        target = np.array(config["start"][6 * k:6 * k + 6], float)
        ticks, stats = replay.convert(f, [target.tolist()])
        worst = max(w for _, w in stats.values())
        if worst > replay.CLAMP_REFUSE_DEG:
            raise ValueError(f"the {side} arm's start is {worst:.1f} deg outside its limits")
        out[side] = policy_run.pose(ticks[0])
        distance, _, parts = replay.nearest_to_itself(collision, f, [out[side]], out[side][replay.GRIPPER_ID])
        if distance < collision.CLEARANCE:
            raise ValueError(f"the {side} arm's start comes within {distance * 1000:.0f} mm of itself ({parts})")
        reason = body.table_fault(policy_run.angles(f, [[out[side][m] for m in MOTOR_IDS]])[0])
        if reason:
            raise ValueError(f"the {side} arm's start: {reason}")
    return out


def closed_loop(followers, policy, phone, config, first, body, options, progress, log, keep=None):
    """The policy's rows, each from the joints read just before it and a picture the phone sent since the last
    look. `keep(frame)`, if given, gets every picture the policy is shown. Returns (0, the last rows sent) when all
    rows were played, (1, the last rows sent) when a check stopped it: then nothing more has been sent."""
    tick = config["row_seconds"]
    sends = max(1, math.ceil(tick / replay.TICK_SECONDS - 1e-9))
    collision, arm = replay.load_collision(), Arm()
    command = {side: dict(first[side]) for side in SIDES}
    history, before, chunk = deque(maxlen=policy.obs_horizon), None, None
    phone.wait()                    # a picture taken now: the last one may be from before the cloth was laid down
    seen, fresh = phone.count, time.monotonic()
    t0 = time.monotonic()

    def stop(k, fault, present):
        log.step("stopped", k, command, present)
        log.write({"phase": "stopped", "row": k, "reason": fault})
        print(f"\nStopped at row {k}: {fault}. Nothing more sent; torque left on, the arms hold the last row sent.")
        if options.torque_limit < 100:
            print(f"The {options.torque_limit}% torque cap stays on the motors until the next run or a power cycle.")
        return 1, command

    for k in range(config["rows"]):
        row_start = t0 + k * tick
        replay.sleep_until(row_start)
        progress["step"] = k
        present = {side: followers[side].read_present() for side in SIDES}
        now, read_at = time.monotonic(), time.time()
        angles = lambda side, pose: policy_run.angles(followers[side], [[pose[m] for m in MOTOR_IDS]])[0]   # noqa: E731
        measured = {side: angles(side, present[side]) for side in SIDES}
        sent = {side: angles(side, command[side]) for side in SIDES}
        for side in SIDES:
            off = np.degrees(np.abs(measured[side] - sent[side]))
            if off.max() > options.track_deg:
                return stop(k, f"the {side} arm's {replay.JOINTS[int(np.argmax(off))]} is {off.max():.1f} deg from the "
                               f"row sent before (limit {options.track_deg:g})", present)
        row, before = rig_dataset.proprio_row(arm, now, np.concatenate([measured[side] for side in SIDES]), before)
        history.append(row)
        if k % options.replan == 0:
            frame = phone.latest()
            if frame is not None and phone.count != seen:
                seen, fresh = phone.count, now
            if frame is None or now - fresh > options.stale:
                return stop(k, f"no new picture from the phone for {now - fresh:.1f} s", present)
            if list(np.shape(frame.rgb)) != [*config["frame_shape"], 3]:
                return stop(k, f"the phone's picture is {list(np.shape(frame.rgb))}, the policy learned on "
                               f"{[*config['frame_shape'], 3]} (the phone turned, or another camera mode)", present)
            if keep is not None:
                keep(frame)
            obs = np.stack([history[0]] * (policy.obs_horizon - len(history)) + list(history))[None]
            try:
                chunk = np.asarray(policy.predict(obs, [{"main": rig_dataset.squash(frame.rgb, config["size"])}]))[0]
            except Exception as err:            # whatever the network or the picture did: the arms must not be left to it
                return stop(k, f"the policy failed on this picture ({type(err).__name__}: {err})", present)
        action = np.asarray(chunk[k % options.replan], float) * config["action_scale"]
        new = {}
        for side in SIDES:
            i = SIDES.index(side)
            target = action[6 * i:6 * i + 6]
            if np.all(np.isfinite(target)):     # a step over the rate cap is cut to it, as replay.py's play does
                target = sent[side] + np.clip(target - sent[side], -STEP_LIMIT, STEP_LIMIT)
            try:
                reason, new[side] = policy_run.row_fault(followers[side], target, command[side], collision)
                # the pose that will be sent: row_fault's limits may have moved it up to CLAMP_REFUSE_DEG
                reason = reason or body.table_fault(angles(side, new[side]))
            except SystemExit as err:           # replay.py's collision check gives up this way
                reason = f"the self-collision check failed ({err})"
            if reason:
                return stop(k, f"the {side} arm's next row: {reason}", present)
        late = time.monotonic() - row_start
        if late > replay.TICK_SECONDS:      # a slow read or a slow network: the rest starts later, at its usual pace
            t0 += late
            row_start += late
        # replay.StepLog's line, with when the joints were read: the line itself is written after the policy has
        # thought, and rig_dataset.py must give a later policy the rates this one saw
        log.write({"phase": "fold", "row": k, "read_t": read_at,
                   "target": {side: log.angles(followers[side], new[side]) for side in SIDES},
                   "measured": {side: log.angles(followers[side], present[side]) for side in SIDES}})
        for j in range(1, sends + 1):
            replay.sleep_until(row_start + (j - 1) * tick / sends)
            for side in SIDES:
                followers[side].send_goals({m: command[side][m] + (new[side][m] - command[side][m]) * j / sends
                                            for m in MOTOR_IDS})
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


def play(followers, policy, phone, config, first, body, options, progress):
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
        code, last = closed_loop(followers, policy, phone, config, first, body, options, progress, log, keep)
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
        try:
            run.close()
        except Exception as err:    # a picture that could not be written must not hide how the run ended
            print(f"The recording lost pictures ({err}); the step log is whole.")
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
    if list(followers) != list(SIDES):
        sys.exit(f"followers must be {list(SIDES)}, in that order")
    if not options.dry_run and not (hasattr(phone, "count") and hasattr(phone, "wait")):
        sys.exit("The phone must count the frames it is sent (rgbd.Phone): without that a stalled camera cannot be told.")
    body = Body()
    try:
        first = start_rows(config, followers, body)
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
        return play(followers, policy, phone, config, first, body, options, progress)
    except (KeyboardInterrupt, OSError) as err:
        reason = "Ctrl-C" if isinstance(err, KeyboardInterrupt) else f"bus or disk error: {err}"
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
