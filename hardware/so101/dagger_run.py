"""The camera fold with a learned policy at the joints, corrected by the scripted fold while it runs (DAgger).

    .venv-hw/bin/python hardware/so101/dagger_run.py [--trust-deg DEG] [--policy WEIGHTS.npz] [--no-train] \
        --left arm2 --right arm1 --speed 0.7 --torque-limit 50 [anything else `fold_camera.py fold` takes]

This runs `fold_camera.py fold` itself, as it is: the look, the plan, look.png, its questions, then replay.py with
its checks, gains, torque cap, glides and log. No file of the scripted fold is changed. Two of its functions are
stood in for while it runs, in this process only:

  fold_traj.fold    is called as before; what it was given (each arm's pinch and place, the pinch heights) is
                    kept, because those are the network's inputs.
  replay.play       the loop that sends the plan's rows. Here, every row and for each arm, the network
                    (dagger_net.py) reads the arm's joints and proposes the row. Its row is sent when every
                    joint is within --trust-deg of the plan's row, the fingertip is no more than LOWER under
                    the plan's (the plan's pinch heights are where the table is, so lower is into the table),
                    and the pose is as clear of itself as replay.py asks of a plan; otherwise the plan's row is
                    sent. Either way the row goes through replay.py's rate cap and is reached in the same equal
                    moves. The last row is the plan's, and rows are added until the arms are on it, so the glide
                    back starts where replay.py expects.

So the arms never leave a band of --trust-deg around the scripted fold, and --trust-deg 0 is the scripted fold
with the network only watching. The scripted fold drives every row, and nothing is learned from the run, when the
plan is one the network was not trained for (--grasp-check, another --closed, --no-gripper).

Without --trust-deg the network has to earn the fold. The scripted fold drives (a band of 0) until, on the fold
before, the network's rows were within TRUST_DEG of the plan's on at least EARNED of the rows; then the band is
TRUST_DEG. A fold where it falls under that share hands the next fold back to the scripted fold. So the first fold
with a new store is always the scripted fold.

The phone's pictures and depth are saved through the fold, as a panel fold saves them (rig_record.py): this script
opens the phone, takes the picture for the look from it (handed to fold_camera.py with --image, as the panel does),
and records into the run folder beside steps.jsonl, then asks whether the cloth folded. With --image, --log or
--dry-run given, the fold runs as fold_camera.py runs it and no pictures are saved.

Every row is kept: what the network read, what it proposed, the plan's row, and which was sent. They are saved
to rows_<time>.npz in --store (default runs/dagger) and the network is trained on with all the rows saved there
(dagger_train.py: a few seconds), giving round_<n>.npz, which the next run starts from. Training needs the planned
folds of dagger_data.py at --base; without that file the rows are saved and the command to make it is printed.

The intervention button (dagger_nudge.py): press the Xbox button on the controller and the fold waits, with the
arms driven by the gamepad teleop (gamepad.py, its controls and its checks); press it again and the fold goes on
with each fingertip as far off the plan as it was left, an offset that is kept for the folds after it. The
corrected rows take the plan's place everywhere above: they are what the band is measured from and what the
network is taught. --no-nudge leaves the controller alone; --forget-nudges starts again from no offsets.

With --dry-run nothing moves (replay.py's own dry run); the network is then played against the plan in the arm
model, each joint taken to be on the row before, and the share of rows it would have driven is printed.
"""

import argparse
import inspect
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import dagger_net
import dagger_nudge
import fold_camera
import fold_traj
import replay
from kinematics import Arm

HERE = Path(__file__).parent
START = HERE / "policies" / "camera_fold_v0.npz"
BASE = HERE.parents[1] / "outputs" / "hardware" / "dagger" / "base.npz"
TRUST_DEG = 2.0
EARNED = 0.90           # share of the last fold's rows the network must have had inside TRUST_DEG to drive the next
LOWER = 0.003           # m: how far under the plan's fingertip the network's row may put the fingertip
MOTOR_IDS = replay.MOTOR_IDS


def latest(store):
    """The newest round_<n>.npz in `store`, or the weights the repo ships."""
    rounds = sorted(p for p in Path(store).glob("round_*.npz") if p.with_suffix(".json").exists())
    return rounds[-1] if rounds else START


def next_round(store):
    """The path of the round after every round_<n>.npz in `store`."""
    numbers = [int(p.stem.split("_")[1]) for p in Path(store).glob("round_*.npz") if p.stem.split("_")[1].isdigit()]
    return Path(store) / f"round_{max(numbers, default=0) + 1:03d}.npz"


def earned(store):
    """The share of the newest saved fold's rows on which the network's row was within TRUST_DEG of the plan's
    (whoever drove); None with no fold saved."""
    files = sorted(Path(store).glob("rows_*.npz"))
    if not files:
        return None
    rows = np.load(files[-1])
    away = np.degrees(np.abs(rows["policy"] - rows["teacher"])).max(axis=1)
    return float(np.mean(np.nan_to_num(away, nan=np.inf) <= TRUST_DEG))


def capped(wanted, before, cap):
    """(`wanted` ({motor: ticks}) with no motor more than `cap` ticks from `before`, how many were held back)."""
    goal, held = {}, 0
    for m in wanted:
        delta = wanted[m] - before[m]
        if abs(delta) > cap:
            held += 1
            delta = math.copysign(cap, delta)
        goal[m] = before[m] + delta
    return goal, held


def eased(wanted, before, share):
    """`wanted` ({motor: ticks}) with every motor only `share` of the way there from `before`."""
    goal = {}
    for m in wanted:
        goal[m] = before[m] + share * (wanted[m] - before[m])
    return goal


class Driver:
    """Stands in for fold_traj.fold (`fold`) and replay.play (`play`) during one fold."""

    def __init__(self, policy, trust_deg, plan, play, nudges=None, button=False, smooth=1.0):
        self.policy, self.trust, self.smooth = policy, trust_deg, smooth
        self.plan, self.scripted_play = plan, play
        self.nudges, self.wants_button, self.check = nudges, button, 0
        self.teleop = self.button = None
        self.inputs = None
        self.why_scripted = "the planner was never asked for a fold"
        self.rows = []          # one per arm per row played: see save_rows
        self.arm = Arm()

    def fold(self, *args, **kwargs):
        """fold_traj.fold, keeping what it was given when the network can act on it."""
        traj = self.plan(*args, **kwargs)
        self.inputs = None
        self.check = traj["marks"]["check"]
        try:
            given = inspect.signature(self.plan).bind(*args, **kwargs)
            given.apply_defaults()
            points, closed = given.arguments["points"], given.arguments["closed"]
            empty, pinch_z = given.arguments["empty_reading"], given.arguments["pinch_z"]
        except (TypeError, KeyError) as err:
            self.why_scripted = f"fold_traj.fold is not called as this file expects ({err})"
            return traj
        if abs(closed - self.policy.meta["closed"]) > 1e-9:
            self.why_scripted = (f"the jaw shuts to {closed} rad and the network was trained for "
                                 f"{self.policy.meta['closed']}")
        elif empty:
            self.why_scripted = "the plan shuts the jaw on nothing first (--grasp-check), which the network never saw"
        else:
            self.inputs = dagger_net.fold_inputs(points, pinch_z)
        return traj

    def choose(self, follower, collision, teacher, wanted, before, cap, last):
        """(the row to send {motor: ticks}, who drives it: "policy" or why the plan does, deg the network's row is
        from the plan's, motors the rate cap held back)."""
        if not np.all(np.isfinite(wanted)):
            goal, held = self.plans_step(follower, collision, teacher, before, cap)
            return goal, "not a number", float("inf"), held
        asked = dict(zip(MOTOR_IDS, replay.convert(follower, [list(map(float, wanted))])[0][0]))
        away = max(abs(asked[m] - teacher[m]) for m in teacher) / replay.STEPS_PER_DEGREE
        why = "watching" if self.trust == 0 else "last row" if last else "strayed" if away > self.trust else None
        if why is None:
            goal, held = self.step(asked, before, cap)
            pose = [follower.to_rad(m, goal[m]) for m in MOTOR_IDS]
            planned = [follower.to_rad(m, teacher[m]) for m in MOTOR_IDS]
            if self.arm.tip(pose[:5])[0][2] < self.arm.tip(planned[:5])[0][2] - LOWER:
                why = "fingertip under the plan's"
            else:
                try:
                    distance = collision.closest([pose])[0]
                except ValueError:
                    distance = -1.0
                if distance >= collision.CLEARANCE:
                    return goal, "policy", away, held
                why = "too close to itself"
        goal, held = self.plans_step(follower, collision, teacher, before, cap)
        return goal, why, away, held

    def step(self, row, before, cap):
        """(the pose sent for `row`: `smooth` of the way to it from `before`, no motor moved more than `cap`; motors
        the rate cap held back)."""
        goal, held = capped(row, before, cap / self.smooth)
        return eased(goal, before, self.smooth), held

    def plans_step(self, follower, collision, teacher, before, cap):
        """`step` for the plan's row. A smoothed pose lies off the rows the plan was checked on: one too close to
        the arm itself is not sent, the plan's own row is."""
        goal, held = self.step(teacher, before, cap)
        if self.smooth < 1.0:
            try:
                distance = collision.closest([[follower.to_rad(m, goal[m]) for m in MOTOR_IDS]])[0]
            except ValueError:
                distance = -1.0
            if distance < collision.CLEARANCE:
                return capped(teacher, before, cap)
        return goal, held

    def intervene(self, k, followers, command, motors, log):
        """While the Xbox button has the fold waiting: hold row k and let gamepad.py's loop drive the arms, 50
        times a second. Then keep where each fingertip was left as its offset. `command` ({side: {motor: ticks}})
        ends on what was sent last. Returns the seconds spent, 0 when the fold was not told to wait."""
        if not self.button.read():
            return 0.0
        started, n, shown = time.monotonic(), 0, None
        self.teleop.take(command)
        print(f"\nWaiting on row {k}: the controller has the arms, as in gamepad mode. The Xbox button again "
              "goes on with the fold.")
        while self.button.read():
            try:
                status = self.teleop.tick()
            except (ValueError, np.linalg.LinAlgError) as err:
                print(f"  The gamepad control stopped ({err}): the fold goes on.")
                self.button.waiting = False
                break
            if status != shown:
                print(f"  {status}")
                shown = status
            n += 1
            replay.sleep_until(started + n * replay.TICK_SECONDS)
        for name in followers:
            command[name] = {m: self.teleop.command(name)[m] for m in motors}
            outcome = self.nudges.take(name, k, command[name])
            if outcome == "kept":           # the rows still to come, with the offsets as they are now
                self.nudges.rebuild(name, k)
            elif outcome != "not moved":
                print(f"  The {name} arm's fingertip is {outcome}.")
        if self.nudges.changed:
            self.nudges.save()
        print(f"Going on. Offsets now: {self.nudges.describe()}.")
        if hasattr(log, "write"):
            log.write({"phase": "nudge", "row": k, "seconds": round(time.monotonic() - started, 2),
                       "offsets": {side: {phase: offset.tolist() for phase, offset in phases.items()}
                                   for side, phases in self.nudges.offsets.items()}})
        return time.monotonic() - started

    def play(self, followers, ticks, dt, speed, motors, progress, steps=None, log=None):
        """replay.play with the network proposing each row. Same arguments, same return (times the rate cap
        engaged). Rows asked for out of order (a grasp-check retry), a run without the gripper, and a plan the
        network was not trained for are played by replay.py's own loop."""
        order = list(range(len(next(iter(ticks.values()))))) if steps is None else list(steps)
        forward = all(b - a == 1 for a, b in zip(order, order[1:]))
        if self.inputs is None or not forward or not order or replay.GRIPPER_ID not in motors:
            if self.inputs is not None:
                self.why_scripted = ("the gripper is left out (--no-gripper)" if replay.GRIPPER_ID not in motors
                                     else "the rows were asked for out of order")
            return self.scripted_play(followers, ticks, dt, speed, motors, progress, steps, log)
        import collision
        first = next(iter(followers))       # build the clearance model now, not inside the first row's time
        collision.closest([[followers[first].to_rad(m, ticks[first][order[0]][m - 1]) for m in MOTOR_IDS]])
        tick = dt / speed
        cap = replay.MAX_DEG_S * tick * speed * replay.STEPS_PER_DEGREE
        sends = max(1, math.ceil(tick / replay.TICK_SECONDS - 1e-9))
        final = len(next(iter(ticks.values()))) - 1
        command = {name: {m: ticks[name][order[0]][m - 1] for m in motors} for name in followers}
        engaged = 0
        if self.nudges is not None:         # the plan's rows with the stored offsets: the rows to follow from here
            self.nudges.begin(followers, ticks, self.check, collision)
            rows = self.nudges.rows
            if self.wants_button and self.button is None:
                try:
                    self.teleop, self.button = dagger_nudge.controls(followers)
                    print("The Xbox button on the controller makes the fold wait and hands the arms to the gamepad.")
                except ValueError as err:
                    print(f"No intervention button in this fold: {err}.")
        else:
            rows = {name: [dict(zip(MOTOR_IDS, row)) for row in ticks[name]] for name in followers}
        t0 = time.monotonic()
        for i, k in enumerate(order):
            if self.button is not None:
                t0 += self.intervene(k, followers, command, motors, log)
            row_start = t0 + i * tick
            replay.sleep_until(row_start)
            progress["step"] = k
            present = {name: f.read_present() for name, f in followers.items()}
            measured = {name: np.array([f.to_rad(m, present[name][m]) for m in MOTOR_IDS])
                        for name, f in followers.items()}
            before = {name: dict(command[name]) for name in followers}
            sent = {name: np.array([f.to_rad(m, before[name][m]) for m in MOTOR_IDS]) for name, f in followers.items()}
            wanted = self.policy.targets(self.inputs, measured, sent, k * dt)
            who, away = {}, {}
            for name, f in followers.items():
                teacher = {m: rows[name][k][m] for m in motors}
                command[name], who[name], away[name], held = self.choose(
                    f, collision, teacher, wanted[name], before[name], cap, k == final)
                engaged += held
                mirror = dagger_net.MIRROR_JOINTS if name == "right" else 1.0
                self.rows.append({"obs": dagger_net.observation(self.inputs, name, measured[name], sent[name],
                                                                k * dt),
                                  "teacher": np.array([f.to_rad(m, teacher[m]) for m in MOTOR_IDS]) * mirror,
                                  "policy": np.asarray(wanted[name], float) * mirror,
                                  "drove": who[name] == "policy", "why": who[name], "row": k, "side": name})
            if hasattr(log, "step"):            # replay.StepLog: the rows as sent, then who drove them
                log.step("fold", k, command, present)
                log.write({"phase": "dagger", "row": k, "who": who,
                           "away_deg": {name: round(a, 3) if math.isfinite(a) else None for name, a in away.items()}})
            elif log is not None:
                log.append((k, time.monotonic(), present))
            for j in range(1, sends + 1):
                replay.sleep_until(row_start + (j - 1) * tick / sends)
                for name, f in followers.items():
                    f.send_goals({m: before[name][m] + (command[name][m] - before[name][m]) * j / sends
                                  for m in motors})
        if self.nudges is not None and self.nudges.changed:     # every row of this fold is taught the rows as
            for name in followers:                              # they stand now, the ones before the nudge too
                self.nudges.rebuild(name, 0)
            for r in self.rows:
                if r["side"] in followers:
                    f, mirror = followers[r["side"]], dagger_net.MIRROR_JOINTS if r["side"] == "right" else 1.0
                    r["teacher"] = np.array([f.to_rad(m, rows[r["side"]][r["row"]][m]) for m in MOTOR_IDS]) * mirror
        if order[-1] == final:              # a row the rate cap held short of the plan's last: step onto it
            last = {name: {m: ticks[name][final][m - 1] for m in motors} for name in followers}
            short = max(abs(last[name][m] - command[name][m]) for name in followers for m in motors)
            for i in range(len(order), len(order) + math.ceil(short / cap - 1e-9)):
                before = command
                command = {name: capped(last[name], before[name], cap)[0] for name in followers}
                for j in range(1, sends + 1):
                    replay.sleep_until(t0 + i * tick + (j - 1) * tick / sends)
                    for name, f in followers.items():
                        f.send_goals({m: before[name][m] + (command[name][m] - before[name][m]) * j / sends
                                      for m in motors})
        return engaged


def save_rows(store, rows, policy, trust):
    """rows_<time>.npz in `store`: per row and arm the observation, the plan's row ("teacher") and the network's
    ("policy"), both rad in the network's convention, whether the network's was sent, the row and the side."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"rows_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    part = path.with_suffix(".part")
    with open(part, "wb") as file:      # written whole or not at all: a half file would stop every later training
        np.savez_compressed(file, obs=np.array([r["obs"] for r in rows]),
                            teacher=np.array([r["teacher"] for r in rows]),
                            policy=np.array([r["policy"] for r in rows]),
                            drove=np.array([r["drove"] for r in rows]), why=np.array([r["why"] for r in rows]),
                            row=np.array([r["row"] for r in rows]), side=np.array([r["side"] for r in rows]),
                            weights=str(policy.path), trust_deg=trust)
    part.replace(path)
    return path


def report(rows, trust):
    """Print, per arm, how many rows the network drove, why the plan drove the others, and how far apart they were."""
    for side in ("left", "right"):
        mine = [r for r in rows if r["side"] == side]
        if not mine:
            continue
        off = np.degrees(np.array([r["policy"] - r["teacher"] for r in mine]))
        finite = np.where(np.isfinite(off), off, 0.0)
        worst = np.unravel_index(np.argmax(np.abs(finite)), finite.shape)
        reasons = {}
        for r in mine:
            if not r["drove"]:
                reasons[r["why"]] = reasons.get(r["why"], 0) + 1
        drove = sum(r["drove"] for r in mine)
        print(f"{side} arm: the network drove {drove} of {len(mine)} rows ({100 * drove / len(mine):.0f}%, within "
              f"{trust:g} deg of the plan); rms {np.sqrt((finite ** 2).mean()):.2f} deg from it, most "
              f"{abs(finite[worst]):.1f} deg ({replay.JOINTS[worst[1]]}, row {mine[worst[0]]['row']})"
              + ("" if not reasons else "; the plan drove: " + ", ".join(f"{n} {why}" for why, n in reasons.items())))


def rehearse(policy, inputs, traj, trust):
    """The rows a dry run would have given: the network against the plan in the arm model, every joint reaching
    the row it is sent, and the plan's row sent in place of one further than `trust` from it."""
    rows = []
    for side in ("left", "right"):
        plan = np.array(traj["arms"][side]["target"])
        mirror = dagger_net.MIRROR_JOINTS if side == "right" else 1.0
        sent = plan[0]
        for k in range(len(plan)):
            wanted = policy.targets(inputs, {side: sent}, {side: sent}, k * traj["dt"])[side]
            drove = bool(np.degrees(np.abs(wanted - plan[k])).max() <= trust)
            rows.append({"obs": dagger_net.observation(inputs, side, sent, sent, k * traj["dt"]),
                         "teacher": plan[k] * mirror, "policy": wanted * mirror, "drove": drove,
                         "why": "strayed", "row": k, "side": side})
            sent = wanted if drove else plan[k]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--policy", help="weights to drive with (default: the newest round in --store, else the repo's)")
    ap.add_argument("--trust-deg", type=float,
                    help="most a joint of the network's row may be from the plan's row and still be sent; 0 plays "
                         f"the scripted fold with the network watching. Default: {TRUST_DEG:g} once the network "
                         f"was that close on {100 * EARNED:.0f}%% of the last fold's rows, 0 until then")
    ap.add_argument("--device", type=int, default=0, help="Record3D device index")
    ap.add_argument("--no-nudge", action="store_true", help="do not read the controller's intervention button")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="share of the way to each row the arms are sent, 0.1 to 1: under 1 rounds the fold's "
                         "corners and the arms trail the rows, by one row at 0.5. Default 1, the rows as they are")
    ap.add_argument("--forget-nudges", action="store_true", help="put the stored offsets aside and start from none")
    ap.add_argument("--store", default=str(replay.RUNS_DIR / "dagger"), help="rows and rounds are kept here")
    ap.add_argument("--base", default=str(BASE), help="the planned folds the network is trained on with the rows")
    ap.add_argument("--no-train", action="store_true", help="save the rows and leave the network as it is")
    ap.add_argument("--steps", type=int, help="training steps after the fold (default: dagger_train.TUNE_STEPS)")
    args, fold_args = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not 0.1 <= args.smooth <= 1.0:
        ap.error("--smooth is 0.1 to 1")
    policy = dagger_net.Policy(args.policy or latest(args.store))
    if args.trust_deg is None:
        share = earned(args.store)
        args.trust_deg = TRUST_DEG if share is not None and share >= EARNED else 0.0
        print(f"Network: {policy.path}. " + (
            "No fold saved yet: the scripted fold drives this one and the network watches." if share is None else
            f"On the last fold its rows were within {TRUST_DEG:g} deg of the scripted fold's on {share:.0%} of rows"
            + (f": it drives this one inside {TRUST_DEG:g} deg." if args.trust_deg else
               f", under {EARNED:.0%}: the scripted fold drives this one and the network watches.")))
    elif not 0.0 <= args.trust_deg <= 5.0:
        ap.error("--trust-deg is 0 to 5")
    else:
        print(f"Network: {policy.path}; its rows are sent while within {args.trust_deg:g} deg of the scripted fold's.")
    stored = Path(args.store) / "nudges.json"
    if args.forget_nudges and stored.exists():
        stored.replace(stored.with_name(f"nudges_until_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    nudges = dagger_nudge.Nudges(stored)
    print(f"Stored offsets: {nudges.describe()}.")
    driver = Driver(policy, args.trust_deg, fold_traj.fold, replay.play, nudges, not args.no_nudge, args.smooth)
    pictures = not any(flag in fold_args for flag in ("--image", "--log", "--dry-run"))
    recorded, look = None, None
    fold_traj.fold, replay.play = driver.fold, driver.play
    try:
        if pictures:
            import rgbd
            import rig_record
            from PIL import Image
            phone = rgbd.Phone(args.device)
            phone.start()
            handle, look = tempfile.mkstemp(suffix=".png", prefix="dagger_look_")
            os.close(handle)
            Image.fromarray(phone.wait().rgb).save(look)
            code, recorded = rig_record.record(lambda argv: fold_camera.main(["fold", *argv]),
                                               ["--image", look, *fold_args], phone, replay.RUNS_DIR, input)
        else:
            code = fold_camera.main(["fold", "--device", str(args.device), *fold_args])
    finally:
        fold_traj.fold, replay.play = driver.plan, driver.scripted_play
        for part in (driver.button, driver.teleop):
            if part is not None:
                part.close()
        if look is not None:
            os.unlink(look)
        if driver.rows and nudges.changed:  # rows of earlier folds were taught other rows: kept, not trained on
            aside = Path(args.store) / f"before_nudge_{time.strftime('%Y%m%d_%H%M%S')}"
            aside.mkdir(parents=True)
            for old in Path(args.store).glob("rows_*.npz"):
                old.replace(aside / old.name)
            print(f"The offsets changed in this fold ({nudges.describe()}): the rows of the folds before it are "
                  f"put aside in {aside.name}.")
        if nudges.kept_plan:
            print(f"{nudges.kept_plan} rows could not take their offset (out of reach, a joint limit, or too "
                  "close to the arm itself) and were the plan's.")
        if driver.rows:
            saved = save_rows(args.store, driver.rows, policy, args.trust_deg)
            report(driver.rows, args.trust_deg)
            print(f"Rows saved: {saved}")
            if recorded is not None:        # which rows and weights belong to this recorded run
                (recorded / "dagger.json").write_text(json.dumps(
                    {"rows": str(saved), "weights": str(policy.path), "trust_deg": args.trust_deg}))
    if not driver.rows:
        if "--no-gripper" in fold_args:
            print("The scripted fold drives every row and nothing is learned: the gripper is left out (--no-gripper).")
        elif driver.inputs is None or "--dry-run" not in fold_args:
            print(f"The scripted fold drove every row and nothing was learned: {driver.why_scripted}.")
        else:
            print("Dry run. In the arm model, with every joint reaching the row it is sent:")
            report(rehearse(policy, driver.inputs, json.loads((fold_camera.HERE / "last_look.json").read_text()),
                            args.trust_deg), args.trust_deg)
        return code
    if args.no_train:
        return code
    if not Path(args.base).exists():
        print(f"Not trained: {args.base} is missing. Make it once (about 10 minutes, no arms needed):\n"
              f"  python hardware/so101/dagger_data.py --episodes 900 --seed 0 --out {args.base}")
        return code
    import dagger_train
    out = next_round(args.store)
    summary = dagger_train.train(args.base, out, start=policy.path, rows=args.store, steps=args.steps, quiet=True)
    was, now = summary["on_rows_from_arms_before"]["rms_deg"], summary["on_rows_from_arms"]["rms_deg"]
    print(f"Trained on {summary['rows_from_arms']} rows from the arms in {summary['seconds']} s: on those rows the "
          f"network is now {max(now.values()):.2f} deg rms from the scripted fold at its worst joint "
          f"(was {max(was.values()):.2f}). Those rows were trained on: the share of rows the network drives on the "
          f"next fold is the test. The next run starts from {out.name}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
