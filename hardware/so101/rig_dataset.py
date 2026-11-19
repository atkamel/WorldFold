"""Recorded rig runs as training episodes: every control row's camera picture, measured joints and the targets sent.

    python rig_dataset.py list [RUNS]                       # what each run folder holds and whether it can be used
    python rig_dataset.py label RUN good|bad [--note ...]   # did it fold: a run is only a demonstration if it did
    python rig_dataset.py build --out ROOT --version rig_v0 [RUNS] [--left arm2 --right arm1]
    python -m imitation.train --policy vision --root ROOT --dataset rig_v0 --run outputs/rig/runs/vision_v0

It reads the run folders the rig already writes, whoever moved the arms:
    a fold played by replay.py (the panel's fold, fold_camera.py, rig_record.py replay): steps.jsonl rows with a
        "phase", joint angles in radians, the planned row and the joints read just before it was sent
    a fold played by policy_run.py: the same, the row being what the policy sent (policy_meta.json beside it)
    fold_live.py: recorder.py's steps, radians
    a teleop recording from the panel's record button: encoder steps per arm name, turned into radians through each
        arm's tick map; the target is the goal the panel was sending
Pictures come from frames.jsonl / frames/ (recorder.py). A run with no pictures, or with one arm only, is listed and
left out of a dataset.

`build` writes a version of the write-once store the simulator's students train from (imitation/data/schema.py), so
imitation.train reads it unchanged. What goes where, per control row:
    images["main"]   the phone's picture taken last before the row, squashed to --size x --size, uint8 [3, S, S]
    obs (139)        only the sensor proprio slots are filled, per arm (left at 0, right at 25):
                       +0..4 the 5 joint angles read (rad), +5..9 their rate (rad/s, from the rows' own times),
                       +10 the jaw angle read (rad), +11..13 the fingertip in the arm's own base frame (m),
                       +14..16 the fingers' direction, +21..23 the fingertip's speed (m/s); the rest stay 0
    actions (12)     the joint targets sent, left then right, each arm's 5 joints and its jaw, in units of
                       ACTION_SCALE rad: absolute targets, so a row's action does not depend on how far the servo
                       lags (the simulator's store holds joint steps; a policy trained here is for the rig only)
The rates and speeds are per second of wall time, so a policy sees the speed its demonstrations were played at.
"""

import argparse
import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np

import recorder
from kinematics import Arm

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SIDES = ("left", "right")
ACTION_SCALE = 2.0          # rad per unit of action: every joint's range (within +-1.92) and the jaw fit in +-1
SIZE = 128                  # pixels a side the pictures are stored at
MIN_ROWS = 20               # fewer rows than this is not a fold
LABEL = "label.json"
OBS_DIM, ARM_SLOTS = 139, 25        # imitation.spec.OBS_DIM and its proprio block per arm (checked in build)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def replay_module():
    """replay.py, for its tick map and rate cap. The servo SDK it imports is not needed for those, so a machine
    without it (where datasets are built) gets an empty stand-in."""
    try:
        import replay
    except ImportError:
        sys.modules.setdefault("scservo_sdk", types.ModuleType("scservo_sdk"))
        import replay
    return replay


def arm_names(steps, left, right):
    """{"left": arm name, "right": arm name} of a teleop recording: as given, else arms.json's followers in order."""
    recorded = list(steps[0]["measured"])
    if left and right:
        names = {"left": left, "right": right}
    else:
        import arms
        leaders = set(arms.CONFIG.get("leaders", {}).values())
        followers = [n for n in arms.CONFIG["arms"] if n in recorded and n not in leaders]
        if len(followers) != 2:
            raise ValueError(f"cannot tell left from right among {recorded}: pass --left and --right")
        names = dict(zip(SIDES, followers))
    for name in names.values():
        if name not in recorded:
            raise ValueError(f"arm {name} is not in this recording (it has {recorded})")
    return names


def load_run(path, left=None, right=None):
    """One run folder as {"kind", "t" [T], "measured" [T, 12], "target" [T, 12] (rad, left arm then right),
    "frame" [T] (index of the picture taken last before each row, -1 for none), "label", "meta", "path"}.
    ValueError says why a folder cannot be used."""
    path = Path(path)
    steps = read_jsonl(path / "steps.jsonl")
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    if not steps:
        raise ValueError("no steps.jsonl")
    if meta.get("task") == "teleop":
        kind = "teleop"
        names = arm_names(steps, left, right)
        maps = {side: replay_module().Follower(name) for side, name in names.items()}
        to_rad = lambda side, ticks: [maps[side].to_rad(m, v) for m, v in zip(range(1, 7), ticks)]    # noqa: E731
        rows = [(s["t"], [to_rad(side, s["measured"][names[side]]) for side in SIDES],
                 [to_rad(side, s["target"][names[side]]) for side in SIDES]) for s in steps]
    else:
        policy = (path / "policy_meta.json").exists()
        kind = "policy" if policy else "live" if "plan" in steps[0] else "script"
        folds = [s for s in steps if s.get("phase", "fold") == "fold"]
        whole = lambda q: q is not None and len(q) == 6 and None not in q                            # noqa: E731
        if any(not whole(s["target"].get(side)) or not whole(s["measured"].get(side)) for s in folds for side in SIDES):
            raise ValueError("a row is missing an arm or a joint (one arm only, or the gripper left out)")
        rows = [(s["t"], [s["measured"][side] for side in SIDES], [s["target"][side] for side in SIDES]) for s in folds]
        if kind == "script" and (path / "plan.json").exists() and len(rows) > 1:
            # replay.py logs the planned row. When a joint is asked for more than its rate cap in one row, what was
            # sent lags the plan, and the rows logged are not what the arm was told
            cap = np.radians(replay_module().MAX_DEG_S) * json.loads((path / "plan.json").read_text())["dt"]
            sent = np.array([np.concatenate(r[2]) for r in rows])
            fastest = np.abs(np.diff(sent, axis=0)).max()
            if fastest > cap + 1e-6:
                raise ValueError(f"the plan moves a joint {np.degrees(fastest):.1f} deg in one row, over replay.py's "
                                 f"{np.degrees(cap):.1f} deg cap: the rows logged are not the rows sent")
    if len(rows) < MIN_ROWS:
        raise ValueError(f"{len(rows)} fold rows")
    t = np.array([r[0] for r in rows], float)
    frames = read_jsonl(path / "frames.jsonl")
    frame = np.full(len(t), -1)
    if frames:
        times, ids = np.array([f["t"] for f in frames], float), np.array([f["i"] for f in frames])
        last = np.searchsorted(times, t, side="right") - 1          # the picture taken last before the row
        frame = np.where(last >= 0, ids[np.clip(last, 0, None)], ids[0] if t[0] >= times[0] - 1.0 else -1)
    label = json.loads((path / LABEL).read_text()) if (path / LABEL).exists() else None
    for name in ("replay_meta.json", "policy_meta.json"):
        if (path / name).exists():
            meta = meta | json.loads((path / name).read_text())
    return {"kind": kind, "t": t, "measured": np.array([np.concatenate(r[1]) for r in rows], float),
            "target": np.array([np.concatenate(r[2]) for r in rows], float), "frame": frame, "label": label,
            "meta": meta, "path": path}


def picture(path, i, size=SIZE):
    """Frame i of a run as uint8 [3, size, size]: the whole picture squashed square, the same for every run."""
    from PIL import Image
    rgb = recorder.load_frame(path, int(i))[0]
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR)).transpose(2, 0, 1).copy()


def proprio(run):
    """The observation rows [T, 139] of a run: each arm's block as the module docstring lays it out."""
    arm, t, T = Arm(), run["t"], len(run["t"])
    dt = np.gradient(t) if T > 1 else np.ones(1)
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 1.0     # two rows logged in the same instant
    rate = lambda x: np.gradient(x, axis=0) / dt[:, None]                                            # noqa: E731
    obs = np.zeros((T, OBS_DIM), np.float32)
    for k in range(2):
        q, b = run["measured"][:, 6 * k:6 * k + 6], ARM_SLOTS * k
        tips = [arm.tip(row[:5]) for row in q]
        tip, fingers = np.array([p for p, _ in tips]), np.array([z for _, z in tips])
        obs[:, b:b + 5], obs[:, b + 5:b + 10], obs[:, b + 10] = q[:, :5], rate(q[:, :5]), q[:, 5]
        obs[:, b + 11:b + 14], obs[:, b + 14:b + 17], obs[:, b + 21:b + 24] = tip, fingers, rate(tip)
    return obs


def episode(run, size=SIZE):
    """A run as an imitation.data.schema.Episode. Needs a label and a picture for every row."""
    from imitation.data.schema import Episode
    if run["label"] is None:
        raise ValueError("not labelled: say whether it folded with `rig_dataset.py label`")
    if (run["frame"] < 0).any():
        raise ValueError("no camera picture for some rows")
    T, good = len(run["t"]), bool(run["label"]["success"])
    actions = (run["target"] / ACTION_SCALE).astype(np.float32)
    if np.abs(actions).max() > 1.0:
        raise ValueError(f"a target of {np.abs(run['target']).max():.2f} rad is outside +-{ACTION_SCALE} rad")
    pictures = {int(i): picture(run["path"], i, size) for i in np.unique(run["frame"])}
    obs = proprio(run)
    last = np.zeros(T, bool)
    last[-1] = True
    name = run["path"].name
    return Episode(
        obs=obs, actions=actions, rewards=np.zeros(T, np.float32), stage=np.zeros(T, np.int8),
        fold_score=np.zeros(T, np.float32), grasped=np.zeros((T, 2), bool),
        actor=np.full(T, 1 if run["kind"] == "policy" else 0, np.int8), final_obs=obs[-1].copy(),
        terminated=last & good, truncated=last & (not good), discount=np.where(last & good, 0.0, 1.0).astype(np.float32),
        images={"main": np.stack([pictures[int(i)] for i in run["frame"]])},
        # the store names an episode by its source and seed: the seed is a number made from the run folder's name
        meta={"seed": zlib.crc32(name.encode()), "source": "student" if run["kind"] == "policy" else "expert",
              "success": good, "termination_reason": "folded" if good else "not folded", "run": name,
              "kind": run["kind"], "note": run["label"].get("note", ""), "speed": run["meta"].get("speed"),
              "seconds": round(float(run["t"][-1] - run["t"][0]), 2), "pictures": len(pictures)})


def run_folders(where):
    """The run folders under `where` (or `where` itself if it is one), oldest first."""
    where = Path(where)
    if (where / "steps.jsonl").exists():
        return [where]
    return sorted(p for p in where.iterdir() if (p / "steps.jsonl").exists()) if where.exists() else []


def describe(path, left=None, right=None):
    """One line about a run folder for `list`, and whether `build` would take it."""
    try:
        run = load_run(path, left, right)
    except (ValueError, KeyError) as err:
        return f"{Path(path).name:22s} unusable: {err}", False
    pictures = len(np.unique(run["frame"][run["frame"] >= 0]))
    label = "unlabelled" if run["label"] is None else "good" if run["label"]["success"] else "bad"
    usable = run["label"] is not None and (run["frame"] >= 0).all()
    return (f"{Path(path).name:22s} {run['kind']:7s} {len(run['t']):4d} rows {run['t'][-1] - run['t'][0]:5.1f} s "
            f"{pictures:4d} pictures  {label:10s}{'' if usable else '  (not usable yet)'}"), usable


def build(folders, out, version, size=SIZE, left=None, right=None, failures=False):
    """Write the usable runs among `folders` as dataset `version` under `out` and freeze it. Good folds only unless
    `failures`. Returns (the manifest, [(run name, why it was left out)])."""
    sys.path.insert(0, str(HERE.parents[1]))
    from imitation import spec
    from imitation.data.schema import DatasetWriter
    assert spec.OBS_DIM == OBS_DIM and spec.ACTION_DIM == 12 and spec.GRIPPER_CTRL_DIMS == (10, 10 + ARM_SLOTS)
    writer = DatasetWriter(out, version, config={"tool": "hardware/so101/rig_dataset.py", "size": size,
                                                 "action": f"joint targets / {ACTION_SCALE} rad", "camera": "main"})
    left_out = []
    for folder in folders:
        try:
            run = load_run(folder, left, right)
            if run["label"] is not None and not run["label"]["success"] and not failures:
                raise ValueError("labelled bad")
            writer.add(episode(run, size), spec.OBS_DIM, spec.ACTION_DIM)
        except (ValueError, KeyError) as err:
            left_out.append((Path(folder).name, str(err)))
    if not writer.new_entries:
        raise ValueError("no usable run: " + "; ".join(f"{n} ({why})" for n, why in left_out[:5]))
    return writer.freeze(), left_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["list", "label", "build"])
    ap.add_argument("runs", nargs="*", help="run folders, or a folder of them (default: runs/)")
    ap.add_argument("--left", help="teleop recordings: the left arm's name in arms.json")
    ap.add_argument("--right")
    ap.add_argument("--note", default="", help="label: a few words kept with the label")
    ap.add_argument("--out", help="build: the dataset store's root")
    ap.add_argument("--version", help="build: the new version's name")
    ap.add_argument("--size", type=int, default=SIZE, help="build: pixels a side the pictures are stored at")
    ap.add_argument("--failures", action="store_true", help="build: keep the runs labelled bad too, marked as failures")
    args = ap.parse_args(argv)
    if args.mode == "label":
        if len(args.runs) != 2 or args.runs[1] not in ("good", "bad"):
            sys.exit("label takes a run folder and good or bad")
        folder = Path(args.runs[0])
        if not (folder / "steps.jsonl").exists():
            sys.exit(f"{folder} is not a run folder")
        (folder / LABEL).write_text(json.dumps({"success": args.runs[1] == "good", "note": args.note}))
        print(f"{folder.name}: {args.runs[1]}")
        return 0
    folders = [f for where in (args.runs or [RUNS]) for f in run_folders(where)]
    if args.mode == "list":
        usable = 0
        for folder in folders:
            line, ok = describe(folder, args.left, args.right)
            usable += ok
            print(line)
        print(f"{len(folders)} runs, {usable} usable")
        return 0
    if not args.out or not args.version:
        sys.exit("build needs --out and --version")
    try:
        manifest, left_out = build(folders, args.out, args.version, args.size, args.left, args.right, args.failures)
    except (ValueError, FileExistsError) as err:
        sys.exit(f"Not built: {err}")
    for name, why in left_out:
        print(f"left out {name}: {why}")
    print(f"{args.version}: {manifest['n_episodes']} episodes ({manifest['n_success']} good), {manifest['n_steps']} "
          f"rows, in {Path(args.out) / args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
