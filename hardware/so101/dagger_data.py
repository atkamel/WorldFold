"""Planned folds to start the learned camera fold from (dagger_net.py), before it has seen the arms: the scripted
planner (fold_traj.fold) run on cloths drawn over the places and saved settings the camera fold meets on the rig.
No arm and no camera are involved, so this runs anywhere the planner does.

    .venv/bin/python hardware/so101/dagger_data.py --episodes 900 --seed 0 --out outputs/hardware/dagger/base.npz
    .venv/bin/python hardware/so101/dagger_data.py --join a.npz b.npz --out both.npz     # two sets as one

A draw: a rectangular cloth in front of the two bases (X right, Y forward from the line of the base origins), its
far edge FAR and its depth DEPTH from the bases, WIDTH wide, off centre by up to SHIFT, turned by up to TURN_DEG;
bases SPACING apart; and the saved settings of fold_camera.py: the inset, each arm's overshoot and pinch height,
and for a share of the draws a pinch pulled toward its arm as --pinch-tilt pulls it. The points go through
fold_camera.fold_points and fold_camera.aimed, so the planner is given what the camera fold gives it. Folds the
planner refuses are skipped and counted, as are plans whose two arms differ in length (their release lifts lean differently), which
replay.py refuses to play.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import fold_camera
import fold_traj
from dagger_net import CLOSED, fold_inputs

FAR = (0.29, 0.41)          # m: the cloth's far edge in front of the base origins
DEPTH = (0.20, 0.31)        # m: near edge to far edge
WIDTH = (0.22, 0.36)        # m
SHIFT = 0.05                # m off the middle between the bases
TURN_DEG = 15.0
SPACING = (0.27, 0.33)      # m between the base origins
INSET = (0.0, 0.04)         # m, fold_camera's --inset
OVERSHOOT = 0.015           # m either way, forward and outward, each arm its own
PINCH_Z = (-0.02, 0.01)     # m, each arm its own (fold_camera.PINCH_HEIGHTS starts at -0.02)
PULLED = 0.3                # share of draws with the pinches pulled toward their arms
PULL = 0.08                 # m, the most a pinch is pulled
WORKERS = 6


def draw(rng):
    """One fold's ({side: (pinch, place)} as the planner is sent them, {side: pinch height})."""
    far, depth, width = (rng.uniform(*r) for r in (FAR, DEPTH, WIDTH))
    centre = np.array([rng.uniform(-SHIFT, SHIFT), far - depth / 2])
    a = np.radians(rng.uniform(-TURN_DEG, TURN_DEG))
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = {name: centre + turn @ np.array(xy) for name, xy in
               (("far_left", (-width / 2, depth / 2)), ("near_left", (-width / 2, -depth / 2)),
                ("far_right", (width / 2, depth / 2)), ("near_right", (width / 2, -depth / 2)))}
    points = fold_camera.fold_points(corners, rng.uniform(*SPACING), rng.uniform(*INSET))
    if rng.random() < PULLED:
        for side in ("left", "right"):
            pinch, place = points[side]
            reach = float(np.hypot(*pinch))
            scale = (reach - rng.uniform(0.0, PULL)) / reach
            points[side] = ((pinch[0] * scale, pinch[1] * scale), place)
    overshoot = {side: tuple(rng.uniform(-OVERSHOOT, OVERSHOOT, 2)) for side in ("left", "right")}
    heights = {side: float(rng.uniform(*PINCH_Z)) for side in ("left", "right")}
    return fold_camera.aimed(points, overshoot), heights


def plan_one(task):
    """(index, points, heights, the fold or the planner's reason for refusing it). Runs in a worker."""
    index, points, heights = task
    try:
        traj = fold_traj.fold(points, CLOSED, False, heights)
    except ValueError as err:
        return index, points, heights, str(err)
    except Exception as err:        # counted and named, so a planner fault cannot end the job early without a word
        return index, points, heights, f"{type(err).__name__}: {err}"
    if len(traj["arms"]["left"]["target"]) != len(traj["arms"]["right"]["target"]):
        return index, points, heights, "the two arms' plans differ in length, which replay.py refuses"
    return index, points, heights, traj


def generate(episodes, seed, workers=WORKERS):
    """(the folds planned, the draws refused), each a list of plan_one's results in draw order."""
    rng = np.random.default_rng(seed)
    tasks = [(i, *draw(rng)) for i in range(episodes)]
    done, started = [], time.time()
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for result in pool.imap_unordered(plan_one, tasks):
            done.append(result)
            if len(done) % 50 == 0 or len(done) == episodes:
                print(f"  {len(done)}/{episodes} planned, {time.time() - started:.0f} s", flush=True)
    if len(done) != episodes:
        raise RuntimeError(f"only {len(done)} of {episodes} draws came back from the workers")
    done.sort(key=lambda r: r[0])
    return [r for r in done if isinstance(r[3], dict)], [r for r in done if not isinstance(r[3], dict)]


def save(path, planned, refused, seed):
    """One .npz: per fold its inputs [2 arms, 5] (dagger_net.fold_inputs: left, right) and the planner's report;
    the target rows of every fold one after another [rows, 2 arms, 6], fold e being rows starts[e] to
    starts[e + 1]; how many draws were refused and why."""
    rows, starts = [], [0]
    for *_, traj in planned:
        rows.append(np.stack([np.array(traj["arms"][side]["target"]) for side in ("left", "right")], axis=1))
        starts.append(starts[-1] + len(rows[-1]))
    inputs = [np.stack([fold_inputs(points, heights)[side] for side in ("left", "right")])
              for _, points, heights, _ in planned]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, seed=seed, dt=fold_traj.DT, closed=CLOSED, inputs=np.array(inputs).reshape(-1, 2, 5),
        targets=(np.concatenate(rows) if rows else np.zeros((0, 2, 6))).astype(np.float32), starts=np.array(starts),
        source=np.array([json.dumps(r[3]["source"]) for r in planned]),
        refused_reason=np.array([r[3] for r in refused], dtype=str))


def load(path):
    """The saved file as a dict of arrays, with "folds" [(inputs [2, 5], targets [T, 2, 6])]."""
    data = dict(np.load(path))
    starts = data["starts"]
    data["folds"] = [(data["inputs"][e], data["targets"][starts[e]:starts[e + 1]]) for e in range(len(starts) - 1)]
    return data


def join(paths, out):
    """One file from several of save()'s, their folds one after another. The seed kept is the first file's."""
    parts = [dict(np.load(path)) for path in paths]
    starts = [0]
    for part in parts:
        starts += (part["starts"][1:] + starts[-1]).tolist()
    np.savez_compressed(out, seed=parts[0]["seed"], dt=parts[0]["dt"], closed=parts[0]["closed"],
                        starts=np.array(starts),
                        **{key: np.concatenate([part[key] for part in parts])
                           for key in ("inputs", "targets", "source", "refused_reason")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, help="cloths drawn; the ones the planner refuses are skipped")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--join", nargs="+", metavar="FILE", help="put these files' folds into --out and plan nothing")
    args = ap.parse_args()
    if args.join:
        join(args.join, args.out)
        print(f"{len(load(args.out)['folds'])} folds: {args.out}")
        return
    if args.episodes is None or args.seed is None:
        ap.error("--episodes and --seed, or --join")
    started = time.time()
    planned, refused = generate(args.episodes, args.seed, args.workers)
    save(args.out, planned, refused, args.seed)
    reasons = {}
    for *_, reason in refused:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  refused {count}: {reason}")
    print(f"{len(planned)} folds, {len(refused)} refused by the planner, {time.time() - started:.0f} s: {args.out}")


if __name__ == "__main__":
    main()
