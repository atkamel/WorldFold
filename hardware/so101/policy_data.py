"""Training data for the learned half fold (policy_net.py): planned folds (fold_traj.fold) for a 30 cm cloth moved
about its ruler place, in both rig layouts. Each episode keeps the pinch and place points the planner was given,
both arms' target rows, the marks and the planner's report. Poses the planner refuses are skipped and counted.

    .venv/bin/python hardware/so101/policy_data.py --episodes 600 --seed 0 --out outputs/policy/train.npz

Layouts: "behind", the arms side by side behind the cloth's near edge (fold_traj.trajectory with cloth 0.30, near
0.08, spacing 0.30), and "sides", an arm at the middle of each side edge facing in (fold_traj.sides_trajectory with
cloth 0.30, gap 0.16). The cloth is shifted by up to SHIFT in X and Y and turned by up to TURN_DEG about its centre.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import fold_traj

LAYOUTS = ("behind", "sides")
CLOTH, NEAR, SPACING, GAP = 0.30, 0.08, 0.30, 0.16
SHIFT = 0.03          # m the cloth may move in X and in Y
TURN_DEG = 8.0        # deg it may turn about its centre
WORKERS = 6           # one fold takes 2 to 10 s


def bases(layout, cloth, near, spacing, gap):
    """{side: (base origin (X, Y), heading rad)} in the table frame: X right and Y toward the cloth's far edge, seen
    from its near edge, origin at the ruler place of the cloth's centre. Heading is the base's forward direction."""
    half = cloth / 2
    if layout == "behind":
        return {"left": ((-spacing / 2, -(near + half)), np.pi / 2), "right": ((spacing / 2, -(near + half)), np.pi / 2)}
    if layout == "sides":
        return {"left": ((-(half + gap), 0.0), 0.0), "right": ((half + gap, 0.0), np.pi)}
    raise ValueError(f"unknown layout {layout!r}")


def cloth_points(layout, cloth=CLOTH, near=NEAR, spacing=SPACING, gap=GAP, shift=(0.0, 0.0), turn_deg=0.0):
    """{"left": (pinch, place), "right": (pinch, place)}, each (x, y) m in that arm's base frame, for the cloth moved
    by `shift` (X, Y m) and turned by `turn_deg` (counterclockwise from above) about its centre. Each arm pinches the
    far corner on its side and places it on the near one. Unmoved, these are the points fold_traj.trajectory
    (behind) and fold_traj.sides_trajectory (sides) fold."""
    half = cloth / 2
    a = np.radians(turn_deg)
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = {name: turn @ np.array(xy) + np.asarray(shift, float) for name, xy in
               (("far_left", (-half, half)), ("near_left", (-half, -half)),
                ("far_right", (half, half)), ("near_right", (half, -half)))}
    points = {}
    for side, (origin, heading) in bases(layout, cloth, near, spacing, gap).items():
        forward = np.array([np.cos(heading), np.sin(heading)])
        left = np.array([-np.sin(heading), np.cos(heading)])
        frame = [(float(forward @ (corners[c] - origin)), float(left @ (corners[c] - origin)))
                 for c in (f"far_{side}", f"near_{side}")]
        points[side] = tuple(frame)
    return points


def plan_one(task):
    """(index, layout, shift, turn, points, fold or the planner's reason for refusing). Runs in a worker."""
    index, layout, shift, turn = task
    points = cloth_points(layout, shift=shift, turn_deg=turn)
    try:
        return index, layout, shift, turn, points, fold_traj.fold(points)
    except ValueError as err:
        return index, layout, shift, turn, points, str(err)


def generate(episodes, seed):
    """Draw `episodes` cloth poses (layouts alternating), plan each on WORKERS processes. Returns (accepted, refused),
    each a list of (index, layout, shift, turn, points, fold or reason) in draw order."""
    rng = np.random.default_rng(seed)
    tasks = [(i, LAYOUTS[i % 2], tuple(rng.uniform(-SHIFT, SHIFT, 2)), float(rng.uniform(-TURN_DEG, TURN_DEG)))
             for i in range(episodes)]
    done = []
    started = time.time()
    with multiprocessing.get_context("spawn").Pool(WORKERS) as pool:
        for result in pool.imap_unordered(plan_one, tasks):
            done.append(result)
            if len(done) % 25 == 0 or len(done) == episodes:
                print(f"  {len(done)}/{episodes} planned, {time.time() - started:.0f} s", flush=True)
        pool.close()
        pool.join()
    done.sort(key=lambda r: r[0])
    return [r for r in done if isinstance(r[5], dict)], [r for r in done if not isinstance(r[5], dict)]


def save(path, accepted, refused, seed):
    """One .npz: per episode its layout, shift (m), turn (deg), points [2 arms, pinch/place, xy] (left, right), marks
    [hover, pinch, check, placed] and planner report (JSON); the target rows of every episode one after another,
    [rows, 2 arms, 6], episode e being rows starts[e] to starts[e + 1]. The refused poses with the planner's reason."""
    rows, starts = [], [0]
    for *_, traj in accepted:
        both = np.stack([np.array(traj["arms"][side]["target"]) for side in ("left", "right")], axis=1)
        rows.append(both)
        starts.append(starts[-1] + len(both))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, seed=seed, dt=fold_traj.DT,
        layout=np.array([r[1] for r in accepted]), shift=np.array([r[2] for r in accepted]).reshape(-1, 2),
        turn_deg=np.array([r[3] for r in accepted]),
        points=np.array([[r[4][side] for side in ("left", "right")] for r in accepted]).reshape(-1, 2, 2, 2),
        marks=np.array([[r[5]["marks"][k] for k in ("hover", "pinch", "check", "placed")] for r in accepted]),
        source=np.array([json.dumps(r[5]["source"]) for r in accepted]),
        targets=np.concatenate(rows) if rows else np.zeros((0, 2, 6)), starts=np.array(starts),
        refused_layout=np.array([r[1] for r in refused], dtype=str),
        refused_shift=np.array([r[2] for r in refused]).reshape(-1, 2),
        refused_turn_deg=np.array([r[3] for r in refused]), refused_reason=np.array([r[5] for r in refused], dtype=str))


def load(path):
    """The saved file as a dict of arrays, with "episodes" [(layout, points, targets [T, 2, 6], marks)]."""
    data = dict(np.load(path))
    starts = data["starts"]
    data["episodes"] = [(str(data["layout"][e]), data["points"][e], data["targets"][starts[e]:starts[e + 1]],
                         data["marks"][e]) for e in range(len(starts) - 1)]
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, required=True, help="cloth poses drawn; refused ones are skipped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    started = time.time()
    accepted, refused = generate(args.episodes, args.seed)
    for _, layout, shift, turn, _, reason in refused:
        print(f"  refused: {layout}, shift {np.round(np.array(shift) * 100, 1)} cm, turn {turn:.1f} deg: {reason}")
    save(args.out, accepted, refused, args.seed)
    print(f"{len(accepted)} folds, {len(refused)} refused by the planner, {time.time() - started:.0f} s: {args.out}")


if __name__ == "__main__":
    main()
