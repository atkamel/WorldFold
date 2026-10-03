"""No-physics reachability of the Isaac half fold for a cloth XY offset (isaac/half_fold_demo.py plan() geometry).

For each arm: pinch target at the far corner, the pose 4 cm above it, and the place target on the near corner's start,
all solved with isaac.pinch.PinchIK from the demo's IK seeds. An offset is reachable when every position error is under
REACH_TOL. The sweep draws offsets as the Isaac id_hard rule (imitation.seeds.shifted_pose_isaac) with R_max = r and
reports the largest r for which all of them are reachable. Run from the repo root in the Isaac venv:

    python isaac/reach_check.py --n 200 --out outputs/isaac/reach.json
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.isaac_env import CLOTH_CENTER, cloth_grid_mesh   # noqa: E402
from isaac.pinch import PinchIK   # noqa: E402
from mujuco.cloth_params import TABLE_TOP_Z   # noqa: E402

PINCH_HEIGHT = 0.010
PLACE_HEIGHT = 0.011
SEED_Q = {"left_": [0.2614, 1.0654, -1.1908, 1.6581, -2.046], "right_": [-0.2619, 1.0622, -1.1853, 1.6581, 2.1429]}
# HalfFoldEnv.stages[0]: (arm, carried vertex, goal vertex)
MOVES = (("left_", 10, 0), ("right_", 120, 110))
REACH_TOL = 0.004
RADII_CM = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
MIN_AXIS = 0.010


def sample_offset(rng, r):
    """Same rule as imitation.seeds.shifted_pose_isaac with R_max = r (metres)."""
    pose = rng.uniform(-r, r, size=2)
    axis = rng.integers(2)
    pose[axis] = np.sign(pose[axis] or 1.0) * rng.uniform(min(MIN_AXIS, r), r)
    return pose


def rest_grid(offset):
    pts = np.array(cloth_grid_mesh(1)[0], float)
    pts[:, :2] += np.asarray(CLOTH_CENTER) + np.asarray(offset)
    pts[:, 2] = TABLE_TOP_Z + 0.002
    return pts


def max_error(ik, offset):
    """(largest pinch/above error, largest place error) in metres over both arms."""
    grid = rest_grid(offset)
    center = grid.mean(axis=0)
    grasp = place_err = 0.0
    for p, carried, goal_v in MOVES:
        corner, goal = grid[carried], grid[goal_v]
        jaw = np.r_[center[:2] - corner[:2], 0.0]
        jaw /= np.linalg.norm(jaw)
        off = -0.005 * jaw[:2]
        pinch = np.array([corner[0] + off[0], corner[1] + off[1], TABLE_TOP_Z + PINCH_HEIGHT])
        place = np.array([goal[0] + off[0], goal[1] - off[1], TABLE_TOP_Z + PLACE_HEIGHT])
        q = SEED_Q[p]
        for target in (pinch + [0, 0, 0.04], pinch):
            q, err = ik.solve(p, target, jaw, q)
            grasp = max(grasp, err)
        _, err = ik.solve(p, place, jaw, q, orientation_weight=0.05)
        place_err = max(place_err, err)
    return grasp, place_err


def demo_envelope(ik):
    """Worst (grasp, place) errors over the +-1 cm reset-jitter box (corners, edge midpoints, centre): offsets the
    scripted demo is known to fold from. The place target at the fold line is stretched for the arm, so even the
    nominal cloth misses it by several mm; a flat REACH_TOL on it would call every offset unreachable."""
    pts = [(x, y) for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01)]
    errs = np.array([max_error(ik, p) for p in pts])
    return float(errs[:, 0].max()), float(errs[:, 1].max())


_IK = None


def _work(offset):
    global _IK
    _IK = _IK or PinchIK()
    return max_error(_IK, offset)


def sweep(n=200, seed=0, radii_cm=RADII_CM, workers=6):
    """Reachable: pinch/above error under max(REACH_TOL, demo envelope's) and place error within the demo envelope's."""
    ik = PinchIK()
    pool = mp.Pool(workers) if workers > 1 else None
    env_grasp, env_place = demo_envelope(ik)
    tol_grasp, tol_place = max(REACH_TOL, env_grasp), env_place
    print(f"demo envelope (+-1 cm box): grasp {1000 * env_grasp:.2f} mm, place {1000 * env_place:.2f} mm", flush=True)
    rows = []
    for r_cm in radii_cm:
        rng = np.random.default_rng([seed, int(round(r_cm * 100))])
        offsets = [sample_offset(rng, r_cm / 100.0) for _ in range(n)]
        errs = np.array(pool.map(_work, offsets, chunksize=4) if pool else [max_error(ik, o) for o in offsets])
        ok = int(((errs[:, 0] <= tol_grasp) & (errs[:, 1] <= tol_place)).sum())
        rows.append({"r_cm": r_cm, "reachable": ok / n, "n": n, "worst_err_mm": round(1000 * float(errs.max()), 3),
                     "worst_grasp_mm": round(1000 * float(errs[:, 0].max()), 3),
                     "worst_place_mm": round(1000 * float(errs[:, 1].max()), 3)})
        print(f"r={r_cm:5.2f} cm  reachable {ok}/{n}  worst grasp {1000 * errs[:, 0].max():.2f} mm "
              f"place {1000 * errs[:, 1].max():.2f} mm", flush=True)
    if pool:
        pool.close()
    full = [row["r_cm"] for row in rows if row["reachable"] == 1.0]
    return {"r_max_cm": max(full) if full else 0.0, "rows": rows,
            "tol_grasp_mm": round(1000 * tol_grasp, 3), "tol_place_mm": round(1000 * tol_place, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac/reach.json")
    args = ap.parse_args()
    res = sweep(args.n, args.seed, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("R_max =", res["r_max_cm"], "cm")


if __name__ == "__main__":
    main()
