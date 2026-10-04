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
    from isaac.pinch import PinchIK
    _IK = _IK or PinchIK()
    return max_error(_IK, offset)


def sweep(n=200, seed=0, radii_cm=RADII_CM, workers=6):
    """Reachable: pinch/above error under max(REACH_TOL, demo envelope's) and place error within the demo envelope's."""
    from isaac.pinch import PinchIK
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


# ---- profile "weld" (Phase W, W2): MuJoCo's FoldExpert waypoints with position-only IK ----
# The far corners sit ~45 cm from the arm bases, at the SO101's reach limit: MuJoCo's own solve_ik misses the descend
# waypoint by 8 mm (nominal) to 30 mm (+2 cm offset) on the same targets (docs/results.md, W2), and the expert still
# folds 100 / 97 % because the weld catches any corner within GRASP_RADIUS and a phase advances within 2 cm. Those
# are the bars here: descend within GRASP_RADIUS, every other waypoint within ADVANCE_DIST.
ADVANCE_DIST = 0.02       # FoldExpert._maybe_advance
WAYPOINTS = ("approach", "descend", "lift", "carry", "place")
CLOTH_REST_DZ = 0.004     # the settled Isaac cloth's height above the table top (weld_check: z 0.424)


def fold_expert_targets(grid, carried, goal_v, overshoot):
    """FoldExpert._plan's moving waypoints for one arm: approach, descend, lift, carry, place (goal + overshoot)."""
    from cloth_fold_rl.fold_env import LIFT_TARGET_Z
    corner, goal = grid[carried], grid[goal_v] + overshoot
    return [corner + [0, 0, 0.06], corner + [0, 0, 0.005], np.r_[corner[:2], LIFT_TARGET_Z],
            np.r_[goal[:2], LIFT_TARGET_Z], goal + [0, 0, 0.02]]


def mujoco_errors(ik, offset, seed):
    """Per-waypoint IK error (m, worst of both arms, WAYPOINTS order) for a MuJoCo-profile cloth offset, solved the
    way the expert does: chained from the home pose, 12 restarts."""
    from cloth_fold_rl.quarter_fold_expert import OVERSHOOT
    pts = np.array(cloth_grid_mesh(1)[0], float)
    pts[:, :2] += np.asarray(offset)               # weld profile: cloth centred at (0, 0)
    pts[:, 2] = TABLE_TOP_Z + CLOTH_REST_DZ
    rng, worst = np.random.default_rng(seed), np.zeros(len(WAYPOINTS))
    for p, carried, goal_v in MOVES:
        q = np.zeros(5)
        for k, target in enumerate(fold_expert_targets(pts, carried, goal_v, OVERSHOOT[(0, p)])):
            q, err = ik.solve_position(p, target, q, rng=rng)
            worst[k] = max(worst[k], err)
    return worst


def _work_mujoco(arg):
    global _IK
    from isaac.pinch import PinchIK
    _IK = _IK or PinchIK()
    return mujoco_errors(_IK, *arg)


_MJ = None


def _work_mujoco_sim(arg):
    """The same waypoints on MuJoCo's own ClothFoldEnv with the expert's solve_ik (MuJoCo venv): the parity
    reference. The cloth grid is read after MuJoCo's reset (it rests at z 0.430, the Isaac cloth at 0.424); each solve
    starts from the previous waypoint's solution, as the live expert's does."""
    global _MJ
    from cloth_fold_rl.expert import solve_ik
    from cloth_fold_rl.quarter_fold_expert import OVERSHOOT
    if _MJ is None:
        from imitation.tasks import make_env
        _MJ = make_env("mujoco")
        _MJ.unwrapped.domain_randomization = False
    offset, seed = arg
    b = _MJ.unwrapped
    _MJ.reset(seed=0, options={"cloth_pose": np.asarray(offset)})
    grid = np.array([b.data.xpos[i] for i in b._cloth_body_ids])
    rng, worst = np.random.default_rng(seed), np.zeros(len(WAYPOINTS))
    for p, carried, goal_v in MOVES:
        adr = b._arm_qpos_adr[p]
        ranges = [b.model.joint(f"{p}{j}").range for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                           "wrist_flex", "wrist_roll")]
        home = np.array([b.data.qpos[a] for a in adr])
        for k, target in enumerate(fold_expert_targets(grid, carried, goal_v, OVERSHOOT[(0, p)])):
            q, err = solve_ik(b.model, b.data, b._site_id[p], adr, b._arm_dof_adr[p], ranges, target, rng=rng)
            for a, v in zip(adr, q):
                b.data.qpos[a] = v
            worst[k] = max(worst[k], err)
        for a, v in zip(adr, home):
            b.data.qpos[a] = v
    return worst


def sweep_mujoco(n=200, seed=0, workers=6, sim="isaac"):
    """MuJoCo's own start distributions: id_easy (+-CLOTH_JITTER uniform) and id_hard (imitation.seeds.shifted_pose,
    2.5-4 cm). Reachable: descend within GRASP_RADIUS (the weld engages) and every other waypoint within
    ADVANCE_DIST. sim="isaac": LeHome's kinematics (PinchIK); sim="mujoco": the MuJoCo arm, for parity."""
    from cloth_fold_rl.fold_env import CLOTH_JITTER
    from cloth_fold_rl.quarter_fold_env import GRASP_RADIUS
    from imitation.seeds import shifted_pose
    rng = np.random.default_rng([seed, 2])
    sets = {"id_easy": [rng.uniform(-CLOTH_JITTER, CLOTH_JITTER, size=2) for _ in range(n)],
            "id_hard": [shifted_pose(seed * 100_000 + i)["cloth_pose"] for i in range(n)]}
    tol = np.full(len(WAYPOINTS), ADVANCE_DIST)
    tol[WAYPOINTS.index("descend")] = GRASP_RADIUS
    pool = mp.Pool(workers) if workers > 1 else None
    work = _work_mujoco if sim == "isaac" else _work_mujoco_sim
    res = {"profile": "weld", "sim": sim, "tol_mm": dict(zip(WAYPOINTS, (1000 * tol).round(1).tolist())), "sets": {}}
    for name, offsets in sets.items():
        args = [(o, i) for i, o in enumerate(offsets)]
        errs = np.array(pool.map(work, args, chunksize=4) if pool else [work(a) for a in args])
        ok = (errs <= tol).all(axis=1)
        res["sets"][name] = {"reachable": int(ok.sum()), "n": n, "ok": ok.astype(int).tolist(),
                             "worst_mm": dict(zip(WAYPOINTS, (1000 * errs.max(0)).round(2).tolist())),
                             "median_mm": dict(zip(WAYPOINTS, (1000 * np.median(errs, 0)).round(2).tolist()))}
        print(f"{name}: reachable {int(ok.sum())}/{n}  worst mm {res['sets'][name]['worst_mm']}", flush=True)
    if pool:
        pool.close()
    res["all_reachable"] = all(s["reachable"] == s["n"] for s in res["sets"].values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", choices=("lehome", "weld"), default="lehome")
    ap.add_argument("--sim", choices=("isaac", "mujoco"), default="isaac",
                    help="profile weld only: whose arm solves the waypoints (mujoco = the parity reference, .venv)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.profile == "weld":
        res = sweep_mujoco(args.n, args.seed, workers=args.workers, sim=args.sim)
        out = Path(args.out or f"outputs/isaac/reach_mujoco{'' if args.sim == 'isaac' else '_ref'}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("all reachable:", res["all_reachable"])
        return
    args.out = args.out or "outputs/isaac/reach.json"
    res = sweep(args.n, args.seed, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("R_max =", res["r_max_cm"], "cm")


if __name__ == "__main__":
    main()
