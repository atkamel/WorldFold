"""No-physics reachability of the quarter fold's SECOND fold (stage 1) on Isaac, LeHome cloth position (Phase Q2).

After stage 0 the cloth is folded in half along y: the left half (iy <= 5) carries a second layer, corners 0 and 10
stacked at vertex 0's start, the crease along iy = 5. Stage 1 (cloth_fold_rl.quarter_fold_env.STAGES[1]) folds the
west short edge (ix = 0) onto the east one: the right arm carries the stack (0+10) onto vertex 110's start, the left
arm carries the crease end, vertex 5, onto vertex 115's start. The fold axis is x here, so the pinch offset is
mirrored in x for the place target (the stage-0 code mirrors y).

Each arm's pinch (and 4 cm above it) and place targets are solved with isaac.pinch.PinchIK from the demo seeds, on the
stage-0 rest geometry approximated as the start grid (the stack is pinched one layer higher). Reported per arm and
role: worst position errors, and the share of offsets under the stage-0 tolerances (isaac/reach_check.py). Also the
swapped roles (left: stack, right: vertex 5) so the decision to swap is made on numbers. Isaac venv, repo root:

    python isaac/reach_check_qf.py --n 200 --out outputs/isaac/reach_qf.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac.reach_check import PINCH_HEIGHT, PLACE_HEIGHT, SEED_Q, demo_envelope, rest_grid, sample_offset  # noqa
from mujuco.cloth_params import TABLE_TOP_Z  # noqa: E402

LAYER = 0.004     # one cloth layer at rest (reach_check.CLOTH_REST_DZ)
RESTARTS = 6
N = 11
# (arm, carried vertices, goal vertex, mirror axis of the pinch offset at the place: 0 = x, 1 = y)
STAGE1 = {"as_defined": (("right_", (0, 10), 110, 0), ("left_", (5,), 115, 0)),
          "swapped": (("left_", (0, 10), 110, 0), ("right_", (5,), 115, 0)),
          # the alternative: fold the strip in half again along y (crease edge onto the open edge), 4 layers
          "parallel": (("left_", (5,), 0, 1), ("right_", (115,), 110, 1)),
          # a perpendicular fold by handover: each arm's share of the 30 cm carry, west half then east half
          "handover_west": (("left_", (0, 10), 55, 0), ("left_", (5,), 60, 0)),
          "handover_east": (("right_", (55,), 110, 0), ("right_", (60,), 115, 0))}


def _best(ik, p, target, jaw, q, restarts=RESTARTS, **kw):
    """PinchIK is a local solver: a miss from one seed can be a local minimum, not the arm's limit. Best of the chained
    seed, the demo seed and `restarts` seeds spread around it (fixed rng, so the check is deterministic)."""
    rng = np.random.default_rng(0)
    seeds = [np.asarray(q, float), np.asarray(SEED_Q[p], float)]
    seeds += [np.asarray(SEED_Q[p], float) + rng.uniform(-0.8, 0.8, 5) for _ in range(restarts)]
    best = None
    for s in seeds:
        sol = ik.solve(p, target, jaw, s, **kw)
        if best is None or sol[1] < best[1]:
            best = sol
        if best[1] < 1e-4:
            break
    return best


def stage1_errors(ik, offset, moves):
    """{arm: (grasp error, place error)} in metres for the stage-1 moves on a cloth offset."""
    grid = rest_grid(offset)
    folded = grid[[ix * N + iy for ix in range(N) for iy in range(N // 2 + 1)]]   # the two-layer half
    center = folded.mean(axis=0)
    out = {}
    for p, carried, goal_v, mirror in moves:
        corner = grid[list(carried)].mean(axis=0)
        jaw = np.r_[center[:2] - corner[:2], 0.0]
        jaw /= np.linalg.norm(jaw)
        off = -0.005 * jaw[:2]
        layers = len(carried)                       # the stack: pinch the top of two layers
        pinch = np.array([corner[0] + off[0], corner[1] + off[1], TABLE_TOP_Z + PINCH_HEIGHT + LAYER * (layers - 1)])
        goal = grid[goal_v]
        sign = np.array([-1.0, 1.0]) if mirror == 0 else np.array([1.0, -1.0])
        place = np.array([goal[0] + sign[0] * off[0], goal[1] + sign[1] * off[1], TABLE_TOP_Z + PLACE_HEIGHT + LAYER])
        q, grasp = SEED_Q[p], 0.0
        for target in (pinch + [0, 0, 0.04], pinch):
            q, err = _best(ik, p, target, jaw, q)
            grasp = max(grasp, err)
        _, place_err = _best(ik, p, place, jaw, q, orientation_weight=0.05)
        out[(p, tuple(carried))] = (grasp, place_err)
    return out


def main():
    from isaac.pinch import PinchIK
    from imitation.seeds import ISAAC_HARD_RMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac/reach_qf.json")
    args = ap.parse_args()
    ik = PinchIK()
    env_grasp, env_place = demo_envelope(ik)       # stage 0's own errors over the jitter box: the tolerance
    tol_g, tol_p = max(0.004, env_grasp), env_place
    rng = np.random.default_rng([args.seed, 7])
    offsets = [np.zeros(2)] + [sample_offset(rng, ISAAC_HARD_RMAX) for _ in range(args.n - 1)]
    res = {"tol_grasp_mm": round(1000 * tol_g, 2), "tol_place_mm": round(1000 * tol_p, 2), "n": len(offsets),
           "roles": {}}
    print(f"stage-0 envelope: grasp {1000 * env_grasp:.2f} mm, place {1000 * env_place:.2f} mm", flush=True)
    for role, moves in STAGE1.items():
        errs = [stage1_errors(ik, o, moves) for o in offsets]
        role_res = {}
        for p, carried, _, _ in moves:
            e = np.array([x[(p, tuple(carried))] for x in errs])
            ok = (e[:, 0] <= tol_g) & (e[:, 1] <= tol_p)
            role_res[f"{p}{list(carried)}"] = {"carries": list(carried), "reachable": int(ok.sum()),
                           "nominal_mm": [round(1000 * v, 2) for v in e[0]],
                           "worst_grasp_mm": round(1000 * float(e[:, 0].max()), 2),
                           "worst_place_mm": round(1000 * float(e[:, 1].max()), 2)}
            r = role_res[f"{p}{list(carried)}"]
            print(f"{role:13s} {p:6s} carries {carried}: reachable {ok.sum()}/{len(ok)}  nominal "
                  f"{r['nominal_mm']} mm  worst grasp {r['worst_grasp_mm']} "
                  f"place {r['worst_place_mm']} mm", flush=True)
        res["roles"][role] = role_res
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
