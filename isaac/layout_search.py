"""No-physics layout search for the quarter fold on Isaac: arm base poses around the 30 cm cloth from which both folds'
pinches, carry arcs and places are reachable, with the arms clear of each other.

Cloth-local frame, cloth centred at the origin, h = 0.15 m. Fold 1 (north onto south) is two-armed: arm 1 carries
the west corner (-h, h) onto (-h, -h) (move A0), arm 2 the east corner (B0). Fold 2 (west onto east) on the half
sheet y in [-h, 0] is either two-armed, the stack corner (-h, -h) (S) and the crease end (-h, 0) (C) carried to
their mirrors, or one-armed from the short edge's midpoint (-h, -h/2) (M). Each move follows IsaacArmExpert's plan:
pinch PINCH_INSET in from the point toward the sheet's centre, straight down at table + PINCH_HEIGHT, from ABOVE
over it; a half-ellipse arc rising ARC_HEIGHT to the place, the pinch point's mirror across the fold line; retreat.
IK is isaac.pinch.PinchIK with the expert's orientation weights; the arm base sits at table + 6 cm (as now), facing
the cloth centre. Run in either venv (with --assets pointing at a LeHome Assets copy when lehome isn't installed):

    python isaac/layout_search.py --assets outputs/isaac/assets --out outputs/isaac/layout_search.json
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mujuco.cloth_params import ARM_BASE_LEFT, CLOTH_COUNT, CLOTH_SPACING, TABLE_TOP_Z   # noqa: E402

H = (CLOTH_COUNT - 1) * CLOTH_SPACING / 2
BASE_Z = ARM_BASE_LEFT[2]
PINCH_HEIGHT, PLACE_HEIGHT, PINCH_INSET, ABOVE, RETREAT = 0.005, 0.011, 0.005, 0.04, 0.05   # IsaacArmExpert
ARC_HEIGHT, ARC_WAYPOINTS = 0.12, 8
PINCH_TOL, PLACE_TOL, ARC_TOL = 0.00466, 0.006, 0.02        # reach_check's grasp bound; a tight place; path
# PinchIK trades orientation for position, so reach is bounded by tilt. Bounds = what today's half fold already does:
# over the +-1.5 cm shifted poses (F3 expert id_hard 94%) its pinches tilt up to 16.3 deg and its places 32.4 deg
# (nominal 5.5 / 26). The friction pinch needs a near-vertical jaw; the README's 35 deg tilted pinch didn't hold.
PINCH_TILT, PLACE_TILT = 16.5, 33.0
# Above the pinch and at it, elbow_flex and wrist_flex stay at or under PINCH_FOLD: folded tighter, the jaw meets the
# arm's own upper link (self-collision on) and the joints stop short (2026-10-15 run q4: the north arm's pinch 11 cm
# from its base asked elbow 1.67 / wrist 1.40 and got 1.56 / 1.14). The half fold pinches at elbow -0.07, wrist 0.92.
PINCH_FOLD = 1.3
OFFSETS = [(dx, dy) for dx in (-0.01, 0.0, 0.01) for dy in (-0.01, 0.0, 0.01)]   # cloth reset jitter, +-1 cm
CLEARANCE = 0.06                                            # link centrelines: ~2 cm link radius each + 2 cm
# move: (pinch point, place point's mirror axis, jaw aim point); the place is the pinch's mirror across x = 0 or y = 0
MOVES = {
    "A0": ((-H, H), "y", (0.0, 0.0)),
    "B0": ((H, H), "y", (0.0, 0.0)),
    "S": ((-H, -H), "x", (0.0, -H / 2)),
    "C": ((-H, 0.0), "x", (0.0, -H / 2)),
    "M": ((-H, -H / 2), "x", (0.0, -H / 2)),
}
SEED_REST = ([1.0654, -1.1908, 1.6581, -2.046], [1.0622, -1.1853, 1.6581, 2.1429])   # SEED_Q without the pan
_IK = None


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _waypoints(move):
    corner, axis, aim = MOVES[move]
    corner, aim = np.array(corner), np.array(aim)
    jaw = np.r_[(aim - corner) / np.linalg.norm(aim - corner), 0.0]
    pinch = np.r_[corner + PINCH_INSET * jaw[:2], TABLE_TOP_Z + PINCH_HEIGHT]
    place = pinch.copy()
    place[0 if axis == "x" else 1] *= -1
    place[2] = TABLE_TOP_Z + PLACE_HEIGHT
    arc = []
    for s in np.linspace(0.0, 1.0, ARC_WAYPOINTS + 1)[1:]:
        tip = pinch + (place - pinch) * (1.0 - math.cos(math.pi * s)) / 2.0
        tip[2] = pinch[2] + (place[2] - pinch[2]) * s + ARC_HEIGHT * math.sin(math.pi * s)
        arc.append(tip)
    return jaw, pinch, arc, place


def _configure(base_dz, half):
    """Mount height change and cloth half-size for this process (module constants, so pool workers need it too)."""
    global BASE_Z, H, MOVES
    BASE_Z, H = ARM_BASE_LEFT[2] + base_dz, half
    MOVES = {"A0": ((-H, H), "y", (0.0, 0.0)), "B0": ((H, H), "y", (0.0, 0.0)), "S": ((-H, -H), "x", (0.0, -H / 2)),
             "C": ((-H, 0.0), "x", (0.0, -H / 2)), "M": ((-H, -H / 2), "x", (0.0, -H / 2))}


def _init_worker(assets, base_dz, half, offsets=None):
    _configure(base_dz, half)
    if offsets:
        OFFSETS[:] = offsets
    _init(assets)


def _init(assets):
    global _IK
    try:
        import lehome.utils.constant   # noqa: F401
    except ImportError:
        mod = types.ModuleType("lehome.utils.constant")
        mod.ASSETS_ROOT = str(Path(assets).resolve())
        sys.modules.update({"lehome": types.ModuleType("lehome"), "lehome.utils": types.ModuleType("lehome.utils"),
                            "lehome.utils.constant": mod})
    from isaac.pinch import PinchIK
    _IK = PinchIK()


def plan_move(base_xy, move):
    """IK along one move from a base at base_xy facing the cloth centre: errors (m) and the joint path."""
    ik, p = _IK, "left_"
    yaw = math.atan2(-base_xy[1], -base_xy[0])
    ik.root[p] = (np.array([base_xy[0], base_xy[1], BASE_Z]), _yaw_quat(yaw))
    jaw, pinch, arc, place = _waypoints(move)
    starts = []
    for pan in np.linspace(-1.8, 1.8, 7):
        for rest in SEED_REST:
            q, e = ik.solve(p, pinch + [0, 0, ABOVE], jaw, [pan] + rest, iters=150)
            starts.append((e, -ik.site_pose(p, q)[1][2, 2], q))
    starts.sort(key=lambda t: (round(t[0], 4), -t[1]))          # closest, then most upright
    runs = [_follow(ik, p, q, e, jaw, pinch, arc, place) for e, _, q in starts[:6]]   # the chain can flip branch
    return min(runs, key=_badness)


def _follow(ik, p, q, e_above, jaw, pinch, arc, place):
    path = [q]
    q, e_pinch = ik.solve(p, pinch, jaw, q)
    tilt = math.degrees(math.acos(min(1.0, -ik.site_pose(p, q)[1][2, 2])))   # fingers off vertical at the pinch
    fold = float(max(path[0][2], path[0][3], q[2], q[3]))                     # elbow / wrist, above and at the pinch
    path.append(q)
    arc_err = []
    for tip in arc:
        q, e = ik.solve(p, tip, jaw, q, orientation_weight=0.05, iters=150)
        path.append(q)
        arc_err.append(e)
    q, _ = ik.solve(p, place + [0, 0, RETREAT], jaw, q, orientation_weight=0.02, iters=150)
    path.append(q)
    place_tilt = math.degrees(math.acos(min(1.0, -ik.site_pose(p, path[-2])[1][2, 2])))   # at the place
    return {"pinch": max(e_above, e_pinch), "tilt": tilt, "fold": fold, "place_tilt": place_tilt, "arc": max(arc_err[:-1]),
            "place": arc_err[-1],
            "path": [list(map(float, x)) for x in path]}


def _job(args):
    base_xy, move = args
    return (tuple(base_xy), move, plan_move(base_xy, move))


def _robust_job(args):
    base_xy, move = args
    return (tuple(base_xy), move, max(_badness(plan_move((base_xy[0] - dx, base_xy[1] - dy), move))
                                      for dx, dy in OFFSETS))


def _reachable(r):
    return _badness(r) <= 1.0


def _badness(r):
    return max(r["pinch"] / PINCH_TOL, r["tilt"] / PINCH_TILT, r["place"] / PLACE_TOL, r["place_tilt"] / PLACE_TILT,
               r["arc"] / ARC_TOL, r["fold"] / PINCH_FOLD)


def arm_points(base_xy, q):
    """World points along the arm's chain (base, every joint origin, the gripperframe site) for clearance."""
    ik = _IK
    yaw = math.atan2(-base_xy[1], -base_xy[0])
    from isaac.isaac_env import _matrix_from_quat
    R0, p0 = _matrix_from_quat(_yaw_quat(yaw)), np.array([base_xy[0], base_xy[1], BASE_Z])
    T, k, pts = np.eye(4), 0, [p0.copy()]
    from isaac.pinch import _rot
    for origin, axis, _ in ik.chain:
        T = T @ origin
        if axis is not None:
            R = np.eye(4)
            R[:3, :3] = _rot(axis, q[k])
            T, k = T @ R, k + 1
        pts.append(p0 + R0 @ T[:3, 3])
    return np.array(pts)


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    s = np.clip((b * t - c) / a, 0, 1) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def clearance(base1, path1, base2, path2):
    """Smallest distance between the two arms' chains with both paths run in lockstep (resampled to one length)."""
    n = max(len(path1), len(path2))
    worst = np.inf
    for i in range(n):
        a = arm_points(base1, path1[round(i * (len(path1) - 1) / (n - 1))])
        b = arm_points(base2, path2[round(i * (len(path2) - 1) / (n - 1))])
        for j in range(1, len(a) - 1):          # skip the base -> shoulder pedestal stub
            for k in range(1, len(b) - 1):
                worst = min(worst, _seg_dist(a[j], a[j + 1], b[k], b[k + 1]))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="outputs/isaac/assets")
    ap.add_argument("--out", default="outputs/isaac/layout_search.json")
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--margin", type=float, default=0.05, help="base centre at least this far outside the cloth")
    ap.add_argument("--base-dz", type=float, default=0.0, help="arm mount height change (m)")
    ap.add_argument("--cloth", type=float, default=2 * H, help="cloth side (m)")
    ap.add_argument("--jitter", type=float, default=0.01, help="cloth offset (+-m, both axes) the robust check covers")
    ap.add_argument("--robust", type=int, default=15, help="re-check this many top layouts per variant over OFFSETS")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()
    _configure(args.base_dz, args.cloth / 2)
    OFFSETS[:] = [(dx, dy) for dx in (-args.jitter, 0.0, args.jitter) for dy in (-args.jitter, 0.0, args.jitter)]
    _init(args.assets)

    grid = np.arange(-0.5, 0.5 + 1e-9, args.step)
    jobs = []
    for x in grid:
        for y in grid:
            outside = max(abs(x) - H, abs(y) - H)
            if outside < args.margin:
                continue
            for move in MOVES:
                _, pinch, _, place = _waypoints(move)
                d = [math.hypot(pt[0] - x, pt[1] - y) for pt in (pinch, place)]
                if max(d) <= 0.40 and min(d) >= 0.06:
                    jobs.append(((round(float(x), 4), round(float(y), 4)), move))
    print(f"{len(jobs)} (base, move) IK jobs", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.assets, args.base_dz, args.cloth / 2)) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    table = {}
    for base, move, r in res:
        table.setdefault(base, {})[move] = r
    ok = {m: [b for b in table if m in table[b] and _reachable(table[b][m])] for m in MOVES}
    print("bases reaching each move:", {m: len(v) for m, v in ok.items()}, flush=True)

    # pairings: arm 1 does A0 (+ its fold-2 move), arm 2 does B0 (+ its fold-2 move)
    variants = {"S/C": ("S", "C"), "C/S": ("C", "S"), "M/-": ("M", None), "-/M": (None, "M")}
    found = []
    for name, (m1, m2) in variants.items():
        b1s = [b for b in ok["A0"] if m1 is None or b in ok[m1]]
        b2s = [b for b in ok["B0"] if m2 is None or b in ok[m2]]
        for b1 in b1s:
            for b2 in b2s:
                if math.hypot(b1[0] - b2[0], b1[1] - b2[1]) < 0.12:
                    continue
                found.append((name, b1, b2))
    print(f"{len(found)} reach-feasible layouts before clearance", flush=True)
    rows = []
    for name, b1, b2 in found:
        m1, m2 = variants[name]
        c0 = clearance(b1, table[b1]["A0"]["path"], b2, table[b2]["B0"]["path"])
        if m1 and m2:
            c1 = clearance(b1, table[b1][m1]["path"], b2, table[b2][m2]["path"])
        else:   # the idle arm parked where fold 1 left it (its retreat pose)
            mover, idle = (b1, b2) if m1 else (b2, b1)
            idle_q = table[idle]["A0" if idle == b1 else "B0"]["path"][-1]
            c1 = clearance(mover, table[mover]["M"]["path"], idle, [idle_q])
        worst_place = max(table[b][m]["place"] for b, m in ((b1, "A0"), (b2, "B0"), (b1, m1), (b2, m2)) if m)
        margin = max(_badness(table[b][m]) for b, m in ((b1, "A0"), (b2, "B0"), (b1, m1), (b2, m2)) if m)
        rows.append({"variant": name, "arm1": b1, "arm2": b2, "clear_fold1": c0, "clear_fold2": c1,
                     "worst_place_mm": 1000 * worst_place, "badness": margin})
    rows.sort(key=lambda r: -min(r["clear_fold1"], r["clear_fold2"]))
    good = [r for r in rows if min(r["clear_fold1"], r["clear_fold2"]) >= CLEARANCE]
    print(f"{len(good)} layouts with clearance >= {CLEARANCE} m", flush=True)
    check = [r for v in variants for r in sorted((r for r in good if r["variant"] == v),
                                                 key=lambda r: r["badness"])[:args.robust]]   # most reach margin first
    need = {(tuple(r[a]), m) for r in check for a, m in zip(("arm1", "arm2", "arm1", "arm2"),
                                                            ("A0", "B0") + variants[r["variant"]]) if m}
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.assets, args.base_dz, args.cloth / 2, list(OFFSETS))) as pool:
        worst = {(b, m): w for b, m, w in pool.map(_robust_job, sorted(need), chunksize=1)}
    for r in check:
        r["robust_badness"] = max(worst[(tuple(r[a]), m)] for a, m in zip(("arm1", "arm2", "arm1", "arm2"),
                                                                          ("A0", "B0") + variants[r["variant"]]) if m)
    print(f"{sum(r['robust_badness'] <= 1.0 for r in check)} of {len(check)} hold over +-1 cm offsets", flush=True)
    rows = sorted(check, key=lambda r: r["robust_badness"]) + [r for r in rows if r not in check]
    for v in variants:
        best = [r for r in rows if r["variant"] == v and "robust_badness" in r][:1]
        for r in best:
            print("best", v, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
    for r in rows[:15]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "ok": {m: v for m, v in ok.items()},
                                          "table": {f"{b[0]},{b[1]}": {m: {k: v for k, v in r.items() if k != "path"}
                                                                         for m, r in t.items()} for b, t in table.items()}},
                                         indent=1))


if __name__ == "__main__":
    main()
