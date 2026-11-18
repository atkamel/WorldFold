"""An open-loop two-arm half fold as a joint trajectory for replay.py, planned on the SO-101 model alone (no camera,
no simulator): each arm pinches its far corner from above, carries it in an arc onto the near corner on its side,
opens slowly, lifts and stands back up. The cloth is placed by ruler where this prints. Above the far corner the jaw
shuts once on nothing, so replay.py --grasp-check has an empty reading to compare the lifted jaw against ("marks").

    .venv/bin/python hardware/so101/fold_traj.py --out outputs/hardware/half_fold.json [--cloth 0.24] [--near 0.10]
    .venv-hw/bin/python hardware/so101/replay.py outputs/hardware/half_fold.json --left left --right right --dry-run

Rig: both bases flat on the table, facing the same way, --spacing apart; the cloth square and centred between them,
its near edge --near in front of each base's origin (the base's front edge is 6.5 cm in front of the origin).
Angles are so101_new_calib's (so101_nexus), radians, the convention replay.py maps to ticks.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import collision
from kinematics import Arm

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DT = 0.05
STAND = np.array([0.0, -0.2, -1.34, -1.2, 0.0])   # isaac_env.QUARTER_HOME: the arm stands over its base
OPEN, PART_OPEN = 1.0, 0.3     # jaw (rad): open, and how far the release opens with the arm still
HOVER = 0.05          # m above the table before the pinch
PINCH_Z = 0.005       # fingertip height at the pinch
PLACE_Z = 0.012       # fingertip height at the release: the held corner rests on the bottom layer
RETREAT = 0.05        # lift after the release (straight up where the arm can, see plan)
STEP = 0.004          # m of fingertip travel per step (8 cm/s at dt; replay.py's default speed halves it)
JOINT_STEP = 0.04     # rad per step on joint-space moves
JOINT_RATE = np.radians(55) * DT    # rad per step a joint may be asked for in the carry (replay.py caps at 60 deg/s)
BASE_FRONT = 0.065    # m from the base's origin to its front edge
SHORT_STEP = 0.005    # m: the release backs away from the base in steps of this
MAX_SHORT = 0.10      # m the release may stop short of the place it was asked for
MIN_FOLD = 0.02       # m of fold that must be left after shortening


def plan(arm, pinch, place, closed, counts=None):
    """One arm's fold from `pinch` to `place` ((x, y) m in its base frame, x forward, y left): target rows [T, 6],
    the worst (tip error, tilt), marks {"empty": the jaw shut in the air, "hover": above the corner with the jaw open
    again, "pinch": down on the corner, "check": 2 cm into the lift, "placed": the carry's end} as row indices, and the step counts of its three variable moves. `counts`
    stretches those moves to given lengths, so two arms' plans line up row for row."""
    pinch, place = np.asarray(pinch, float), np.asarray(place, float)
    reach = float(np.linalg.norm(place - pinch))
    seed = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])     # arm out, forearm level, fingers down
    rows, worst, used = [], [0.0, 0.0], {}

    def joint_move(name, q_goal, jaw):
        q0 = np.array(rows[-1][:5]) if rows else STAND
        used[name] = max(1, int(np.ceil(np.max(np.abs(q_goal - q0)) / JOINT_STEP)))
        n = max(used[name], (counts or {}).get(name, 0))
        rows.extend([*(q0 + (q_goal - q0) * k / n), jaw] for k in range(1, n + 1))

    def line(points, jaw):
        q = np.array(rows[-1][:5])
        for p in points:
            q, err, tilt = arm.solve(np.asarray(p), q)
            worst[0], worst[1] = max(worst[0], err), max(worst[1], tilt)
            rows.append([*q, jaw])

    def hold(steps, jaws):
        rows.extend([*rows[-1][:5], j] for j in np.broadcast_to(jaws, (steps,)))

    def move_jaw(start, end, least):
        """The arm still, the jaw from `start` to `end` in at least `least` rows and never faster than JOINT_RATE:
        replay.py holds a faster jaw back, and it then lags the rest of the plan."""
        steps = max(least, int(np.ceil(abs(end - start) / JOINT_RATE)) + 1)
        hold(steps, np.linspace(start, end, steps))

    def segment(a, b):
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / STEP)))
        return [np.asarray(a) + np.subtract(b, a) * k / n for k in range(1, n + 1)]

    rows.append([*STAND, OPEN])
    q = seed                                                # the pinch first, then up from it: one branch
    for p in [np.array([*pinch, PINCH_Z])] + segment((*pinch, PINCH_Z), (*pinch, HOVER)):
        q, err, _ = arm.solve(p, q)
        worst[0] = max(worst[0], err)
    joint_move("approach", q, OPEN)
    move_jaw(OPEN, closed, 10)
    hold(6, closed)
    marks = {"empty": len(rows) - 1}
    move_jaw(closed, OPEN, 10)
    marks["hover"] = len(rows) - 1
    line(segment((*pinch, HOVER), (*pinch, PINCH_Z)), OPEN)
    marks["pinch"] = len(rows) - 1
    move_jaw(OPEN, closed, 12)
    hold(8, closed)
    lift = len(rows)
    peak = min(0.10, 0.45 * reach)                          # under the cloth's half length: the carry never pulls it

    def arc(n):
        return [(*(pinch + (place - pinch) * k), PINCH_Z + (PLACE_Z - PINCH_Z) * k + peak * np.sin(np.pi * k))
                for k in np.arange(1, n + 1) / n]

    n = int(np.ceil(1.3 * reach / STEP))
    if counts is None:                                      # with the wrist at its limit the tip's speed asks more of
        q, fastest = np.array(rows[-1][:5]), 0.0            # the other joints than replay.py lets them give: slow the
        for p in arc(n):                                    # carry until the fastest joint step is under JOINT_RATE
            new = arm.solve(np.asarray(p), q)[0]
            q, fastest = new, max(fastest, float(np.max(np.abs(new - q))))
        n = int(np.ceil(n * max(1.0, fastest / JOINT_RATE)))
    used["carry"] = n
    line(arc(max(n, (counts or {}).get("carry", 0))), closed)
    marks["check"] = next(k for k in range(lift, len(rows)) if arm.tip(rows[k][:5])[0][2] >= PINCH_Z + 0.02)
    marks["placed"] = len(rows) - 1
    hold(10, closed)
    move_jaw(closed, PART_OPEN, 14)                         # gentle release: the jaw opens part way, arm still
    out = place / np.linalg.norm(place)                     # away from the base
    for lean in np.arange(0.0, RETREAT + 1e-9, 0.01):       # under ~12 cm from the base the arm cannot lift straight
        lift = segment((*place, PLACE_Z), (*(place + lean * out), PLACE_Z + RETREAT))   # up: lean the lift outward
        q, miss = np.array(rows[-1][:5]), 0.0               # by the least that lets the tip follow it
        for p in lift:
            q, err, _ = arm.solve(p, q)
            miss = max(miss, err)
        if miss < 1e-3:
            break
    line(lift, PART_OPEN)
    move_jaw(PART_OPEN, OPEN, 8)
    joint_move("back", STAND, OPEN)
    return np.array(rows), worst, marks, used


def release_pose_is_clear(arm, pinch, place, closed):
    """True if the arm keeps clear of itself with the fingertip at the release point `place`, jaw shut and jaw open.
    One pose, reached along the table from `pinch`: a quick test before a whole plan is made and checked."""
    q = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])
    start = np.array([pinch[0], pinch[1], PINCH_Z])
    end = np.array([place[0], place[1], PLACE_Z])
    steps = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.02)))
    for k in range(steps + 1):
        q = arm.solve(start + (end - start) * k / steps, q)[0]
    angles = [float(angle) for angle in q]
    distance = collision.closest([angles + [closed], angles + [OPEN]])[0]
    return distance >= collision.CLEARANCE


def clear_release(arm, pinch, place, closed):
    """(release point, m it stops short of `place`, the plan's step counts): the point nearest `place`, on the line
    back toward `pinch`, whose whole plan keeps the arm clear of itself. ValueError if there is none."""
    pinch = np.asarray(pinch, float)
    place = np.asarray(place, float)
    reach = float(np.linalg.norm(pinch - place))
    if reach < MIN_FOLD:
        raise ValueError("the pinch and the place are the same point: nothing to fold")
    back = (pinch - place) / reach
    short = 0.0
    while short <= MAX_SHORT and reach - short >= MIN_FOLD:
        candidate = place + short * back
        if release_pose_is_clear(arm, pinch, candidate, closed):
            rows, _, _, used = plan(arm, pinch, candidate, closed)
            if collision.closest(rows.tolist())[0] >= collision.CLEARANCE:
                return candidate, short, used
        short += SHORT_STEP
    raise ValueError(f"no release within {MAX_SHORT * 100:.0f} cm of the place keeps the arm clear of itself")


def fold(points, closed=-0.17):
    """A two-arm trajectory for replay.py from {"left": (pinch, place), "right": (pinch, place)}, each (x, y) m in
    that arm's base frame. A place the arm cannot reach without hitting its own base is moved back toward the pinch
    ("release_short_mm"). Both plans get the same step counts; the grasp check comes when both have lifted 2 cm."""
    arm = Arm()
    asked = points
    points = {}
    shorts = {}
    natural = []
    for side in ("left", "right"):
        pinch, place = asked[side]
        release, short, used = clear_release(arm, pinch, place, closed)
        points[side] = (pinch, (float(release[0]), float(release[1])))
        shorts[side] = short
        natural.append(used)
    counts = {name: max(n[name] for n in natural) for name in natural[0]}
    arms, report, checks = {}, {}, []
    for side in ("left", "right"):
        target, (err, tilt), marks, _ = plan(arm, *points[side], closed, counts)
        clearance = collision.closest(target.tolist())[0]
        if clearance < collision.CLEARANCE:
            raise ValueError(f"the {side} arm comes within {clearance * 1000:.0f} mm of itself")
        checks.append(marks["check"])
        q = np.vstack([target[:1], target])                  # replay.py wants T + 1 rows; open loop, q = the targets
        tips = np.array([arm.tip(r[:5])[0] for r in q])
        pinch_err = float(np.linalg.norm(tips[marks["pinch"] + 1, :2] - np.asarray(points[side][0])))
        arms[side] = {"target": target.tolist(), "q": q.tolist(),
                      "tip": [[-p[1], p[0], p[2]] for p in tips.tolist()]}   # right, forward, up
        place_err = float(np.linalg.norm(tips[marks["placed"] + 1, :2] - np.asarray(points[side][1])))
        report[side] = {"tip_error_mm": round(err * 1000, 1), "max_tilt_deg": round(tilt, 1),
                        "pinch_error_mm": round(pinch_err * 1000, 1), "place_error_mm": round(place_err * 1000, 1),
                        "release_short_mm": round(shorts[side] * 1000, 1),
                        "self_clearance_mm": round(clearance * 1000, 1)}
    return {"dt": DT, "joints": JOINTS, "arms": arms, "marks": marks | {"check": max(checks)},
            "source": {"tool": "hardware/so101/fold_traj.py", "closed_rad": closed, **report}}


def trajectory(cloth=0.24, near=0.10, spacing=0.30, closed=-0.17):
    """The fold for a cloth placed by ruler: square, centred between the bases, near edge `near` in front of them."""
    inset = spacing / 2 - cloth / 2          # the cloth's corner column sits this far toward the middle of the rig
    traj = fold({side: ((near + cloth, y), (near, y)) for side, y in (("left", -inset), ("right", inset))}, closed)
    traj["layout"] = {"frame": "each arm's base, on the table", "base_spacing": spacing, "cloth_side": cloth,
                      "near_edge_from_base_origin": near, "near_edge_from_base_front": round(near - BASE_FRONT, 3)}
    return traj


def sides_trajectory(cloth=0.30, gap=0.16, closed=-0.17):
    """The fold with an arm at the middle of each side edge of the cloth, facing in, its base origin `gap` outside
    the edge. Each arm pinches the far corner on its side and places it on the near one, so it reaches across half
    the cloth and not all of it. "left" is the arm on the left seen from the near edge, the one folded onto."""
    half = cloth / 2
    traj = fold({"left": ((gap, half), (gap, -half)), "right": ((gap, -half), (gap, half))}, closed)
    traj["layout"] = {"frame": "each arm's base, on the table", "arms": "at the middle of the side edges, facing in",
                      "base_spacing": cloth + 2 * gap, "cloth_side": cloth, "side_edge_from_base_origin": gap}
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cloth", type=float, default=0.24, help="cloth side (m)")
    ap.add_argument("--near", type=float, default=0.10, help="near edge in front of the base origin (m)")
    ap.add_argument("--spacing", type=float, default=0.30, help="between the two base origins (m)")
    ap.add_argument("--closed", type=float, default=-0.17, help="jaw target at the pinch (rad); the URDF closed limit is -0.1745")
    ap.add_argument("--sides", type=float, metavar="GAP", help="the arms at the middle of the cloth's side edges, "
                    "facing in, each base origin GAP (m) outside its edge (0.16 or more); --near and --spacing unused")
    args = ap.parse_args()
    if args.sides is None:
        traj = trajectory(args.cloth, args.near, args.spacing, args.closed)
    else:
        traj = sides_trajectory(args.cloth, args.sides, args.closed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(traj))
    T = len(traj["arms"]["left"]["target"])
    print(f"{T} steps, {T * DT:.0f} s at full speed. IK: {traj['source']['left']} / {traj['source']['right']}")
    if args.sides is None:
        print(f"Cloth: {args.cloth * 100:.0f} cm square, centred between the bases, near edge "
              f"{(args.near - BASE_FRONT) * 100:.1f} cm in front of the bases' front edges.")
    else:
        print(f"Bases facing each other, origins {(args.cloth + 2 * args.sides) * 100:.0f} cm apart. Cloth: "
              f"{args.cloth * 100:.0f} cm square, centred between them, its middle on the line through the two "
              f"origins, each side edge {(args.sides - BASE_FRONT) * 100:.1f} cm in front of a base's front edge.")
    for side in ("left", "right"):
        short = traj["source"][side]["release_short_mm"]
        if short > 0:
            print(f"The {side} arm releases {short / 10:.1f} cm short of the near edge: nearer, it would hit its own base.")


if __name__ == "__main__":
    main()
