"""An open-loop two-arm half fold as a joint trajectory for replay.py, planned on the SO-101 model alone (no camera,
no simulator): each arm pinches its far corner from above, carries it in an arc onto the near corner on its side,
opens slowly, lifts and stands back up. The cloth is placed by ruler where this prints.

    .venv/bin/python hardware/so101/fold_traj.py --out outputs/hardware/half_fold.json [--cloth 0.24] [--near 0.10]
    .venv-hw/bin/python hardware/so101/replay.py outputs/hardware/half_fold.json --left left --right right --dry-run

Rig: both bases flat on the table, facing the same way, --spacing apart; the cloth square and centred between them,
its near edge --near in front of each base's origin (the base's front edge is 6.5 cm in front of the origin).
Angles are so101_new_calib's (so101_nexus), radians, the convention replay.py maps to ticks.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DT = 0.05
STAND = np.array([0.0, -0.2, -1.34, -1.2, 0.0])   # isaac_env.QUARTER_HOME: the arm stands over its base
OPEN, PART_OPEN = 1.0, 0.3     # jaw (rad): open, and how far the release opens with the arm still
HOVER = 0.05          # m above the table before the pinch
PINCH_Z = 0.005       # fingertip height at the pinch
PLACE_Z = 0.012       # fingertip height at the release: the held corner rests on the bottom layer
RETREAT = 0.05        # straight lift after the release
STEP = 0.004          # m of fingertip travel per step (8 cm/s at dt; replay.py's default speed halves it)
JOINT_STEP = 0.04     # rad per step on joint-space moves
BASE_FRONT = 0.065    # m from the base's origin to its front edge
TILT_WEIGHT = 0.03    # IK: metres of tip error per unit of finger-direction error (soft: far pinches may tilt)


class Arm:
    def __init__(self):
        root = importlib.util.find_spec("so101_nexus").submodule_search_locations[0]
        self.model = mujoco.MjModel.from_xml_path(os.path.join(root, "assets", "SO101", "so101_new_calib.xml"))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("gripperframe").id
        self.low, self.high = self.model.jnt_range[:5].T

    def tip(self, q):
        self.data.qpos[:5] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3)[:, 2].copy()

    def solve(self, target, seed, iters=200):
        """Arm angles for the fingertip at `target` (base frame: x forward, y left, z up), fingers as near straight
        down as the reach allows, on the seed's branch; wrist_roll stays 0 (free, it flipped half a turn at the edge
        of reach). Returns (q, tip error m, tilt deg)."""
        down = np.array([0.0, 0.0, -1.0])

        def error(q):
            p, z = self.tip(q)
            return np.concatenate([target - p, TILT_WEIGHT * (down - z)])

        q = np.clip(np.asarray(seed, float), self.low, self.high)
        q[4] = 0.0
        for _ in range(iters):
            e = error(q)
            J = np.column_stack([(e - error(q + dq)) / 1e-6 for dq in np.eye(5)[:4] * 1e-6])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            dq *= min(1.0, 0.1 / max(float(np.max(np.abs(dq))), 1e-12))
            q[:4] = np.clip(q[:4] + dq, self.low[:4], self.high[:4])
        p, z = self.tip(q)
        return q, float(np.linalg.norm(target - p)), float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))


def plan(arm, y, near, cloth, closed):
    """One arm's fold: target rows [T, 6] for the corner column at lateral offset y (m, left positive)."""
    far = near + cloth
    seed = np.radians([0.0, 40.0, -30.0, 80.0, 0.0])     # arm out, forearm level, fingers down
    rows, worst = [], [0.0, 0.0]

    def joint_move(q_goal, jaw):
        q0 = np.array(rows[-1][:5]) if rows else STAND
        n = max(1, int(np.ceil(np.max(np.abs(q_goal - q0)) / JOINT_STEP)))
        rows.extend([*(q0 + (q_goal - q0) * k / n), jaw] for k in range(1, n + 1))

    def line(points, jaw):
        q = np.array(rows[-1][:5])
        for p in points:
            q, err, tilt = arm.solve(np.asarray(p), q)
            worst[0], worst[1] = max(worst[0], err), max(worst[1], tilt)
            rows.append([*q, jaw])

    def hold(steps, jaws):
        rows.extend([*rows[-1][:5], j] for j in np.broadcast_to(jaws, (steps,)))

    def segment(a, b):
        n = max(1, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / STEP)))
        return [np.asarray(a) + np.subtract(b, a) * k / n for k in range(1, n + 1)]

    rows.append([*STAND, OPEN])
    q = seed                                                # the pinch first, then up from it: one branch
    for p in [np.array([far, y, PINCH_Z])] + segment((far, y, PINCH_Z), (far, y, HOVER)):
        q, err, _ = arm.solve(p, q)
        worst[0] = max(worst[0], err)
    joint_move(q, OPEN)
    line(segment((far, y, HOVER), (far, y, PINCH_Z)), OPEN)
    hold(12, np.linspace(OPEN, closed, 12))
    hold(8, closed)
    peak = min(0.10, 0.45 * cloth)                          # under the cloth's half length: the carry never pulls it
    n = int(np.ceil(1.3 * cloth / STEP))
    s = np.arange(1, n + 1) / n
    line([(far - cloth * k, y, PINCH_Z + (PLACE_Z - PINCH_Z) * k + peak * np.sin(np.pi * k)) for k in s], closed)
    hold(10, closed)
    hold(14, np.linspace(closed, PART_OPEN, 14))            # gentle release: the jaw opens part way, arm still
    line(segment((near, y, PLACE_Z), (near, y, PLACE_Z + RETREAT)), PART_OPEN)
    hold(8, np.linspace(PART_OPEN, OPEN, 8))
    joint_move(STAND, OPEN)
    return np.array(rows), worst


def trajectory(cloth=0.24, near=0.10, spacing=0.30, closed=-0.17):
    arm = Arm()
    inset = spacing / 2 - cloth / 2          # the cloth's corner column sits this far toward the middle of the rig
    arms, report = {}, {}
    for side, y in (("left", -inset), ("right", inset)):
        target, (err, tilt) = plan(arm, y, near, cloth, closed)
        q = np.vstack([target[:1], target])                  # replay.py wants T + 1 rows; open loop, q = the targets
        tips = [arm.tip(r[:5])[0] for r in q]
        arms[side] = {"target": target.tolist(), "q": q.tolist(),
                      "tip": [[-p[1], p[0], p[2]] for p in np.array(tips).tolist()]}   # right, forward, up
        report[side] = {"tip_error_mm": round(err * 1000, 1), "max_tilt_deg": round(tilt, 1)}
    return {"dt": DT, "joints": JOINTS, "arms": arms,
            "layout": {"frame": "each arm's base, on the table", "base_spacing": spacing, "cloth_side": cloth,
                       "near_edge_from_base_origin": near, "near_edge_from_base_front": round(near - BASE_FRONT, 3)},
            "source": {"tool": "hardware/so101/fold_traj.py", "closed_rad": closed, **report}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cloth", type=float, default=0.24, help="cloth side (m)")
    ap.add_argument("--near", type=float, default=0.10, help="near edge in front of the base origin (m)")
    ap.add_argument("--spacing", type=float, default=0.30, help="between the two base origins (m)")
    ap.add_argument("--closed", type=float, default=-0.17, help="jaw target at the pinch (rad); the URDF closed limit is -0.1745")
    args = ap.parse_args()
    traj = trajectory(args.cloth, args.near, args.spacing, args.closed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(traj))
    T = len(traj["arms"]["left"]["target"])
    print(f"{T} steps, {T * DT:.0f} s at full speed. IK: {traj['source']['left']} / {traj['source']['right']}")
    print(f"Cloth: {args.cloth * 100:.0f} cm square, centred between the bases, near edge "
          f"{(args.near - BASE_FRONT) * 100:.1f} cm in front of the bases' front edges.")


if __name__ == "__main__":
    main()
