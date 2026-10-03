"""SO-101 kinematics from the challenge URDF, in plain numpy (no simulator needed, so it can be tested offline).

FK of the jaw-tip frame (gripper_frame_link) and a damped-least-squares IK for
[tip position (3) + approach direction (the tip frame's +z, i.e. where the jaws point)].
World pose of each arm base comes from the env config (garment_bi_cfg_v2.py).
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_new_calib.urdf")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripper_frame_link"

# env config: both bases at z=0.5. The USD robot's forward axis differs from the URDF's, net world = Rz(+90 deg) * URDF base
# (checked against recorded teacher joint states: tips land at table height on the towel corners at grasp time).
# The USD "base" body origin is not the URDF base_link origin: every sim link sits (-0.0208, -0.0157, +0.0325) m from
# the URDF prediction, identically for both arms (measured in oracle run 1, see kin_check in oracle_fold.py).
_USD_OFFSET = np.array([-0.0208, -0.0157, 0.0325])
BASE_POS = {"left": np.array([-0.23, -0.25, 0.5]) + _USD_OFFSET, "right": np.array([0.23, -0.25, 0.5]) + _USD_OFFSET}
BASE_ROT = np.array([[0.0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


class SO101:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz").split()]
            rpy = [float(v) for v in o.get("rpy").split()]
            lim = j.find("limit")
            self.joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                T=_T(_rpy(*rpy), xyz),
                lim=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None)
        self.by_child = {v["child"]: k for k, v in self.joints.items()}
        # chain base_link -> TIP
        chain, link = [], TIP
        while link in self.by_child:
            name = self.by_child[link]; chain.append(name); link = self.joints[name]["parent"]
        self.chain = chain[::-1]
        self.lo = np.array([self.joints[n]["lim"][0] for n in ARM_JOINTS])
        self.hi = np.array([self.joints[n]["lim"][1] for n in ARM_JOINTS])
        self.grip_lim = self.joints["gripper"]["lim"]

    def fk_base(self, q):
        """q: 5 arm joints (rad). Returns 4x4 pose of the jaw-tip frame in the arm's base_link frame."""
        qd = dict(zip(ARM_JOINTS, q))
        M = np.eye(4)
        for name in self.chain:
            j = self.joints[name]
            M = M @ j["T"]
            if j["type"] == "revolute":
                M = M @ _rotz(qd[name])
        return M

    def fk(self, q, side):
        """Jaw-tip pose in the world frame for the 'left' or 'right' arm."""
        return _T(BASE_ROT, BASE_POS[side]) @ self.fk_base(q)

    def tip(self, q, side):
        M = self.fk(q, side)
        return M[:3, 3], M[:3, 2]          # position, approach direction (+z of tip frame)

    def _res(self, q, side, pos, direction, w_dir):
        p, a = self.tip(q, side)
        r = p - pos
        if direction is not None and w_dir > 0:
            r = np.concatenate([r, w_dir * (a - direction)])
        return r

    def ik(self, pos, side, q0, direction=(0, 0, -1), w_dir=0.005, iters=200, damp=1e-4, tol=3e-4, free=(0, 1, 2, 3)):
        """DLS IK over the `free` joints (default: all but wrist_roll, which only spins the jaws).
        direction = desired approach unit vector (None = position only); w_dir (metres per unit of
        direction error) makes orientation a soft goal, so position wins near the edge of reach.
        Returns (q, pos_err_m)."""
        pos = np.asarray(pos, float)
        direction = None if direction is None else np.asarray(direction, float) / np.linalg.norm(direction)
        free = list(free)
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        for _ in range(iters):
            r = self._res(q, side, pos, direction, w_dir)
            if np.linalg.norm(r[:3]) < tol and (direction is None or np.linalg.norm(r[3:]) < 0.03 * w_dir):
                break
            J = np.zeros((len(r), len(free))); eps = 1e-5
            for c, i in enumerate(free):
                dq = q.copy(); dq[i] += eps
                J[:, c] = (self._res(dq, side, pos, direction, w_dir) - r) / eps
            step = -np.linalg.solve(J.T @ J + damp * np.eye(len(free)), J.T @ r)
            n = np.linalg.norm(step)
            if n < 2e-4:
                break
            if n > 0.2:
                step *= 0.2 / n
            q[free] = np.clip(q[free] + step, self.lo[free], self.hi[free])
        return q, float(np.linalg.norm(self.tip(q, side)[0] - pos))

    def roll_for_axis(self, q, side, axis, near=None):
        """wrist_roll that best aligns the jaw-opening axis (tip frame x) with `axis` (sign-free).
        Of the two equivalent rolls (180 deg apart) returns the one nearest `near` (default: current)."""
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        near = q[4] if near is None else near
        best = None
        for r in np.linspace(self.lo[4], self.hi[4], 361):
            qq = np.asarray(q, float).copy(); qq[4] = r
            a = abs(float(self.fk(qq, side)[:3, 0] @ axis))
            score = a - 0.02 * abs(r - near)
            if best is None or score > best[0]:
                best = (score, r, a)
        return best[1], best[2]

    def ik_best(self, pos, side, seeds, **kw):
        """Try several start configurations, return the solution with the smallest position error."""
        best = None
        for s in seeds:
            q, e = self.ik(pos, side, s, **kw)
            if best is None or e < best[1] - 1e-5:
                best = (q, e)
        return best


    def roll_for_slide(self, q, side, slide_dir, near=None):
        """wrist_roll that aligns the jaw-opening axis with slide_dir AND puts the moving finger ahead (+slide_dir).

        Grasp mechanics (measured): with the gripper open the fixed finger rests on the table and bulldozes cloth
        while the moving finger hovers ~1 cm up, ~5 cm away. Cloth only ends up between the fingers when the moving
        finger leads the slide (8/8 grasps in the live session followed this). The moving finger's side is the gripper
        link's +x = the tip frame's -x. Returns (roll, signed alignment in [-1, 1])."""
        d = np.asarray(slide_dir, float); d = d / np.linalg.norm(d)
        near = q[4] if near is None else near
        best = None
        for r in np.linspace(self.lo[4], self.hi[4], 721):
            qq = np.asarray(q, float).copy(); qq[4] = r
            a = float(-self.fk(qq, side)[:3, 0] @ d)
            score = a - 0.02 * abs(r - near)
            if best is None or score > best[0]:
                best = (score, r, a)
        return best[1], best[2]

    def moving_side(self, q, side):
        """World unit vector from the fixed finger toward the moving finger (also the fixed pad's face normal)."""
        return -self.fk(q, side)[:3, 0]

# rest pose used by the challenge (common.py), arm joints only
HOME = {"left": np.array([-1.2363, -1.7135, 1.4979, 1.0534, -0.085]),
        "right": np.array([1.2363, -1.7135, 1.4979, 1.0534, -0.085])}
GRIP_OPEN, GRIP_CLOSED = 0.55, -0.17       # rad; teacher opens to ~0.5 rad, closed limit is -0.1745
