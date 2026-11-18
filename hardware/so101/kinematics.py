"""Forward and inverse kinematics of the SO-101 arm in plain numpy (no simulator at runtime). The chain is copied
from so101-nexus 0.4.8, assets/SO101/so101_new_calib.xml, and test_kinematics.py checks it against MuJoCo.
Base frame: x forward, y left, z up, origin at the base's origin. Angles in radians."""

import numpy as np

# Per arm joint (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll): the body it turns, with the body's
# position and quaternion (w, x, y, z) in its parent's frame, and the joint range. Each joint turns about the body's z.
BODIES = [
    ([0.0388353, -8.97657e-09, 0.0624], [3.56167e-16, 1.22818e-15, -1, -4.14635e-16]),
    ([-0.0303992, -0.0182778, -0.0542], [0.5, -0.5, -0.5, -0.5]),
    ([-0.11257, -0.028, 1.73763e-16], [0.707107, -5.98613e-17, -2.58051e-17, 0.707107]),
    ([-0.1349, 0.0052, 3.62355e-17], [0.707107, 9.58722e-16, -7.51313e-16, -0.707107]),
    ([5.55112e-17, -0.0611, 0.0181], [0.0172091, -0.0172091, 0.706897, 0.706897]),
]
RANGES = [
    (-1.9198621771937616, 1.9198621771937634),
    (-1.7453292519943224, 1.7453292519943366),
    (-1.69, 1.69),
    (-1.6580628494556928, 1.6580627293335335),
    (-2.7438472969992493, 2.841206309382605),
]
SITE = ([-0.0079, -0.000218121, -0.0981274], [0, 0, 1, 0])     # gripperframe on the last body; z is the finger direction


def rotation(quat):
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)    # the XML's quaternions are not exactly unit
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Arm:
    def __init__(self):
        self.low, self.high = np.array(RANGES).T
        self.links = [(np.array(pos), rotation(quat)) for pos, quat in BODIES]
        self.site = (np.array(SITE[0]), rotation(SITE[1]))

    def tip(self, q):
        """(fingertip position (3,), finger direction (3,)) for the 5 arm angles `q`."""
        R, p = np.eye(3), np.zeros(3)
        for (pos, rot), angle in zip(self.links, q):
            p = p + R @ pos
            R = R @ rot @ turn(angle)
        return p + R @ self.site[0], (R @ self.site[1])[:, 2]

    def joint_points(self, q):
        """(elbow joint centre, wrist joint centre, fingertip), each a position (3,), for the 5 arm angles `q`."""
        R = np.eye(3)
        p = np.zeros(3)
        origins = []
        for (pos, rot), angle in zip(self.links, q):
            p = p + R @ pos
            origins.append(p)
            R = R @ rot @ turn(angle)
        fingertip = p + R @ self.site[0]
        return origins[2], origins[3], fingertip

    def solve(self, target, seed, iters=200):
        """Arm angles for the fingertip at `target` (base frame: x forward, y left, z up), on the seed's branch;
        wrist_roll stays 0 (free, it flipped half a turn at the edge of reach). The tip position comes first: the four
        joints leave one freedom at a reachable target, and it goes to keeping the fingers as near straight down as
        that position allows. A joint at its limit is held there and the others go on. Returns (q, tip error m, tilt
        deg); a tip error means the target is out of reach on this branch."""
        down = np.array([0.0, 0.0, -1.0])
        q = np.clip(np.asarray(seed, float), self.low, self.high)
        q[4] = 0.0
        for _ in range(iters):
            f = np.concatenate(self.tip(q))
            J = np.column_stack([(np.concatenate(self.tip(q + dq)) - f) / 1e-6 for dq in np.eye(5)[:4] * 1e-6])
            e, lean = target - f[:3], down - f[3:]
            free = np.ones(4, bool)
            for _ in range(4):                             # drop the joints a limit holds, solve with the rest
                dq = np.zeros(4)
                if not free.any():
                    break
                Jp, Jz = J[:3, free], J[3:, free]
                inv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 1e-6 * np.eye(3))
                null = np.eye(free.sum()) - inv @ Jp       # joint motions that leave the tip where it is
                reach, upright = inv @ e, null @ (Jz.T @ lean)
                dq[free] = reach * min(1.0, 0.1 / max(float(np.max(np.abs(reach))), 1e-12)) \
                    + upright * min(1.0, 0.05 / max(float(np.max(np.abs(upright))), 1e-12))
                held = free & (((q[:4] <= self.low[:4]) & (dq < 0)) | ((q[:4] >= self.high[:4]) & (dq > 0)))
                if not held.any():
                    break
                free &= ~held
            if np.max(np.abs(dq)) < 1e-6:                  # converged: a warm start from the last waypoint takes a
                break                                      # few steps
            q[:4] = np.clip(q[:4] + dq, self.low[:4], self.high[:4])
        p, z = self.tip(q)
        return q, float(np.linalg.norm(target - p)), float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))
