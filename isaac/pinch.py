"""Top-down pinch poses for IsaacClothFoldEnv's arms: forward kinematics read from LeHome's SO101 URDF, damped
least squares IK. (LeHome's own pinocchio helper does not import in its locked env: the lock's "pinocchio" is an
unrelated PyPI package.)

The gripperframe site (the URDF's gripper_frame_link, at the fixed fingertip) goes to `tip` with the fingers pointing
straight down and the jaw opening along `jaw_dir` (the site's -x axis), so closing sweeps the jaw toward the fixed
finger across whatever lies between them. Import inside the Isaac process: it reads LeHome's assets.
"""

import os
import xml.etree.ElementTree as ET   # trusted input: LeHome's URDF from its pinned asset download

import numpy as np

from lehome.utils.constant import ASSETS_ROOT

from mujuco.cloth_params import ARM_BASE_LEFT, ARM_BASE_RIGHT, ARM_BASE_QUAT, ARM_JOINTS
from isaac.isaac_env import _matrix_from_quat

URDF = os.path.join(ASSETS_ROOT, "robots", "so101_new_calib.urdf")
SITE_LINK = "gripper_frame_link"
ORIENTATION_WEIGHT = 0.1     # metres of position error per radian of orientation error, at the pinch


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _origin(joint):
    o = joint.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    r, p, y = [float(v) for v in o.get("rpy", "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = _rot([0, 0, 1], y) @ _rot([0, 1, 0], p) @ _rot([1, 0, 0], r)
    T[:3, 3] = xyz
    return T


class PinchIK:

    def __init__(self):
        joints = {j.find("child").get("link"): j for j in ET.parse(URDF).getroot().findall("joint")}
        chain, link = [], SITE_LINK
        while link in joints:
            chain.append(joints[link])
            link = joints[link].find("parent").get("link")
        self.chain = []
        for j in reversed(chain):
            axis = j.find("axis")
            moving = j.get("type") in ("revolute", "continuous")
            axis = [float(v) for v in axis.get("xyz").split()] if moving else None
            self.chain.append((_origin(j), axis, j.get("name")))
        names = [name for _, axis, name in self.chain if axis is not None]
        assert names == ARM_JOINTS, names
        limits = {j.get("name"): j.find("limit") for j in chain if j.find("limit") is not None}
        self.low = np.array([float(limits[n].get("lower")) for n in ARM_JOINTS])
        self.high = np.array([float(limits[n].get("upper")) for n in ARM_JOINTS])
        # the URDF base frame is the MJCF base frame, at ARM_BASE_* (fitted on six joint poses: position residual
        # under 0.3 mm, gripper frame turned 2.8 deg from the USD link's)
        self.root = {"left_": (np.asarray(ARM_BASE_LEFT), ARM_BASE_QUAT),
                     "right_": (np.asarray(ARM_BASE_RIGHT), ARM_BASE_QUAT)}

    def _fk(self, q):
        T, k = np.eye(4), 0
        for origin, axis, _ in self.chain:
            T = T @ origin
            if axis is not None:
                R = np.eye(4)
                R[:3, :3] = _rot(axis, q[k])
                T, k = T @ R, k + 1
        return T

    def site_pose(self, prefix, q):
        """World position and rotation of the gripperframe site for the arm's 5 joint angles (rad)."""
        pos, quat = self.root[prefix]
        R0 = _matrix_from_quat(quat)
        T = self._fk(np.asarray(q[:5], float))
        return pos + R0 @ T[:3, 3], R0 @ T[:3, :3]

    def solve(self, prefix, tip, jaw_dir, seed_q, orientation_weight=ORIENTATION_WEIGHT, iters=300):
        """Joint angles (rad, 5) for the pinch pose, and the reached site position's error (m). Lift and carry poses
        can pass a lower orientation_weight: there the fingers may tilt a little to keep the position."""
        x = -np.array([jaw_dir[0], jaw_dir[1], 0.0])
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, -1.0])
        Rt = np.column_stack([x, np.cross(z, x), z])
        tip = np.asarray(tip, float)

        def error(q):
            p, R = self.site_pose(prefix, q)
            w = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            return np.concatenate([tip - p, orientation_weight * w])

        q = np.clip(np.asarray(seed_q[:5], float), self.low, self.high)
        for _ in range(iters):
            e = error(q)
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.column_stack([(e - error(q + dq)) / 1e-6 for dq in np.eye(5) * 1e-6])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.low, self.high)
        return q, float(np.linalg.norm(self.site_pose(prefix, q)[0] - tip))
