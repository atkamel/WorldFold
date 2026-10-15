"""Collision geometry for layout checks without physics: the SO101 URDF's collision meshes (STL, from the SO-ARM100
repo the URDF comes from), downsampled to points and posed by forward kinematics. Three checks, each a clearance in m:

- arms: smallest distance between two arms' points
- table: lowest point of every link except the gripper and jaw, above the table top
- self: the gripper and jaw against the arm's base, shoulder and upper arm (a tight fold puts the jaw into the upper
  arm; on Isaac the joints then stop short, run q4)

    from isaac.arm_collision import ArmGeometry
    g = ArmGeometry("outputs/isaac/assets/robots/so101_new_calib.urdf")
    pts = g.points(base_pos, base_quat, q5, jaw=1.0)   # {link: (n, 3)}
"""

import os
import xml.etree.ElementTree as ET   # trusted input: the pinned SO101 URDF

import numpy as np
from scipy.spatial import cKDTree

from isaac.isaac_env import _matrix_from_quat
from isaac.pinch import _origin, _rot
from mujuco.cloth_params import ARM_JOINTS, TABLE_TOP_Z

VOXEL = 0.004
HAND = ("gripper_link", "moving_jaw_so101_v1_link")
BODY = ("base_link", "shoulder_link", "upper_arm_link")


def _stl_points(path):
    """Vertices of a binary STL (m), voxel-downsampled."""
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        tri = np.frombuffer(f.read(50 * n), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = tri["v"].reshape(-1, 3).astype(float)
    return np.unique(np.round(v / VOXEL), axis=0) * VOXEL


class ArmGeometry:

    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        here = os.path.dirname(urdf)
        self.meshes = {}            # link -> points in the link frame
        for link in root.findall("link"):
            pts = []
            for c in link.findall("collision"):
                m = c.find("geometry/mesh")
                if m is None:
                    continue
                T = _origin(c)
                p = _stl_points(os.path.join(here, m.get("filename")))
                pts.append(p @ T[:3, :3].T + T[:3, 3])
            if pts:
                self.meshes[link.get("name")] = np.concatenate(pts)
        self.joints = []            # (parent, child, origin, axis or None, name), parents before children
        for j in root.findall("joint"):
            axis = j.find("axis")
            moving = j.get("type") in ("revolute", "continuous")
            self.joints.append((j.find("parent").get("link"), j.find("child").get("link"), _origin(j),
                                [float(v) for v in axis.get("xyz").split()] if moving else None, j.get("name")))
        order, done = [], {"base_link"}
        while len(order) < len(self.joints):
            for j in self.joints:
                if j not in order and j[0] in done:
                    order.append(j)
                    done.add(j[1])
        self.joints = order

    def points(self, base_pos, base_quat, q5, jaw=1.0):
        """{link: world points} for the 5 arm joints (rad) and the jaw angle."""
        angles = dict(zip(ARM_JOINTS, q5)) | {"gripper": jaw}
        T0 = np.eye(4)
        T0[:3, :3] = _matrix_from_quat(np.asarray(base_quat, float))
        T0[:3, 3] = base_pos
        frames = {"base_link": T0}
        for parent, child, origin, axis, name in self.joints:
            T = frames[parent] @ origin
            if axis is not None:
                R = np.eye(4)
                R[:3, :3] = _rot(axis, angles.get(name, 0.0))
                T = T @ R
            frames[child] = T
        return {k: v @ frames[k][:3, :3].T + frames[k][:3, 3] for k, v in self.meshes.items()}


def clearance(a, b):
    """Smallest distance between two point sets (each a {link: points} or an array)."""
    a = np.concatenate(list(a.values())) if isinstance(a, dict) else a
    b = np.concatenate(list(b.values())) if isinstance(b, dict) else b
    return float(cKDTree(b).query(a, k=1)[0].min())


def table_clearance(pts):
    return float(min(v[:, 2].min() for k, v in pts.items() if k not in HAND) - TABLE_TOP_Z)


def self_clearance(pts):
    return clearance({k: pts[k] for k in HAND if k in pts}, {k: pts[k] for k in BODY if k in pts})
