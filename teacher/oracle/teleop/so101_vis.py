"""SO-101 visual meshes posed from joint angles, for drawing the robot on the laptop (teleop 'state' view).

Uses so101_kin's URDF model and base placement (the same kinematics the controller uses, calibrated against the Isaac
bodies), so the drawn arm is where Isaac's arm is to within that calibration (mm). Poses are in Isaac world coordinates.
"""
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

import so101_kin as sk

JOINTS6 = sk.ARM_JOINTS + ["gripper"]       # order of the 6 values per arm in observation.state


def load_stl(path):
    """Binary STL -> (vertices (n,3) float32, faces (m,3) int32), duplicate corners merged."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        rec = np.frombuffer(f.read(50 * n), dtype=np.dtype([("nrm", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    corners = rec["v"].reshape(-1, 3)
    verts, inv = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
    return verts.astype(np.float32), inv.reshape(-1, 3).astype(np.int32)


class RobotVis:
    """visuals: list of (link, 4x4 offset in the link frame, mesh file name); poses(q6, side) -> one 4x4 per visual."""

    def __init__(self):
        self.K = sk.SO101()
        root = ET.parse(sk.URDF).getroot()
        self.visuals = []
        for link in root.findall("link"):
            for vis in link.findall("visual"):
                o, m = vis.find("origin"), vis.find("geometry/mesh")
                if m is None:
                    continue
                xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
                self.visuals.append((link.get("name"), sk._T(sk._rpy(*rpy), xyz), os.path.basename(m.get("filename"))))
        self.children = {}
        for name, j in self.K.joints.items():
            self.children.setdefault(j["parent"], []).append(name)
        self.root = next(l for l in (j["parent"] for j in self.K.joints.values())
                         if l not in {j["child"] for j in self.K.joints.values()})

    def link_poses(self, q6, side):
        """World pose of every link for the 6 joint values (5 arm joints + gripper) of one arm."""
        qd = dict(zip(JOINTS6, q6))
        out = {self.root: sk._T(sk.BASE_ROT, sk.BASE_POS[side])}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for jn in self.children.get(parent, []):
                j = self.K.joints[jn]
                M = out[parent] @ j["T"]
                if j["type"] == "revolute":
                    M = M @ sk._rotz(qd.get(jn, 0.0))
                out[j["child"]] = M
                stack.append(j["child"])
        return out

    def poses(self, q6, side):
        L = self.link_poses(q6, side)
        return [L[link] @ off for link, off, _ in self.visuals]
