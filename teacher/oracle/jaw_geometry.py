"""World-frame geometry of the two flat pads for a given arm pose (offline, numpy + pxr for the pad file).

Tip frame (URDF gripper_frame_link) comes from so101_kin; the gripper link frame is recovered through the URDF's
fixed gripper_frame_joint (xyz -0.0079 -0.000218 -0.0981, rpy 0 pi 0). Pads are taken from the flat-pad USD
written by pads.py (fixed pad on the gripper link, moving pad on the jaw link, jaw at gripper angle q).
"""
import os

import numpy as np
from pxr import Usd, UsdGeom

import pads
from so101_kin import SO101

K = SO101()
HERE = os.path.dirname(os.path.abspath(__file__))
PAD_USD = os.path.join(HERE, "assets", "so101_flatpads.usd")


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


T_LINK_TIP = np.eye(4)
T_LINK_TIP[:3, 3] = [-0.0079, -0.000218, -0.0981]
T_LINK_TIP = T_LINK_TIP @ _ry(np.pi)

_st = Usd.Stage.Open(PAD_USD)


def _local(prim_path, body_path):
    M = lambda p: np.array(UsdGeom.Xformable(_st.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    return np.linalg.inv(M(body_path)) @ M(prim_path)


T_G_FIXEDPAD = _local("/so101_new_calib/gripper/FlatPad", "/so101_new_calib/gripper")
T_J_MOVINGPAD = _local("/so101_new_calib/jaw/FlatPad", "/so101_new_calib/jaw")
CORNERS = np.array([[x, y, z, 1.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])


def pads_world(q_arm, side, grip_rad):
    """Return dict with 8 world corners of each pad, their face normals (pointing to the other pad), and frames."""
    T_w_tip = K.fk(q_arm, side)
    T_w_link = T_w_tip @ np.linalg.inv(T_LINK_TIP)
    T_w_fixed = T_w_link @ T_G_FIXEDPAD
    T_w_moving = T_w_link @ pads.jaw_in_gripper(_st, np.degrees(grip_rad)) @ T_J_MOVINGPAD
    out = {}
    for name, T, n_sign in (("fixed", T_w_fixed, +1), ("moving", T_w_moving, -1)):
        pts = (T @ CORNERS.T).T[:, :3]
        n = T[:3, 0] / np.linalg.norm(T[:3, 0]) * n_sign          # +x face of the fixed pad / -x face of the moving pad
        out[name] = dict(corners=pts, centre=T[:3, 3], normal=n, z_min=float(pts[:, 2].min()))
    out["tip"] = T_w_tip[:3, 3]
    out["link_x"] = T_w_link[:3, 0]
    return out
