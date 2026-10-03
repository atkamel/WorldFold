"""Flat fingertip pads for the LeHome SO-101 (`so101_follower_good.usd`).

The challenge's robot touches cloth only through two round capsule colliders (r = 10 mm) that cross each other
when the gripper closes (~5 mm overlap at the tips), so a fingertip pinch is impossible. This writes a copy of the
robot USD with those capsules' collision disabled and a thin flat box pad on each jaw tip instead. The pad faces are
parallel and touch at the gripper's closed limit (-10 deg), like the real jaws; a single cloth layer (~8 mm of
collision thickness in the sim) then holds the jaw ~6 deg open, so the gripper drive actually squeezes.

    python pads.py so101_follower_good.usd so101_flatpads.usd      # needs pxr (usd-core or Isaac's python)
"""
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = "/so101_new_calib"
PAD = dict(x_face=-0.005, z_center=-0.089, thick=0.004, width=0.016, length=0.022, q_closed_deg=-10.0)


def _m(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def _qmat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]
    return T


def joint_frame(stage):
    j = [p for p in stage.Traverse() if p.GetName() == "gripper" and p.IsA(UsdPhysics.RevoluteJoint)][0]
    q = j.GetAttribute("physics:localRot0").Get()
    T0 = np.eye(4); T0[:3, :3] = _qmat(q.GetReal(), *q.GetImaginary())
    T0[:3, 3] = np.array(j.GetAttribute("physics:localPos0").Get())
    return T0


def jaw_in_gripper(stage, q_deg):
    """4x4 pose of the jaw link frame in the gripper link frame at gripper angle q (deg)."""
    return joint_frame(stage) @ _rz(np.radians(q_deg))


def _add_box(stage, parent, name, T_parent_box, size):
    box = UsdGeom.Cube.Define(stage, f"{parent}/{name}")
    box.CreateSizeAttr(1.0)
    R = T_parent_box[:3, :3]; t = T_parent_box[:3, 3]
    m = Gf.Matrix3d(*R.T.ravel().tolist())          # Gf is row-major with row vectors
    qd = m.ExtractRotation().GetQuat()
    xf = UsdGeom.Xformable(box)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*t.tolist()))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qd.GetReal(), qd.GetImaginary()))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    UsdGeom.Imageable(box).CreatePurposeAttr(UsdGeom.Tokens.guide)   # invisible in renders
    return box


def make(src, dst, pad=PAD):
    st = Usd.Stage.Open(src)
    for link in ("gripper", "jaw"):
        cap = st.GetPrimAtPath(f"{ROOT}/{link}/Capsule")
        cap.GetAttribute("physics:collisionEnabled").Set(False)
    size = (pad["thick"], pad["width"], pad["length"])
    # fixed pad: gripper link frame, inner face at x = x_face (faces +x, toward the moving jaw)
    Tf = np.eye(4); Tf[:3, 3] = [pad["x_face"] - pad["thick"] / 2, 0.0, pad["z_center"]]
    _add_box(st, f"{ROOT}/gripper", "FlatPad", Tf, size)
    # moving pad: at the closed limit its inner face coincides with the fixed one; expressed in the jaw link frame
    Tm_g = np.eye(4); Tm_g[:3, 3] = [pad["x_face"] + pad["thick"] / 2, 0.0, pad["z_center"]]
    Tm_j = np.linalg.inv(jaw_in_gripper(st, pad["q_closed_deg"])) @ Tm_g
    _add_box(st, f"{ROOT}/jaw", "FlatPad", Tm_j, size)
    st.GetRootLayer().Export(dst)
    return dst


def face_gap(stage_path, q_deg, pad=PAD):
    """Gap (m) between the two pad faces, measured at the pad centre along the fixed pad's normal (+x)."""
    st = Usd.Stage.Open(stage_path)
    g = st.GetPrimAtPath(f"{ROOT}/gripper/FlatPad"); j = st.GetPrimAtPath(f"{ROOT}/jaw/FlatPad")
    Mg = _m(st.GetPrimAtPath(f"{ROOT}/gripper")); Mj = _m(st.GetPrimAtPath(f"{ROOT}/jaw"))
    Tg = np.linalg.inv(Mg) @ _m(g)                       # fixed pad in gripper frame
    Tj = np.linalg.inv(Mj) @ _m(j)                       # moving pad in jaw frame
    Tj_g = jaw_in_gripper(st, q_deg) @ Tj                # moving pad in gripper frame at angle q
    fixed_face = (Tg @ np.array([0.5, 0, 0, 1]))[:3]     # +x face centre of the fixed pad (cube size 1, scaled)
    moving_face = (Tj_g @ np.array([-0.5, 0, 0, 1]))[:3] # -x face centre of the moving pad
    n = Tg[:3, 0] / np.linalg.norm(Tg[:3, 0])
    return float((moving_face - fixed_face) @ n), moving_face


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    make(src, dst)
    for q in (30, 10, 0, -3.5, -5, -8, -10):
        gap, mf = face_gap(dst, q)
        print(f"q={q:6.1f} deg: pad face gap {gap * 1000:6.1f} mm (moving face centre {np.round(mf * 1000, 1)} mm)")
