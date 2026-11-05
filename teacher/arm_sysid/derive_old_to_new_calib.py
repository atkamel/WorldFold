"""Derive the joint mapping between TheRobotStudio so101_old_calib.xml and so101_new_calib.xml.

Both files describe the same CAD; only body frames / joint zeros / limits differ. For random
old-calib joint vectors we solve for the new-calib joint vector that puts the same physical points
(body centres of mass + joint anchors) in the same place up to one rigid transform (the base frames
differ), then fit q_new = sign * q_old + offset per joint.

Also reports the wrist_roll angle in new_calib at which the moving-jaw hinge axis is horizontal
when the arm is at the zero pose ("gripper upright").

Run: python derive_old_to_new_calib.py
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from check_sim_conventions import JOINTS, TRS_XML, load_without_meshes, qpos_for

OLD_XML = Path(
    "C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
    "c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/ext_trs/so101_old_calib.xml"
)
BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]


def points(m, d, q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    pts = [d.body(b).xipos.copy() for b in BODIES]  # COMs (physical points)
    pts += [d.xanchor[m.joint(j).id].copy() for j in JOINTS]
    return np.array(pts)


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    dd = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, dd])
    R = U @ D @ Vt
    return Pc @ R - Qc


def main():
    np.set_printoptions(precision=4, suppress=True)
    mo, mn = load_without_meshes(OLD_XML), load_without_meshes(TRS_XML)
    do, dn = mujoco.MjData(mo), mujoco.MjData(mn)
    rng = np.random.default_rng(1)
    Qo, Qn, res = [], [], []
    for k in range(40):
        q_old_deg = {j: rng.uniform(*np.rad2deg(mo.joint(j).range) * 0.8) for j in JOINTS}
        P = points(mo, do, qpos_for(mo, q_old_deg))

        def f(x):
            return kabsch_rmsd(points(mn, dn, x), P).ravel()

        best = None
        for _ in range(8):  # multi-start
            x0 = np.array([rng.uniform(*mn.joint(j).range) for j in JOINTS])
            r = least_squares(f, x0)
            if best is None or r.cost < best.cost:
                best = r
        Qo.append([q_old_deg[j] for j in JOINTS])
        Qn.append(np.rad2deg(best.x))
        res.append(np.sqrt(2 * best.cost / len(best.fun)))
    Qo, Qn = np.array(Qo), np.array(Qn)
    print(f"point RMS residual after solve: median {np.median(res)*1e3:.4f} mm, max {np.max(res)*1e3:.4f} mm")
    print("joint           sign   offset_deg (q_new = sign*q_old + offset)   fit RMS deg")
    for i, j in enumerate(JOINTS):
        best = None
        for s in (+1, -1):
            # wrap differences to (-180,180]
            diff = (Qn[:, i] - s * Qo[:, i] + 180) % 360 - 180
            off = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(diff)))))
            rms = np.sqrt(np.mean(((diff - off + 180) % 360 - 180) ** 2))
            if best is None or rms < best[2]:
                best = (s, off, rms)
        print(f"  {j:13s}  {best[0]:+d}   {best[1]:+9.3f}                               {best[2]:.4f}")

    # wrist_roll angle at which the moving-jaw hinge axis is horizontal (arm at zero pose)
    def tilt(qr_deg):
        dn.qpos[:] = qpos_for(mn, {"wrist_roll": qr_deg})
        mujoco.mj_forward(mn, dn)
        return dn.xaxis[mn.joint("gripper").id][2]

    lo, hi = -20.0, 20.0
    if np.sign(tilt(lo)) != np.sign(tilt(hi)):
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.sign(tilt(mid)) == np.sign(tilt(lo)) else (lo, mid)
        print(f"new_calib wrist_roll at which gripper hinge axis is horizontal (arm at zero): {0.5*(lo+hi):+.3f} deg; "
              f"hinge-axis z at roll=0: {tilt(0.0):+.4f}")


if __name__ == "__main__":
    main()
