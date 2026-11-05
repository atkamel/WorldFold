#!/usr/bin/env python3
"""Close-up of the grasp from a grip_bench.py recording: cloth near the tip seen along the jaw hinge (side view of
the fingers), at the frames around the close and the lift. Needs the jaw meshes for the outlines.

  python closeup.py results/newton/run3/grip_deep_sq1.47_mu0.4.npz --menagerie ~/wf/assets/so101_menagerie
"""
import argparse
import math
import os

import numpy as np

FIXED_JAW_GEOM = ((0.0, -0.000218, 0.00095), np.diag([1.0, -1.0, -1.0]))
JAW_JOINT_P = np.array([0.0202, 0.0188, -0.0234])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
MOVING_JAW_OFF = np.array([0.0, 0.0, 0.0189])
TCP = np.array([-0.0079, -0.000218121, -0.0981274])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--menagerie", default=os.path.expanduser("~/wf/assets/so101_menagerie"))
    p.add_argument("--yaw", type=float, default=None, help="gripper yaw (rad); default: from the slide direction")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    d = np.load(a.npz)
    t, jaw, w = d["t"], d["jaw"], d["wrist"]
    fr = d["frames"].astype(np.float64) * float(d["scale"]) + d["origin"]
    tip = w[:, :3]
    # the slide: the tip's horizontal motion while low (z < 2 cm) gives the yaw (moving finger leads = +x)
    low = np.nonzero(tip[:, 2] < 0.02)[0]
    yaw = a.yaw
    if yaw is None:
        mv = tip[low[-1], :2] - tip[low[0], :2] if len(low) > 1 else np.array([1.0, 0.0])
        yaw = math.atan2(mv[1], mv[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    fixed = trimesh.load(os.path.join(a.menagerie, "assets", "wrist_roll_follower_so101_v1.stl")).vertices
    moving = trimesh.load(os.path.join(a.menagerie, "assets", "moving_jaw_so101_v1.stl")).vertices
    fixed_g = fixed @ FIXED_JAW_GEOM[1].T + np.array(FIXED_JAW_GEOM[0])

    def moving_g(q):
        rq = np.array([[math.cos(q), -math.sin(q), 0], [math.sin(q), math.cos(q), 0], [0, 0, 1]])
        return ((moving + MOVING_JAW_OFF) @ rq.T) @ RX90.T + JAW_JOINT_P

    # frames: from the landing to the end of the lift
    first = low[0] if len(low) else 0
    idx = [i for i in range(first, len(t)) if tip[i, 2] < 0.09][:40:3][:12]
    n = len(idx)
    fig, ax = plt.subplots(2, (n + 1) // 2, figsize=(3.3 * ((n + 1) // 2), 7))
    ax = ax.ravel()
    for k, i in enumerate(idx):
        g0 = tip[i] - rz @ TCP                                    # gripper body origin
        v = (fr[i] - g0) @ rz                                     # cloth in the gripper frame
        sel = (np.abs(v[:, 1]) < 0.008) & (v[:, 0] > -0.05) & (v[:, 0] < 0.08) & (v[:, 2] < -0.04)
        z_table = -g0[2]
        A = ax[k]
        for pts, col in ((fixed_g, "0.6"), (moving_g(jaw[i]), "tab:orange")):
            m = (np.abs(pts[:, 1]) < 0.006) & (pts[:, 2] < -0.04)
            A.scatter(pts[m, 0], pts[m, 2], s=0.2, c=col)
        A.scatter(v[sel, 0], v[sel, 2], s=2, c="tab:blue")
        A.axhline(z_table, color="k", lw=0.8)
        A.set_xlim(-0.05, 0.08)
        A.set_ylim(z_table - 0.005, z_table + 0.06)
        A.set_aspect("equal")
        A.set_title(f"t={t[i]:.1f}s jaw={math.degrees(jaw[i]):.0f}deg give={w[i, 3]:.1f}mm", fontsize=7)
        A.tick_params(labelsize=5)
    for k in range(n, len(ax)):
        ax[k].axis("off")
    fig.suptitle(os.path.basename(a.npz) + ": cloth (blue) within 8 mm of the jaw plane, fixed jaw (grey), "
                 "moving jaw (orange), table (black)", fontsize=8)
    fig.tight_layout()
    out = a.npz.replace(".npz", "_closeup.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
