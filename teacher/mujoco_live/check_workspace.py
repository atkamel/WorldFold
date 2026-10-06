"""Robot-side check of the reduced action space (x, y, height per arm, gripper pointing down).

For every point on a grid over the table and several heights, per arm:
  1. reachable?  IK error < 1 mm and gripper within 5 deg of vertical
  2. well-conditioned?  Jacobian of the task map  q (4 joints) -> [tip x, y, z, tilt]  has full rank 4,
     measured by its smallest singular value (0 = singular: some direction of your mouse can't be produced)
  3. margin to joint limits
Prints a 1 cm-per-cell map per arm and summary numbers.
"""
import numpy as np, mujoco
from server import Sim, JN

SEEDS = [np.array([0.0, 0.0, 0.0, 1.2]), np.array([0.0, -0.6, 0.8, 1.3]), np.array([0.0, 0.6, -0.6, 1.4])]
s = Sim()
m, ik = s.m, s.ik


def task(a, q):
    ik.qpos[s.qadr[a][:4]] = q
    ik.qpos[s.qadr[a][4]] = 0.0
    mujoco.mj_kinematics(m, ik)
    p = ik.site_xpos[s.site[a]].copy()
    g = ik.xpos[s.gbody[a]]
    ax = (p - g) / np.linalg.norm(p - g)
    tilt = np.arccos(np.clip(-ax[2], -1, 1))  # angle from straight down
    return p, ax, tilt


def jac(a, q):
    """d[x, y, z, approach-axis horizontal components] / dq  (5 x 4); rank 4 = full control."""
    def f(qq):
        p, ax, _ = task(a, qq)
        return np.r_[p, 0.05 * ax[:2]]  # 0.05 m lever arm: puts angles on the same scale as metres
    f0 = f(q)
    J = np.zeros((5, 4))
    for k in range(4):
        dq = np.zeros(4); dq[k] = 1e-5
        J[:, k] = (f(q + dq) - f0) / 1e-5
    return J


lo = {a: np.array([m.jnt_range[m.joint(f"{a}_{j}").id][0] for j in JN[:4]]) for a in "LR"}
hi = {a: np.array([m.jnt_range[m.joint(f"{a}_{j}").id][1] for j in JN[:4]]) for a in "LR"}
xs = np.arange(-0.24, 0.2401, 0.02)
ys = np.arange(-0.04, 0.4001, 0.02)
shirt = lambda x, y: (-0.20 <= x <= 0.20) and (0.08 <= y <= 0.30) and (abs(x) <= 0.10 or y >= 0.22)

for z in (0.008, 0.03, 0.06, 0.10):
    print(f"\n=== height {z*100:.1f} cm ===  (# reachable+well-conditioned, + reachable but weak, . unreachable; rows = far side on top)")
    res = {}
    for a in "LR":
        grid = []
        stats = []
        for y in ys[::-1]:
            row = ""
            for x in xs:
                goal = np.array([x, y, z])
                best = None
                for seed in [SEEDS[0]] + SEEDS[1:]:
                    q, perr, tilt = s.solve_ik(a, goal, seed.copy(), iters=120)
                    if best is None or perr < best[1]:
                        best = (q, perr, tilt)
                q, perr, tilt = best
                ok = perr < 0.001 and tilt < 5
                if ok:
                    sv = np.linalg.svd(jac(a, q), compute_uv=False)
                    smin = sv[-1]
                    margin = np.degrees(np.min(np.minimum(q - lo[a], hi[a] - q)))
                    stats.append((x, y, smin, margin))
                    row += "#" if smin > 0.01 and margin > 5 else "+"
                else:
                    row += "."
            grid.append(row)
        res[a] = (grid, stats)
    for (gl, gr) in zip(res["L"][0], res["R"][0]):
        print(f"L {gl}    R {gr}")
    for a in "LR":
        st = np.array([t[2:] for t in res[a][1]]) if res[a][1] else np.zeros((0, 2))
        on_shirt = [t for t in res[a][1] if shirt(t[0], t[1])]
        print(f"{a}: reachable cells {len(st)}; smallest singular value min {st[:,0].min():.4f} median {np.median(st[:,0]):.4f}; "
              f"joint-limit margin min {st[:,1].min():.1f} deg; reachable shirt cells {len(on_shirt)}")
    # union coverage of the shirt by either arm
    cells = [(x, y) for x in xs for y in ys if shirt(x, y)]
    cov = {(round(t[0], 3), round(t[1], 3)) for a in "LR" for t in res[a][1]}
    n = sum((round(x, 3), round(y, 3)) in cov for x, y in cells)
    print(f"shirt grid points reachable by at least one arm: {n}/{len(cells)}")
