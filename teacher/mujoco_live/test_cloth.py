"""Physics checks for cloth.py with virtual grippers (no robot): idle creep, sleeve fold, two-hand
bottom fold, one-corner drag. Prints numbers that say whether the cloth behaves like cloth.
Usage: python test_cloth.py [key=value ...]   e.g. bend_compliance=0.5 substeps=20"""
import sys, time
import numpy as np
from cloth import Cloth
from shirt import shirt_mesh, landmark_ids

kw = {k: float(v) if "." in v or "e" in v else int(v) for k, v in (a.split("=") for a in sys.argv[1:])}
FPS = 60
pts, tris = shirt_mesh(0.01)
L = landmark_ids(pts)


def new():
    return Cloth(pts, tris, 0.01, **kw)


def run(c, seconds, grips, jaws=None):
    t0 = time.perf_counter()
    for _ in range(int(seconds * FPS)):
        c.step(1 / FPS, grips, jaws)
    return (time.perf_counter() - t0) / (seconds * FPS) * 1000


def move(c, grips, gid, goal, speed=0.15):
    """Move gripper gid in a straight line to goal at speed (m/s), stepping the cloth."""
    start = grips[gid].copy()
    n = max(1, int(np.linalg.norm(goal - start) / speed * FPS))
    for k in range(n):
        grips[gid] = start + (goal - start) * (k + 1) / n
        c.step(1 / FPS, grips)


def move2(c, grips, goals, speed=0.15):
    starts = grips.copy()
    n = max(1, int(max(np.linalg.norm(goals[g] - starts[g]) for g in (0, 1)) / speed * FPS))
    for k in range(n):
        for g in (0, 1):
            grips[g] = starts[g] + (goals[g] - starts[g]) * (k + 1) / n
        c.step(1 / FPS, grips)


def layers(c, top_mask, bot_mask):
    """For particles of the folded part lying over the base part: gap to the base layer under them."""
    x = c.x
    top, bot = np.where(top_mask)[0], np.where(bot_mask)[0]
    gaps = []
    for i in top:
        d = np.linalg.norm(x[bot, :2] - x[i, :2], axis=1)
        k = np.argmin(d)
        if d[k] < 0.006:
            gaps.append(x[i, 2] - x[bot[k], 2])
    g = np.array(gaps)
    if not len(g):
        return "no overlap"
    return f"{len(g)} overlapping pts, gap median {np.median(g)*1000:.1f} mm, through/under the lower layer: {(g < 0.5 * c.h).sum()}"


def report(name, c, ms, extra=""):
    x = c.x
    print(f"{name:22s} {ms:5.2f} ms/frame | lowest point {(x[:, 2].min() - c.h / 2) * 1000:+.2f} mm vs table "
          f"| max speed {np.linalg.norm(c.v, axis=1).max():.3f} m/s | {extra}", flush=True)


rest = pts
# A. idle
c = new(); grips = np.array([[-0.3, 0, 0.1], [0.3, 0, 0.1]])
ms = run(c, 0.2, grips)  # includes JIT warm-up
ms = run(c, 3.0, grips)
report("A idle 3 s", c, ms, f"creep max {np.linalg.norm(c.x[:, :2] - rest, axis=1).max() * 1000:.2f} mm")

# B. left sleeve folded over the body by one gripper, then 3 s rest
c = new(); grips = np.array([[-0.195, 0.26, 0.03], [0.3, 0, 0.1]])
move(c, grips, 0, np.array([-0.195, 0.26, 0.004]))
n = c.grab(0, grips[0]);
move(c, grips, 0, np.array([-0.15, 0.26, 0.05]))
move(c, grips, 0, np.array([-0.02, 0.26, 0.03]))
move(c, grips, 0, np.array([-0.02, 0.26, 0.012]), 0.05)
rel = c.x[L["sleeve_L_lo"]].copy()
stretch = (np.linalg.norm(c.x[c.edges[:, 0]] - c.x[c.edges[:, 1]], axis=1) / c.rest_e).max()
x_rel = c.x.copy()
c.release(0)
move(c, grips, 0, np.array([-0.02, 0.26, 0.06]), 0.05)
ms = run(c, 3.0, grips)
fb = np.linalg.norm(c.x - x_rel, axis=1)
sl = rest[:, 0] < -0.105
report("B sleeve fold", c, ms, f"grabbed {n}; max stretch while carried {100 * (stretch - 1):.0f}%; FLOWBACK after release: "
       f"avg {fb.mean() * 1000:.1f} mm, worst 5% {np.percentile(fb, 95) * 1000:.1f} mm, cuff {np.linalg.norm(c.x[L['sleeve_L_lo']] - rel) * 1000:.1f} mm; "
       + layers(c, sl, (np.abs(rest[:, 0]) <= 0.095)))
xB = c.x.copy()

# C. bottom-up fold with two grippers together
c = new(); grips = np.array([[-0.06, 0.085, 0.03], [0.06, 0.085, 0.03]])
move2(c, grips, {0: np.array([-0.06, 0.085, 0.004]), 1: np.array([0.06, 0.085, 0.004])})
n0, n1 = c.grab(0, grips[0]), c.grab(1, grips[1])
move2(c, grips, {0: np.array([-0.06, 0.15, 0.08]), 1: np.array([0.06, 0.15, 0.08])})
move2(c, grips, {0: np.array([-0.06, 0.27, 0.03]), 1: np.array([0.06, 0.27, 0.03])})
move2(c, grips, {0: np.array([-0.06, 0.27, 0.012]), 1: np.array([0.06, 0.27, 0.012])}, 0.05)
stretch = (np.linalg.norm(c.x[c.edges[:, 0]] - c.x[c.edges[:, 1]], axis=1) / c.rest_e).max()
x_rel = c.x.copy()
c.release(0); c.release(1)
move2(c, grips, {0: np.array([-0.06, 0.27, 0.06]), 1: np.array([0.06, 0.27, 0.06])}, 0.05)
ms = run(c, 3.0, grips)
fb = np.linalg.norm(c.x - x_rel, axis=1)
report("C bottom-up 2 hands", c, ms, f"grabbed {n0}+{n1}; max stretch {100 * (stretch - 1):.0f}%; FLOWBACK avg {fb.mean() * 1000:.1f} mm, "
       f"worst 5% {np.percentile(fb, 95) * 1000:.1f} mm; hem now at y={c.x[L['hem_mid'], 1]:.3f} (target ~0.27); "
       + layers(c, (rest[:, 1] < 0.17) & (np.abs(rest[:, 0]) < 0.095), (rest[:, 1] > 0.20) & (np.abs(rest[:, 0]) < 0.095)))

# D. drag one corner 8 cm along the table, release, watch it settle
c = new(); grips = np.array([[-0.10, 0.08, 0.03], [0.3, 0, 0.1]])
far0 = c.x[L["sleeve_R_hi"]].copy()
move(c, grips, 0, np.array([-0.10, 0.08, 0.004]))
c.grab(0, grips[0])
move(c, grips, 0, np.array([-0.18, 0.08, 0.006]), 0.10)
c.release(0)
ke = []
for k in range(90):
    c.step(1 / FPS, grips)
    ke.append(np.sum(np.linalg.norm(c.v, axis=1) ** 2))
report("D drag corner 8 cm", c, ms, f"far corner moved {np.linalg.norm(c.x[L['sleeve_R_hi']] - far0) * 1000:.1f} mm; "
       f"motion after release: 0.1 s {ke[5]:.2e}, 0.5 s {ke[30]:.2e}, 1.5 s {ke[-1]:.2e}")
np.save("cloth_B.npy", xB)
