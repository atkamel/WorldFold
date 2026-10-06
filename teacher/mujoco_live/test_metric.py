"""Does the fold score separate a neat fold from a crumple (new cloth, virtual grippers)?"""
import numpy as np
from cloth import Cloth
from shirt import shirt_mesh
from neat_metric import fold_score, flat_fraction

pts, tris = shirt_mesh(0.01)
H, FPS = 0.004, 60
FLAT = fold_score(np.c_[pts, np.full(len(pts), H / 2)], tris, 1.0, H)["area_ratio"]


def sim():
    return Cloth(pts, tris, 0.01, thickness=H), np.array([[-0.5, 0, 0.1], [0.5, 0, 0.1]])


def move(c, grips, goals, speed=0.15):
    starts = grips.copy()
    n = max(1, int(max(np.linalg.norm(goals[g] - starts[g]) for g in goals) / speed * FPS))
    for s in range(n):
        for g in goals:
            grips[g] = starts[g] + (goals[g] - starts[g]) * (s + 1) / n
        c.step(1 / FPS, grips)


def fold(c, grips, pairs, lift=0.05, drop_h=0.012):
    move(c, grips, {g: np.r_[p, 0.03] for g, (p, q) in pairs.items()}, 0.5)
    move(c, grips, {g: np.r_[p, 0.004] for g, (p, q) in pairs.items()})
    for g, (p, q) in pairs.items():
        c.grab(g, grips[g])
    move(c, grips, {g: np.r_[(np.array(p) + q) / 2, lift] for g, (p, q) in pairs.items()})
    move(c, grips, {g: np.r_[q, max(drop_h, 0.02)] for g, (p, q) in pairs.items()})
    move(c, grips, {g: np.r_[q, drop_h] for g, (p, q) in pairs.items()}, 0.05)
    for g in pairs:
        c.release(g)
    move(c, grips, {g: np.r_[q, 0.08] for g, (p, q) in pairs.items()}, 0.1)
    for _ in range(150):
        c.step(1 / FPS, grips)




def report(name, c):
    s = fold_score(c.x, tris, FLAT, H)
    print(f"{name:42s} area {100 * s['area_ratio']:3.0f}%  flat {100 * s['flat']:3.0f}%  rect {s['rect']:.2f}  "
          f"height {s['height_mm']:4.1f} mm  valleys {100 * s['valleys']:.0f}% (mean {s['valley_mm']:.1f} mm)  -> SCORE {s['score']:5.1f}", flush=True)


c, g = sim()
fold(c, g, {0: ((-0.195, 0.26), (-0.01, 0.26)), 1: ((0.195, 0.26), (0.01, 0.26))})
fold(c, g, {0: ((-0.08, 0.085), (-0.08, 0.285)), 1: ((0.08, 0.085), (0.08, 0.285))}, lift=0.07)
report("neat: sleeves in + bottom-up (both hands)", c)

c, g = sim()
fold(c, g, {0: ((-0.195, 0.24), (0.0, 0.11))})
fold(c, g, {1: ((0.195, 0.24), (0.0, 0.11))})
fold(c, g, {0: ((-0.07, 0.085), (-0.07, 0.285)), 1: ((0.07, 0.085), (0.07, 0.285))}, lift=0.07)
report("Roy's: diagonal sleeves + bottom-up", c)

c, g = sim()  # crumple: pull corners to the middle and drop them from height, then push the edges in
for (p, q) in [((-0.195, 0.29), (0.0, 0.19)), ((0.195, 0.29), (0.02, 0.18)), ((-0.095, 0.085), (0.0, 0.2)),
               ((0.095, 0.085), (-0.01, 0.21)), ((0.0, 0.30), (0.0, 0.17))]:
    fold(c, g, {0: (p, q)}, lift=0.09, drop_h=0.06)
report("crumple: corners dragged to the middle", c)
