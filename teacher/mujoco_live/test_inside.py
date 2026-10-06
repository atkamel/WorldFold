"""Inside vs outside valleys on Roy's two old states and three new-cloth folds."""
import json
import numpy as np
from cloth import Cloth
from shirt import shirt_mesh
from neat_metric import fold_score, inside_voids, valley_share, flat_fraction

pts, tris = shirt_mesh(0.01)
H, FPS = 0.004, 60


def line(name, x):
    vin, air, lay = inside_voids(x, tris, H)
    vout, vmm = valley_share(x, tris, H)
    print(f"{name:40s} OUTSIDE valleys {100 * vout:3.0f}% | INSIDE air pockets {100 * vin:3.0f}% of the footprint "
          f"(avg {air:4.1f} mm of air, {lay:.1f} layers) | lying flat {100 * flat_fraction(x, tris):3.0f}%", flush=True)


recs = [json.loads(l) for l in open(r"demos\20261003_213517.jsonl")]
drops = [r for r in recs if r.get("episode") == 3 and r.get("event") == "drop"]
s26 = min((r for r in drops if r["tick"] - drops[0]["tick"] < 3600), key=lambda r: r["score"]["area_cm2"])
line("your 26% (old cloth)", np.array(s26["cloth_cm"]) / 100)
line("your 22% (old cloth)", np.array(drops[-1]["cloth_cm"]) / 100)


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


c, g = sim()
line("flat shirt (new cloth)", c.x)
fold(c, g, {0: ((-0.195, 0.26), (-0.01, 0.26)), 1: ((0.195, 0.26), (0.01, 0.26))})
fold(c, g, {0: ((-0.08, 0.085), (-0.08, 0.285)), 1: ((0.08, 0.085), (0.08, 0.285))}, lift=0.07)
line("neat: sleeves + bottom-up (new)", c.x)
c, g = sim()
fold(c, g, {0: ((-0.195, 0.24), (0.0, 0.11))}); fold(c, g, {1: ((0.195, 0.24), (0.0, 0.11))})
fold(c, g, {0: ((-0.07, 0.085), (-0.07, 0.285)), 1: ((0.07, 0.085), (0.07, 0.285))}, lift=0.07)
line("one-corner diagonal + bottom-up (new)", c.x)
c, g = sim()
for (p, q) in [((-0.195, 0.29), (0.0, 0.19)), ((0.195, 0.29), (0.02, 0.18)), ((-0.095, 0.085), (0.0, 0.2)),
               ((0.095, 0.085), (-0.01, 0.21)), ((0.0, 0.30), (0.0, 0.17))]:
    fold(c, g, {0: (p, q)}, lift=0.09, drop_h=0.06)
line("crumple (new)", c.x)
