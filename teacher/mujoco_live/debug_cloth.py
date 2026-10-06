"""Bottom-up fold with given cloth settings: watch the 6 s after release - does it settle or jitter?"""
import sys, numpy as np
from cloth import Cloth
from shirt import shirt_mesh

kw = {k: float(v) if "." in v or "e" in v else int(v) for k, v in (a.split("=") for a in sys.argv[1:])}
pts, tris = shirt_mesh(0.01)
c = Cloth(pts, tris, 0.01, **kw)
FPS = 60
grips = np.array([[-0.06, 0.085, 0.03], [0.06, 0.085, 0.03]])


def move2(goals, speed=0.15):
    starts = grips.copy()
    n = max(1, int(max(np.linalg.norm(goals[g] - starts[g]) for g in (0, 1)) / speed * FPS))
    for s in range(n):
        for g in (0, 1):
            grips[g] = starts[g] + (goals[g] - starts[g]) * (s + 1) / n
        c.step(1 / FPS, grips)


move2({0: np.array([-0.06, 0.085, 0.004]), 1: np.array([0.06, 0.085, 0.004])})
c.grab(0, grips[0]); c.grab(1, grips[1])
move2({0: np.array([-0.06, 0.15, 0.08]), 1: np.array([0.06, 0.15, 0.08])})
move2({0: np.array([-0.06, 0.27, 0.03]), 1: np.array([0.06, 0.27, 0.03])})
move2({0: np.array([-0.06, 0.27, 0.012]), 1: np.array([0.06, 0.27, 0.012])}, 0.05)
c.release(0); c.release(1)
move2({0: np.array([-0.06, 0.27, 0.06]), 1: np.array([0.06, 0.27, 0.06])}, 0.05)
prev = c.x.copy()
for k in range(1, 361):
    c.step(1 / FPS, grips)
    if k % 30 == 0:
        sp = np.linalg.norm(c.v, axis=1)
        moved = np.linalg.norm(c.x - prev, axis=1)
        prev = c.x.copy()
        print(f"t={k / FPS:4.1f}s side flips so far {c.stats[0]:5d} | max speed {sp.max():.3f} m/s | points faster than 1 cm/s: {(sp > 0.01).sum():3d} | moved in last 0.5 s: max {moved.max() * 1000:5.1f} mm, avg {moved.mean() * 1000:.2f} mm")
