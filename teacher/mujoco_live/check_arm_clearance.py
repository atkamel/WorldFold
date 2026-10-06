"""How close can the two grippers work before the arms hit each other? Both grippers are sent to (+-s, y)
at 3 cm height; prints how far each tip stays from its target (big = blocked by the other arm)."""
import numpy as np, mujoco
from world import World, ARMS

w = World()
print("rows: y (cm), columns: half-spacing s (cm) -> worst tip error (mm); X = blocked (>8 mm)")
ss = [3, 4, 5, 6, 7, 8, 10]
print("y\\s " + "".join(f"{s:>6d}" for s in ss))
for y in [10, 14, 18, 22, 26, 29]:
    row = []
    for s in ss:
        w.reset()
        goals = {"L": np.array([-s, y, 3.0]) / 100, "R": np.array([s, y, 3.0]) / 100}
        q = {a: w.q(a) for a in ARMS}
        # approach in small steps like the game does
        start = {a: w.tip(a) for a in ARMS}
        for k in range(1, 81):
            for a in ARMS:
                tgt = start[a] + (goals[a] - start[a]) * k / 80
                q[a], err, tilt = w.ik(a, tgt, q[a])
                w.set_arm(a, q[a])
            w.step(1 / 60)
        for _ in range(30):
            w.step(1 / 60)
        e = max(np.linalg.norm(w.tip(a) - goals[a]) for a in ARMS) * 1000
        row.append(f"{'X' if e > 8 else ''}{e:5.0f}")
    print(f"{y:>3d} " + "".join(f"{v:>6s}" for v in row), flush=True)
