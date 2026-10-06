"""Trace parallel tandem: does the right arm follow the left arm's mouse?"""
import numpy as np
from game import Game

g = Game(log=False)
M1, M2 = 101, 202
g.tick({M1: (0, 0, 0, True, False, ["left_down"])}); g.tick({M2: (0, 0, 0, True, False, ["left_down"])})


def ev(h, dx=0, dy=0, events=()):
    return {h: (dx, dy, 0, False, False, list(events))}


def goto(h, arm, xy, steps=200, show=False):
    for k in range(steps):
        d = np.array(xy) - g.cursor[arm][:2] * 100
        if np.linalg.norm(d) < 0.1 and np.linalg.norm(g.cmd[arm][:2] * 100 - xy) < 0.3:
            break
        st = d / max(1.0, np.linalg.norm(d) / 1.2)
        g.tick(ev(h, st[0] / 100 / g.gain[h], -st[1] / 100 / g.gain[h]))
        if show and k % 4 == 0:
            print(f"  k{k:3d} L cur {np.round(g.cursor['L'][:2]*100,1)} aim {np.round(g.cmd['L']*100,1)} | R cur {np.round(g.cursor['R'][:2]*100,1)} "
                  f"aim {np.round(g.cmd['R']*100,1)} tip {np.round(g.w.tip('R')*100,1)} flashR {g.reach_flash['R']>0} phase {g.phase['R']}")
    for _ in range(30):
        g.tick({})


goto(M1, "L", [-6, 8.5]); goto(M2, "R", [6, 8.5])
g.key("T")
g.tick({M2: (0, 0, 0, True, False, ["left_down"])})
for _ in range(60):
    g.tick({})
print("held", g.w.held)
goto(M1, "L", [-6, 28], steps=150, show=True)
