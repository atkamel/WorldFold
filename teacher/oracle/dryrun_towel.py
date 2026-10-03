"""Offline dry run of the towel plan (no simulator): checks IK errors, tilt, speeds and tick counts.

    python dryrun_towel.py
"""
import time

import numpy as np

import fold_plans as fp
from so101_kin import HOME

N = 72; hx, hy = 0.135, 0.1485; cx, cy = 0.03, 0.05
xs, ys = np.linspace(-hx, hx, N) + cx, np.linspace(hy, -hy, N) + cy
verts = np.array([(x, y, 0.502) for y in ys for x in xs])
t = time.time()
plan = fp.Plan({"left": HOME["left"], "right": HOME["right"]}, {"left": -0.17, "right": -0.17})
info, errs = fp.towel_drag(plan, verts)
print("drag dy", round(info["dy"], 3), {f"{k[0]}:{k[1]}": round(v * 1000, 1) for k, v in errs.items()}, "ticks", len(plan.ticks))
v2 = verts.copy(); v2[:, 1] += info["dy"]
info2, errs2 = fp.towel_fold(plan, v2)
print("fold place", info2["place_left"], info2["place_right"], {f"{k[0]}:{k[1]}": round(v * 1000, 1) for k, v in errs2.items()})
plan.home()
print("ticks", len(plan.ticks), "= sim seconds", round(len(plan.ticks) / 90, 1), "| build s", round(time.time() - t, 1))
for l in plan.log:
    print(l)
for side in ("left", "right"):
    tags = {}
    for tk in plan.ticks:
        p, a = fp.K.tip(tk["q"][side], side)
        tags.setdefault(tk["tag"], []).append((p, np.degrees(np.arccos(np.clip(-a[2], -1, 1)))))
    for k, v in tags.items():
        P = np.array([x[0] for x in v])
        print(f"{side[0]} {k:15s} n={len(v):4d} start={P[0].round(3)} end={P[-1].round(3)} zmax={P[:, 2].max():.3f} tilt_max={max(x[1] for x in v):3.0f}")
dq = np.array([max(np.abs(b["q"][s] - a["q"][s]).max() for s in ("left", "right")) for a, b in zip(plan.ticks[:-1], plan.ticks[1:])])
print("max joint step per tick (rad)", dq.max().round(3))
