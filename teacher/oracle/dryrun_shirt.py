"""Offline dry run of the shirt plan (no simulator): the cloth is assumed to follow the ideal fold after each
step. Checks reach (IK error), jaw tilt and tick counts for a given placement of the shirt on the table.

    python dryrun_shirt.py Top_Long_Seen_0 [cx cy]
"""
import sys
import time

import numpy as np

import fold_metric as fm
import fold_plans as fp
import garment_keys as gk
from so101_kin import HOME

name = sys.argv[1] if len(sys.argv) > 1 else "Top_Long_Seen_0"
cx, cy = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.0, 0.0)
d = np.load(f"garments/{name}.npz")
scale = float(d["scale"][0])
keys = gk.top_keys(d["points"].astype(float), d["check_point"])
v = d["points"].astype(float) * scale
v = np.c_[v[:, 0] + cx, v[:, 1] + cy, np.full(len(v), 0.503)]
flat = v.copy()
folds = gk.top_folds(flat[:, :2], keys)
arm_of = fp.shirt_arms(flat, keys)
print(name, "scale", scale, "arm_of", arm_of, "extent cm", ((flat.max(0) - flat.min(0)) * 100).round(1)[:2],
      "y range", flat[:, 1].min().round(3), flat[:, 1].max().round(3))
total = 0
for i, fname in enumerate(["sleeve_L", "sleeve_R", "bottom_up"]):
    f = {x["name"]: x for x in gk.top_folds(v[:, :2], keys)}[fname]
    plan = fp.Plan({"left": HOME["left"], "right": HOME["right"]}, {"left": -0.17, "right": -0.17})
    spec = fp.shirt_spec(v, keys, f, arm_of, {})
    t = time.time()
    errs = plan.pick_place(spec, fname)
    plan.home(sides=list(spec))
    total += len(plan.ticks)
    tilt = {}
    for tk in plan.ticks:
        for a in spec:
            _, ap = fp.K.tip(tk["q"][a], a)
            ph = tk["tag"].split(":")[-1]
            tilt[(a, ph)] = max(tilt.get((a, ph), 0), np.degrees(np.arccos(np.clip(-ap[2], -1, 1))))
    print(f"{fname}: arms {list(spec)} ticks {len(plan.ticks)} plan {time.time() - t:.0f}s")
    for a, sp in spec.items():
        print(f"   {a}: grasp {np.round(sp['grasp'], 3)} -> place {np.round(sp['place'], 3)} slide {np.round(sp['slide'], 2)}")
    print("   ik err mm:", {f"{a}:{ph}": round(e * 1000, 1) for (a, ph), e in errs.items()})
    print("   max tilt deg:", {f"{a}:{ph}": round(t_) for (a, ph), t_ in tilt.items() if ph in ("slide", "carry", "settle")})
    v[:, :2] = fm.ideal_fold(flat[:, :2], folds[:i + 1])[0]
print("total ticks", total, "= sim s", round(total / 90, 1))
