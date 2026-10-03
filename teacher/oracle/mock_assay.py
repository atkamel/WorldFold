"""Dry-run grip_assay + fold_best against a fake env (static cloth) to catch code errors without Isaac."""
import os, sys, json, tempfile, types
import numpy as np
src = open("mock_test.py").read().split("out = tempfile.mkdtemp()")[0]
exec(src)                                  # defines of (oracle_fold module), Env, Obj, d, name, W

class PC:                                  # stand-in for the OmegaConf particle config
    def __init__(self):
        self.objects = {"garment_config": {"particle_mass": 0.01}, "particle_material": {"gravity_scale": 2}}

class P2(Obj._P):
    def GetAttribute(s, n):
        if n == "points":
            class A:
                def Get(s2): return W
            return A()
        return super().GetAttribute(n)
Obj._prim = P2()

class Cfg: garment_name = name
e = Env(); e.cfg = Cfg(); e.reset = lambda: None; e.particle_config = PC()
e.switch_garment = lambda n: None
out = tempfile.mkdtemp()
job = {"garment": name, "mode": "grip_assay", "reps": 1, "speeds": [0.001, 0.0035, 0.010]}
rows = of.grip_assay(e, None, job, name, out)
print("ROWS", len(rows), [ (r["cloth"], r["speed_mm_step"], r.get("error"), r.get("steps")) for r in rows])
e._assay_rows = rows
print("BEST", of.best_variant(rows))
r = of.shirt_trial(e, None, {"settle": 3, "final_settle": 3, "pause": 2, "carry_ds": 0.0035, "inset": 0.015}, name, "fold", out)
print("FOLD_MOCK_OK neat", r["neat"], sorted(os.listdir(out)))
