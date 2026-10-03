"""Dry-run mechanics_assay, grasp_rule_check and the live grasp/release ops against the fake env."""
import os, sys, json, tempfile, threading, time, pickle
import numpy as np
src = open("mock_assay.py").read().split(chr(10) + "out = tempfile.mkdtemp()" + chr(10) + "job")[0]
exec(src)
out = tempfile.mkdtemp()
rows = of.mechanics_assay(e, None, {"reps": 1, "opens": [0.55, 1.4], "aways": [0.0, 0.02], "lift_dss": [0.004]}, name, out)
print("MECH rows", len(rows), [(r["cloth"], r["open"], r["away_cm"], r.get("error"), r.get("stuck"), r.get("between_pads"), r.get("jaw_deg_obs")) for r in rows])
rr = of.grasp_rule_check(e, None, {"reps": 1}, name, out)
print("RULE rows", [(r["lead"], r.get("align"), r.get("held_closed"), r.get("error")) for r in rr])
# live ops
dlive = tempfile.mkdtemp(); os.environ["ORACLE_LIVE_DIR"] = dlive; of.OUT = tempfile.mkdtemp()
e.left_arm = e.right_arm = None
th = threading.Thread(target=of.live, args=(e, None), daemon=True); th.start()
def rpc(n, cmd=None):
    if cmd is not None: json.dump(cmd, open(f"{dlive}/cmd_{n}.json", "w"))
    p = f"{dlive}/res_{n}.pkl"
    for _ in range(4000):
        if os.path.exists(p): time.sleep(0.05); return pickle.load(open(p, "rb"))
        time.sleep(0.05)
    raise SystemExit("timeout %d" % n)
r = rpc(0)
r = rpc(1, {"op": "grasp", "arm": "left", "target": [-0.228, -0.0423], "dir": [0.884, 0.467], "slide": 0.03, "lift": 0.05}); print("GRASP", r["info"])
r = rpc(2, {"op": "release", "arm": "left", "z": 0.534, "open": 1.4, "away": 0.02, "lift_ds": 0.004}); print("RELEASE", r["info"])
r = rpc(3, {"op": "quit"}); th.join(10); print("LIVE_OPS_OK")
