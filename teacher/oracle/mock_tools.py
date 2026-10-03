"""Dry-run the agent toolset (fold / move_corner / save / restore / map) through the live loop on the fake env."""
import os, sys, json, tempfile, threading, time, pickle, types
import numpy as np
src = open("mock_assay.py").read().split(chr(10) + "out = tempfile.mkdtemp()" + chr(10) + "job")[0]
exec(src)
rl = types.ModuleType("scripts.utils.remote_loop")
rl.capture_physics_state = lambda env: {"garment_points": W.tolist()}
rl.restore_physics_state = lambda env, d, mode, label="": True
sys.modules["scripts.utils.remote_loop"] = rl
dlive = tempfile.mkdtemp(); os.environ["ORACLE_LIVE_DIR"] = dlive; of.OUT = tempfile.mkdtemp()
e.left_arm = e.right_arm = None
th = threading.Thread(target=of.live, args=(e, None), daemon=True); th.start()
n = [0]
def rpc(cmd=None):
    if cmd is not None: json.dump(cmd, open(f"{dlive}/cmd_{n[0]}.json", "w"))
    p = f"{dlive}/res_{n[0]}.pkl"
    for _ in range(6000):
        if os.path.exists(p):
            time.sleep(0.05); r = pickle.load(open(p, "rb")); n[0] += 1; return r
        time.sleep(0.05)
    raise SystemExit("timeout")
r = rpc(); print("START map bytes:", len(r["obs"].get("map_jpg", b"")), "score", r["obs"].get("score"), "done", r["obs"].get("done"))
open("mock_map.jpg", "wb").write(r["obs"]["map_jpg"])
for cmd in ({"op": "save", "name": "s0"}, {"op": "fold", "step": "sleeve_L"}, {"op": "restore", "name": "s0"},
            {"op": "fold", "step": "sleeve_L", "params": {"inset": 0.03}}, {"op": "fold", "step": "sleeve_R"},
            {"op": "fold", "step": "bottom_up"}, {"op": "fold", "step": "half"},
            {"op": "move_corner", "arm": "left", "key": "L.hem_out", "dir": [1, 0], "to": [-0.10, 0.10]},
            {"op": "score"}, {"op": "quit"}):
    r = rpc(cmd); i = r["info"]
    print(cmd["op"], cmd.get("step", ""), json.dumps(i)[:260])
th.join(10); print("TOOLS_MOCK_OK")
