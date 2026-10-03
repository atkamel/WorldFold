"""Laptop side of a live folding session (see modal_teacher.py::isaac_live).

    python live_client.py wait                      # block until the sim is up; saves the first images
    python live_client.py '{"op": "move", "arms": {"left": {"pos": [x, y, z]}}}'
    python live_client.py '{"op": "grip", "arms": {"left": -0.17}}'
    python live_client.py '{"op": "quit"}'

Each call sends one command, waits for the reply, saves the camera images under OUT_DIR
(step_<n>_top.jpg / _left.jpg / _right.jpg) and prints the state as JSON (images stripped).
Commands: move (arms -> pos [m], optional via [[x,y,z],...], jaw_axis [dx,dy]; ds = metres per tick),
grip (arms -> gripper angle rad; 0.55 open, -0.17 closed), wait (n ticks), home, score, reset, quit.
"""
import json
import os
import sys
import time

import modal

OUT_DIR = os.environ.get("LIVE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "teacher", "results", "isaac", "live"))
CMD_Q, RES_Q = "oracle-live-cmd", "oracle-live-res"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    q_cmd = modal.Queue.from_name(CMD_Q, create_if_missing=True)
    q_res = modal.Queue.from_name(RES_Q, create_if_missing=True)
    arg = sys.argv[1]
    t0 = time.time()
    if arg != "wait":
        q_cmd.put(json.loads(arg))
    try:
        r = q_res.get(timeout=float(os.environ.get("LIVE_TIMEOUT", "600" if arg == "wait" else "360")))
    except Exception as e:
        print(json.dumps({"error": f"no reply: {e!r}"})); return
    if r is None:
        print(json.dumps({"error": "no reply (timeout)"})); return
    n = r.get("n", -1)
    obs = r.get("obs") or {}
    for k in ("top", "left", "right", "map"):
        b = obs.pop(k + "_jpg", None)
        if isinstance(b, (bytes, bytearray)):
            with open(os.path.join(OUT_DIR, f"step_{n:03d}_{k}.jpg"), "wb") as f:
                f.write(b)
    print(json.dumps({"n": n, "secs": round(time.time() - t0, 1), "info": r.get("info"), "obs": obs}))


if __name__ == "__main__":
    main()
