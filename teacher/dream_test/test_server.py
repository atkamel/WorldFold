"""Offline test of server.py's session protocol: a stand-in for LeHome's remote_loop (with dream_isaac_patch.py's
calls) drives the smoke battery against the server's --fake-dream client. No Isaac, no API calls.

    <python with numpy + opencv> test_server.py
"""
import argparse
import base64
import http.server
import json
import pathlib
import sys
import tempfile
import threading
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server  # noqa: E402
import tasks  # noqa: E402


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def img(shade):
    a = np.full((480, 640, 3), shade, np.uint8)
    return {"base64": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": "uint8"}


def main():
    out = pathlib.Path(tempfile.mkdtemp(prefix="dreamtest_"))
    args = argparse.Namespace(battery="smoke", shard=0, shards=1, port=0, out=str(out), ch=".", python=sys.executable,
                              model="x", fake_dream=True, no_sim=True)
    sess = server.Session(args)
    srv = http.server.HTTPServer(("127.0.0.1", 0), server.make_handler(sess))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    assert post(url, "/ping", {})["ok"]
    dt, n_eps, fails = 1 / 90, 0, []
    state = np.array([0.25, -0.6, 0.9, 0.9, 0.0, 0.55, -0.25, -0.6, 0.9, 0.9, 0.0, 0.55])
    while True:
        task = post(url, "/next_task", {})
        if task.get("shutdown"):
            break
        n_eps += 1
        post(url, "/reset", {"ep_idx": task["ep_idx"]})
        cfg = post(url, "/dream_episode", {"garment": task["garment_name"], "check_points": [1, 2, 3, 4, 5, 6],
                                           "verts": np.random.rand(50, 3).tolist(), "local": np.random.rand(50, 3).tolist(),
                                           "table_z": 0.5, "physics_dt": dt})
        spec = sess.ep.spec
        if cfg["block"] != (spec["kind"] == "block"):
            fails.append(f"block flag wrong for {spec['kind']}")
        steps, length, bz = 0, 0, 0.52
        while length < 3000:
            body = {"observation.state": state.tolist(), "observation.images.top_rgb": img(90),
                    "observation.images.left_rgb": img(120), "observation.images.right_rgb": img(150),
                    "dream_physics_dt": dt}
            if cfg["block"]:
                bz = min(bz + 0.001, 0.60)
                body["dream_block_pos"] = [0.0, 0.0, bz]
            resp = post(url, "/infer", body)
            if resp.get("stop"):
                break
            a = np.asarray(resp["actions"])
            if a.shape != (12,):
                fails.append(f"action shape {a.shape}")
            state = a
            steps += resp["n_steps"]
            length += 1
        # the grasp logger's line for this episode (the sim prints it after the step loop)
        sess.grasp_line({"side": "left", "n_pinch": 30, "max_lift_cm": 6.0, "t_lift4": 10, "t_open": 60,
                         "lift_open_cm": 5.0, "outcome": "held_lifted", "layers": 2})
        want = round(spec["seconds"] / dt)
        if abs(steps - want) > 5:
            fails.append(f"{spec['kind']}: {steps} physics steps, want ~{want}")
    lines = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    vids = sorted(p.name for p in (out / "videos").glob("*.mp4"))
    starts = sorted(p.name for p in (out / "start").glob("*.npz"))
    exp = len(tasks.battery("smoke"))
    checks = [
        (n_eps == exp and len(lines) == exp, f"{n_eps} episodes, {len(lines)} result lines, want {exp}"),
        (len(vids) == exp and len(starts) == exp, f"{len(vids)} videos, {len(starts)} start shapes"),
        (all(l["lifted"] and l["held_2s"] for l in lines if l["kind"] != "block"), "grasp line scored lifted + held"),
        (all(l.get("block_lift_cm", 0) >= 4 for l in lines if l["kind"] == "block"), "block lift scored"),
        (not fails, "; ".join(fails[:5])),
    ]
    for ok, msg in checks:
        print(("ok  " if ok else "FAIL"), msg)
    print("example:", {k: lines[0][k] for k in ("idx", "kind", "sim_s", "rows", "calls", "lifted", "block_lift_cm")})
    ok = all(c[0] for c in checks)
    print("ALL SERVER TESTS PASSED" if ok else "FAILED", out)
    srv.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
