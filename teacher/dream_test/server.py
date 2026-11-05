"""Dream-1 grasp test session: serves LeHome's remote protocol (remote_loop.py + dream_isaac_patch.py) to one Isaac sim
it launches, runs the battery (tasks.py) through Dream-1 (bridge.py), and writes per try: a result line, the video of
what Dream saw, and the garment's start shape for scoring (score.py).

    <dream venv python> server.py --battery smoke --out DIR --ch /opt/lehome_solution/lehome-challenge
    --fake-dream: a stand-in client (arms hold still, grip closes and opens) to test the plumbing with no API calls

Output lines a watcher can key on: DREAMRESULT {...} per try (written as soon as it ends), DREAMDONE at the end.
"""
import argparse
import base64
import http.server
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bridge  # noqa: E402
import tasks  # noqa: E402


def log(msg):
    print(msg, flush=True)


class FakeClient:
    """Stands in for dream.DreamClient: holds the arms where they are, closes then reopens the grippers."""

    def __init__(self):
        self.obs, self.last_timing = None, None

    def reset_episode(self):
        self.obs = None

    def set_observation(self, frames, observation, timestamp=None):
        assert set(frames) == {"video.overhead", "video.wrist_l", "video.wrist_r"}
        self.obs = observation

    def get_action(self, commanded=None):
        o = self.obs
        g = np.r_[np.linspace(1, 0, 24), np.linspace(0, 1, 24)]
        mk = lambda arm: np.c_[np.tile(np.asarray(arm, float), (48, 1)), g]
        return {"q_l_chunk": mk(o["state.left_arm"]), "q_r_chunk": mk(o["state.right_arm"])}


class Episode:
    def __init__(self, spec, out):
        self.spec, self.out = spec, out
        self.t, self.rows, self.dt, self.wall0 = 0.0, 0, None, time.time()
        self.grasps, self.block_z0, self.block_zmax, self.success, self.error = [], None, None, False, None
        self.video, self.clock = None, None

    def name(self):
        s = self.spec
        return f"ep{s['idx']:03d}_{s['kind']}_{s['garment']}"


class Session:
    def __init__(self, args):
        self.args = args
        self.out = pathlib.Path(args.out)
        (self.out / "videos").mkdir(parents=True, exist_ok=True)
        (self.out / "start").mkdir(exist_ok=True)
        eps = tasks.battery(args.battery)
        self.todo = [e for e in eps if e["idx"] % args.shards == args.shard]
        self.next_i, self.ep, self.done, self.lock = 0, None, [], threading.Lock()
        self.sim, self.errors = None, 0
        if args.fake_dream:
            self.client = FakeClient()
        else:
            import dream
            self.client = dream.DreamClient()
            self.client.policy(bridge.make_config(args.model))
        self.bridge = bridge.DreamBridge(self.client, log=log)

    # ---- protocol ------------------------------------------------------------------------------------------------
    def next_task(self, body):
        time.sleep(0.5)                       # let the sim's GRASP lines for the episode that just ended arrive
        self.finish()
        if self.next_i >= len(self.todo) or self.errors >= 3:
            return {"shutdown": True}
        spec = self.todo[self.next_i]
        self.next_i += 1
        self.ep = Episode(spec, self.out)
        gtype = "_".join(spec["garment"].split("_")[:2]).lower()
        return {"garment_name": spec["garment"], "garment_type": gtype, "seed": spec["seed"], "ep_idx": spec["idx"],
                "stop_on_success": spec["kind"] == "fold"}

    def reset(self, body):
        self.bridge.begin(self.ep.spec["prompt"])
        return {}

    def dream_episode(self, body):
        ep = self.ep
        ep.dt = float(body.get("physics_dt") or 1 / 90)
        np.savez_compressed(self.out / "start" / f"{ep.name()}.npz", verts=np.asarray(body["verts"], np.float32),
                            local=np.asarray(body["local"], np.float32),
                            check_points=np.asarray(body.get("check_points", []), np.int64),
                            table_z=float(body.get("table_z", 0.0)), garment=body.get("garment", ""))
        return {"block": ep.spec["kind"] == "block"}

    def infer(self, body):
        ep = self.ep
        if ep.dt is None:
            ep.dt = float(body.get("dream_physics_dt") or 1 / 90)
        if ep.clock is None:
            ep.clock = bridge.RowClock(ep.dt)
        bp = body.get("dream_block_pos")
        if bp is not None:
            ep.block_z0 = bp[2] if ep.block_z0 is None else ep.block_z0
            ep.block_zmax = bp[2] if ep.block_zmax is None else max(ep.block_zmax, bp[2])
        if ep.t >= ep.spec["seconds"] or ep.error:
            return {"stop": True}
        images = {k: bridge.decode_image(body[k]) for k in bridge.CAMERAS if k in body}
        try:
            small = self.bridge.observe(images, body["observation.state"], ep.t)
            self._video(ep, small)
            action = self.bridge.next_action()
        except Exception as e:                # Dream refused or failed: end this try, count it
            ep.error = repr(e)[:300]
            self.errors += 1
            log(f"[dream] try {ep.name()} stopped: {ep.error}")
            return {"stop": True}
        self.errors = 0
        n = ep.clock.next()
        ep.t += n * ep.dt
        ep.rows += 1
        return {"actions": action.tolist(), "n_steps": n}

    def check_status(self, body):
        if self.ep is not None:
            self.ep.success = True
        return {}

    # ---- per try ---------------------------------------------------------------------------------------------------
    def _video(self, ep, small):
        import cv2
        frame = np.hstack([small["overhead"], small["wrist_l"], small["wrist_r"]])[..., ::-1]
        if ep.video is None:
            ep.video = cv2.VideoWriter(str(self.out / "videos" / f"{ep.name()}.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), bridge.ROW_HZ, (frame.shape[1], frame.shape[0]))
            cv2.imwrite(str(self.out / "videos" / f"{ep.name()}_first.png"), frame)
        ep.video.write(np.ascontiguousarray(frame))

    def grasp_line(self, rec):
        with self.lock:
            if self.ep is not None:
                self.ep.grasps.append(rec)

    def finish(self):
        ep, self.ep = self.ep, None
        if ep is None:
            return
        if ep.video is not None:
            ep.video.release()
        with self.lock:
            gs = list(ep.grasps)
        lifted = [g for g in gs if g.get("n_pinch", 0) >= 5 and g.get("max_lift_cm", 0) >= 4.0]
        held = [g for g in lifted if g.get("t_lift4") is not None and (g["t_open"] - g["t_lift4"]) >= 2 * bridge.ROW_HZ
                and g.get("lift_open_cm", 0) >= 2.0 and g.get("outcome") == "held_lifted"]
        calls = self.bridge.calls
        res = dict(idx=ep.spec["idx"], kind=ep.spec["kind"], garment=ep.spec["garment"], prompt=ep.spec["prompt"],
                   target=ep.spec["target"], sim_s=round(ep.t, 2), rows=ep.rows, wall_s=round(time.time() - ep.wall0, 1),
                   calls=len(calls), rtt_ms_median=float(np.median([c["rtt_ms"] for c in calls])) if calls else None,
                   closes=len(gs), pinched=sum(g.get("n_pinch", 0) >= 5 for g in gs), lifted=len(lifted) > 0,
                   lifted_arms=sorted({g["side"] for g in lifted}), held_2s=len(held) > 0,
                   max_lift_cm=max([g.get("max_lift_cm", 0) for g in gs], default=0.0),
                   layers=[g.get("layers") for g in lifted], lehome_success=ep.success, error=ep.error)
        if ep.block_z0 is not None:
            res["block_lift_cm"] = round((ep.block_zmax - ep.block_z0) * 100, 1)
            res["lifted"] = res["block_lift_cm"] >= 4.0
        with open(self.out / "results.jsonl", "a") as f:
            f.write(json.dumps(dict(res, grasps=gs, call_log=calls)) + "\n")
        self.done.append(res)
        log("DREAMRESULT " + json.dumps(res))


def make_handler(sess):
    routes = {"/ping": lambda b: {"ok": True}, "/next_task": sess.next_task, "/reset": sess.reset,
              "/dream_episode": sess.dream_episode, "/infer": sess.infer, "/snapshot": lambda b: {"status": "ok"},
              "/update_check_status": sess.check_status}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            fn = routes.get(self.path)
            try:
                out = fn(body) if fn else {"error": f"unknown {self.path}"}
            except Exception as e:
                import traceback
                traceback.print_exc()
                out = {"stop": True, "error": repr(e)}
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    return H


def launch_sim(args, sess, url):
    first = sess.todo[0]
    gtype = "_".join(first["garment"].split("_")[:2]).lower()
    cmd = [args.python, "-u", "-m", "scripts.eval", "--policy_type", "remote", "--remote_url", url,
           "--garment_type", gtype, "--garment_name", first["garment"], "--num_episodes", "9999",
           "--max_steps", "3000", "--headless", "--enable_cameras", "--device", "cpu", "--seed", str(first["seed"]),
           "--session_id", f"dream{args.shard}"]
    env = dict(os.environ, LEHOME_DISABLE_KEYBOARD="1", PYTHONUNBUFFERED="1", LEHOME_NO_DEPTH="1",
               LEHOME_REMOTE_URL=url, DREAM_SESSION="1", DREAM_CAMS="1")
    logf = open(sess.out / f"sim{args.shard}.log", "a")
    sim = subprocess.Popen(cmd, cwd=args.ch, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           bufsize=1, preexec_fn=os.setsid)

    def read():
        for line in sim.stdout:
            logf.write(line)
            logf.flush()
            if line.startswith("GRASP {"):
                try:
                    sess.grasp_line(json.loads(line[6:]))
                except ValueError:
                    pass
            elif any(k in line for k in ("Traceback", "Error", "[dream]", "ATTACH", "[arm]", "EXCEPTION")):
                log(f"[sim{args.shard}] {line.rstrip()[:300]}")
    threading.Thread(target=read, daemon=True).start()
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="smoke")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ch", default="/opt/lehome_solution/lehome-challenge")
    ap.add_argument("--python", default="/opt/lehome_solution/lehome-challenge/.venv/bin/python")
    ap.add_argument("--model", default=bridge.CHECKPOINT)
    ap.add_argument("--fake-dream", action="store_true")
    ap.add_argument("--no-sim", action="store_true", help="serve only (offline test drives it)")
    args = ap.parse_args()
    sess = Session(args)
    port = args.port + args.shard
    srv = http.server.HTTPServer(("127.0.0.1", port), make_handler(sess))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"[dream] shard {args.shard}/{args.shards}: {len(sess.todo)} tries, model "
        f"{'FAKE' if args.fake_dream else args.model}, serving on {port}")
    if args.no_sim:
        return sess, srv
    sim = launch_sim(args, sess, f"http://127.0.0.1:{port}")
    t0 = time.time()
    while sim.poll() is None:
        time.sleep(2)
        if sess.next_i >= len(sess.todo) and sess.ep is None and time.time() - t0 > 5:
            time.sleep(20)                     # shutdown was sent; give the sim a moment to exit on its own
            break
    sess.finish()
    if sim.poll() is None:
        os.killpg(os.getpgid(sim.pid), signal.SIGKILL)
    res = sess.done
    summary = {"tries": len(res), "lifted": sum(r["lifted"] for r in res), "held_2s": sum(r["held_2s"] for r in res),
               "errors": sum(bool(r["error"]) for r in res), "sim_exit": sim.poll()}
    (sess.out / f"summary{args.shard}.json").write_text(json.dumps(summary, indent=1))
    log("DREAMDONE " + json.dumps(summary))
    os._exit(0)


if __name__ == "__main__":
    main()
