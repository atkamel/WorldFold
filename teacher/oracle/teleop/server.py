"""Teleop server: runs the controller + physics backend in real time, takes mouse/key input from one client
over TCP, streams JPEG frames, the on-screen info and the demo log back. Backend-agnostic: the caller passes
the controller (teleop.controller.Game on some world) and a render(game) -> RGB uint8 image function.

Safety (it may run on a paid GPU): stops after `idle_s` without any input message, at `max_s` total, and
when the client disconnects. The demo log is also written to `log_path` on the server side.
"""
import json
import socket
import threading
import time
import queue

from . import protocol


def jpeg_encoder(quality=80):
    """Fastest available JPEG encoder: OpenCV, else Pillow."""
    try:
        import cv2

        def enc(img):
            ok, buf = cv2.imencode(".jpg", img[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf.tobytes()
        return enc
    except ImportError:
        import io
        from PIL import Image

        def enc(img):
            b = io.BytesIO()
            Image.fromarray(img).save(b, format="JPEG", quality=quality)
            return b.getvalue()
        return enc


def merge_inputs(msgs):
    """Several input messages that arrived within one tick -> one mice dict, all keys, newest echo."""
    mice, keys, ext, echo, tune = {}, [], {}, None, []
    for m in msgs:
        for h, (dx, dy, wh, l, r, ev) in m.get("mice", {}).items():
            if h in mice:
                pdx, pdy, pwh, _, _, pev = mice[h]
                mice[h] = (pdx + dx, pdy + dy, pwh + wh, l, r, pev + list(ev))
            else:
                mice[h] = (dx, dy, wh, l, r, list(ev))
        keys += m.get("keys", [])
        tune += m.get("tune", [])
        ext.update(m.get("ext", {}))
        echo = m.get("ct", echo)
    return mice, keys, ext, echo, tune


class TeleopServer:
    def __init__(self, game, render, hello, port=7777, dt=1 / 60, frame_every=2, log_path=None,
                 idle_s=300, max_s=3300, on_key=None, log=print, wall_speed=0.0, state=None):
        self.g, self.render = game, render
        self.hello = dict(hello, tune=game.tune.spec()) if hasattr(game, "tune") else hello
        self.port, self.dt, self.frame_every = port, dt, frame_every
        self.idle_s, self.max_s = idle_s, max_s
        self.on_key = on_key            # server-side keys the controller doesn't know (e.g. "V" camera view)
        self.wall_speed = wall_speed    # max arm-speed scale-up when the sim is slower than real time (0 = off)
        self.state = state              # state(game) -> bytes: send state for the laptop to draw instead of a JPEG
        self.log = log
        self.log_path = log_path
        self.outbox = []
        self.inbox = queue.Queue()
        self.ext = {}
        self.encode = jpeg_encoder()
        game.sink = self._sink

    def _sink(self, rec):
        self.outbox.append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=protocol._default) + "\n")

    def _reader(self, conn):
        while True:
            try:
                m = protocol.recv(conn)
            except OSError:
                m = None
            if m is None or m[0].get("t") == "bye":
                self.inbox.put(None)
                return
            self.inbox.put(m[0])

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(1)
        srv.settimeout(self.idle_s)
        self.log(f"TELEOP listening on port {self.port}")
        t_start = time.time()
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            self.log("TELEOP nobody connected"); return "nobody connected"
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.log(f"TELEOP client {addr}")
        protocol.send(conn, dict(t="hello", **self.hello))
        threading.Thread(target=self._reader, args=(conn,), daemon=True).start()
        why, last_input, echo = "client left", time.time(), None
        t_next = time.perf_counter()
        n, fps_t, fps_n, fps = 0, time.time(), 0, 0.0
        perf = dict(tick=0.0, sim=0.0, render=0.0, encode=0.0, n=0, t=time.time())   # where each tick's time goes
        try:
            while True:
                msgs, closed = [], False
                while not self.inbox.empty():
                    m = self.inbox.get()
                    if m is None:
                        closed = True
                        break
                    msgs.append(m)
                if closed:
                    break
                if any(m.get("mice") or m.get("keys") or m.get("tune") for m in msgs):   # real input, not keep-alive
                    last_input = time.time()
                elif time.time() - last_input > self.idle_s:
                    why = f"no input for {self.idle_s}s"; break
                if time.time() - t_start > self.max_s:
                    why = f"max {self.max_s}s reached"; break
                mice, keys, ext, e, tune = merge_inputs(msgs)
                self.ext.update(ext)
                echo = e if e is not None else echo
                for k in keys:
                    if self.on_key and self.on_key(k):
                        continue
                    self.g.key(k)
                for op in tune:              # live tuning (tuning.py); "apply_cloth" rebuilds the shirt
                    try:
                        self.g.tune_op(op)
                    except Exception as ex:  # a bad op must never end a paid session
                        self.g.msg = f"tuning failed: {ex!r}"[:120]
                        self.log(f"TELEOP tune op {op} failed: {ex!r}")
                t0 = time.perf_counter()
                sim0 = getattr(self.g.w, "t_sim", 0.0)
                # arm speeds are per simulated second; when a tick takes longer than the sim time it advances, scale them up
                # so the arms follow the mouse at wall-clock pace (capped; the simulated motors' speed limit still applies)
                if self.wall_speed:
                    now_t = time.perf_counter()
                    if getattr(self, "_t_last", None) is not None:
                        per = now_t - self._t_last
                        self._per = per if getattr(self, "_per", None) is None else 0.9 * self._per + 0.1 * per
                        self.g.speed_scale = float(min(self.wall_speed, max(1.0, self._per / self.dt)))
                    self._t_last = now_t
                self.g.tick(mice, self.dt, lambda h: self.ext.get(h, True))
                t1 = time.perf_counter()
                perf["tick"] += t1 - t0
                perf["sim"] += getattr(self.g.w, "t_sim", 0.0) - sim0
                n += 1
                if self.outbox:
                    protocol.send(conn, dict(t="log", recs=self.outbox)); self.outbox = []
                if n % self.frame_every == 0:
                    if self.state:                   # the laptop draws: send the state (joint angles + cloth points)
                        payload, kind = self.state(self.g), "state"
                        t2 = time.perf_counter()
                    else:                            # the server draws: rendered picture as JPEG
                        img = self.render(self.g)
                        t2 = time.perf_counter()
                        payload, kind = self.encode(img), "frame"
                    perf["render"] += t2 - t1; perf["encode"] += time.perf_counter() - t2
                    fps_n += 1
                    if time.time() - fps_t > 1.0:
                        fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
                    protocol.send(conn, dict(t=kind, hud=self.g.hud(), echo=echo, sim_t=round(self.g.time, 3),
                                             fps=round(fps, 1)), payload)
                perf["n"] += 1
                if time.time() - perf["t"] > 5.0:
                    k = 1000 / perf["n"]
                    self.log(f"TELEOP perf: {perf['n'] / (time.time() - perf['t']):.1f} ticks/s | per tick: sim {perf['sim'] * k:.0f} ms, "
                             f"controller {(perf['tick'] - perf['sim']) * k:.0f} ms, frame {perf['render'] * k:.0f} ms, "
                             f"jpeg {perf['encode'] * k:.0f} ms")
                    perf.update(tick=0.0, sim=0.0, render=0.0, encode=0.0, n=0, t=time.time())
                t_next += self.dt
                slack = t_next - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -0.25:          # backend slower than real time: don't try to catch up
                    t_next = time.perf_counter()
        except OSError as e:
            why = f"connection error {e!r}"
        finally:
            try:
                protocol.send(conn, dict(t="end", why=why))
            except OSError:
                pass
            conn.close(); srv.close()
        self.log(f"TELEOP end: {why} after {n} ticks")
        return why
