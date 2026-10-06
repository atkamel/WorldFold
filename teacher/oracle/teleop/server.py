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
                 idle_s=300, max_s=3300, on_key=None, log=print, wall_speed=0.0, state=None, profile=None, window=3,
                 keepalive=None, keepalive_every=48):
        self.g, self.render = game, render
        self.hello = dict(hello, tune=game.tune.spec()) if hasattr(game, "tune") else hello
        self.port, self.dt, self.frame_every = port, dt, frame_every
        self.idle_s, self.max_s = idle_s, max_s
        self.on_key = on_key            # server-side keys the controller doesn't know (e.g. "V" camera view)
        self.wall_speed = wall_speed    # max arm-speed scale-up when the sim is slower than real time (0 = off)
        self.state = state              # state(game) -> bytes, or a function returning bytes (packed on the sender
                                        # thread): send state for the laptop to draw instead of a JPEG
        self.log = log
        self.log_path = log_path
        self.window = window            # compressed stream: max unconfirmed frames in flight
        # keepalive(): lets the backend run its app loop (Isaac without cameras never does on its own); called about
        # once a second while waiting for the client and every `keepalive_every` ticks while playing
        self.keepalive, self.keepalive_every, self._ka_s, self._ka_n = keepalive, keepalive_every, 0.0, 0
        self.profile = profile          # (first tick, n ticks, stats path): cProfile those ticks, log the top functions
        self.outbox = []
        self.inbox = queue.Queue()
        self.ext = {}
        self.encode = jpeg_encoder()
        game.sink = self._sink
        # sender thread: the main loop only hands messages over, so packing + network never hold up the next physics step
        self._cv = threading.Condition()
        self._logs, self._frame, self._send_err, self._send_s = [], None, None, 0.0
        # compressed state stream (protocol.StateEncoder): switched on when the client's inputs carry "got"
        self._enc, self._sent_n, self._sent_bytes = None, 0, 0

    def _frame_ready(self):
        f = self._frame
        if f is None:
            return False
        if f is False or self._enc is None or f[0].get("t") != "state":
            return True
        return self._enc.can_send()      # flow control: at most `window` unconfirmed frames on the way

    def _sender(self, conn):
        """Sends queued log messages in order, then the newest frame (an older unsent frame is dropped: it's stale)."""
        while True:
            with self._cv:
                while not self._logs and not self._frame_ready():
                    self._cv.wait()
                logs, self._logs = self._logs, []
                frame = None
                if self._frame_ready():
                    frame, self._frame = self._frame, None
            try:
                for h in logs:
                    if h.get("t") == "log":
                        self._send_logs(conn, h)
                    else:
                        protocol.send(conn, h)
                if frame is False:           # stop signal (after the last logs went out)
                    return
                if frame is not None:
                    t = time.perf_counter()
                    h, payload = frame
                    payload = payload() if callable(payload) else payload
                    if self._enc is not None and h.get("t") == "state":
                        extra, payload = self._enc.encode(payload)
                        h = dict(h, **extra)
                    protocol.send(conn, h, payload)
                    self._send_s += time.perf_counter() - t
                    self._sent_n += 1; self._sent_bytes += len(payload)
            except OSError as e:
                self._send_err = e
                return

    def _post(self, header=None, frame=None):
        if self._send_err:
            raise self._send_err
        with self._cv:
            if header is not None:
                self._logs.append(header)
            if frame is not None:
                self._frame = frame
            self._cv.notify()

    def _profile_tick(self, n):
        if not self.profile:
            return
        import cProfile, io, pstats
        first, count, path = self.profile
        if n == first:
            self._pr, self._pr_t = cProfile.Profile(), time.perf_counter()
            self._pr.enable()
            self.log(f"TELEOP profile: ticks {first}..{first + count}")
        elif n == first + count and getattr(self, "_pr", None):
            self._pr.disable()
            wall = time.perf_counter() - self._pr_t
            self._pr.dump_stats(path)
            self.log(f"TELEOP profile: {count} ticks in {wall:.1f} s = {wall / count * 1000:.1f} ms/tick (with profiler "
                     f"overhead); stats in {path}")
            for key in ("cumulative", "tottime"):
                b = io.StringIO()
                pstats.Stats(self._pr, stream=b).sort_stats(key).print_stats(40)
                for line in b.getvalue().splitlines():
                    if line.strip():
                        self.log("PROF " + line[:220])
            self._pr = None

    def _keepalive(self):
        t = time.perf_counter()
        try:
            self.keepalive()
        except Exception as e:           # never end a session over this
            self.log(f"TELEOP keepalive failed: {e!r}")
            self.keepalive = None
        self._ka_s += time.perf_counter() - t
        self._ka_n += 1

    def _sink(self, rec):
        self.outbox.append(rec)          # the sender thread writes it to log_path and sends it (_send_logs)

    def _send_logs(self, conn, h):
        """Demo-log records: turned into JSON once (a record with a cloth snapshot is ~340 kB of text), appended to
        log_path and sent to the client with that same text."""
        js = [json.dumps(r, default=protocol._default) for r in h["recs"]]
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write("\n".join(js) + "\n")
        protocol.send_json(conn, '{"t":"log","recs":[' + ",".join(js) + "]}")

    def _reader(self, conn):
        while True:
            try:
                m = protocol.recv(conn)
            except OSError:
                m = None
            if m is None or m[0].get("t") == "bye":
                self.inbox.put(None)
                return
            if "got" in m[0]:                # client confirms frames: compressed stream + flow control
                with self._cv:
                    if self._enc is None:
                        self._enc = protocol.StateEncoder(window=self.window)
                    self._enc.ack(m[0]["got"])
                    self._cv.notify()
            self.inbox.put(m[0])

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(1)
        self.log(f"TELEOP listening on port {self.port}")
        t_start = time.time()
        srv.settimeout(1.0 if self.keepalive else self.idle_s)
        while True:
            try:
                conn, addr = srv.accept()
                break
            except socket.timeout:
                if time.time() - t_start > self.idle_s:
                    self.log("TELEOP nobody connected"); return "nobody connected"
                self._keepalive()
        conn.settimeout(None)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.log(f"TELEOP client {addr}")
        protocol.send(conn, dict(t="hello", **self.hello))
        threading.Thread(target=self._reader, args=(conn,), daemon=True).start()
        sender = threading.Thread(target=self._sender, args=(conn,), daemon=True)
        sender.start()
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
                self._profile_tick(n)
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
                    self._post(dict(t="log", recs=self.outbox)); self.outbox = []
                if n % self.frame_every == 0:
                    if self.state:                   # the laptop draws: send the state (joint angles + cloth points)
                        payload, kind = self.state(self.g), "state"   # may be a function: packed on the sender thread
                        t2 = time.perf_counter()
                    else:                            # the server draws: rendered picture as JPEG
                        img = self.render(self.g)
                        t2 = time.perf_counter()
                        payload, kind = self.encode(img), "frame"
                    perf["render"] += t2 - t1; perf["encode"] += time.perf_counter() - t2
                    fps_n += 1
                    if time.time() - fps_t > 1.0:
                        fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
                    self._post(frame=(dict(t=kind, hud=self.g.hud(), echo=echo, sim_t=round(self.g.time, 3),
                                           fps=round(fps, 1)), payload))
                perf["n"] += 1
                if self.keepalive and self.keepalive_every and n % self.keepalive_every == 0:
                    self._keepalive()
                if time.time() - perf["t"] > 5.0:
                    k = 1000 / perf["n"]
                    self.log(f"TELEOP perf: {perf['n'] / (time.time() - perf['t']):.1f} ticks/s | per tick: sim {perf['sim'] * k:.0f} ms, "
                             f"controller {(perf['tick'] - perf['sim']) * k:.0f} ms, frame {perf['render'] * k:.0f} ms, "
                             f"jpeg {perf['encode'] * k:.0f} ms | sender thread {self._send_s * k:.0f} ms (overlapped)"
                             f" | sent {self._sent_n / (time.time() - perf['t']):.1f} frames/s, "
                             f"{self._sent_bytes / max(self._sent_n, 1) / 1000:.1f} kB each"
                             f"{f' (compressed, window {self._enc.window})' if self._enc else ''}"
                             f"{f' | app keepalive {self._ka_s / max(self._ka_n, 1) * 1000:.1f} ms x {self._ka_n}' if self._ka_n else ''}")
                    self._send_s, self._sent_n, self._sent_bytes, self._ka_s, self._ka_n = 0.0, 0, 0, 0.0, 0
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
            with self._cv:                   # stop the sender after it flushes what's queued
                self._frame = False
                self._cv.notify()
            sender.join(timeout=5)
            try:
                protocol.send(conn, dict(t="end", why=why))
            except OSError:
                pass
            conn.close(); srv.close()
        self.log(f"TELEOP end: {why} after {n} ticks")
        return why
