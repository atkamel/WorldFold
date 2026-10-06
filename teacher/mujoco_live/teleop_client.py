"""Teleop client: the two-mouse game window, connected to a teleop server (local MuJoCo test or Isaac on RunPod).
    python teleop_client.py HOST PORT [--fresh]
Two views, picked by the server's hello: the server's JPEG frames in a pygame window (this file), or, for hello
"view": "state", the laptop draws the scene itself from streamed joint angles + cloth points (teleop_client_ps.py).
Sends both mice + keys ~60x/s, shows the streamed frames, keeps its own copy of the demo log in demos/,
scores every drop locally (fold score) and shows the measured round-trip lag.
Keys as in the local game; Esc pause (frees pointer), F11 full screen / window, H help, Q quit.
Live tuning (teleop/tuning.py): Tab opens the settings panel - Up/Down pick, Left/Right change, 0 back to default,
C rebuild the shirt with the new cloth settings. The settings are saved in tuning_last.json and restored next
session (--fresh: start from the server's values)."""
import os, sys, io, json, time, socket, threading, queue, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
from teleop import protocol  # noqa: E402

FORWARD = {"t": "T", "z": "Z", "r": "R", "a": "A", "return": "ENTER", "keypad enter": "ENTER", "s": "S",
           "=": "+", "+": "+", "[+]": "+", "-": "-", "[-]": "-", "[": "[", "]": "]", "v": "V"}


TUNE_FILE = os.path.join(HERE, "tuning_last.json")


class Client:
    def __init__(self, host, port, fullscreen=True, restore_tuning=True, conn=None):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        try:
            self.dpi = max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96)
        except Exception:
            self.dpi = 1.0
        from rawmouse import MiceReader
        self.dec = protocol.StateDecoder()
        self.sock, self.hello = conn or self._connect(host, port)
        self.FW, self.FH = self.hello.get("frame_size", [960, 600])
        self.tris = np.array(self.hello["tris"], np.int64) if self.hello.get("tris") else None
        if "flat_area_m2" not in self.hello and self.tris is not None and self.hello.get("flat_cloth_cm"):
            from neat_metric import fold_score   # flat footprint of this garment, measured the same way as the score
            flat = np.array(self.hello["flat_cloth_cm"]) / 100
            self.hello["flat_area_m2"] = fold_score(flat, self.tris, 1.0, self.hello.get("thickness", 0.004))["area_ratio"]
        self.mice = MiceReader()
        self.ext_sent = set()
        self.frame, self.hud, self.echo, self.srv_fps, self.new = None, {}, None, 0.0, False
        self.pending = []           # (client time, {mouse: (dx, dy)}) sent but not yet in a received frame
        self.cam = self.hello.get("cam")   # lets us draw the cursor locally (no network delay on the cursor)
        self.lag = None
        self.score, self.best = None, None
        self.ended = None
        self.show_help, self.playing = True, True
        os.makedirs(os.path.join(HERE, "demos"), exist_ok=True)
        self.log_path = os.path.join(HERE, "demos", time.strftime("%Y%m%d_%H%M%S") + f"_remote_{self.hello.get('backend')}.jsonl")
        self.scoreq = queue.Queue()
        self.lock = threading.Lock()
        # live tuning
        self.tspec = self.hello.get("tune") or []
        self.tune_open, self.tune_sel, self.tune_ops = False, 0, []
        if restore_tuning:
            self.tune_ops = self._restore_tuning()
        self._init_ui(fullscreen)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._scorer, daemon=True).start()

    @staticmethod
    def _connect(host, port, wait_s=900):
        """Keep trying until the server is up (Isaac takes ~3 min to boot behind the tunnel)."""
        t0 = time.time()
        while True:
            try:
                s = socket.create_connection((host, port), timeout=10)
                s.settimeout(30)
                hello = protocol.recv(s)
                if hello and hello[0].get("t") == "hello":
                    s.settimeout(None)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    print(f"connected: {hello[0].get('backend')} {hello[0].get('garment', '')}")
                    return s, hello[0]
                s.close()
            except OSError:
                pass
            if time.time() - t0 > wait_s:
                raise SystemExit(f"no teleop server at {host}:{port} after {wait_s}s")
            print(f"waiting for the server at {host}:{port} ... {time.time() - t0:.0f}s", flush=True)
            time.sleep(3)

    # ---------------- window / pointer (pygame view; the Polyscope view overrides these)
    def _init_ui(self, fullscreen):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(f"Fold it - remote ({self.hello.get('backend')})")
        self.font = pygame.font.SysFont("consolas", 15)
        self.big = pygame.font.SysFont("consolas", 19, bold=True)
        self.set_window(fullscreen)

    def set_window(self, fullscreen):
        pg = self.pg
        self.fullscreen = fullscreen
        flags = (pg.FULLSCREEN if fullscreen else pg.RESIZABLE) | pg.SCALED
        self.screen = pg.display.set_mode((self.FW, self.FH), flags)   # GPU stretches the stream to the window
        self.mice.register()
        self.set_play(self.playing)

    def set_play(self, on):
        pg = self.pg
        self.playing = on
        pg.mouse.set_visible(not on)
        pg.event.set_grab(on)
        self.mice.register()
        self.mice.take()

    # ---------------- network
    def _reader(self):
        while True:
            try:
                m = protocol.recv(self.sock)
            except OSError:
                m = None
            if m is None:
                self.ended = self.ended or "connection closed"
                return
            h, p = m
            t = h.get("t")
            if t == "state":
                p = self.dec.decode(h, p)   # compressed difference -> the full state (protocol.StateDecoder)
            if t in ("frame", "state"):     # JPEG (server draws) or state bytes (we draw)
                with self.lock:
                    self.frame, self.hud, self.srv_fps, self.new = p, h.get("hud", {}), h.get("fps", 0.0), True
                    self.echo = h.get("echo") or self.echo
                    if h.get("echo"):
                        self.lag = time.time() - h["echo"]
            elif t == "log":
                with open(self.log_path, "a") as f:
                    for r in h["recs"]:
                        f.write(json.dumps(r) + "\n")
                        if r.get("event") == "drop" and "cloth_cm" in r:
                            self.scoreq.put(r["cloth_cm"])
            elif t == "end":
                self.ended = h.get("why", "server ended")

    def _scorer(self):
        if self.tris is None:
            return
        from neat_metric import fold_score
        while True:
            c = self.scoreq.get()
            try:
                s = fold_score(np.array(c) / 100, self.tris, self.hello["flat_area_m2"], self.hello["thickness"])
                self.score = s
                if self.best is None or s["score"] > self.best["score"]:
                    self.best = s
            except Exception as e:
                print("scoring failed:", repr(e))

    def send_input(self, mice, keys):
        ext = {}
        for h in mice:
            if h not in self.ext_sent:
                ext[str(h)] = bool(self.mice.is_external_mouse(h)); self.ext_sent.add(h)
        msg = dict(t="input", mice={str(h): list(v) for h, v in mice.items()}, keys=keys, ext=ext, ct=time.time(),
                   got=self.dec.last)       # confirms frames: the server then sends compressed differences, never a backlog
        moved = {str(h): (v[0], v[1]) for h, v in mice.items() if v[0] or v[1]}
        if moved:
            with self.lock:
                self.pending.append((msg["ct"], moved))
                del self.pending[:-240]
        if self.tune_ops:
            msg["tune"], self.tune_ops = self.tune_ops, []
        protocol.send(self.sock, msg)

    # ---------------- live tuning
    def _restore_tuning(self):
        """Last session's settings -> 'set' ops (+ one cloth rebuild if any cloth value differs)."""
        try:
            saved = json.load(open(TUNE_FILE))
        except (OSError, ValueError):
            return []
        known = {s["key"]: s for s in self.tspec}
        ops = [["set", k, v] for k, v in saved.items() if k in known]
        if any(known[k]["kind"] == "cloth" and abs(v - known[k]["default"]) > 1e-9 for k, v in saved.items() if k in known):
            ops.append(["apply_cloth"])
        if ops:
            print(f"restoring {len(ops)} tuned settings from {TUNE_FILE} (--fresh to skip)")
        return ops

    def _save_tuning(self, values):
        try:
            with open(TUNE_FILE, "w") as f:
                json.dump(values, f, indent=1)
        except OSError:
            pass

    def tune_action(self, act):
        """Panel actions: 'up', 'down', 'left', 'right', 'reset', 'apply'."""
        n = len(self.tspec)
        if not n:
            return
        k = self.tspec[self.tune_sel]["key"]
        if act in ("up", "down"):
            self.tune_sel = (self.tune_sel + (-1 if act == "up" else 1)) % n
        elif act in ("left", "right"):
            self.tune_ops.append(["adjust", k, 1 if act == "right" else -1])
        elif act == "reset":
            self.tune_ops.append(["reset", k])
        elif act == "apply":
            self.tune_ops.append(["apply_cloth"])

    def tune_key(self, e):
        """pygame keys while the panel is open. Returns True if the key was used."""
        pg = self.pg
        act = {pg.K_UP: "up", pg.K_DOWN: "down", pg.K_LEFT: "left", pg.K_RIGHT: "right", pg.K_0: "reset",
               pg.K_KP0: "reset", pg.K_BACKSPACE: "reset", pg.K_c: "apply"}.get(e.key)
        if act and self.tspec:
            self.tune_action(act)
            return True
        return False

    def draw_tuning(self, hud):
        vals = hud.get("tune") or {}
        rows, kind = [], None
        for i, s in enumerate(self.tspec):
            if s["kind"] != kind:
                kind = s["kind"]
                rows.append({"control": "-- controls (instant)", "grip": "-- grab / release (next grab)",
                             "cloth": "-- cloth physics (press C to rebuild the shirt)"}[kind])
            v = vals.get(s["key"], s["default"])
            shown = v * s["scale"]
            txt = f"{shown:.0f}" if abs(shown) >= 100 else f"{shown:.3g}"
            mark = "*" if abs(v - s["default"]) > 1e-9 else " "
            rows.append(f"{'>' if i == self.tune_sel else ' '}{mark}{s['label']:<30s}{txt:>8s} {s['unit']}")
        if hud.get("cloth_pending"):
            rows.append("CLOTH CHANGED - press C to apply (fresh shirt)")
        rows.append("Up/Down pick  Left/Right change  0 default  C apply cloth  Tab close   (* = changed)")
        self.text("\n".join(rows), (6, 70), bg=(0, 0, 40, 200))

    def track_tuning(self, hud):
        """Save the server's values whenever they change (panel open or not)."""
        vals = hud.get("tune")
        if vals and vals != getattr(self, "_saved_tune", None):
            self._saved_tune = dict(vals)
            self._save_tuning(vals)

    # ---------------- drawing
    def text(self, s, pos, col=(240, 240, 240), font=None, bg=(0, 0, 0, 150)):
        pg = self.pg
        f = font or self.font
        lines = s.split("\n")
        wmax = max(f.size(l)[0] for l in lines) + 12
        box = pg.Surface((wmax, len(lines) * f.get_linesize() + 8), pg.SRCALPHA)
        box.fill(bg)
        self.screen.blit(box, pos)
        for i, l in enumerate(lines):
            self.screen.blit(f.render(l, True, col), (pos[0] + 6, pos[1] + 4 + i * f.get_linesize()))

    def local_cursors(self, hud):
        """Where each arm's aim cursor is right now: the server's cursor in the last frame plus the mouse movement the
        server has not applied yet (same gain as the controller). Drawing only - the arms follow the server."""
        arms = hud.get("arms") or {}
        mice = hud.get("mice") or {}
        if not arms or not mice:
            return {}
        with self.lock:
            echo = self.echo or 0
            self.pending = [x for x in self.pending if x[0] > echo]
            pend = list(self.pending)
        tandem = hud.get("tandem", "off")
        cur = {a: np.array(st.get("cursor", st.get("aim", [0, 0])[:2]), float) for a, st in arms.items()}
        for _, moved in pend:
            for h, (dx, dy) in moved.items():
                if h not in mice:
                    continue
                a, g = mice[h]
                d = np.array([dx * g, -dy * g])
                if tandem == "off":
                    cur[a] = cur[a] + d
                else:
                    cur["L"] = cur["L"] + d
                    cur["R"] = cur["R"] + d * (np.array([-1, 1]) if tandem == "mirror" else 1)
        lim = hud.get("lim")
        if lim:
            for a in cur:
                cur[a] = np.array([np.clip(cur[a][0], *lim[0]), np.clip(cur[a][1], *lim[1])])
        return cur

    def draw_local_cursors(self, hud):
        cam = self.cam
        if not cam:
            return
        P = np.array(cam["P"], float).reshape(3, 4); O = np.array(cam["O"], float)
        W, H = cam["size"]
        sx, sy = self.FW / W, self.FH / H
        for a, xy in self.local_cursors(hud).items():
            z = (hud["arms"][a].get("aim") or [0, 0, 0.03])[2]
            p = P @ np.r_[xy[0] + O[0], xy[1] + O[1], z + O[2], 1.0]
            if p[2] <= 0:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if cam.get("flip"):
                u, v = W - 1 - u, H - 1 - v
            col = (255, 140, 40) if a == "L" else (70, 220, 100)
            c = (int(u * sx), int(v * sy))
            self.pg.draw.circle(self.screen, col, c, 9, 2)
            self.pg.draw.line(self.screen, col, (c[0] - 13, c[1]), (c[0] + 13, c[1]), 1)
            self.pg.draw.line(self.screen, col, (c[0], c[1] - 13), (c[0], c[1] + 13), 1)

    def draw(self):
        with self.lock:
            frame, hud, new = self.frame, dict(self.hud), self.new
            self.new = False
        self.draw_scene(frame if new else None, hud)
        self.draw_hud(hud)
        self.present()

    def draw_scene(self, frame, hud):
        """pygame view: the server's JPEG, then the locally predicted cursors on top."""
        if frame is not None:
            self.surf = self.pg.image.load(io.BytesIO(frame), "f.jpg").convert()
        if getattr(self, "surf", None) is not None:
            self.screen.blit(self.surf, (0, 0))
        self.draw_local_cursors(hud)

    def present(self):
        self.pg.display.flip()

    def draw_hud(self, hud):
        s = self.score
        if s:
            self.text(f"FOLD SCORE {s['score']:4.0f}   best {self.best['score']:.0f}\n"
                      f"area {100 * s['area_ratio']:.0f}%  flat {100 * s['flat']:.0f}%  rect {s['rect']:.2f}  "
                      f"valleys {100 * s['valleys']:.0f}%  air {s['air_mm']:.1f} mm", (6, 6), font=self.big)
        else:
            self.text("FOLD SCORE -  (scored after your first drop)", (6, 6), font=self.big)
        rows = []
        for a, st in hud.get("arms", {}).items():
            lim = "  AT REACH LIMIT" if st.get("reach_limited") else ""
            rows.append(f"{a}: mouse {st.get('mouse')} speed {st.get('speed')} {st.get('phase', ''):8s} "
                        f"carry {st.get('carry_cm')} cm jaws {st.get('jaw_deg')}{lim}")
        lag = f"{self.lag * 1000:.0f} ms" if self.lag is not None else "-"
        rows.append(f"tandem {hud.get('tandem')}   server {self.srv_fps:.0f} fps   lag {lag}   {self.hello.get('backend')}")
        self.text("\n".join(rows), (self.FW - 540, 6))
        fb = hud.get("flowback")
        if fb:
            col = (140, 255, 140) if fb[1] < 5 else (255, 230, 120) if fb[1] < 15 else (255, 140, 120)
            self.text(f"flowback after last drop: avg {fb[0]:.1f} mm, worst 5% {fb[1]:.1f} mm", (6, self.FH - 64), col=col)
        msg = self.ended and f"SERVER ENDED: {self.ended}" or hud.get("msg", "connecting...")
        self.text(msg, (6, self.FH - 32), col=(255, 230, 120))
        self.track_tuning(hud)
        if self.tune_open:
            self.draw_tuning(hud)
        elif self.show_help:
            self.text("HOLD LEFT grab top layer + lift | WHEEL-CLICK whole stack | RIGHT drag low (still = pin)\n"
                      "WHEEL turn jaws   [ ] carry height   T tandem   Z undo   R new shirt   V camera\n"
                      "+/- mouse speed   A re-pick mice   Tab settings   F11 full screen   Esc pause   H hide   Q quit",
                      (6, 70))

    def run(self):
        pg = self.pg
        t_next = time.perf_counter()
        while True:
            keys = []
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_q):
                    return self.close()
                if e.type == pg.KEYDOWN:
                    name = pg.key.name(e.key)
                    if e.key == pg.K_TAB and self.tspec:
                        self.tune_open = not self.tune_open
                    elif self.tune_open and self.tune_key(e):
                        pass
                    elif e.key == pg.K_ESCAPE:
                        self.set_play(not self.playing)
                    elif e.key in (pg.K_F11, pg.K_f):
                        self.set_window(not self.fullscreen)
                    elif e.key == pg.K_h:
                        self.show_help = not self.show_help
                    elif name in FORWARD:
                        keys.append(FORWARD[name])
                if e.type == pg.WINDOWFOCUSLOST and self.playing:
                    self.set_play(False)
                if e.type == pg.MOUSEBUTTONDOWN and not self.playing:
                    self.set_play(True)
            mice = self.mice.take() if self.playing else {}
            if not self.ended:
                try:
                    self.send_input(mice, keys)
                except OSError:
                    self.ended = "connection lost"
            self.draw()
            t_next += 1 / 60
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.2:
                t_next = time.perf_counter()

    def close(self):
        try:
            protocol.send(self.sock, dict(t="bye"))
        except OSError:
            pass
        self.pg.quit()
        print("demo log:", self.log_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    host = args[0] if args else "127.0.0.1"
    port = int(args[1]) if len(args) > 1 else 7777
    conn = Client._connect(host, port)
    if conn[1].get("view") == "state":
        from teleop_client_ps import PolyscopeClient as View
    else:
        View = Client
    View(host, port, restore_tuning="--fresh" not in sys.argv, conn=conn).run()
