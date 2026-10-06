"""Fold it: a two-mouse cloth folding game. One USB mouse per robot arm, first-person view.

  MOUSE (one per arm)
    move                 slide that gripper over the table (height is automatic)
    hold LEFT button     gripper drops, grabs the TOP layer under it, lifts it; move to carry
    release LEFT         lowers the cloth and lets go
    hold WHEEL-CLICK     same, but grabs the WHOLE STACK (to fold an already-folded shirt again)
    hold RIGHT button    same, but drags the cloth low along the table (smoothing / flattening);
                         keep the mouse still = pin the cloth down while the other arm folds
    wheel                turn that arm's jaws (15 deg per notch); held cloth turns with them
    [ / ]                carry height of the arm you moved last
  KEYS
    T  tandem: off -> both arms move together -> mirrored (either mouse drives both)
    Z  undo last grab     R  reset shirt      V  camera view     N  minimap
    +/-  speed of the mouse you moved last     A  re-pick mice     Enter  play with one mouse (S swaps arm)
    H  controls panel     Esc  pause (frees your pointer)       Q  quit     F11  full screen / window
Everything is logged to demos/<session>.jsonl (arm targets 20x per second, cloth state 1x per second
and at every grab/drop, plus undo/reset), ready to train on.
"""
import os, sys, json, time
import numpy as np
import mujoco
from world import World, ARMS, THICK
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle"))
from teleop import controller  # noqa: E402  (shared game logic: local game, bridge test and Isaac)
from teleop.controller import TANDEM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = [  # lookat, distance, elevation, azimuth, fovy
    ([0, 0.20, 0.0], 0.46, -49, 90, 65),   # first person: head just behind and between the arms
    ([0, 0.19, 0.0], 0.60, -70, 90, 45),   # leaning over the table
    ([0, 0.18, 0.0], 0.62, -89.5, 90, 45),  # top-down
]
VIEW_NAMES = ["first person", "over the table", "top-down"]
ARM_RGB = {"L": (1.0, 0.45, 0.1), "R": (0.2, 0.75, 0.3)}


class Game(controller.Game):
    """The shared controller (WorldFold/teacher/oracle/teleop/controller.py) on the local MuJoCo world,
    logging to demos/<session>.jsonl."""

    def __init__(self, log=True):
        self.log_path = None
        if log:
            os.makedirs(os.path.join(HERE, "demos"), exist_ok=True)
            self.log_path = os.path.join(HERE, "demos", time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
        super().__init__(World(), sink=self._to_file)

    def _to_file(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")


# ==================================================================== window (pygame)
class App:
    def __init__(self, size=(1180, 690), fullscreen=True):
        import ctypes
        # Windows display scaling (150% on this laptop): declare the game DPI-aware BEFORE creating the
        # window, otherwise "full screen" only covers 1280x800 of the real 1920x1200 pixels
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        try:
            self.dpi_scale = max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96)
        except Exception:
            self.dpi_scale = 1.0
        import pygame
        from rawmouse import MiceReader
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Fold it - two-mouse cloth folding")
        self.font = pygame.font.SysFont("consolas", 17)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self.g = Game()
        self.mice = MiceReader()
        self.windowed_size = size
        self.renderer = None
        self.set_window(fullscreen)
        self.mini = mujoco.Renderer(self.g.w.m, 200, 280)
        self.view, self.show_help, self.show_mini, self.playing = 0, True, True, True
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.mini_opt = mujoco.MjvOption(); self.mini_opt.geomgroup[2] = 0
        self.mini_cam = mujoco.MjvCamera(); self.mini_cam.lookat[:] = [0, 0.18, 0]
        self.mini_cam.distance, self.mini_cam.elevation, self.mini_cam.azimuth = 0.62, -89.5, 90
        self.set_play(True)

    def set_window(self, fullscreen, size=None):
        """Full screen (F11) or a resizable window; the 3D view is re-rendered at the window's size."""
        pg = self.pg
        self.fullscreen = fullscreen
        if fullscreen:
            # render at the scaled size (1280x800 here) and let SDL stretch it over every real pixel on the GPU
            dw, dh = pg.display.get_desktop_sizes()[0]
            size_fs = (int(dw / self.dpi_scale), int(dh / self.dpi_scale))
            self.screen = pg.display.set_mode(size_fs, pg.FULLSCREEN | pg.SCALED)
        else:  # a window you can drag to any size; the picture is stretched to fit
            self.screen = pg.display.set_mode(size or self.windowed_size, pg.RESIZABLE | pg.SCALED)
        self.W, self.H = self.screen.get_size()
        m = self.g.w.m
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, self.H)
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(m, self.H, self.W)
        self.mice.register()
        if getattr(self, "playing", False):
            self.set_play(True)

    def set_play(self, on):
        pg = self.pg
        self.playing = on
        pg.mouse.set_visible(not on)
        pg.event.set_grab(on)  # keep the pointer inside the window. NOT relative mode: SDL would take over
        self.mice.register()   # Windows' raw mouse input, which is how we tell the two mice apart
        self.mice.take()
        if not on:
            for a in ARMS:  # drop anything held so nothing hangs while paused
                if self.g.w.held[a]:
                    self.g.phase[a] = "place"
            self.g.msg = "PAUSED - pointer free.  Click the window or press Esc to play"

    def markers(self, scn, with_rays=True):
        g, w = self.g, self.g.w
        eye = np.eye(3).ravel()
        for a in ARMS:
            c = g.cmd[a]
            top = w.surface_under(c[:2])
            rgb = list(ARM_RGB[a]) if not w.held[a] else [0.9, 0.1, 0.1]
            if g.reach_flash[a] > 0:
                rgb = [0.6, 0.6, 0.6]
            h = max(c[2] - top, 0.001)
            items = [(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.016, 0.0006, 0], [c[0], c[1], top + 0.001], 0.5)]
            if with_rays:
                items.append((mujoco.mjtGeom.mjGEOM_CYLINDER, [0.0012, h / 2, 0], [c[0], c[1], top + h / 2], 0.6))
            for typ, size, pos, alpha in items:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.array(size, float), np.array(pos, float), eye,
                                    np.array(rgb + [alpha], np.float32))
                scn.ngeom += 1
            # jaw bar: the line the jaws close along (turn it with the wheel)
            if scn.ngeom < scn.maxgeom:
                yaw = w.jaw_yaw(a)
                u = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.022
                z = top + 0.0015
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                                    np.array(rgb + [0.9], np.float32))
                mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025,
                                     np.array([c[0], c[1], z]) - u, np.array([c[0], c[1], z]) + u)
                scn.ngeom += 1

    def draw(self):
        pg, g, w = self.pg, self.g, self.g.w
        la, dist, el, az, fov = VIEWS[self.view]
        self.cam.lookat[:] = la
        self.cam.distance, self.cam.elevation, self.cam.azimuth = dist, el, az
        w.m.vis.global_.fovy = fov
        self.renderer.update_scene(w.d, self.cam, self.opt)
        self.markers(self.renderer.scene)
        img = self.renderer.render()
        self.screen.blit(pg.image.frombuffer(img.tobytes(), (self.W, self.H), "RGB"), (0, 0))
        if self.show_mini:
            self.n_draw = getattr(self, "n_draw", 0) + 1
            if self.n_draw % 3 == 1 or not hasattr(self, "mini_surf"):  # minimap every 3rd frame (saves ~3 ms/frame)
                w.m.vis.global_.fovy = 45
                self.mini.update_scene(w.d, self.mini_cam, self.mini_opt)
                self.markers(self.mini.scene, with_rays=False)
                mi = self.mini.render()
                self.mini_surf = pg.image.frombuffer(mi.tobytes(), (280, 200), "RGB")
            self.screen.blit(self.mini_surf, (self.W - 290, self.H - 260))
            pg.draw.rect(self.screen, (30, 30, 30), (self.W - 291, self.H - 261, 282, 202), 2)
        self.hud()
        pg.display.flip()

    def text(self, s, pos, col=(240, 240, 240), font=None, bg=(0, 0, 0, 150)):
        pg = self.pg
        f = font or self.font
        lines = s.split("\n")
        wmax = max(f.size(l)[0] for l in lines) + 16
        hh = len(lines) * (f.get_linesize()) + 10
        box = pg.Surface((wmax, hh), pg.SRCALPHA)
        box.fill(bg)
        self.screen.blit(box, pos)
        for i, l in enumerate(lines):
            self.screen.blit(f.render(l, True, col), (pos[0] + 8, pos[1] + 5 + i * f.get_linesize()))

    def hud(self):
        g = self.g
        s = g.score
        self.text(f"FOLD SCORE {s.get('fold_score', 0):4.0f}     best this shirt {g.best.get('fold_score', 0):.0f}\n"
                  f"area {100 * s.get('area_vs_flat', 1):.0f}% of flat (no credit below 25%)  x  lying flat {100 * s.get('flat', 1):.0f}%"
                  f"  x  rectangular {s['rectangularity']:.2f}\n"
                  f"valleys outside {100 * s.get('valleys', 0):.0f}%  x  air inside {s.get('air_mm', 0):.1f} mm "
                  f"({s.get('layers', 1):.1f} layers)   (lower is better)\n"
                  f"box {s['rect_cm'][0]} x {s['rect_cm'][1]} cm", (8, 8), font=self.big)
        now = time.time()
        rows = []
        for a in ARMS:
            hs = [h for h, b in g.devmap.items() if b == a]
            ms = "NO MOUSE (A)" if not hs else ("moving" if now - g.last_input.get(hs[0], 0) < 0.4 else "idle  ")
            sp = f"{g.gain.get(hs[0], 0) * 1e5:.0f}" if hs else "-"
            lim = "  AT REACH LIMIT" if g.reach_limited[a] else ""
            rows.append(f"{a}: mouse {ms} speed {sp} {g.phase[a]:6s} carry {g.carry[a] * 100:.1f} cm jaws {np.degrees(g.yaw_cmd[a]):+4.0f}{lim}")
        rows.append(f"tandem: {TANDEM[g.tandem]}    view: {VIEW_NAMES[self.view]}")
        self.text("\n".join(rows), (self.W - 560, 8))
        if g.last_flowback:
            avg, worst = g.last_flowback
            col = (140, 255, 140) if worst < 5 else (255, 230, 120) if worst < 15 else (255, 140, 120)
            self.text(f"flowback after last drop: avg {avg:.1f} mm, worst 5% {worst:.1f} mm", (8, self.H - 80), col=col)
        self.text(g.msg, (8, self.H - 40), col=(255, 230, 120))
        if self.show_help:
            self.text("HOLD LEFT          grab TOP layer + lift   (release = drop)\n"
                      "HOLD WHEEL-CLICK   grab WHOLE STACK + lift (fold a folded shirt again)\n"
                      "HOLD RIGHT         grab + drag along the table (keep still = pin the cloth down)\n"
                      "WHEEL              turn the jaws (held cloth turns too)    [ ]  carry height\n"
                      "T tandem (off / parallel / mirror)   Z undo   R new shirt\n"
                      "V camera   N minimap   +/- mouse speed   A re-pick mice\n"
                      "F11 full screen / window   Esc pause (frees pointer)   H hide this   Q quit", (8, 110))

    def run(self):
        pg, g = self.pg, self.g
        keymap = {pg.K_t: "T", pg.K_z: "Z", pg.K_r: "R", pg.K_a: "A", pg.K_RETURN: "ENTER", pg.K_KP_ENTER: "ENTER",
                  pg.K_s: "S", pg.K_EQUALS: "+", pg.K_PLUS: "+", pg.K_KP_PLUS: "+", pg.K_MINUS: "-", pg.K_KP_MINUS: "-",
                  pg.K_LEFTBRACKET: "[", pg.K_RIGHTBRACKET: "]"}
        dt = 1 / 60
        t_next = time.perf_counter()
        frame = 0
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_q):
                    return
                if e.type == pg.KEYDOWN:
                    if e.key in (pg.K_F11, pg.K_f):
                        self.set_window(not self.fullscreen)
                        g.msg = "full screen (F11 for a window)" if self.fullscreen else "window (F11 for full screen; drag the edges to resize)"
                    elif e.key == pg.K_ESCAPE:
                        self.set_play(not self.playing)
                    elif e.key == pg.K_h:
                        self.show_help = not self.show_help
                    elif e.key == pg.K_n:
                        self.show_mini = not self.show_mini
                    elif e.key == pg.K_v:
                        self.view = (self.view + 1) % len(VIEWS)
                        g.msg = f"view: {VIEW_NAMES[self.view]}"
                    elif e.key in keymap:
                        g.key(keymap[e.key])
                if e.type == pg.WINDOWFOCUSLOST and self.playing:
                    self.set_play(False)
                if e.type == pg.MOUSEBUTTONDOWN and not self.playing:
                    self.set_play(True)
            mice = self.mice.take() if self.playing else {}
            g.tick(mice, dt, self.mice.is_external_mouse)
            frame += 1
            if frame % 60 == 0 and not self.mice.listening():  # someone took raw mouse input: take it back
                self.mice.register()
            if frame % 2 == 0:
                self.draw()
            t_next += dt
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.2:
                t_next = time.perf_counter()


if __name__ == "__main__":
    App().run()
