"""Polyscope view for the teleop client: the laptop draws the scene itself from the server's state stream (hello
"view": "state"): 12 measured joint angles + the cloth points each frame. Isaac does physics only.

Same session logic as teleop_client.Client (network, two raw mice, keys, live tuning, scoring, local cursor
prediction); only the window, drawing and key reading differ. Camera = the MuJoCo game's first-person view.
"""
import ctypes
import os
import sys
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from teleop_client import Client, FORWARD, HERE

sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
from teleop.so101_vis import RobotVis, load_stl  # noqa: E402

TITLE = "Fold it - Isaac (local view)"
CAMERA = (0.17, 0.58, 80.0, 58.0)     # lookat y (m), distance (m), degrees down, vertical fov (deg)
ARM_COL = {"L": (1.0, 0.55, 0.15), "R": (0.27, 0.86, 0.39)}
MESH_DIRS = [os.path.join(HERE, "assets_lo"), os.path.join(HERE, "..", "oracle", "assets")]
# ImGui key -> the key name the pygame client forwards (FORWARD maps names to the controller's key codes)
KEYS = {"ImGuiKey_T": "t", "ImGuiKey_Z": "z", "ImGuiKey_R": "r", "ImGuiKey_A": "a", "ImGuiKey_Enter": "return",
        "ImGuiKey_S": "s", "ImGuiKey_Equal": "=", "ImGuiKey_Minus": "-", "ImGuiKey_LeftBracket": "[",
        "ImGuiKey_RightBracket": "]", "ImGuiKey_V": "v"}
TUNE_KEYS = {"ImGuiKey_UpArrow": "up", "ImGuiKey_DownArrow": "down", "ImGuiKey_LeftArrow": "left",
             "ImGuiKey_RightArrow": "right", "ImGuiKey_0": "reset", "ImGuiKey_Backspace": "reset", "ImGuiKey_C": "apply"}


def screen_hz():
    """The monitor's refresh rate (Windows), 60 if unknown: the loop draws this often so the local cursors move as
    smoothly as the screen allows (the arms and cloth still change only when the server's state arrives)."""
    try:
        u32, gdi = ctypes.windll.user32, ctypes.windll.gdi32
        dc = u32.GetDC(0)
        hz = gdi.GetDeviceCaps(dc, 116)          # VREFRESH
        u32.ReleaseDC(0, dc)
        return hz if 30 <= hz <= 360 else 60
    except Exception:
        return 60


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class PolyscopeClient(Client):
    # ---------------- window
    def _init_ui(self, fullscreen):
        h = self.hello
        self.FW, self.FH = 1280, 800
        self.O = np.array(h["origin"], float)
        self.quant = float(h["quant"])
        ps.set_program_name(TITLE)
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_front_dir("y_front")
        if os.environ.get("SO101_SHADOWS", "1") != "0":    # soft shadows on the table (~1 ms on the laptop GPU)
            ps.set_ground_plane_mode("shadow_only")
            ps.set_shadow_darkness(0.35)
        else:
            ps.set_ground_plane_mode("none")
        ps.set_SSAA_factor(1)
        ps.set_window_size(self.FW, self.FH)
        ps.set_give_focus_on_show(True)
        ps.set_build_default_gui_panels(False)        # no Polyscope side panels: only our HUD
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_user_callback(self._frame_ui)          # ImGui (HUD text, key presses) only works inside this callback
        self._hud, self._keys_down = {}, []
        try:
            ps.set_navigation_style("none")        # the mice drive the arms, not the camera
        except Exception:
            pass
        # first-person view like the MuJoCo game's, framed for Isaac's robots (LeHome puts the bases at x = +-0.23 m,
        # 8 cm behind the table edge, vs +-0.17 m / 2 cm in the MuJoCo game): CAMERA = lookat (table frame), distance,
        # degrees down, vertical field of view. SO101_CAM="y,dist,el,fov" overrides it.
        y, dist, el_deg, fov = (float(v) for v in os.environ.get("SO101_CAM", ",".join(map(str, CAMERA))).split(","))
        el = np.radians(el_deg)
        target = np.array([0.0, y, 0.0])
        ps.look_at(target + dist * np.array([0.0, -np.cos(el), np.sin(el)]), target)
        ps.set_vertical_fov_degrees(fov)
        # table
        x0, x1, y0, y1 = -0.45, 0.45, -0.25, 0.60
        ps.register_surface_mesh("table", np.array([[x0, y0, -0.002], [x1, y0, -0.002], [x1, y1, -0.002], [x0, y1, -0.002]]),
                                 np.array([[0, 1, 2], [0, 2, 3]]), color=(0.86, 0.85, 0.82), smooth_shade=False)
        # cloth: hello carries its triangles and flat shape; the vertices are replaced every frame
        flat = np.array(h["flat_cloth_cm"], float) / 100
        self.n_cloth = len(flat)
        self.font = self.big = None                  # text sizes are the pygame view's; ImGui uses one font
        self.cloth = ps.register_surface_mesh("cloth", flat, self.tris, color=(0.55, 0.70, 0.90), edge_width=0.0,
                                              smooth_shade=True, back_face_policy="custom")
        self.cloth.set_back_face_color((0.40, 0.52, 0.72))
        # robots: one mesh per URDF visual, posed every frame from the measured joint angles
        self.robot = RobotVis()
        meshes = {}
        for _, _, fn in self.robot.visuals:
            if fn not in meshes:
                path = next(os.path.join(d, fn) for d in MESH_DIRS if os.path.exists(os.path.join(d, fn)))
                meshes[fn] = load_stl(path)
        self.parts = {}
        for side, a in (("left", "L"), ("right", "R")):
            self.parts[a] = [ps.register_surface_mesh(f"{a}_{i}_{fn}", *meshes[fn], color=ARM_COL[a], smooth_shade=False)
                             for i, (_, _, fn) in enumerate(self.robot.visuals)]
        # cursors: where each arm is aiming (drawn locally, no network delay)
        self.cursor_pc = ps.register_point_cloud("cursors", np.zeros((2, 3)), radius=0.006)
        self.cursor_pc.add_color_quantity("arm", np.array([ARM_COL["L"], ARM_COL["R"]]), enabled=True)
        self.playing = True
        self._hud_i = 0
        self.set_play(True)

    def _hwnd(self):
        return ctypes.windll.user32.FindWindowW(None, TITLE)

    def set_window(self, fullscreen):          # Polyscope manages its own window
        pass

    def set_play(self, on):
        """Keep the system cursor inside our window while playing (the two raw mice still drive the arms)."""
        self.playing = on
        u32 = ctypes.windll.user32
        hwnd = self._hwnd()
        if on and hwnd:
            r = _RECT()
            u32.GetWindowRect(hwnd, ctypes.byref(r))
            u32.ClipCursor(ctypes.byref(r))
        else:
            u32.ClipCursor(None)
        self.mice.register()
        self.mice.take()

    # ---------------- drawing
    def text(self, s, pos, col=(240, 240, 240), font=None, bg=(0, 0, 0, 150)):
        """One borderless ImGui box per call (same signature as the pygame view's text())."""
        self._hud_i += 1
        # anchor to the nearest window edge (ImGui's font is wider than the pygame layout these positions came from)
        right, bottom = pos[0] > self.FW / 2, pos[1] > self.FH / 2
        x = self.FW - 6 if right else pos[0]
        y = self.FH - (self.FH - pos[1] - 26) if bottom else pos[1]
        if not right and not bottom and pos[1] >= 50:   # help / tuning panel: start below the top-right status box
            y = pos[1] + 70
        psim.SetNextWindowPos((float(x), float(min(y, self.FH - 6))), 0, (1.0 if right else 0.0, 1.0 if bottom else 0.0))
        psim.SetNextWindowBgAlpha(bg[3] / 255 if len(bg) == 4 else 0.6)
        flags = (psim.ImGuiWindowFlags_NoDecoration | psim.ImGuiWindowFlags_AlwaysAutoResize |
                 psim.ImGuiWindowFlags_NoInputs | psim.ImGuiWindowFlags_NoFocusOnAppearing | psim.ImGuiWindowFlags_NoNav)
        psim.Begin(f"##hud{self._hud_i}", True, flags)
        c = tuple(v / 255 for v in col) + (1.0,)
        try:
            psim.PushFont(psim.GetFont(), 14.0)         # smaller than Polyscope's default UI font
            pushed = True
        except Exception:
            pushed = False
        for line in s.split("\n"):
            psim.TextColored(c, line)
        if pushed:
            psim.PopFont()
        psim.End()

    def draw_scene(self, frame, hud):
        if frame is not None:
            q = np.frombuffer(frame[:48], "<f4").astype(float)
            v = np.frombuffer(frame[48:], "<i2").reshape(-1, 3).astype(np.float32) * self.quant
            if len(v) == self.n_cloth:
                self.cloth.update_vertex_positions(v)
            shift = np.eye(4)
            shift[:3, 3] = -self.O                       # Isaac world -> table frame
            for side, a, q6 in (("left", "L", q[0:6]), ("right", "R", q[6:12])):
                for part, M in zip(self.parts[a], self.robot.poses(q6, side)):
                    part.set_transform(shift @ M)
        cur = self.local_cursors(hud)
        if cur:
            pts = [np.r_[cur[a][0], cur[a][1], (hud["arms"][a].get("aim") or [0, 0, 0.03])[2]] for a in ("L", "R")]
            self.cursor_pc.update_point_positions(np.array(pts))

    def draw(self):
        with self.lock:
            frame, hud, new = self.frame, dict(self.hud), self.new
            self.new = False
        self.draw_scene(frame if new else None, hud)
        self._hud = hud
        ps.frame_tick()                                # runs _frame_ui: HUD + key reading

    def _frame_ui(self):
        """Inside Polyscope's frame: read this frame's key presses and draw the HUD (the shared draw_hud)."""
        self._hud_i = 0
        self._keys_down += [k for k in list(KEYS) + list(TUNE_KEYS) + ["ImGuiKey_Q", "ImGuiKey_Tab", "ImGuiKey_Escape",
                                                                        "ImGuiKey_H"] if self._pressed(k)]
        self.draw_hud(self._hud)

    # ---------------- input + loop
    def _pressed(self, name):
        return psim.IsKeyPressed(getattr(psim, name), False)

    def run(self):
        hz = screen_hz()
        print(f"drawing at {hz} Hz (screen refresh)")
        t_next = time.perf_counter()
        while not ps.window_requests_close():
            down, self._keys_down = self._keys_down, []    # keys pressed during the last frame
            keys = []
            if "ImGuiKey_Q" in down:
                break
            for k in down:
                if k == "ImGuiKey_Tab" and self.tspec:
                    self.tune_open = not self.tune_open
                elif k == "ImGuiKey_Escape":
                    self.set_play(not self.playing)
                elif k == "ImGuiKey_H":
                    self.show_help = not self.show_help
                elif self.tune_open and k in TUNE_KEYS:
                    self.tune_action(TUNE_KEYS[k])
                elif k in KEYS:
                    keys.append(FORWARD[KEYS[k]])
            mice = self.mice.take() if self.playing else {}
            if not self.ended:
                try:
                    self.send_input(mice, keys)
                except OSError:
                    self.ended = "connection lost"
            self.draw()
            t_next += 1 / hz
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.2:
                t_next = time.perf_counter()
        self.close()

    def close(self):
        from teleop import protocol
        try:
            protocol.send(self.sock, dict(t="bye"))
        except OSError:
            pass
        ctypes.windll.user32.ClipCursor(None)
        print("demo log:", self.log_path)
