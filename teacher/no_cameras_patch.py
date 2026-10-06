"""Let LeHome's garment env run without its three cameras (live teleop only: the laptop draws the scene itself).

Runtime patch of garment_bi_v2.py, gated by LEHOME_NO_CAMERAS=1 (OFF = original behaviour). With the gate on, the
env creates no TiledCamera sensors and its observations hold only the action + joint state, so Isaac can be launched
without --enable_cameras: IsaacLab then loads its headless no-rendering experience (isaaclab.python.headless.kit).
Recorded demos get their camera images later, by replaying them with the normal settings.

    python no_cameras_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [no-cameras-patch]"
p = ROOT / "lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py"
s = p.read_text()
if MARK in s:
    print("already patched: no-cameras")
    sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, f"no-cameras patch: expected 1 match for {old[:60]!r}, got {s.count(old)}"
    s = s.replace(old, new)


rep('''        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
''', f'''        if os.environ.get("LEHOME_NO_CAMERAS") == "1":  {MARK}
            self.top_camera = self.left_camera = self.right_camera = None
        else:
            self.top_camera = TiledCamera(self.cfg.top_camera)
            self.left_camera = TiledCamera(self.cfg.left_wrist)
            self.right_camera = TiledCamera(self.cfg.right_wrist)
''')
rep('''        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera
''', '''        if self.top_camera is not None:
            self.scene.sensors["top_camera"] = self.top_camera
            self.scene.sensors["left_camera"] = self.left_camera
            self.scene.sensors["right_camera"] = self.right_camera
''')
# observations: the image block (rgb + optional depth) runs only when the cameras exist
start = s.index('        top_camera_rgb = self.top_camera.data.output["rgb"]')
end = s.index("        # Per-condition success check status")
block = s[start:end]
indented = "".join(("    " + ln) if ln.strip() else ln for ln in block.splitlines(keepends=True))
s = s[:start] + '''        if self.top_camera is None:  # no cameras: action + joint state only
            observations = {
                "action": action.cpu().detach().numpy(),
                "observation.state": joint_pos.cpu().detach().numpy(),
            }
        else:
''' + indented + s[end:]
p.write_text(s)
print("patched: no-cameras (gate LEHOME_NO_CAMERAS=1)")
