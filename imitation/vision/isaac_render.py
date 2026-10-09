"""Camera rig on Isaac Sim (Phase I, I1.3): the same interface as imitation.vision.render.CameraRig.

The cameras themselves live in the Isaac scene (isaac/lab_scene.py rig_camera_cfg: `main` at the shared pose, one
camera per wrist at the so101-nexus MJCF wrist_cam mount) and are rendered once per control step; this class only
reads them. No visual domain randomization in this pass (roadmap I-DR), so reset(seed) changes nothing.
"""

from __future__ import annotations

import time


class IsaacCameraRig:
    def __init__(self, env, cameras):
        self.base = env.unwrapped
        self.cameras = dict(cameras)
        missing = set(self.cameras) - set(self.base.rig)
        if missing or any(self.base.rig[c] != n for c, n in self.cameras.items()):
            raise ValueError(f"the Isaac env was built with rig {self.base.rig}, asked for {self.cameras}")
        self.last_render_s = 0.0

    def reset(self, seed):
        """Visual randomization is deferred (I-DR); kept for CameraRig's interface."""

    def render(self):
        t0 = time.perf_counter()
        frames = self.base.render_rig()
        self.last_render_s = time.perf_counter() - t0
        return {c: frames[c] for c in self.cameras}
