"""Camera rig for the sensor-only student (roadmap M4.1).

Renders the robot's real cameras -- the fixed `main` camera and one camera per wrist --
as uint8 [3, H, W] arrays. Visual domain randomization (cloth / table / floor colour,
light intensity, a few mm of main-camera jitter) is drawn per episode from the episode
seed. It only touches render-side model fields, so physics -- and therefore replaying a
frozen dataset's actions -- is unaffected.
"""

from __future__ import annotations

import mujoco
import numpy as np

CAMERAS = {"main": 96, "left_wrist_cam": 64, "right_wrist_cam": 64}   # name -> square size


class CameraRig:
    def __init__(self, env, cameras=CAMERAS, randomize=True):
        self.base = env.unwrapped
        m = self.base.model
        self.cameras, self.randomize = dict(cameras), randomize
        self.renderers = {}
        for size in set(self.cameras.values()):
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, size)
            m.vis.global_.offheight = max(m.vis.global_.offheight, size)
            self.renderers[size] = mujoco.Renderer(m, size, size)
        self.cam_ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in self.cameras}
        self._geom = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("table", "floor")}
        self._nominal = {"flex_rgba": m.flex_rgba.copy(), "geom_rgba": m.geom_rgba.copy(),
                         "light_diffuse": m.light_diffuse.copy(), "cam_pos": m.cam_pos.copy()}

    def reset(self, seed):
        """Restore the nominal look, then (if randomizing) redraw it for this episode."""
        m = self.base.model
        m.flex_rgba[:] = self._nominal["flex_rgba"]
        m.geom_rgba[:] = self._nominal["geom_rgba"]
        m.light_diffuse[:] = self._nominal["light_diffuse"]
        m.cam_pos[:] = self._nominal["cam_pos"]
        if not self.randomize:
            return
        rng = np.random.default_rng([int(seed), 4243])
        m.flex_rgba[:, :3] = np.clip(m.flex_rgba[:, :3] + rng.uniform(-0.15, 0.15, 3), 0, 1)
        for gid in self._geom.values():
            m.geom_rgba[gid, :3] = np.clip(m.geom_rgba[gid, :3] + rng.uniform(-0.1, 0.1, 3), 0, 1)
        m.light_diffuse[:] *= rng.uniform(0.7, 1.2)
        main = self.cam_ids.get("main", -1)
        if main >= 0:
            m.cam_pos[main] += rng.uniform(-0.01, 0.01, 3)

    def render(self) -> dict[str, np.ndarray]:
        out = {}
        for name, size in self.cameras.items():
            r = self.renderers[size]
            r.update_scene(self.base.data, camera=self.cam_ids[name])
            out[name] = np.ascontiguousarray(r.render().transpose(2, 0, 1))   # [3, H, W] uint8
        return out
