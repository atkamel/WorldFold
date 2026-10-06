"""The half fold on Isaac Sim: imitation.tasks.HalfFoldEnv (same 139-D observation, action, reward and success) over
isaac.isaac_env.IsaacClothFoldEnv instead of MuJoCo's ClothFoldEnv. Import inside an Isaac Sim process.

Only state observations: the camera student's rig (imitation.vision.render) renders with MuJoCo. `video_size` turns on
the base env's `main` camera, for demo videos (base._render_image); the policy still sees only the state.
"""

from __future__ import annotations

from imitation.sim import PROFILE, PROFILES, SIM
from imitation.tasks.half_fold import HalfFoldEnv

ISAAC_MAX_STEPS = PROFILE["max_steps"] if SIM == "isaac" else PROFILES["isaac"]["max_steps"]


def make_isaac_env(max_episode_steps=ISAAC_MAX_STEPS, seed=None, obs_mode="state", cameras=None, video_size=None,
                   **kwargs):
    if obs_mode != "state":
        raise NotImplementedError("the Isaac half fold has no camera rig yet; use obs_mode='state'")
    kwargs.pop("domain_randomization", None)      # the Isaac env has none to switch
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import CLOTH_JITTER, IsaacClothFoldEnv
    base = IsaacClothFoldEnv(observation_mode="hybrid" if video_size else "state",
                             image_size=(video_size, video_size) if video_size else (84, 84),
                             max_episode_steps=max_episode_steps, grasp_corners=GRASP_CORNERS,
                             grasp_radius=GRASP_RADIUS)
    return HalfFoldEnv(max_episode_steps=max_episode_steps, seed=seed, domain_randomization=False,
                       base_env=base, cloth_jitter=CLOTH_JITTER, **kwargs)
