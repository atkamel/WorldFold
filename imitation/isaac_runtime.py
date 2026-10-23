"""Isaac Sim backend settings and the base-env factory (Phase I). Importable without Isaac Sim.

The Isaac half fold takes ~230 control steps (the arm moves at most 0.05 rad per step and lags), so
episodes run to 400 instead of MuJoCo's 250; the cloth start jitter is +-1 cm because the fold already
spans the arms' straight-down reach (isaac/README.md). One Isaac env per process (IsaacLab singleton).
"""

from __future__ import annotations

import os
import sys

# concurrent Isaac processes per job on the 16 GB laptop GPU (measured in I0.3); a job can lower it with
# WORLDFOLD_N_ISAAC.
N_ISAAC = int(os.environ.get("WORLDFOLD_N_ISAAC", "2"))
ISAAC_MAX_STEPS = 400
ISAAC_CLOTH_JITTER = 0.01   # = isaac.isaac_env.CLOTH_JITTER (kept literal so this module stays Isaac-free)
# backend name -> IsaacClothFoldEnv profile. "isaac" = LeHome friction grasp (400-step cap, 1 cm jitter);
# "isaac_friction" = MuJoCo's setup with LeHome's friction grasp (Phase F): MuJoCo eval sets, jitter and DR, with a
# FRICTION_MAX_STEPS cap, since a physical pinch (close, settle, lift) is slower than MuJoCo's weld.
ISAAC_PROFILES = {"isaac": "lehome", "isaac_friction": "friction"}
FRICTION_MAX_STEPS = 400
KIT_TICK_S = 30.0           # idle workers tick Kit this often; its hang detector allows 120 s


def make_isaac_base(max_episode_steps: int = ISAAC_MAX_STEPS, cameras: dict | None = None,
                    profile: str = "lehome"):
    """IsaacClothFoldEnv in state mode, built the way the half fold's wrapper expects (its grasp corners); `cameras`
    ({name: size}) adds the imitation camera rig to the scene (I1.3). `profile` is an ISAAC_PROFILES value."""
    # AppLauncher/Kit read sys.argv; in a spawned rollout worker that is the parent CLI's argv
    sys.argv = sys.argv[:1]
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import IsaacClothFoldEnv
    return IsaacClothFoldEnv(observation_mode="state", max_episode_steps=max_episode_steps,
                             grasp_corners=GRASP_CORNERS, grasp_radius=GRASP_RADIUS, cameras=cameras,
                             profile=profile)


def make_isaac_batch(n: int, max_episode_steps: int = ISAAC_MAX_STEPS, cameras: dict | None = None,
                     profile: str = "friction"):
    """IsaacClothFoldBatch of n sub-envs, each built as make_isaac_base builds its env (milestone V)."""
    sys.argv = sys.argv[:1]
    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import IsaacClothFoldBatch
    return IsaacClothFoldBatch(n, observation_mode="state", max_episode_steps=max_episode_steps,
                               grasp_corners=GRASP_CORNERS, grasp_radius=GRASP_RADIUS, cameras=cameras,
                               profile=profile)
