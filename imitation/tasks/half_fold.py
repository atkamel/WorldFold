"""Half fold: stage 0 of the quarter fold, ending in success when it settles.

Both arms carry the north corners onto the south corners (a fold about y = 0),
release, and the corners must stay placed for SETTLE_STEPS. Everything else --
observation (139-D state), action (12-D joint deltas + grippers), reward shaping,
drag/instability termination -- is inherited from QuarterFoldEnv unchanged, so
QuarterFoldExpert drives it as-is.
"""

from __future__ import annotations

from cloth_fold_rl.quarter_fold_env import STAGES, QuarterFoldEnv
from imitation.spec import ACTION_DIM, OBS_DIM  # noqa: F401 (re-exported)

HALF_FOLD_MAX_STEPS = 250


class HalfFoldEnv(QuarterFoldEnv):
    stages = STAGES[:1]

    def __init__(self, max_episode_steps=HALF_FOLD_MAX_STEPS, seed=None, domain_randomization=True, **kwargs):
        super().__init__(max_episode_steps=max_episode_steps, seed=seed, **kwargs)
        self.unwrapped.domain_randomization = domain_randomization

    def reset(self, seed=None, options=None):
        # the base env samples a random task one-hot into the observation; for
        # a single-task policy that is pure noise, so pin it
        opts = dict(options or {})
        opts.setdefault("task", 0)
        return super().reset(seed=seed, options=opts)


def make_env(**kwargs):
    return HalfFoldEnv(**kwargs)
