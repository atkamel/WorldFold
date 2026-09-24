"""Disjoint seed ranges, so training data never shares a cloth start with evaluation.

Design doc 12.1 evaluation sets:
  id_easy    same start distribution as the demos
  id_hard    larger cloth offset than the demos ever saw (+-4 cm vs +-2.5 cm)
  recovery   id_easy starts, but the student is knocked off course mid-episode
"""

from __future__ import annotations

import numpy as np

from imitation.rollout import Perturbation

TRAIN_SEED_BASE = 0            # expert demos: 0 ..
DAGGER_SEED_BASE = 50_000      # DAgger rollouts: 50_000 + 1000 * round ..
DISTILL_SEED_BASE = 70_000     # sensor-only distillation rollouts (Phase 4): 70_000 + 1000 * round ..
HARVEST_SEED_BASE = 400_000    # student rollout harvest for offline RL (Phase 5): 400_000 ..
EVAL_SEED_BASE = {"id_easy": 100_000, "id_hard": 200_000, "recovery": 300_000}
HARD_JITTER = 0.04


def eval_set(name, n):
    """(seeds, reset_options fn or None, perturb_fn or None) for a named evaluation set."""
    seeds = list(range(EVAL_SEED_BASE[name], EVAL_SEED_BASE[name] + n))
    if name == "id_easy":
        return seeds, None, None
    if name == "id_hard":
        def hard_pose(seed):
            rng = np.random.default_rng([seed, 17])
            # a ring outside the training jitter: at least 2.5 cm off in one axis
            pose = rng.uniform(-HARD_JITTER, HARD_JITTER, size=2)
            axis = rng.integers(2)
            pose[axis] = np.sign(pose[axis] or 1.0) * rng.uniform(0.025, HARD_JITTER)
            return {"cloth_pose": pose}
        return seeds, hard_pose, None
    if name == "recovery":
        def knock(seed, rng):
            return Perturbation(t=int(rng.integers(15, 60)), k=int(rng.integers(8, 16)))
        return seeds, None, knock
    raise KeyError(name)
