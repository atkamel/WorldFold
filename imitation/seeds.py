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
SHIFT_SEED_BASE = 500_000      # shifted-pose training rollouts (M5b.2): 500_000 + 1000 * round ..
TUNE_SEED_BASE = 600_000       # Isaac grasp-tuning blocks: 600_000 ..
# the perturbation suite (Phase F3b, docs/imitation.md section 4) shares the old recovery block, 10k seeds each
EVAL_SEED_BASE = {"id_easy": 100_000, "id_hard": 200_000, "recovery": 300_000, "knock_arm": 310_000,
                  "drop": 320_000, "joint_noise": 330_000, "overshoot": 340_000}
PERTURB_SETS = ("knock_arm", "drop", "joint_noise", "overshoot")
# tuning copies of the suite (and of the legacy knock) inside the tune block, so expert tuning never sees eval seeds
TUNE_SET_BASE = {"tune_recovery": 610_000, "tune_knock_arm": 620_000, "tune_drop": 630_000,
                 "tune_joint_noise": 640_000, "tune_overshoot": 650_000, "tune_id_easy": 660_000}
JOINT_NOISE_SIGMA = 0.15         # of the max joint delta (actions are normalised to [-1, 1])
OVERSHOOT_GAIN = (1.2, 1.4)
DROP_K = 3
HARD_JITTER = 0.04

# [start, end) of every seed range; tests/imitation/test_seeds.py checks they never overlap
SEED_RANGES = {"train": (TRAIN_SEED_BASE, 10_000), "dagger": (DAGGER_SEED_BASE, 70_000),
               "distill": (DISTILL_SEED_BASE, 100_000), "id_easy": (100_000, 200_000),
               "id_hard": (200_000, 300_000), "recovery": (300_000, 310_000),
               "knock_arm": (310_000, 320_000), "drop": (320_000, 330_000), "joint_noise": (330_000, 340_000),
               "overshoot": (340_000, 350_000),
               "harvest": (HARVEST_SEED_BASE, 500_000), "shift": (SHIFT_SEED_BASE, 600_000),
               "tune": (TUNE_SEED_BASE, 700_000)}


def shifted_pose(seed):
    """Cloth start outside the demos' jitter: at least 2.5 cm off in one axis, up to 4 cm.
    The id_hard eval set and shifted-pose training (M5b.2) share this distribution on
    disjoint seeds."""
    rng = np.random.default_rng([seed, 17])
    pose = rng.uniform(-HARD_JITTER, HARD_JITTER, size=2)
    axis = rng.integers(2)
    pose[axis] = np.sign(pose[axis] or 1.0) * rng.uniform(0.025, HARD_JITTER)
    return {"cloth_pose": pose}


ISAAC_HARD_RMAX = 0.01   # m; largest cloth offset radius all of whose id_hard draws are reachable (isaac/reach_check.py)
ISAAC_HARD_MIN = 0.01    # m; the forced axis is at least this far off
ISAAC_RECOVERY_T = (35, 140)   # provisional: MuJoCo 15-60 scaled to the ~230-step Isaac fold (revisit at I2.1)


def shifted_pose_isaac(seed):
    """Isaac id_hard start: reset jitter is 1 cm, so one axis is forced to 1 cm .. ISAAC_HARD_RMAX and the other is
    within +-ISAAC_HARD_RMAX (the fold only spans the arms' reach for a couple of cm)."""
    r = ISAAC_HARD_RMAX
    rng = np.random.default_rng([seed, 17])
    pose = rng.uniform(-r, r, size=2)
    axis = rng.integers(2)
    pose[axis] = np.sign(pose[axis] or 1.0) * rng.uniform(ISAAC_HARD_MIN, r)
    return {"cloth_pose": pose}


# Backends that run LeHome's task distribution (1 cm jitter, the Isaac id_hard ring, the longer fold's knock window):
# the friction pinch reaches only ~1.5 cm of cloth offset at LeHome's cloth position (F2 reach sweep: 39/100 starts
# reachable at MuJoCo's +-2.5 cm jitter, 0/200 of MuJoCo's id_hard), so isaac_friction cannot use
# MuJoCo's sets. isaac_weld keeps MuJoCo's.
LEHOME_TASK_BACKENDS = ("isaac", "isaac_friction")


def perturbation_fn(kind, backend="mujoco"):
    """perturb_fn(seed, rng) -> Perturbation for one kind of the suite; draws from its own seeded rng, so a seed's
    perturbation is the same in eval, demos and collection."""
    isaac = backend in LEHOME_TASK_BACKENDS
    lo, hi = ISAAC_RECOVERY_T if isaac else (15, 60)

    def fn(seed, rng):
        r = np.random.default_rng([seed, 31337])
        t = int(r.integers(lo, hi))
        if kind == "knock_arm":
            return Perturbation(t=t, k=int(r.integers(8, 16)), kind=kind)
        if kind == "drop":
            return Perturbation(t=t, k=DROP_K, kind=kind)
        if kind == "joint_noise":
            return Perturbation(t=0, k=0, kind=kind, sigma=JOINT_NOISE_SIGMA)
        if kind == "overshoot":
            return Perturbation(t=0, k=0, kind=kind, gain=float(r.uniform(*OVERSHOOT_GAIN)))
        raise ValueError(kind)
    return fn


def eval_set(name, n, backend="mujoco"):
    """(seeds, reset_options fn or None, perturb_fn or None) for a named evaluation set."""
    if name in TUNE_SET_BASE:               # same distribution as the eval set it copies, tune-block seeds
        base_name = name[len("tune_"):]
        seeds = list(range(TUNE_SET_BASE[name], TUNE_SET_BASE[name] + n))
        _, opts, fn = eval_set(base_name, n, backend)
        return seeds, opts, fn
    seeds = list(range(EVAL_SEED_BASE[name], EVAL_SEED_BASE[name] + n))
    isaac = backend in LEHOME_TASK_BACKENDS   # isaac_weld uses the MuJoCo sets
    if name == "id_easy":
        return seeds, None, None
    if name == "id_hard":
        return seeds, shifted_pose_isaac if isaac else shifted_pose, None
    if name == "recovery":
        lo, hi = ISAAC_RECOVERY_T if isaac else (15, 60)

        def knock(seed, rng):
            return Perturbation(t=int(rng.integers(lo, hi)), k=int(rng.integers(8, 16)))
        return seeds, None, knock
    if name in PERTURB_SETS:
        return seeds, None, perturbation_fn(name, backend)
    raise KeyError(name)
