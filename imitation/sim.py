"""Which simulator the pipeline runs on: IMITATION_SIM=mujoco (the default) or isaac.

Read from the environment, not a flag, so every entry point (collect, train, evaluate, dagger) and the rollout
workers they spawn agree without threading an argument through each of them. The two sims run the same half fold
but not the same distances: Isaac's cloth sits where its 5-joint arms only just reach both ends of the fold (see
isaac/README.md), so its reset jitter is +-1 cm instead of +-2.5 cm and id_hard shifts the cloth 1-2 cm instead of
2.5-4 cm; and its friction grasp makes the expert slower (about 230-300 steps), so episodes run 400 steps and
recovery knocks land anywhere up to mid-carry.
"""

from __future__ import annotations

import os

SIM = os.environ.get("IMITATION_SIM", "mujoco")

PROFILES = {
    "mujoco": {"max_steps": 250, "hard_jitter": (0.025, 0.04), "eval_knock_t": (15, 60), "collect_knock_t": (10, 70)},
    "isaac": {"max_steps": 400, "hard_jitter": (0.01, 0.02), "eval_knock_t": (15, 200), "collect_knock_t": (10, 200)},
}
if SIM not in PROFILES:
    raise ValueError(f"IMITATION_SIM={SIM!r}: expected one of {sorted(PROFILES)}")
PROFILE = dict(PROFILES[SIM])
if os.environ.get("IMITATION_MAX_STEPS"):       # e.g. cut failing expert demos short while collecting
    PROFILE["max_steps"] = int(os.environ["IMITATION_MAX_STEPS"])
