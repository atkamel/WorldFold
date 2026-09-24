"""Student rollout harvest for offline RL (roadmap M5.1).

    python -m imitation.rl.harvest --ckpt runs/dagger_v2/round_2/final.pt --episodes 1000 --version harvest_v1

Offline RL learns from reward *variance*: expert data is ~all success, so it says nothing
about which of the student's own choices go wrong. This rolls the student out on its own
seed range (HARVEST_SEED_BASE), 30% knocked off course, with optional Gaussian action
noise for coverage, and freezes every episode -- successes and failures -- as a
`student` version. The summary stratifies outcomes by mechanistic failure code.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from imitation.data.collect import DEFAULT_ROOT, recovery_perturbation
from imitation.data.schema import DatasetWriter
from imitation.evaluate import failure_code
from imitation.policies.common import load_policy
from imitation.rollout import EnvPool, PolicyController, rollout
from imitation.seeds import HARVEST_SEED_BASE
from imitation.spec import ACTION_DIM, OBS_DIM


class _Noisy:
    """Adds per-chunk Gaussian noise to a policy's predictions (seeded, clipped)."""

    def __init__(self, policy, sigma, seed=0):
        self.policy, self.sigma, self.rng = policy, sigma, np.random.default_rng(seed)
        self.obs_horizon, self.chunk, self.needs_images = policy.obs_horizon, policy.chunk, policy.needs_images

    def predict(self, obs, *a):
        out = self.policy.predict(obs, *a)
        return np.clip(out + self.rng.normal(0, self.sigma, out.shape), -1, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--version", default="harvest_v1")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.ckpt)
    actor = _Noisy(policy, args.noise) if args.noise > 0 else policy
    writer = DatasetWriter(args.root, args.version, resume=args.resume,
                           config={"checkpoint": args.ckpt, "noise": args.noise, "episodes": args.episodes,
                                   "recovery_fraction": args.recovery_fraction, "seed_base": HARVEST_SEED_BASE})
    seeds = [s for s in range(HARVEST_SEED_BASE, HARVEST_SEED_BASE + args.episodes) if s not in writer.done_seeds]
    codes = Counter()
    t0 = time.time()

    def save(ep):
        writer.add(ep, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        codes[failure_code(ep) or "success"] += 1
        n = sum(codes.values())
        if n % 50 == 0:
            print(f"  {n}/{len(seeds)} {dict(codes)}  {time.time() - t0:.0f}s", flush=True)

    with EnvPool(args.workers) as pool:
        rollout(pool, seeds, PolicyController(actor, replan_every=args.replan_every),
                perturb_fn=recovery_perturbation(args.recovery_fraction), meta_extra={"policy": args.ckpt},
                on_done=save)
    m = writer.freeze()
    print(f"froze {args.version}: {m['n_episodes']} episodes, success {m['n_success']}, by code {dict(codes)}, "
          f"hash {m['content_hash'][:12]}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
