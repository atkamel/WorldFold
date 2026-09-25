"""Student rollout harvest for offline RL (roadmap M5.1, redone for M5b.4).

    python -m imitation.rl.harvest --ckpts runs/bc_v1_s0/final.pt runs/dagger_v2/round_3/final.pt \
        --noise 0.1 0.3 --episodes 2000 --min-per-code 60 --version harvest_v2

Offline RL learns from reward *variance* and from *action contrast*: different actions
from similar states with different outcomes. `harvest_v1` (one policy, sigma 0.1) had the
first but not the second -- the critic ranked states, but advantages were ~0 for every
logged action (results.md, M5.3). So this harvests from several policies x several noise
levels. Each seed is assigned a (policy, sigma) cell, recorded in the episode's meta.
It runs on its own seed range (HARVEST_SEED_BASE), 30% knocked off course, and keeps
successes *and* failures.

Stratified by failure code (M5.1): after `--episodes`, it keeps rolling in batches until
every failure code seen has at least `--min-per-code` episodes, or `--max-episodes` is hit.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from collections import Counter

import numpy as np

from imitation.data.collect import DEFAULT_ROOT, recovery_perturbation
from imitation.data.schema import ACTOR_STUDENT, DatasetWriter, Episode
from imitation.evaluate import failure_code
from imitation.policies.common import load_policy
from imitation.rollout import Controller, EnvPool, Plan, padded_predict, rollout
from imitation.seeds import HARVEST_SEED_BASE
from imitation.spec import ACTION_DIM, OBS_DIM

FAIL_CODES = ("G1", "F1", "S1", "M1")


def cell_of(seed, cells):
    return cells[(seed - HARVEST_SEED_BASE) % len(cells)]


class CellController(Controller):
    """Each env's chunk comes from its seed's (policy, sigma) cell: one batched call per
    policy, then that cell's Gaussian noise from the env's seeded rng."""
    source = "student"

    def __init__(self, policies, cells, replan_every=8):
        self.policies, self.cells, self.replan_every = policies, cells, replan_every
        self.horizon = max(p.obs_horizon for p in policies)

    def plan(self, slots, obs_hist, labels, rngs, images=None):
        cell = [cell_of(self.slot_seeds[s], self.cells) for s in slots]
        chunks = [None] * len(slots)
        for pi, policy in enumerate(self.policies):
            rows = [j for j, c in enumerate(cell) if c[0] == pi]
            if rows:
                out = padded_predict(policy, obs_hist[rows][:, -policy.obs_horizon:])
                for j, c in zip(rows, out):
                    chunks[j] = c
        plans = []
        for j, s in enumerate(slots):
            noisy = np.clip(chunks[j] + rngs[s].normal(0, cell[j][1], chunks[j].shape), -1, 1)
            plans.append(Plan(actions=noisy[:self.replan_every].astype(np.float32), actor=ACTOR_STUDENT))
        return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--noise", nargs="+", type=float, default=[0.1])
    ap.add_argument("--episodes", type=int, default=1000, help="size before the stratification top-up")
    ap.add_argument("--min-per-code", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None, help="hard cap (default 1.5x --episodes)")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--version", default="harvest_v2")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--recovery-fraction", type=float, default=0.3)
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cap = args.max_episodes or int(1.5 * args.episodes)

    policies = [load_policy(c) for c in args.ckpts]
    cells = list(itertools.product(range(len(policies)), args.noise))
    writer = DatasetWriter(args.root, args.version, resume=args.resume,
                           config={"checkpoints": args.ckpts, "noise": args.noise, "episodes": args.episodes,
                                   "min_per_code": args.min_per_code, "recovery_fraction": args.recovery_fraction,
                                   "seed_base": HARVEST_SEED_BASE, "cells": cells})
    codes, by_cell = Counter(), Counter()
    for e in writer.new_entries:          # --resume: count what the killed run already kept
        ep = Episode.load(Path(args.root) / e["file"])
        pi, sigma = cell_of(e["seed"], cells)
        codes[failure_code(ep) or "success"] += 1
        by_cell[f"p{pi} s{sigma}", failure_code(ep) or "success"] += 1
    t0 = time.time()

    def save(ep):
        pi, sigma = cell_of(ep.meta["seed"], cells)
        ep.meta.update(cell=[pi, sigma], policy=args.ckpts[pi])
        writer.add(ep, obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        code = failure_code(ep) or "success"
        codes[code] += 1
        by_cell[f"p{pi} s{sigma}", code] += 1
        if sum(codes.values()) % 50 == 0:
            print(f"  {sum(codes.values())} {dict(codes)}  {time.time() - t0:.0f}s", flush=True)

    def enough():
        n = len(writer.new_entries)
        if n >= cap:
            return True
        if n < args.episodes:
            return False
        return all(codes[c] >= args.min_per_code for c in FAIL_CODES if codes[c] > 0)

    controller = CellController(policies, cells, args.replan_every)
    next_seed, done = HARVEST_SEED_BASE, writer.done_seeds
    with EnvPool(args.workers) as pool:
        while not enough():
            seeds = []
            while len(seeds) < args.batch:
                if next_seed not in done:
                    seeds.append(next_seed)
                next_seed += 1
            rollout(pool, seeds, controller, perturb_fn=recovery_perturbation(args.recovery_fraction), on_done=save)
    m = writer.freeze()
    cell_table = {f"{c} {o}": n for (c, o), n in sorted(by_cell.items())}
    print(f"froze {args.version}: {m['n_episodes']} episodes, success {m['n_success']}, by code {dict(codes)}, "
          f"hash {m['content_hash'][:12]}, {time.time() - t0:.0f}s\nby cell {cell_table}")


if __name__ == "__main__":
    main()
