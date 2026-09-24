"""Training samples from episodes: (obs history) -> (next K actions), plus a loss mask.

Where the targets come from:
  teacher-executed steps   actions[t:t+K]; positions the teacher did not choose
                           (a perturbation started) are masked out, and positions
                           past the episode end are padded with "hold still, keep
                           the gripper command" -- the policy should learn to stop.
  DAgger-labelled steps    the teacher's chunk from that exact student-visited state.
Student-executed steps without a label never become targets.

Splits are by episode (seed), never by frame (design doc 7.2), and the
normalizer is fit on training episodes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import numpy as np

from imitation.data.schema import ACTOR_TEACHER, Episode

GRIPPER_DIMS = (5, 11)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, episodes: list[Episode], min_std=1e-2):
        obs = np.concatenate([e.obs for e in episodes])
        return cls(mean=obs.mean(0).astype(np.float32), std=np.maximum(obs.std(0), min_std).astype(np.float32))


def clamp_fraction(X, mean, std, clip=10.0):
    """Fraction of normalized observation entries at the normalizer's clip. A warm-
    started student keeps its first normalizer, so DAgger states far outside the expert
    data show up here before they silently saturate (M3.1)."""
    z = (X - mean) / std
    return float((z.abs() >= clip).float().mean())


def bc_episodes(episodes, allow_failures=False):
    """Episodes fit to imitate, and how many were dropped (imitation.md 5.3).

    A failed expert episode is not a demonstration: its tail is the expert stuck or
    padded in place. DAgger episodes stay whatever the student's outcome -- their
    teacher labels are correct from the states the student reached.
    """
    if allow_failures:
        return list(episodes), 0
    kept = [e for e in episodes if e.meta["success"] or e.meta["source"] != "expert"]
    return kept, len(episodes) - len(kept)


def is_val_seed(env_seed, val_fraction=0.1, seed=0) -> bool:
    """Per-seed, content-free assignment: a seed's side never depends on which other
    episodes exist, so the val set does not drift as DAgger rounds add data (M3.1)."""
    h = hashlib.sha256(f"{seed}:{int(env_seed)}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64 < val_fraction


def split_episodes(episodes, val_fraction=0.1, seed=0):
    """Deterministic split on the episode's seed (so DAgger episodes of a seed land with it)."""
    val = [e for e in episodes if is_val_seed(e.meta["seed"], val_fraction, seed)]
    if not val and episodes:                  # tiny debug sets: keep at least one val seed
        first = min(e.meta["seed"] for e in episodes)
        val = [e for e in episodes if e.meta["seed"] == first]
    val_ids = {id(e) for e in val}
    return [e for e in episodes if id(e) not in val_ids], val


def _pad_action(last):
    pad = np.zeros_like(last)
    pad[list(GRIPPER_DIMS)] = last[list(GRIPPER_DIMS)]
    return pad


def _history(obs, t, horizon):
    idx = np.clip(np.arange(t - horizon + 1, t + 1), 0, None)
    return obs[idx]


def build_samples(episodes: list[Episode], obs_horizon: int, chunk: int):
    """Returns X [N, H, D], Y [N, K, A], M [N, K] (1 = supervised), W [N] source tag
    (0 expert chunk, 1 DAgger label)."""
    X, Y, M, W = [], [], [], []
    for ep in episodes:
        T = ep.steps
        # Dense chunks come from expert episodes only. In a DAgger episode the executed
        # teacher steps are pieces of different replans; its teacher chunks are exactly
        # the stored labels, used below (M3.1).
        teacher = (ep.actor == ACTOR_TEACHER) & (ep.meta["source"] == "expert")
        padded = np.concatenate([ep.actions, np.repeat(_pad_action(ep.actions[-1])[None], chunk, 0)])
        # "hold still" after the end is only the right target after a true terminal
        valid = np.concatenate([teacher, np.full(chunk, bool(ep.terminated[-1]))])
        for t in np.flatnonzero(teacher):
            m = valid[t:t + chunk].copy()
            # once someone else took over, the rest of this chunk is not the teacher's plan
            if not m.all():
                m[np.argmin(m):] = False
            X.append(_history(ep.obs, t, obs_horizon))
            Y.append(padded[t:t + chunk])
            M.append(m)
            W.append(0)
        for t, label in zip(ep.label_steps, ep.labels):
            X.append(_history(ep.obs, t, obs_horizon))
            Y.append(label[:chunk])
            M.append(np.ones(chunk, dtype=bool))
            W.append(1)
    if not X:
        raise ValueError("no supervised samples in these episodes")
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32),
            np.stack(M).astype(np.float32), np.asarray(W, dtype=np.int8))
