"""Offline-RL transitions over chunk macro-actions (roadmap M5.2).

The imitation policy acts in chunks: every `replan_every` (8) steps it emits a chunk
and executes its first 8 actions open-loop. The RL stage treats exactly that as its
action space, so one transition is one replan interval:

    s   = observation history at t              [H, D]
    a   = the 8 actions executed t .. t+7         [8, A]   (flattened for the critic)
    r   = sum_k gamma_step^k r_{t+k}              (see REWARD below)
    s'  = observation history at t+8
    d   = 1 if the episode *terminated* inside the interval (true MDP end), else 0
    discount = gamma_step ** (steps actually taken)

REWARD (documented in imitation.md section 7). The env's shaped reward has grasp/release
toggle spikes that would dominate a critic, so the RL reward is rebuilt from stored
per-step fields instead of used as-is:
  +1.0                 on the step the episode *terminates with success* (sparse label)
  + clip(delta fold_score, -CLIP, CLIP)   dense progress shaping, spikes clipped
Truncations (step cap) bootstrap, since `terminated` is False there (M1.3 schema).
`unstable` episodes (solver blow-ups, stored as truncated) are *tagged* (`unstable` [N])
and excluded by default: their last states are simulator artefacts, not outcomes.

Every transition also carries its episode's outcome code (`code` [N]: success / G1 / F1 /
S1 / M1, `imitation.evaluate.failure_code`) so a learner can sample stratified by failure
mode (M5.1), and `max_per_episode` subsamples long episodes -- a stall that runs to the
step cap otherwise contributes ~3x the transitions of a success (M5.3 diagnosis).
"""

from __future__ import annotations

import numpy as np

from imitation.data.dataset import _history

SUCCESS_REWARD = 1.0
SHAPING_CLIP = 0.05
GAMMA_STEP = 0.99


def step_rewards(ep) -> np.ndarray:
    """Per-step RL reward (see module doc)."""
    prev = np.concatenate([[0.0], ep.fold_score[:-1]])
    r = np.clip(ep.fold_score - prev, -SHAPING_CLIP, SHAPING_CLIP).astype(np.float32)
    r[0] = 0.0
    if ep.terminated[-1] and ep.meta["success"]:
        r[-1] += SUCCESS_REWARD
    return r


CODES = ("success", "G1", "F1", "S1", "M1")


def build_transitions(episodes, obs_horizon=2, macro=8, gamma_step=GAMMA_STEP, include_unstable=False,
                      max_per_episode=None, seed=0):
    """Macro-action transitions. Returns a dict of arrays: obs [N,H,D], act [N,macro,A],
    rew [N], next_obs [N,H,D], done [N], discount [N], valid [N,macro] (steps that exist --
    the last interval can be short), `source` [N] (0 expert, 1 student/dagger), `code` [N]
    (index into CODES), `unstable` [N] and `episode` [N]."""
    from imitation.evaluate import failure_code
    rng = np.random.default_rng(seed)
    O, Act, R, O2, Dn, G, V, S, C, U, E = [], [], [], [], [], [], [], [], [], [], []
    for ei, ep in enumerate(episodes):
        unstable = ep.meta.get("termination_reason") == "unstable"
        if unstable and not include_unstable:
            continue
        T = ep.steps
        r = step_rewards(ep)
        code = CODES.index(failure_code(ep) or "success")
        obs_ext = np.concatenate([ep.obs, ep.final_obs[None]])          # obs_ext[T] = final
        starts = np.arange(0, T, macro)
        if max_per_episode and len(starts) > max_per_episode:            # keep the terminal interval
            keep = rng.choice(len(starts) - 1, max_per_episode - 1, replace=False)
            starts = np.sort(np.concatenate([starts[keep], starts[-1:]]))
        for t in starts:
            n = min(macro, T - t)
            a = np.zeros((macro, ep.actions.shape[1]), np.float32)
            a[:n] = ep.actions[t:t + n]
            valid = np.zeros(macro, bool)
            valid[:n] = True
            O.append(_history(ep.obs, t, obs_horizon))
            O2.append(_history(obs_ext, t + n, obs_horizon))
            Act.append(a)
            R.append(float(np.sum(r[t:t + n] * gamma_step ** np.arange(n))))
            Dn.append(float(ep.terminated[t + n - 1]))
            G.append(gamma_step ** n)
            V.append(valid)
            S.append(0 if ep.meta["source"] == "expert" else 1)
            C.append(code)
            U.append(unstable)
            E.append(ei)
    return {"obs": np.stack(O).astype(np.float32), "act": np.stack(Act), "rew": np.asarray(R, np.float32),
            "next_obs": np.stack(O2).astype(np.float32), "done": np.asarray(Dn, np.float32),
            "discount": np.asarray(G, np.float32), "valid": np.stack(V), "source": np.asarray(S, np.int8),
            "code": np.asarray(C, np.int8), "unstable": np.asarray(U, bool), "episode": np.asarray(E, np.int32)}


def stratified_weights(code, mode="uniform"):
    """Per-transition sampling weights. "uniform": every outcome code present gets the same
    total weight (M5.1 stratification); "natural": plain uniform over transitions."""
    if mode == "natural":
        return np.ones(len(code), np.float32)
    present, counts = np.unique(code, return_counts=True)
    w = np.zeros(len(code), np.float32)
    for c, n in zip(present, counts):
        w[code == c] = 1.0 / n
    return w / w.sum() * len(code)
