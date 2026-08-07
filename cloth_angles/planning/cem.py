"""CEM planner over imagined world-model rollouts (M4 -- the 'brain').

Closed loop, replanned every control step:

    observe angle field -> bootstrap RSSM state -> sample K action sequences
    -> imagine each forward (deterministic prior rollout) -> decode corner
    displacements via the keypoint head -> cost vs goal corners -> refit CEM
    distribution on elites -> execute the FIRST action of the best mean.

Scope (v1, hybrid): the planner controls the CARRY -- both grippers already
welded to the far corners (the scripted policy handles reach/grasp/lift,
where success is about IK precision, not imagination). Plan space is the two
end-effector position deltas (6 dims/step); grippers held closed, rotations
zero. Cost: mean distance of the two MOVING corners to their goals plus a
penalty on predicted drift of the stationary corners.
"""

from __future__ import annotations

import numpy as np
import torch

from cloth_angles.model.rssm import RSSMState

# rows of the env's corner arrays: [near-left(stay), far-left(move), near-right(stay), far-right(move)]
MOVING_ROWS = (1, 3)
STAY_ROWS = (0, 2)


def _expand_state(state: RSSMState, n: int) -> RSSMState:
    return RSSMState(
        h=state.h.expand(n, -1).contiguous(),
        z=state.z.expand(n, -1).contiguous(),
        prior_logits=state.prior_logits.expand(n, -1, -1).contiguous(),
        posterior_logits=None,
    )


class CEMCarryPlanner:
    def __init__(self, wm, keypoint_head, corners0: np.ndarray, goal_corners: np.ndarray,
                 horizon: int = 10, population: int = 48, elites: int = 6, iters: int = 3,
                 init_std: float = 0.5, stay_penalty: float = 0.5, device=None):
        self.wm = wm
        self.head = keypoint_head
        self.device = device or torch.device("cpu")
        self.corners0 = torch.as_tensor(corners0.reshape(4, 3), dtype=torch.float32,
                                          device=self.device)
        self.goal = torch.as_tensor(goal_corners.reshape(4, 3), dtype=torch.float32,
                                     device=self.device)
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iters = iters
        self.init_std = init_std
        self.stay_penalty = stay_penalty
        # warm-started CEM distribution over [horizon, 6] EE position deltas
        self.mean = np.zeros((horizon, 6), dtype=np.float32)
        self.std = np.full((horizon, 6), init_std, dtype=np.float32)

    @staticmethod
    def to_env_action(plan6: np.ndarray) -> np.ndarray:
        """6-dim plan step -> 14-dim env action: EE pos deltas, rotations 0,
        grippers closed (the carry contract)."""
        action = np.zeros(14, dtype=np.float32)
        action[0:3] = plan6[0:3]     # left EE delta
        action[7:10] = plan6[3:6]    # right EE delta
        action[6] = action[13] = -1.0
        return action

    def _cost(self, features: torch.Tensor) -> torch.Tensor:
        """features: [pop, latent] at the final imagined step -> [pop] costs."""
        delta = self.head(features).reshape(-1, 4, 3)
        corners = self.corners0[None] + delta
        moving = torch.stack([(corners[:, r] - self.goal[r]).norm(dim=-1)
                               for r in MOVING_ROWS], dim=1).mean(dim=1)
        drift = torch.stack([delta[:, r].norm(dim=-1) for r in STAY_ROWS], dim=1).mean(dim=1)
        return moving + self.stay_penalty * drift

    @torch.no_grad()
    def plan(self, field_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """field_flat: current observed angle field [N*N] -> 14-dim env action."""
        obs = torch.as_tensor(field_flat[None], dtype=torch.float32, device=self.device)
        state1 = self.wm.initial_state_from_obs(obs)

        mean, std = self.mean.copy(), self.std.copy()
        best_seq, best_cost = None, float("inf")
        for _ in range(self.iters):
            plans = rng.normal(mean, std, size=(self.population, self.horizon, 6)).astype(np.float32)
            plans = np.clip(plans, -1.0, 1.0)
            # 14-dim action sequences for the world model
            acts = np.zeros((self.population, self.horizon, 14), dtype=np.float32)
            acts[:, :, 0:3] = plans[:, :, 0:3]
            acts[:, :, 7:10] = plans[:, :, 3:6]
            acts[:, :, 6] = acts[:, :, 13] = -1.0

            state = _expand_state(state1, self.population)
            states = self.wm.rssm.imagine(state, torch.as_tensor(acts, device=self.device),
                                           deterministic=True)
            final = states[-1]
            costs = self._cost(final.feature()).cpu().numpy()

            elite_idx = np.argsort(costs)[: self.elites]
            mean = plans[elite_idx].mean(axis=0)
            std = plans[elite_idx].std(axis=0) + 1e-3
            if costs[elite_idx[0]] < best_cost:
                best_cost = float(costs[elite_idx[0]])
                best_seq = plans[elite_idx[0]].copy()

        # warm start next replan: shift the refined mean one step forward
        self.mean = np.concatenate([mean[1:], np.zeros((1, 6), dtype=np.float32)], axis=0)
        self.std = np.full_like(self.std, self.init_std)
        self.last_cost = best_cost
        return self.to_env_action(best_seq[0])
