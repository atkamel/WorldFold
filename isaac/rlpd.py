"""Off-policy SAC learner with RLPD settings (Ball et al. 2023, arXiv 2302.02948): a large critic ensemble with
LayerNorm (REDQ, Chen et al. 2021, arXiv 2101.05982) and many gradient updates per environment step (high
update-to-data ratio), plus symmetric 50/50 sampling between a prior (scripted/noisy) buffer and the online
replay buffer. The actor is a tanh-squashed Gaussian whose mean head starts at zero, so a residual policy added
to a scripted controller starts as a no-op.

Pure torch + numpy; no simulator imports, so this runs the same inside the Isaac container (torch 2.7) and out.
Only long-stable torch APIs are used: nn.Module, nn.Linear, nn.LayerNorm, torch.optim.Adam, torch.distributions.Normal.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
TANH_EPS = 1e-6
INIT_ALPHA = 0.1


class ReplayBuffer:
    """Circular float32 replay buffer. `obs`/`act`/`next_obs` are (capacity, dim); `rew`/`terminal` are
    (capacity,). `terminal` means a true terminal (no bootstrap past it); time-limit truncations go in with
    terminal=False so the critic still bootstraps through them."""

    def __init__(self, obs_dim, act_dim, capacity, seed=0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, terminal):
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        rew = np.asarray(rew, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        terminal = np.asarray(terminal, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
            act = act[None, :]
            next_obs = next_obs[None, :]
            rew = rew.reshape(1)
            terminal = terminal.reshape(1)
        n = obs.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.terminal[idx] = terminal
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, n, device="cpu"):
        idx = self._rng.integers(0, self.size, size=n)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.act[idx], dtype=torch.float32, device=device),
            "rew": torch.as_tensor(self.rew[idx], dtype=torch.float32, device=device).unsqueeze(-1),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "terminal": torch.as_tensor(self.terminal[idx], dtype=torch.float32, device=device).unsqueeze(-1),
        }

    def save(self, path):
        np.savez_compressed(
            path, obs=self.obs, act=self.act, rew=self.rew, next_obs=self.next_obs,
            terminal=self.terminal, ptr=self.ptr, size=self.size,
        )

    def load(self, path):
        d = np.load(path)
        self.obs = d["obs"]
        self.act = d["act"]
        self.rew = d["rew"]
        self.next_obs = d["next_obs"]
        self.terminal = d["terminal"]
        self.capacity = self.obs.shape[0]
        self.ptr = int(d["ptr"])
        self.size = int(d["size"])


def _mixed_batch(online, prior, n, device="cpu"):
    """RLPD symmetric sampling: n // 2 transitions from `online`, the rest from `prior`."""
    n_online = n // 2
    n_prior = n - n_online
    online_batch = online.sample(n_online, device)
    prior_batch = prior.sample(n_prior, device)
    return {k: torch.cat([online_batch[k], prior_batch[k]], dim=0) for k in online_batch}


class _Actor(nn.Module):
    """obs -> hidden -> hidden -> (mean, log_std) of a Gaussian in pre-tanh space. Mean head starts at zero
    (weight and bias) so the deterministic action at init is exactly 0; log_std head starts at a constant
    init_std (zero weight, bias = log(init_std))."""

    def __init__(self, obs_dim, act_dim, hidden, init_std):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        with torch.no_grad():
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(math.log(init_std))

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class _QNet(nn.Module):
    """(obs, act) -> hidden -> hidden -> scalar Q, with LayerNorm after each hidden Linear (RLPD)."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class RLPDAgent:
    """SAC actor + an ensemble of `n_critics` Q-networks, REDQ-style target (min over `n_target` randomly
    chosen critics), trained with RLPD's symmetric online/prior sampling and a high update-to-data ratio."""

    def __init__(self, obs_dim, act_dim, device="cpu", hidden=256, n_critics=10, n_target=2, gamma=0.99,
                 tau=0.005, lr=3e-4, init_std=0.1, target_entropy_scale=1.0, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.n_critics = n_critics
        self.n_target = n_target
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -target_entropy_scale * act_dim

        self.actor = _Actor(obs_dim, act_dim, hidden, init_std).to(device)
        self.critics = nn.ModuleList([_QNet(obs_dim, act_dim, hidden) for _ in range(n_critics)]).to(device)
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for p in self.target_critics.parameters():
            p.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(math.log(INIT_ALPHA), device=device))

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample(self, obs, actor=None):
        actor = actor if actor is not None else self.actor
        mean, log_std = actor(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + TANH_EPS)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mean

    def act(self, obs, deterministic=False):
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if single:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(obs_t)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                action = torch.tanh(mean + std * torch.randn_like(mean))
        action = action.cpu().numpy()
        return action[0] if single else action

    def _q_all(self, critics, obs, act):
        return torch.stack([c(obs, act) for c in critics], dim=0)  # (n_critics, batch, 1)

    def _target(self, next_obs, rew, terminal):
        """REDQ target value: min over `n_target` randomly chosen (distinct) target critics, SAC entropy backup.
        Bootstrapping is zeroed wherever `terminal` is 1 (a true terminal, not a time-limit truncation)."""
        with torch.no_grad():
            next_action, next_log_prob, _ = self._sample(next_obs)
            idx = self.rng.choice(self.n_critics, size=self.n_target, replace=False)
            target_qs = torch.stack([self.target_critics[i](next_obs, next_action) for i in idx], dim=0)
            min_q = target_qs.min(dim=0).values
            target_v = min_q - self.alpha.detach() * next_log_prob
            y = rew + self.gamma * (1.0 - terminal) * target_v
        return y

    def _critic_step(self, batch):
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, terminal = batch["next_obs"], batch["terminal"]
        y = self._target(next_obs, rew, terminal)
        q_preds = self._q_all(self.critics, obs, act)
        loss = sum(((q_preds[i] - y) ** 2).mean() for i in range(self.n_critics))
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            for critic, target in zip(self.critics, self.target_critics):
                for p, tp in zip(critic.parameters(), target.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return float(loss.item())

    def _actor_alpha_step(self, batch):
        obs = batch["obs"]
        action, log_prob, _ = self._sample(obs)
        q_all = self._q_all(self.critics, obs, action)
        q_mean = q_all.mean(dim=0)
        actor_loss = (self.alpha.detach() * log_prob - q_mean).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_mean.mean().item()),
            "entropy": float((-log_prob).mean().item()),
        }

    def update(self, online, prior=None, batch_size=256, utd=10, update_actor=True):
        """`utd` critic updates per call, half the batch from `online` and half from `prior` (when `prior` is
        given and non-empty), else all from `online`. Actor and temperature are updated once, after the critic
        updates (RLPD/DroQ), on the last critic batch."""
        use_mixed = prior is not None and len(prior) > 0
        critic_losses = []
        batch = None
        for _ in range(utd):
            if use_mixed:
                batch = _mixed_batch(online, prior, batch_size, device=self.device)
            else:
                batch = online.sample(batch_size, self.device)
            critic_losses.append(self._critic_step(batch))
        stats = self._actor_alpha_step(batch) if update_actor else {}
        stats["critic_loss"] = float(np.mean(critic_losses))
        return stats

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"])
        self.critics.load_state_dict(d["critics"])
        self.target_critics.load_state_dict(d["target_critics"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"])
        self.actor_optim.load_state_dict(d["actor_optim"])
        self.critic_optim.load_state_dict(d["critic_optim"])
        self.alpha_optim.load_state_dict(d["alpha_optim"])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
