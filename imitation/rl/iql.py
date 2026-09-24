"""Implicit Q-learning over chunk macro-actions, warm-started from the DAgger policy (M5.3).

    python -m imitation.rl.iql --init runs/dagger_v2/round_2/final.pt \
        --versions v1 v1_failures harvest_v1 --out runs/iql_v1

IQL (Kostrikov et al. 2021) never queries the critic on actions outside the data:
  V   expectile regression (tau) onto min of twin target Q(s, a_data)
  Q   TD onto r + discount * (1 - done) * V(s')
  pi  advantage-weighted regression: the chunk policy's first `macro` actions are pulled
      toward the data's executed actions with weight exp(beta * (Q - V)), clipped
Actions are the 8-step executed chunks (`imitation.rl.transitions`). The policy is the
same ChunkMLP the imitation stage produced, so the result drops into evaluate / demo
unchanged. Critics read the policy's own observation normalizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.schema import load_dataset
from imitation.policies.common import default_device, load_policy, save_policy
from imitation.rl.transitions import build_transitions
from imitation.train import git_commit


def _mlp(inp, width=512, depth=3, out=1):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, out))


class Critics(nn.Module):
    def __init__(self, obs_in, act_in):
        super().__init__()
        self.q1, self.q2 = _mlp(obs_in + act_in), _mlp(obs_in + act_in)
        self.v = _mlp(obs_in)

    def q(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()


def train_iql(init, versions, out, root=DEFAULT_ROOT, steps=40_000, batch=1024, tau=0.7, beta=3.0,
              lr_critic=3e-4, lr_policy=1e-4, target_ema=0.005, macro=8, adv_clip=100.0, adv_norm=False, seed=0,
              log=print):
    """adv_norm: weight = exp(A / std(A) / beta) instead of exp(beta * A). With a sparse
    0/1 reward the raw advantages here are ~1e-2, so exp(3 A) ~ 1 and the update is plain
    BC over all data, failures included (iql_v1). Standardizing sets the sharpness per batch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    episodes, hashes = [], {}
    for v in versions:
        m, eps = load_dataset(root, v)
        episodes += eps
        hashes[v] = m["content_hash"]
    seen, uniq = set(), []          # a version lists its ancestors too: keep each episode once
    for e in episodes:
        key = (e.meta["source"], e.meta["seed"], e.meta.get("round"))
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    policy = load_policy(init, device).train()
    tr = build_transitions(uniq, obs_horizon=policy.obs_horizon, macro=macro)
    n = len(tr["rew"])
    log(f"{len(uniq)} episodes -> {n} macro transitions ({int((tr['source'] == 1).sum())} student), "
        f"success-terminal {int(tr['done'].sum())}")
    T = {k: torch.as_tensor(v, device=device) for k, v in tr.items()}
    T["valid"] = T["valid"].float()

    def enc(o):                                     # normalized, flattened obs history
        return policy.normalize_obs(o).flatten(1).detach()

    obs_in = policy.obs_horizon * policy.obs_dim
    critics = Critics(obs_in, macro * policy.action_dim).to(device)
    target = copy.deepcopy(critics).requires_grad_(False)
    opt_c = torch.optim.AdamW(critics.parameters(), lr=lr_critic)
    opt_p = torch.optim.AdamW(policy.parameters(), lr=lr_policy, weight_decay=1e-4)
    t0, history = time.time(), []
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), device=device)
        s, s2 = enc(T["obs"][idx]), enc(T["next_obs"][idx])
        a = (T["act"][idx] * T["valid"][idx, :, None]).flatten(1)
        with torch.no_grad():
            tq = torch.min(*target.q(s, a))
            v_next = target.v(s2).squeeze(-1)
            y = T["rew"][idx] + T["discount"][idx] * (1 - T["done"][idx]) * v_next
        v = critics.v(s).squeeze(-1)
        q1, q2 = critics.q(s, a)
        loss_v = expectile_loss(tq - v, tau)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_c.zero_grad(set_to_none=True)
        (loss_v + loss_q).backward()
        opt_c.step()
        with torch.no_grad():
            for p, tp in zip(critics.parameters(), target.parameters()):
                tp.lerp_(p, target_ema)
            adv = tq - v.detach()
            if adv_norm:
                w = torch.exp(adv / (adv.std() + 1e-6) / beta).clamp(max=adv_clip)
            else:
                w = torch.exp(beta * adv).clamp(max=adv_clip)
        pred = policy(T["obs"][idx])[:, :macro]
        err = ((pred - T["act"][idx]).abs().mean(-1) * T["valid"][idx]).sum(-1) / T["valid"][idx].sum(-1)
        loss_p = (w * err).mean()
        opt_p.zero_grad(set_to_none=True)
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_p.step()
        if step % 2000 == 0 or step == steps:
            row = {"step": step, "loss_v": loss_v.item(), "loss_q": loss_q.item(), "loss_pi": loss_p.item(),
                   "v_mean": v.mean().item(), "adv_mean": adv.mean().item(), "w_mean": w.mean().item()}
            history.append(row)
            log(f"  step {step:6d}  V {row['loss_v']:.4f}  Q {row['loss_q']:.4f}  pi {row['loss_pi']:.4f}  "
                f"v {row['v_mean']:.3f}  w {row['w_mean']:.2f}  {time.time() - t0:.0f}s")
    policy.eval()
    ckpt = out / "final.pt"
    sha = save_policy(policy, ckpt, extra={"iql_from": str(init), "versions": versions})
    torch.save(critics.state_dict(), out / "critics.pt")
    info = {"git_commit": git_commit(), "init": str(init), "datasets": hashes, "n_episodes": len(uniq),
            "n_transitions": n, "iql": {"steps": steps, "batch": batch, "tau": tau, "beta": beta, "macro": macro,
                                        "lr_critic": lr_critic, "lr_policy": lr_policy, "adv_clip": adv_clip, "adv_norm": adv_norm,
                                        "seed": seed},
            "checkpoint": {"path": str(ckpt), "sha256": sha}, "history": history}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="imitation/DAgger checkpoint (chunk_mlp)")
    ap.add_argument("--versions", nargs="+", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--adv-norm", action="store_true", help="standardize advantages (beta is then a temperature)")
    ap.add_argument("--adv-clip", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_iql(args.init, args.versions, args.out, root=args.root, steps=args.steps, tau=args.tau, beta=args.beta,
              adv_norm=args.adv_norm, adv_clip=args.adv_clip, seed=args.seed)


if __name__ == "__main__":
    main()
