"""Supervised training of a chunk policy on a frozen dataset version (GPU when available).

    python -m imitation.train --policy chunk_mlp --dataset v1 --run runs/mlp_v1
    python -m imitation.train --policy diffusion --dataset v1 --run runs/diff_v1 --steps 60000

Writes <run>/final.pt (EMA weights), <run>/tb/ and <run>/run.json with everything
needed to reproduce the run (design doc 12): git commit, dataset version + hash,
config, seed, parameter count, checkpoint hash, final losses.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.dataset import Normalizer, bc_episodes, clamp_fraction, teacher_samples, build_samples, split_episodes
from imitation.data.schema import load_dataset
from imitation.policies.common import build_policy, policy_class, default_device, load_policy, n_params, save_policy
from imitation.spec import OBS_SUBSETS

POLICY_DEFAULTS = {
    "chunk_mlp": {"obs_horizon": 2, "chunk": 16, "width": 512, "depth": 4, "dropout": 0.1},
    "diffusion": {"obs_horizon": 2, "chunk": 16, "cond_dim": 256, "channels": [128, 256],
                  "train_steps": 100, "infer_steps": 10},
    "vision": {"obs_horizon": 2, "chunk": 16, "width": 512, "depth": 3, "dropout": 0.1},
}


def git_commit():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(p, 1 - self.decay)
            else:
                e.copy_(p)


def train(policy_kind, dataset, run, root=DEFAULT_ROOT, steps=30_000, batch=1024, lr=1e-3, weight_decay=1e-4,
          warmup=500, ema=0.999, seed=0, eval_every=2000, init_from=None, max_episodes=None,
          policy_kwargs=None, dagger_weight=1.0, allow_failures=False, obs_subset=None, teacher=None, log=print):
    """dagger_weight: sampling weight of a DAgger label relative to an expert chunk. Labels
    are few (one per replan) next to the dense expert chunks, and they are exactly
    the states the student gets wrong, so they are oversampled.

    obs_subset: hide observation dims (M2.3); None keeps the policy's own default.
    teacher: a privileged checkpoint -- every step's target becomes its chunk from the
    stored observation (Phase 4 distillation) instead of the expert/DAgger labels.
    Image policies (`needs_images`) load the dataset's camera images onto the device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device()
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)

    needs_images = policy_class(load_policy(init_from, "cpu").kind if init_from else policy_kind).needs_images
    manifest, episodes = load_dataset(root, dataset, images=needs_images)
    episodes, n_failed = bc_episodes(episodes, allow_failures)
    if n_failed:
        log(f"dropped {n_failed} failed expert episodes (pass allow_failures to keep them)")
    if max_episodes:
        episodes = episodes[:max_episodes]
    train_eps, val_eps = split_episodes(episodes, val_fraction=0.1)
    if not train_eps:             # a single episode (debugging): validate on what we train on
        train_eps = val_eps
    cfg = POLICY_DEFAULTS.get(policy_kind, {}) | (policy_kwargs or {})
    if init_from:                 # warm start keeps the checkpoint's normalizer: its weights expect it
        policy = load_policy(init_from, device).train()
    else:
        policy = build_policy(policy_kind, **cfg).to(device)
        norm = Normalizer.fit(train_eps)
        policy.set_normalizer(norm.mean, norm.std)
        if obs_subset:
            policy.set_obs_subset(OBS_SUBSETS[obs_subset])
    teacher_policy = load_policy(teacher, device) if teacher else None

    tensors, images = {}, {}
    for name, eps in (("train", train_eps), ("val", val_eps)):
        if teacher_policy is not None:
            X, Y, M, W, I = teacher_samples(eps, teacher_policy, policy.obs_horizon, policy.chunk)
        else:
            X, Y, M, W, I = build_samples(eps, policy.obs_horizon, policy.chunk, index=True)
        tensors[name] = [torch.as_tensor(a, device=device) for a in (X, Y, M)] + [W, torch.as_tensor(I, device=device)]
        if needs_images:          # uint8 on the device; ~50 KB/step, converted to float per batch
            images[name] = {cam: torch.as_tensor(np.concatenate([e.images[cam] for e in eps]), device=device)
                            for cam, _ in policy.cameras}

    def batch_images(name, rows):
        return {cam: img[rows] for cam, img in images[name].items()} if needs_images else None
    n_train = len(tensors["train"][0])
    clamped = clamp_fraction(tensors["train"][0], policy.obs_mean, policy.obs_std)
    if clamped > 0:
        log(f"normalizer clamp: {clamped:.4%} of train obs entries at +-10")
    n_dagger = int((tensors["train"][3] == 1).sum())
    log(f"device {device} | {len(train_eps)} train / {len(val_eps)} val episodes | {n_train} samples "
        f"({n_dagger} DAgger labels) | {policy.kind} {n_params(policy) / 1e6:.2f}M params")

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ema_model = EMA(policy, ema)
    amp = device.type == "cuda"
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run / "tb"))
    except Exception:
        tb = None

    @torch.no_grad()
    def val_loss(model):
        X, Y, M, _, I = tensors["val"]
        total, count, bs = 0.0, 0, 4096 if not needs_images else 512
        for i in range(0, len(X), bs):
            s = slice(i, i + bs)
            args = (X[s], Y[s], M[s]) + ((batch_images("val", I[s]),) if needs_images else ())
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                total += float(model.compute_loss(*args)) * len(X[s])
            count += len(X[s])
        return total / max(count, 1)

    X, Y, M, W, I = tensors["train"]
    weights = torch.as_tensor(np.where(W == 1, dagger_weight, 1.0), dtype=torch.float32, device=device)
    t0, history, running = time.time(), [], 0.0
    policy.train()
    for step in range(1, steps + 1):
        idx = (torch.multinomial(weights, batch, replacement=True) if n_dagger and dagger_weight != 1.0
               else torch.randint(0, n_train, (batch,), device=device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            args = (X[idx], Y[idx], M[idx]) + ((batch_images("train", I[idx]),) if needs_images else ())
            loss = policy.compute_loss(*args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_model.update(policy)
        running += float(loss.detach()) if step % 50 == 0 else 0.0
        if step % eval_every == 0 or step == steps:
            vl = val_loss(ema_model.model)
            tl = running / (eval_every / 50)
            running = 0.0
            history.append({"step": step, "train_loss": tl, "val_loss": vl})
            if tb:
                tb.add_scalar("loss/train", tl, step)
                tb.add_scalar("loss/val_ema", vl, step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
            log(f"  step {step:6d}  train {tl:.4f}  val(ema) {vl:.4f}  {time.time() - t0:.0f}s")

    ckpt = run / "final.pt"
    sha = save_policy(ema_model.model, ckpt, extra={"dataset": dataset, "dataset_hash": manifest["content_hash"]})
    info = {"git_commit": git_commit(), "dataset": {"root": str(root), "version": dataset,
                                                    "content_hash": manifest["content_hash"],
                                                    "n_episodes": len(episodes), "n_train_samples": n_train,
                                                    "n_dagger_labels": n_dagger, "clamped_fraction": clamped,
                                                    "teacher": str(teacher) if teacher else None},
            "policy": {"kind": policy.kind, "config": policy.config(), "n_params": n_params(policy)},
            "train": {"steps": steps, "batch": batch, "lr": lr, "weight_decay": weight_decay, "warmup": warmup,
                      "ema": ema, "seed": seed, "dagger_weight": dagger_weight, "obs_subset": obs_subset, "init_from": str(init_from) if init_from else None,
                      "device": str(device), "amp_bf16": amp, "seconds": round(time.time() - t0, 1)},
            "checkpoint": {"path": str(ckpt), "sha256": sha}, "history": history}
    with open(run / "run.json", "w") as f:
        json.dump(info, f, indent=1)
    if tb:
        tb.close()
    return ckpt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=list(POLICY_DEFAULTS), default="chunk_mlp")
    ap.add_argument("--dataset", default="v1")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--obs-subset", choices=list(OBS_SUBSETS), default=None, help="M2.3 ablation")
    ap.add_argument("--teacher", default=None, help="privileged checkpoint to distill from (Phase 4)")
    ap.add_argument("--dagger-weight", type=float, default=1.0)
    ap.add_argument("--allow-failures", action="store_true", help="train on failed expert episodes too")
    ap.add_argument("--max-episodes", type=int, default=None, help="debug: train on the first N episodes")
    ap.add_argument("--policy-kwargs", default="{}", help='JSON overrides, e.g. \'{"chunk": 8}\'')
    args = ap.parse_args()
    train(args.policy, args.dataset, args.run, root=args.root, steps=args.steps, batch=args.batch, lr=args.lr,
          seed=args.seed, init_from=args.init_from, max_episodes=args.max_episodes, allow_failures=args.allow_failures,
          obs_subset=args.obs_subset, teacher=args.teacher, dagger_weight=args.dagger_weight,
          policy_kwargs=json.loads(args.policy_kwargs))


if __name__ == "__main__":
    main()
