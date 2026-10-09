"""Vision success / fold-score detector (roadmap M4.3).

On hardware nothing reads cloth vertex positions, so success has to be judged from the
camera. This trains a small CNN on the `main` camera frame to predict:
  folded      every corner within SUCCESS_DIST of its goal *and* both grippers open --
              the per-step condition the sim's success criterion holds for 20 steps
  fold_score  the sim's continuous progress measure
Labels are free in sim: `folded` comes from the corner-to-goal block of the stored
observation (dims 107:119) and the `grasped` flags; `fold_score` is stored per step.

    python -m imitation.vision.success train --versions v1_img v1_failures_img --out runs/success_v1
    python -m imitation.vision.success agree --ckpt runs/success_v1/detector.pt --versions eval_...

`agree` reports the exit metric: agreement between the detector's verdict on each
episode's final frame and the sim's success flag, with a Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.dataset import is_val_seed
from imitation.data.schema import load_dataset
from imitation.policies.common import default_device
from imitation.policies.vision import _Encoder, random_shift

CORNER_TO_GOAL = slice(107, 119)      # 4 corners x 3, see imitation.md section 2
SUCCESS_DIST = 0.05
CAMERA = "main"


def folded_labels(ep):
    """Per-step `folded` label from the privileged observation (see module doc)."""
    dist = np.linalg.norm(ep.obs[:, CORNER_TO_GOAL].reshape(-1, 4, 3), axis=2).max(1)
    return (dist < SUCCESS_DIST) & ~ep.grasped.any(1)


def final_folded(ep):
    dist = np.linalg.norm(ep.final_obs[CORNER_TO_GOAL].reshape(4, 3), axis=1).max()
    return bool(dist < SUCCESS_DIST and not ep.grasped[-1].any())


class SuccessDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = _Encoder(256)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(256, 2))
        # per-channel normalization fit on the training frames (M4.1); defaults = fixed /255 - 0.5
        self.register_buffer("img_mean", torch.full((3,), 0.5))
        self.register_buffer("img_std", torch.ones(3))

    def forward(self, img, augment=False):
        x = (img.float() / 255.0 - self.img_mean[:, None, None]) / self.img_std[:, None, None]
        if augment:
            x = random_shift(x)
        out = self.head(self.enc(x))
        return out[:, 0], torch.sigmoid(out[:, 1])        # folded logit, fold score

    @torch.no_grad()
    def predict(self, frames: np.ndarray, device=None):
        device = device or next(self.parameters()).device
        logits, scores = [], []
        for i in range(0, len(frames), 1024):
            lg, sc = self(torch.as_tensor(frames[i:i + 1024], device=device))
            logits.append(lg.float().cpu())
            scores.append(sc.float().cpu())
        return torch.sigmoid(torch.cat(logits)).numpy(), torch.cat(scores).numpy()


def _frames(root, versions):
    frames, folded, score, val = [], [], [], []
    for v in versions:
        _, eps = load_dataset(root, v, images=True)
        for ep in eps:
            frames.append(ep.images[CAMERA])
            folded.append(folded_labels(ep))
            score.append(ep.fold_score)
            val.append(np.full(ep.steps, is_val_seed(ep.meta["seed"])))
    return (np.concatenate(frames), np.concatenate(folded).astype(np.float32),
            np.clip(np.concatenate(score), 0, 1).astype(np.float32), np.concatenate(val))


def train_detector(root, versions, out, steps=8000, batch=256, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    device = default_device()
    X, y, s, val = _frames(root, versions)
    X, y, s = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device), torch.as_tensor(s, device=device)
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    pos = float(y[tr].mean())
    log(f"{len(tr)} train / {len(va)} val frames, {pos:.1%} folded")
    w = torch.where(y[tr] > 0.5, 0.5 / pos, 0.5 / (1 - pos))      # class-balanced sampling
    model = SuccessDetector().to(device)
    sample = X[torch.as_tensor(tr[:4096], device=device)]
    model.img_mean.copy_((sample.float() / 255).mean(dim=(0, 2, 3)))
    model.img_std.copy_((sample.float() / 255).std(dim=(0, 2, 3)).clamp(min=1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr_t = torch.as_tensor(tr, device=device)
    for step in range(1, steps + 1):
        idx = tr_t[torch.multinomial(w, batch, replacement=True)]
        logit, score = model(X[idx], augment=True)
        loss = F.binary_cross_entropy_with_logits(logit, y[idx]) + F.mse_loss(score, s[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            model.eval()
            p, sc = model.predict(X[va], device)
            model.train()
            yv = y[va].cpu().numpy()
            acc = float(((p > 0.5) == (yv > 0.5)).mean())
            mae = float(np.abs(sc - s[va].cpu().numpy()).mean())
            log(f"  step {step:5d}  loss {float(loss):.4f}  val frame acc {acc:.3f}  fold-score MAE {mae:.3f}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "versions": list(versions), "val_frame_acc": acc,
                "val_fold_mae": mae}, out / "detector.pt")
    return out / "detector.pt", {"val_frame_acc": acc, "val_fold_mae": mae, "n_val_frames": int(len(va))}


def load_detector(path, device=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SuccessDetector()
    state = ckpt["state_dict"]
    for k, v in model.state_dict().items():          # detectors saved before the normalizer
        state.setdefault(k, v)
    model.load_state_dict(state)
    return model.to(device or default_device()).eval()


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / 2 / n) / d, z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return c - h, c + h


def agreement(detector, root, versions):
    """Detector verdict on each episode's final frame vs the sim's success flag. Only each
    version's *own* episodes count (its ancestors may be the detector's training data)."""
    rows = []
    for v in versions:
        m, eps = load_dataset(root, v, images=True)
        own = [ep for ep, e in zip(eps, m["episodes"]) if e["file"].startswith(f"{v}/")]
        for ep in own:
            p, _ = detector.predict(ep.images[CAMERA][-1:])
            rows.append((bool(p[0] > 0.5), bool(ep.meta["success"])))
    k, n = sum(a == b for a, b in rows), len(rows)
    tp = sum(a and b for a, b in rows)
    fp = sum(a and not b for a, b in rows)
    fn = sum(b and not a for a, b in rows)
    lo, hi = wilson(k, n)
    return {"n": n, "agree": k, "rate": k / n, "wilson": [lo, hi], "true_pos": tp, "false_pos": fp,
            "false_neg": fn, "sim_success": sum(b for _, b in rows)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--versions", nargs="+", default=["v1_img", "v1_failures_img"])
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=8000)
    a = sub.add_parser("agree")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--versions", nargs="+", required=True)
    for p in (t, a):
        p.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    if args.cmd == "train":
        path, info = train_detector(args.root, args.versions, args.out, steps=args.steps)
        print(json.dumps(info))
    else:
        res = agreement(load_detector(args.ckpt), args.root, args.versions)
        print(json.dumps(res, indent=1))
        Path(args.ckpt).with_name("agreement.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
