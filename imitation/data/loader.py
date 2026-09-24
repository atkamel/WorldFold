"""Lazy training batches (roadmap M4.1 "lazy DataLoader").

`build_samples` materializes every training window -- the obs history is copied H
times and, for image policies, ~50 KB of camera frames per step sat on the GPU. The
sampler keeps each array *once* and builds a batch by index:

    per step     obs [N, D] (device), actions padded per episode [N + E*K, A]
    per sample   the step row t, its episode's first row, its padded-action row p,
                 or a DAgger label index
    per batch    X = obs[clamp(t - H+1 .. t, start)]   Y = act_pad[p .. p+K]   (or label)
                 M = teacher-chosen steps, cut at the first non-teacher step
                 images = camera[t] gathered from pinned CPU memory, then copied over

`batch(idx)` for all indices reproduces `build_samples` / `teacher_samples` exactly
(tested). With a `teacher`, targets are that policy's chunk from the stored
observations, computed per batch on the device -- Phase 4 distillation without
storing an N x K x A label table.
"""

from __future__ import annotations

import numpy as np
import torch

from imitation.data.dataset import _pad_action
from imitation.data.schema import ACTOR_TEACHER


class WindowSampler:
    def __init__(self, episodes, obs_horizon, chunk, device, teacher=None, cameras=None):
        self.H, self.K, self.device, self.teacher = obs_horizon, chunk, device, teacher
        self.cameras = [c for c, _ in cameras] if cameras else []
        obs, act_pad, valid_pad = [], [], []
        dense_t, dense_start, dense_p, dense_w = [], [], [], []
        lab_t, lab_start, labels = [], [], []
        row = prow = 0
        for ep in episodes:
            T = ep.steps
            obs.append(ep.obs)
            act_pad.append(np.concatenate([ep.actions, np.repeat(_pad_action(ep.actions[-1])[None], chunk, 0)]))
            expert = ep.meta["source"] == "expert"
            if teacher is not None:          # every step, labelled by the teacher
                steps = np.arange(T)
                valid = np.ones(T + chunk, bool)
                w = 0 if expert else 1
            else:                            # dense expert chunks (imitation.md section 4)
                teach = (ep.actor == ACTOR_TEACHER) & expert
                steps = np.flatnonzero(teach)
                valid = np.concatenate([teach, np.full(chunk, bool(ep.terminated[-1]))])
                w = 0
            valid_pad.append(valid)
            dense_t.append(row + steps)
            dense_start.append(np.full(len(steps), row))
            dense_p.append(prow + steps)
            dense_w.append(np.full(len(steps), w))
            if teacher is None and len(ep.label_steps):
                lab_t.append(row + ep.label_steps)
                lab_start.append(np.full(len(ep.label_steps), row))
                labels.append(ep.labels[:, :chunk])
            row += T
            prow += T + chunk
        t = lambda xs: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0, np.int64), device=device)
        self.obs = torch.as_tensor(np.concatenate(obs), dtype=torch.float32, device=device)
        self.act_pad = torch.as_tensor(np.concatenate(act_pad), dtype=torch.float32, device=device)
        self.valid_pad = torch.as_tensor(np.concatenate(valid_pad), device=device)
        self.dense_t, self.dense_start, self.dense_p = t(dense_t), t(dense_start), t(dense_p)
        self.lab_t, self.lab_start = t(lab_t), t(lab_start)
        self.labels = (torch.as_tensor(np.concatenate(labels), dtype=torch.float32, device=device) if labels
                       else torch.zeros((0, chunk, self.act_pad.shape[1]), device=device))
        self.n_dense, self.n_label = len(self.dense_t), len(self.lab_t)
        self.weight_tag = np.concatenate([np.concatenate(dense_w) if dense_w else np.zeros(0),
                                          np.ones(self.n_label)]).astype(np.int8)   # W: 1 = oversample
        self.images = {}
        if self.cameras:                     # uint8 frames stay on the host, pinned for fast copies
            pin = device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda")
            for cam in self.cameras:
                arr = torch.as_tensor(np.concatenate([ep.images[cam] for ep in episodes]))
                self.images[cam] = arr.pin_memory() if pin else arr
        self.rows = torch.cat([self.dense_t, self.lab_t])          # each sample's step row

    def __len__(self):
        return self.n_dense + self.n_label

    def _history(self, t, start, obs=None, h=None):
        obs = self.obs if obs is None else obs
        h = self.H if h is None else h
        offs = torch.arange(h - 1, -1, -1, device=self.device)
        idx = torch.maximum(t[:, None] - offs[None], start[:, None])
        return obs[idx]

    def batch(self, idx):
        """(X, Y, M, images or None) for sample indices `idx` (a device LongTensor)."""
        dense = idx < self.n_dense
        di, li = idx[dense], idx[~dense] - self.n_dense
        B, K = len(idx), self.K
        X = torch.empty((B, self.H, self.obs.shape[1]), device=self.device)
        Y = torch.empty((B, K, self.act_pad.shape[1]), device=self.device)
        M = torch.empty((B, K), device=self.device)
        if len(di):
            t, st, p = self.dense_t[di], self.dense_start[di], self.dense_p[di]
            X[dense] = self._history(t, st)
            if self.teacher is not None:
                with torch.no_grad():
                    y = self.teacher.sample(self._history(t, st, h=self.teacher.obs_horizon)).clamp(-1, 1).float()
                Y[dense], M[dense] = y[:, :K], 1.0
            else:
                win = p[:, None] + torch.arange(K, device=self.device)[None]
                Y[dense] = self.act_pad[win]
                # once someone else took over, the rest of this chunk is not the teacher's plan
                M[dense] = torch.cumprod(self.valid_pad[win].float(), dim=1)
        if len(li):
            X[~dense] = self._history(self.lab_t[li], self.lab_start[li])
            Y[~dense], M[~dense] = self.labels[li], 1.0
        imgs = None
        if self.cameras:
            rows = self.rows[idx].cpu()
            imgs = {c: self.images[c][rows].to(self.device, non_blocking=True) for c in self.cameras}
        return X, Y, M, imgs

    def image_stats(self, n=4096, seed=0):
        """Per-camera, per-channel mean/std of the frames (for VisionChunkPolicy's normalizer)."""
        g = torch.Generator().manual_seed(seed)
        out = {}
        for cam, arr in self.images.items():
            pick = torch.randint(0, len(arr), (min(n, len(arr)),), generator=g)
            x = arr[pick].float() / 255.0
            out[cam] = (x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp(min=1e-3))
        return out
