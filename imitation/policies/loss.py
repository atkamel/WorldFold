"""Loss v2 (Phase Q1): a classification gripper, and per-step weights on the chunk mask.

Measured on `isaac_v1_friction` (31,117 steps): the expert's gripper commands are exactly -1 (close, 60%) or +1 (open),
never the env's hold band, so the gripper is a binary choice. Regressing it (L1 / epsilon-MSE) averages the two
modes near a switch, where the env's +-0.3 hysteresis turns an in-between value into "hold" -- a late close or a
release that never comes. With `gripper_head` on, the gripper dims are trained with BCE and emitted as exactly +-1.

The joint deltas saturate (|a| = 1 on 2-32% of steps per dim, p99 = 1), so per-dim action normalization would not
change their relative scale; it is not part of loss v2.

Weights multiply the [B, K] chunk mask, so every policy's masked mean becomes a weighted mean with no policy change:
- horizon: steps before the replan point (the ones that are executed) weigh 1, the rest `horizon_tail`;
- phase: a sample in the placement phase (fold_score >= `phase_fold`) or whose target chunk switches the gripper
  weighs `phase_weight` -- the fine corrections behind the S1 stalls and the grasp/release moments behind G1/G2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

GRIPPER_ACTION_DIMS = (5, 11)    # = imitation.data.dataset.GRIPPER_DIMS


@dataclass(frozen=True)
class LossCfg:
    gripper_head: bool = False
    gripper_weight: float = 1.0      # BCE weight relative to the per-step joint loss
    horizon_tail: float = 1.0        # weight of chunk steps at or past the replan point
    replan_every: int = 8
    phase_weight: float = 1.0
    phase_fold: float = 0.8

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))

    @classmethod
    def parse(cls, text):
        """JSON, or `key=value,key=value` (Windows PowerShell 5.1 strips the double quotes from JSON arguments)."""
        import json
        text = (text or "").strip()
        if not text or text.startswith("{"):
            return cls.from_dict(json.loads(text or "{}"))
        return cls.from_dict({k.strip(): json.loads(v) for k, v in (kv.split("=", 1) for kv in text.split(","))})

    def to_dict(self):
        return asdict(self)

    @property
    def is_baseline(self):
        return self == LossCfg()


def joint_dims(action_dim):
    return [d for d in range(action_dim) if d not in GRIPPER_ACTION_DIMS]


def split_loss(pred, target, gripper_weight=1.0):
    """Per-step [B, K] loss: L1 over the joint dims of `pred`, BCE with the gripper dims of `pred` read as logits
    (target open = +1)."""
    j, g = joint_dims(pred.shape[-1]), list(GRIPPER_ACTION_DIMS)
    joint = (pred[..., j] - target[..., j]).abs().mean(-1)
    grip = F.binary_cross_entropy_with_logits(pred[..., g].float(), (target[..., g] > 0).float(),
                                              reduction="none").mean(-1)
    return joint + gripper_weight * grip


def gripper_mask(action_dim, device=None):
    """Bool [action_dim], True on the gripper dims. Built on the device without a host copy, so it is safe inside
    CUDA graph capture (an index list is not: it is copied from the CPU)."""
    ar = torch.arange(action_dim, device=device)
    keep = torch.zeros(action_dim, dtype=torch.bool, device=device)
    for d in GRIPPER_ACTION_DIMS:
        keep = keep | (ar == d)
    return keep


def gripper_from_logits(out):
    """Replace the gripper-logit dims of a chunk by the command they choose: exactly +1 (open) or -1 (close)."""
    cmd = torch.where(out > 0, 1.0, -1.0).to(out.dtype)
    return torch.where(gripper_mask(out.shape[-1], out.device), cmd, out)


def horizon_weights(chunk, cfg: LossCfg, device=None):
    w = torch.ones(chunk, device=device)
    w[cfg.replan_every:] = cfg.horizon_tail
    return w
