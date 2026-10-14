"""Phase Q1 loss v2: gripper classification head, chunk-mask weights, checkpoint candidates."""

import json

import numpy as np
import pytest
import torch

from imitation.data.loader import WindowSampler
from imitation.data.schema import DatasetWriter
from imitation.policies.common import build_policy, load_policy, save_policy
from imitation.policies.loss import GRIPPER_ACTION_DIMS, LossCfg, gripper_from_logits, horizon_weights, split_loss
from tests.imitation.test_data_and_policies import A, D, make_episode

G = list(GRIPPER_ACTION_DIMS)


def _episode(seed, T=30):
    ep = make_episode(seed, T)
    ep.actions[:, G] = np.where(np.arange(T)[:, None] < T // 2, -1.0, 1.0)   # close, then open at T/2
    return ep


def test_gripper_dims_match_the_dataset_padding_dims():
    from imitation.data.dataset import GRIPPER_DIMS
    assert tuple(GRIPPER_DIMS) == GRIPPER_ACTION_DIMS


def test_gripper_logits_map_to_exact_commands_outside_the_hold_band():
    # the env holds for |a| <= 0.3 (imitation.md 3); the head must always command a side, never the band
    out = gripper_from_logits(torch.tensor([[[0.0] * 5 + [2.0] + [0.0] * 5 + [-0.01]]]))
    assert out[0, 0, 5] == 1.0 and out[0, 0, 11] == -1.0
    assert out[0, 0, :5].abs().max() == 0.0


@pytest.mark.parametrize("kind", ["chunk_mlp", "diffusion"])
def test_baseline_config_is_unchanged_and_head_roundtrips(kind, tmp_path):
    base = build_policy(kind, obs_dim=D, action_dim=A, obs_horizon=2, chunk=16)
    assert "gripper_head" not in base.config()          # pre-Q1 checkpoints and run.json stay as they were
    torch.manual_seed(0)
    p = build_policy(kind, obs_dim=D, action_dim=A, obs_horizon=2, chunk=16, gripper_head=True)
    obs, acts = torch.randn(3, 2, D), torch.rand(3, 16, A) * 2 - 1
    acts[..., G] = torch.sign(acts[..., G])
    loss = p.compute_loss(obs, acts, torch.ones(3, 16))
    loss.backward()
    assert torch.isfinite(loss)
    out = p.predict(obs.numpy())
    assert set(np.unique(out[..., G])) <= {-1.0, 1.0} and np.abs(out).max() <= 1.0
    save_policy(p, tmp_path / "p.pt")
    q = load_policy(tmp_path / "p.pt", device="cpu")
    assert q.gripper_head and q.config()["gripper_head"]


def test_chunk_mlp_baseline_loss_is_the_plain_masked_l1():
    torch.manual_seed(0)
    p = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16, dropout=0.0).eval()
    obs, acts, mask = torch.randn(4, 2, D), torch.rand(4, 16, A) * 2 - 1, (torch.rand(4, 16) > 0.3).float()
    ref = ((p(obs) - acts).abs().mean(-1) * mask).sum() / mask.sum()
    assert torch.equal(p.compute_loss(obs, acts, mask), ref)


def test_split_loss_is_l1_on_joints_plus_bce_on_gripper():
    pred, tgt = torch.zeros(1, 1, A), torch.zeros(1, 1, A)
    tgt[..., G] = 1.0
    # joints exact, gripper logits 0 -> BCE = log 2 per gripper dim
    assert torch.isclose(split_loss(pred, tgt)[0, 0], torch.tensor(np.log(2.0), dtype=torch.float32))


def test_horizon_weights_and_phase_flags():
    cfg = LossCfg(horizon_tail=0.5, replan_every=8)
    w = horizon_weights(16, cfg)
    assert w[:8].eq(1).all() and w[8:].eq(0.5).all()
    ep = _episode(0)
    ep.fold_score[:] = 0.0
    ep.fold_score[25:] = 0.9                            # placement phase at the end
    s = WindowSampler([ep], obs_horizon=2, chunk=4, device="cpu")
    flags = s.phase_flags(0.8).numpy()
    t = s.dense_t.numpy()
    # samples whose 4-step target crosses the switch at t = 15 (t = 12..14), or in the placement phase (t >= 25)
    expect = ((t >= 12) & (t <= 14)) | (t >= 25)
    np.testing.assert_array_equal(flags, expect)


def test_train_with_loss_v2_writes_candidates_and_records_the_config(tmp_path):
    from imitation.train import train
    w = DatasetWriter(tmp_path, "v1")
    for s in range(4):
        w.add(_episode(s))
    w.freeze()
    cfg = {"gripper_head": True, "horizon_tail": 0.5, "phase_weight": 3.0}
    ckpt, info = train("chunk_mlp", "v1", tmp_path / "run", root=tmp_path, steps=40, batch=16, eval_every=20,
                       save_every=20, loss_cfg=cfg, log=lambda *a: None)
    run = tmp_path / "run"
    assert (run / "best_val.pt").exists() and (run / "step_20.pt").exists() and (run / "step_40.pt").exists()
    rec = json.loads((run / "run.json").read_text())["train"]
    assert rec["loss_cfg"]["gripper_head"] and rec["loss_cfg"]["phase_weight"] == 3.0
    assert rec["best_val"]["step"] in (20, 40)
    assert load_policy(ckpt, device="cpu").gripper_head


def test_loss_cfg_parses_json_and_key_value_pairs():
    kv = LossCfg.parse("gripper_head=true,horizon_tail=0.5,phase_weight=3")
    assert kv == LossCfg.parse('{"gripper_head": true, "horizon_tail": 0.5, "phase_weight": 3}')
    assert kv.gripper_head and kv.phase_weight == 3 and LossCfg.parse("").is_baseline


def test_diffusion_head_keeps_the_full_denoising_loss():
    # the gripper head adds BCE on top; the epsilon loss still covers every dim (Q1 regression: dropping the gripper
    # dims from it corrupted the joint channels)
    torch.manual_seed(0)
    base = build_policy("diffusion", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16)
    head = build_policy("diffusion", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16, gripper_head=True)
    head.load_state_dict(base.state_dict(), strict=False)
    obs, acts = torch.randn(4, 2, D), torch.sign(torch.rand(4, 16, A) * 2 - 1)
    torch.manual_seed(1)
    lb = base.compute_loss(obs, acts, torch.ones(4, 16))
    torch.manual_seed(1)
    lh = head.compute_loss(obs, acts, torch.ones(4, 16))
    assert lh > lb            # the same epsilon loss (same seed, same weights) plus a positive BCE term
