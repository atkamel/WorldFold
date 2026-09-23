"""Schema, dataset store, sample building and policy shapes (no simulator needed)."""

import json

import numpy as np
import pytest
import torch

from imitation.data.dataset import Normalizer, build_samples, split_episodes
from imitation.data.schema import (ACTOR_PERTURB, ACTOR_STUDENT, ACTOR_TEACHER, DatasetWriter, Episode,
                                   load_dataset, validate_episode)
from imitation.policies.common import build_policy, load_policy, save_policy

D, A = 141, 12


def make_episode(seed, T=30, source="expert", actor=None, labels=None):
    rng = np.random.default_rng(seed)
    label_steps, lab = (np.zeros(0, np.int32), np.zeros((0, 0, 0), np.float32)) if labels is None else labels
    return Episode(obs=rng.normal(size=(T, D)).astype(np.float32),
                   actions=rng.uniform(-1, 1, (T, A)).astype(np.float32),
                   rewards=np.zeros(T, np.float32), stage=np.zeros(T, np.int8),
                   fold_score=np.linspace(0, 1, T).astype(np.float32), grasped=np.zeros((T, 2), bool),
                   actor=np.full(T, ACTOR_TEACHER, np.int8) if actor is None else actor,
                   final_obs=np.zeros(D, np.float32),
                   meta={"seed": seed, "source": source, "success": True, "termination_reason": "success"},
                   label_steps=label_steps, labels=lab)


def test_validator_catches_bad_episodes():
    assert validate_episode(make_episode(0), D, A) == []
    bad = make_episode(1)
    bad.actions[3, 2] = 1.5
    bad.obs[0, 0] = np.nan
    errs = validate_episode(bad, D, A)
    assert any("outside" in e for e in errs) and any("non-finite" in e for e in errs)


def test_store_is_write_once_aggregates_and_detects_tampering(tmp_path):
    w = DatasetWriter(tmp_path, "v1")
    for s in range(3):
        w.add(make_episode(s), D, A)
    m1 = w.freeze()
    with pytest.raises(FileExistsError):
        DatasetWriter(tmp_path, "v1")
    w2 = DatasetWriter(tmp_path, "v2", parent="v1")
    w2.add(make_episode(10, source="dagger"), D, A)
    m2 = w2.freeze()
    assert (m1["n_episodes"], m2["n_episodes"], m2["n_new"]) == (3, 4, 1)
    _, eps = load_dataset(tmp_path, "v2")
    assert sorted(e.meta["seed"] for e in eps) == [0, 1, 2, 10]
    # silently editing frozen data must be caught
    victim = tmp_path / json.loads((tmp_path / "v1" / "manifest.json").read_text())["episodes"][0]["file"]
    make_episode(99).save(victim)
    with pytest.raises(ValueError, match="hash"):
        load_dataset(tmp_path, "v2")


def test_split_is_by_seed_and_normalizer_uses_train_only():
    eps = [make_episode(s) for s in range(20)] + [make_episode(s, source="dagger") for s in range(5)]
    train, val = split_episodes(eps, val_fraction=0.2)
    assert not {e.meta["seed"] for e in train} & {e.meta["seed"] for e in val}
    for e in val:
        e.obs += 1000.0                       # would drag the mean if val leaked in
    norm = Normalizer.fit(train)
    assert np.abs(norm.mean).max() < 1.0


def test_samples_mask_non_teacher_actions_and_pad_the_end():
    T, K = 12, 4
    actor = np.full(T, ACTOR_TEACHER, np.int8)
    actor[6:8] = ACTOR_PERTURB
    ep = make_episode(0, T=T, actor=actor)
    X, Y, M, W = build_samples([ep], obs_horizon=2, chunk=K)
    assert len(X) == T - 2 and (W == 0).all()
    # chunk starting at t=4 covers steps 4..7; 6 and 7 were the perturbation
    np.testing.assert_array_equal(M[4], [1, 1, 0, 0])
    # the last chunk runs past the end: padded, supervised, joints zero, gripper kept
    last = Y[-1]
    np.testing.assert_array_equal(M[-1], [1, 1, 1, 1])
    assert np.all(last[1:, :5] == 0) and np.all(last[1:, 5] == ep.actions[-1, 5])
    # history at t=0 repeats the first observation
    np.testing.assert_array_equal(X[0, 0], X[0, 1])


def test_dagger_labels_become_samples_and_student_steps_do_not():
    T, K = 10, 4
    labels = (np.array([0, 5], np.int32), np.full((2, K, A), 0.5, np.float32))
    ep = make_episode(0, T=T, source="dagger", actor=np.full(T, ACTOR_STUDENT, np.int8), labels=labels)
    X, Y, M, W = build_samples([ep], obs_horizon=2, chunk=K)
    assert len(X) == 2 and (W == 1).all() and np.all(Y == 0.5)


@pytest.mark.parametrize("kind", ["chunk_mlp", "diffusion"])
def test_policy_shapes_loss_and_checkpoint_roundtrip(kind, tmp_path):
    torch.manual_seed(0)
    p = build_policy(kind, obs_dim=D, action_dim=A, obs_horizon=2, chunk=16)
    obs = torch.randn(3, 2, D)
    acts = torch.rand(3, 16, A) * 2 - 1
    loss = p.compute_loss(obs, acts, torch.ones(3, 16))
    loss.backward()
    assert torch.isfinite(loss)
    out = p.predict(obs.numpy())
    assert out.shape == (3, 16, A) and np.abs(out).max() <= 1.0
    save_policy(p, tmp_path / "p.pt")
    q = load_policy(tmp_path / "p.pt", device="cpu")
    if kind == "chunk_mlp":                   # deterministic: identical outputs after reload
        np.testing.assert_allclose(q.predict(obs.numpy()), p.to("cpu").predict(obs.numpy()), atol=1e-6)
