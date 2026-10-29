"""Schema, dataset store, sample building and policy shapes (no simulator needed)."""

import json

import numpy as np
import pytest
import torch

from imitation.data.dataset import Normalizer, bc_episodes, build_samples, split_episodes, teacher_samples
from imitation.data.schema import (ACTOR_PERTURB, ACTOR_STUDENT, ACTOR_TEACHER, DatasetWriter, Episode,
                                   load_dataset, validate_episode)
from imitation.policies.common import build_policy, load_policy, save_policy
from imitation.spec import ACTION_DIM, OBS_DIM

D, A = OBS_DIM, ACTION_DIM


def make_episode(seed, T=30, source="expert", actor=None, labels=None):
    rng = np.random.default_rng(seed)
    label_steps, lab = (np.zeros(0, np.int32), np.zeros((0, 0, 0), np.float32)) if labels is None else labels
    return Episode(obs=rng.normal(size=(T, D)).astype(np.float32),
                   actions=rng.uniform(-1, 1, (T, A)).astype(np.float32),
                   rewards=np.zeros(T, np.float32), stage=np.zeros(T, np.int8),
                   fold_score=np.linspace(0, 1, T).astype(np.float32), grasped=np.zeros((T, 2), bool),
                   actor=np.full(T, ACTOR_TEACHER, np.int8) if actor is None else actor,
                   final_obs=np.zeros(D, np.float32),
                   terminated=np.arange(T) == T - 1, truncated=np.zeros(T, bool),
                   discount=np.r_[np.ones(T - 1), 0.0].astype(np.float32),
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


def test_writer_streams_and_resumes_after_a_kill(tmp_path):
    w = DatasetWriter(tmp_path, "v1")
    for s in range(3):
        w.add(make_episode(s), D, A)
    del w                                            # process killed before freeze
    # a torn write from the killed process: npz on disk, never journaled
    make_episode(7).save(tmp_path / "v1" / "episodes" / "expert_s7.npz")
    with pytest.raises(FileExistsError, match="resume"):
        DatasetWriter(tmp_path, "v1")
    w = DatasetWriter(tmp_path, "v1", resume=True)
    assert w.done_seeds == {0, 1, 2}
    assert not (tmp_path / "v1" / "episodes" / "expert_s7.npz").exists()
    w.add(make_episode(3), D, A)
    m = w.freeze()
    assert m["n_episodes"] == m["n_new"] == 4
    _, eps = load_dataset(tmp_path, "v1")
    assert [e.meta["seed"] for e in eps] == [0, 1, 2, 3]


def test_writer_rejects_duplicate_seed(tmp_path):
    w = DatasetWriter(tmp_path, "v1")
    w.add(make_episode(0), D, A)
    with pytest.raises(ValueError, match="already"):
        w.add(make_episode(0), D, A)


def test_validator_checks_transition_flags():
    ep = make_episode(0)
    ep.terminated[3] = True                                  # terminal mid-episode
    assert any("last step" in e for e in validate_episode(ep, D, A))
    ep = make_episode(0)
    ep.terminated[-1] = False                                # episode ends with no flag
    assert any("exactly one" in e for e in validate_episode(ep, D, A))
    ep = make_episode(0)
    ep.discount[-1] = 1.0                                    # a true terminal must not bootstrap
    assert any("discount" in e for e in validate_episode(ep, D, A))
    ep = make_episode(0)
    ep.terminated[-1], ep.truncated[-1], ep.discount[-1] = False, True, 1.0
    assert validate_episode(ep, D, A) == []


def test_bc_drops_failed_expert_demos_but_keeps_dagger_labels():
    ok, bad, dag = make_episode(0), make_episode(1), make_episode(2, source="dagger")
    bad.meta["success"] = dag.meta["success"] = False
    kept, n_dropped = bc_episodes([ok, bad, dag])
    assert [e.meta["seed"] for e in kept] == [0, 2] and n_dropped == 1
    kept, n_dropped = bc_episodes([ok, bad, dag], allow_failures=True)
    assert len(kept) == 3 and n_dropped == 0


def test_obs_subset_mask_hides_excluded_dims_and_survives_checkpoint(tmp_path):
    from imitation.spec import OBS_SUBSETS
    policy = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=4)
    policy.set_obs_subset(OBS_SUBSETS["proprio"])
    obs = torch.randn(3, 2, D)
    moved = obs.clone()
    moved[..., 60] += 5.0                                   # a cloth dim, outside the subset
    torch.testing.assert_close(policy.normalize_obs(obs), policy.normalize_obs(moved))
    save_policy(policy, tmp_path / "p.pt")
    assert int(load_policy(tmp_path / "p.pt").obs_mask.sum()) == len(OBS_SUBSETS["proprio"])


def test_val_split_is_stable_when_episodes_are_added():
    base = [make_episode(s) for s in range(40)]
    _, val = split_episodes(base, val_fraction=0.2)
    grown = base + [make_episode(s, source="dagger") for s in range(50_000, 50_030)]
    _, val2 = split_episodes(grown, val_fraction=0.2)
    before = {e.meta["seed"] for e in val}
    assert before and before == {e.meta["seed"] for e in val2 if e.meta["seed"] < 40}


def test_dagger_teacher_steps_train_only_through_their_labels():
    T, K = 16, 4
    labels = (np.array([0, 8], np.int32), np.full((2, K, A), 0.5, np.float32))
    ep = make_episode(0, T=T, source="dagger", actor=np.full(T, ACTOR_TEACHER, np.int8), labels=labels)
    X, Y, M, W = build_samples([ep], obs_horizon=2, chunk=K)
    assert len(X) == 2 and (W == 1).all()


def test_truncated_episode_tail_is_not_padded_as_supervised():
    T, K = 12, 4
    ep = make_episode(0, T=T)
    ep.terminated[-1], ep.truncated[-1], ep.discount[-1] = False, True, 1.0
    X, Y, M, W = build_samples([ep], obs_horizon=2, chunk=K)
    np.testing.assert_array_equal(M[-1], [1, 0, 0, 0])


def test_images_are_stored_beside_the_episode_hashed_and_loaded_on_request(tmp_path):
    ep = make_episode(0, T=5)
    ep.images = {"main": np.random.default_rng(0).integers(0, 255, (5, 3, 8, 8), dtype=np.uint8)}
    w = DatasetWriter(tmp_path, "v1_img")
    w.add(ep, D, A)
    w.freeze()
    _, (plain,) = load_dataset(tmp_path, "v1_img")
    assert plain.images is None
    _, (withimg,) = load_dataset(tmp_path, "v1_img", images=True)
    np.testing.assert_array_equal(withimg.images["main"], ep.images["main"])
    img_file = next((tmp_path / "v1_img" / "episodes").glob("*.img.npz"))
    np.savez_compressed(img_file, main=np.zeros((5, 3, 8, 8), np.uint8))
    with pytest.raises(ValueError, match="hash"):
        load_dataset(tmp_path, "v1_img", images=True)


def test_vision_policy_ignores_privileged_dims_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    p = build_policy("vision", obs_dim=D, action_dim=A, obs_horizon=2, chunk=4)
    obs = np.random.default_rng(0).normal(size=(2, 2, D)).astype(np.float32)
    imgs = [{c: np.random.default_rng(i).integers(0, 255, (3, s, s), dtype=np.uint8) for c, s in p.cameras}
            for i in range(2)]
    out = p.predict(obs, imgs)
    assert out.shape == (2, 4, A) and np.abs(out).max() <= 1.0
    moved = obs.copy()
    moved[..., 60] += 5.0                                    # cloth state is privileged: no effect
    np.testing.assert_allclose(p.predict(moved, imgs), out, atol=1e-6)
    t_imgs = {c: torch.as_tensor(np.stack([im[c] for im in imgs])) for c, _ in p.cameras}
    loss = p.compute_loss(torch.as_tensor(obs), torch.zeros(2, 4, A), torch.ones(2, 4), t_imgs)
    loss.backward()
    save_policy(p, tmp_path / "v.pt")
    np.testing.assert_allclose(load_policy(tmp_path / "v.pt", "cpu").predict(obs, imgs), out, atol=1e-5)


def test_teacher_samples_label_every_step_and_weight_student_episodes():
    class Const:
        obs_horizon = 1
        def predict(self, x):
            return np.full((len(x), 16, A), 0.25, np.float32)
    eps = [make_episode(0, T=5), make_episode(1, T=3, source="dagger")]
    X, Y, M, W, I = teacher_samples(eps, Const(), obs_horizon=2, chunk=4)
    assert len(X) == 8 and Y.shape == (8, 4, A) and np.all(Y == 0.25)
    assert list(W) == [0] * 5 + [1] * 3 and list(I) == list(range(8))


def test_padded_predict_is_independent_of_batch_composition():
    from imitation.rollout import padded_predict
    torch.manual_seed(0)
    p = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=4).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    x = np.random.default_rng(0).normal(size=(9, 2, D)).astype(np.float32)
    alone = padded_predict(p, x[:1])
    np.testing.assert_array_equal(padded_predict(p, x)[:1], alone)
    np.testing.assert_array_equal(padded_predict(p, x[:3])[:1], alone)


def test_folded_label_needs_placed_corners_and_open_grippers():
    from imitation.vision.success import CORNER_TO_GOAL, folded_labels
    ep = make_episode(0, T=3)
    ep.obs[:, CORNER_TO_GOAL] = 0.0
    ep.obs[0, CORNER_TO_GOAL.start] = 0.2          # step 0: a corner 20 cm off
    ep.grasped[1, 0] = True                          # step 1: still holding
    assert list(folded_labels(ep)) == [False, False, True]


def test_images_change_the_version_hash():
    import tempfile
    hashes = []
    for with_images in (False, True):
        root = tempfile.mkdtemp()
        ep = make_episode(0, T=4)
        if with_images:
            ep.images = {"main": np.zeros((4, 3, 8, 8), np.uint8)}
        w = DatasetWriter(root, "v")
        w.add(ep, D, A)
        hashes.append(w.freeze()["content_hash"])
    assert hashes[0] != hashes[1]


def test_policy_teacher_labels_are_the_checkpoint_chunk(tmp_path):
    from imitation.teachers import PolicyTeacher
    torch.manual_seed(0)
    p = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16)
    save_policy(p, tmp_path / "t.pt")
    teacher = PolicyTeacher(None, tmp_path / "t.pt")
    rng = np.random.default_rng(0)
    o0, o1 = rng.normal(size=(2, D)).astype(np.float32)
    teacher.reset()
    teacher.see(o0)
    teacher.see({"state": o1})
    want = load_policy(tmp_path / "t.pt", "cpu").predict(np.stack([o0, o1])[None])[0]
    np.testing.assert_allclose(teacher.label_chunk(None, 16), want, atol=1e-6)
    assert teacher.label_chunk(None, 20).shape == (20, A)


def _key_rows(X, Y, M, W, I):
    order = np.lexsort((I, W))
    return X[order], Y[order], M[order]


def test_lazy_sampler_reproduces_build_samples_exactly():
    from imitation.data.loader import WindowSampler
    T, K = 14, 4
    actor = np.full(T, ACTOR_TEACHER, np.int8)
    actor[6:8] = ACTOR_PERTURB
    a = make_episode(0, T=T, actor=actor)
    b = make_episode(1, T=9)
    b.terminated[-1], b.truncated[-1], b.discount[-1] = False, True, 1.0
    labels = (np.array([0, 5], np.int32), np.full((2, K, A), 0.5, np.float32))
    c = make_episode(2, T=10, source="dagger", actor=np.full(10, ACTOR_STUDENT, np.int8), labels=labels)
    eps = [a, b, c]
    want = _key_rows(*build_samples(eps, obs_horizon=2, chunk=K, index=True))
    s = WindowSampler(eps, 2, K, torch.device("cpu"))
    X, Y, M, _ = s.batch(torch.arange(len(s)))
    got = _key_rows(X.numpy(), Y.numpy(), M.numpy(), s.weight_tag, s.rows.numpy())
    for w, g in zip(want, got):
        np.testing.assert_allclose(g, w, atol=1e-6)


def test_lazy_sampler_teacher_mode_matches_teacher_samples():
    from imitation.data.loader import WindowSampler
    torch.manual_seed(0)
    teacher = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16).eval()
    eps = [make_episode(0, T=6), make_episode(1, T=4, source="dagger")]
    X0, Y0, M0, W0, I0 = teacher_samples(eps, teacher, obs_horizon=2, chunk=4)
    s = WindowSampler(eps, 2, 4, torch.device("cpu"), teacher=teacher)
    X, Y, M, _ = s.batch(torch.arange(len(s)))
    np.testing.assert_allclose(X.numpy(), X0, atol=1e-6)
    np.testing.assert_allclose(Y.numpy(), Y0, atol=1e-5)
    assert list(s.weight_tag) == list(W0)


def test_lazy_sampler_gathers_images_and_fits_per_camera_stats():
    from imitation.data.loader import WindowSampler
    ep = make_episode(0, T=5)
    ep.images = {"main": np.stack([np.full((3, 8, 8), 10 * t, np.uint8) for t in range(5)])}
    s = WindowSampler([ep], 2, 4, torch.device("cpu"), cameras=[("main", 8)])
    _, _, _, imgs = s.batch(torch.tensor([3]))
    assert int(imgs["main"][0, 0, 0, 0]) == 10 * int(s.rows[3])
    mean, std = s.image_stats()["main"]
    assert mean.shape == (3,) and std.shape == (3,)


def test_diffusion_sampling_is_a_deterministic_function_of_the_obs():
    from imitation.rollout import padded_predict
    torch.manual_seed(0)
    p = build_policy("diffusion", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16).eval()
    x = np.random.default_rng(0).normal(size=(5, 2, D)).astype(np.float32)
    alone = padded_predict(p, x[2:3])
    torch.randn(100)                                           # disturb the global RNG
    np.testing.assert_array_equal(padded_predict(p, x)[2:3], alone)


def test_policy_teacher_labels_on_gpu_in_the_controller():
    from imitation.rollout import PolicyController, padded_predict
    torch.manual_seed(0)
    student = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16).eval()
    teacher = build_policy("chunk_mlp", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16).eval()
    c = PolicyController(student, beta=1.0, label=True, teacher_policy=teacher)
    assert not c.needs_labels                       # workers are never asked to label
    rngs = {0: np.random.default_rng(0), 1: np.random.default_rng(1)}
    x = np.random.default_rng(0).normal(size=(2, 2, D)).astype(np.float32)
    plans = c.plan([0, 1], x, None, rngs)
    want = padded_predict(teacher, x)
    np.testing.assert_allclose(plans[0].label, want[0], atol=1e-6)
    np.testing.assert_allclose(plans[1].actions, want[1][:8], atol=1e-6)      # beta = 1: teacher executes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_diffusion_sampling_matches_eager_exactly():
    from imitation.policies.common import ChunkPolicy
    torch.manual_seed(0)
    p = build_policy("diffusion", obs_dim=D, action_dim=A, obs_horizon=2, chunk=16).cuda().eval()
    x = np.random.default_rng(0).normal(size=(16, 2, D)).astype(np.float32)
    eager = ChunkPolicy.predict(p, x)
    np.testing.assert_array_equal(p.predict(x), eager)          # first call captures the graph
    np.testing.assert_array_equal(p.predict(x * 0.5), ChunkPolicy.predict(p, x * 0.5))   # replay, new input
