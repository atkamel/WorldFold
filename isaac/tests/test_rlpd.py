"""isaac.rlpd without Isaac: buffer mechanics, RLPD symmetric sampling, REDQ terminal bootstrap, and a toy
1-D placement task that checks the SAC loop actually learns something in a few thousand env steps."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isaac import rlpd as R   # noqa: E402


def test_initial_policy_outputs_zero():
    agent = R.RLPDAgent(obs_dim=5, act_dim=3, hidden=32, seed=0)
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(8, 5)).astype(np.float32)
    act = agent.act(obs, deterministic=True)
    assert act.shape == (8, 3)
    assert np.all(act == 0.0)

    obs1 = rng.normal(size=5).astype(np.float32)
    act1 = agent.act(obs1, deterministic=True)
    assert act1.shape == (3,)
    assert np.all(act1 == 0.0)


def test_buffer_batch_and_single_add_and_circular():
    buf = R.ReplayBuffer(obs_dim=2, act_dim=1, capacity=10, seed=0)

    # single transition
    buf.add(np.array([1.0, 2.0]), np.array([0.5]), 1.0, np.array([1.1, 2.1]), False)
    assert len(buf) == 1

    # batch of transitions
    obs = np.ones((4, 2), dtype=np.float32)
    act = np.ones((4, 1), dtype=np.float32)
    rew = np.ones(4, dtype=np.float32)
    next_obs = np.ones((4, 2), dtype=np.float32) * 2
    terminal = np.zeros(4, dtype=np.float32)
    buf.add(obs, act, rew, next_obs, terminal)
    assert len(buf) == 5

    # push past capacity to force wraparound
    for i in range(20):
        buf.add(np.array([float(i), float(i)]), np.array([float(i)]), float(i),
                 np.array([float(i) + 1, float(i) + 1]), bool(i % 7 == 0))
    assert len(buf) == 10  # capped at capacity

    batch = buf.sample(6, device="cpu")
    assert set(batch.keys()) == {"obs", "act", "rew", "next_obs", "terminal"}
    assert batch["obs"].shape == (6, 2)
    assert batch["act"].shape == (6, 1)
    assert batch["rew"].shape == (6, 1)
    assert batch["next_obs"].shape == (6, 2)
    assert batch["terminal"].shape == (6, 1)
    assert batch["terminal"].dtype == torch.float32
    assert batch["obs"].dtype == torch.float32

    # save / load round trip
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "buf.npz")
        buf.save(path)
        buf2 = R.ReplayBuffer(obs_dim=2, act_dim=1, capacity=10, seed=1)
        buf2.load(path)
        assert len(buf2) == len(buf)
        assert np.allclose(buf2.obs, buf.obs)
        assert np.allclose(buf2.act, buf.act)
        assert np.allclose(buf2.rew, buf.rew)
        assert np.allclose(buf2.next_obs, buf.next_obs)
        assert np.allclose(buf2.terminal, buf.terminal)


def test_symmetric_sampling_uses_both_buffers():
    online = R.ReplayBuffer(obs_dim=1, act_dim=1, capacity=100, seed=0)
    prior = R.ReplayBuffer(obs_dim=1, act_dim=1, capacity=100, seed=1)

    # distinguishable rewards: online always +1, prior always -1
    for i in range(50):
        online.add(np.array([float(i)]), np.array([0.0]), 1.0, np.array([float(i)]), False)
        prior.add(np.array([float(i)]), np.array([0.0]), -1.0, np.array([float(i)]), False)

    batch = R._mixed_batch(online, prior, 100, device="cpu")
    rew = batch["rew"].numpy().ravel()
    n_online = int((rew == 1.0).sum())
    n_prior = int((rew == -1.0).sum())
    assert n_online == 50
    assert n_prior == 50
    assert n_online + n_prior == 100


def test_targets_ignore_bootstrap_on_terminal():
    agent = R.RLPDAgent(obs_dim=2, act_dim=1, hidden=16, seed=0)
    next_obs = torch.zeros((4, 2), dtype=torch.float32)
    rew = torch.full((4, 1), 5.0, dtype=torch.float32)
    terminal = torch.ones((4, 1), dtype=torch.float32)
    y = agent._target(next_obs, rew, terminal)
    assert torch.allclose(y, rew)


def test_save_load_roundtrip():
    agent = R.RLPDAgent(obs_dim=4, act_dim=2, hidden=32, seed=0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(5, 4)).astype(np.float32)

    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "agent.pt")
        agent.save(path)
        agent2 = R.RLPDAgent(obs_dim=4, act_dim=2, hidden=32, seed=99)  # different seed/init
        agent2.load(path)

    a1 = agent.act(obs, deterministic=True)
    a2 = agent2.act(obs, deterministic=True)
    assert np.allclose(a1, a2)

    a1s = agent.act(obs, deterministic=False)
    torch.manual_seed(0)
    # not asserting stochastic equality here; deterministic path is the meaningful round trip check
    assert a1s.shape == a2.shape


# ---------------------------------------------------------------------------
# Toy 1-D placement task
# ---------------------------------------------------------------------------

def _phi(x):
    return -np.log(np.abs(x) + 0.05)


def _toy_step(x, a, rng, noise_scale=0.02):
    a = np.clip(a, -1.0, 1.0)
    noise = rng.normal(0.0, noise_scale, size=np.shape(x))
    return np.clip(x + 0.1 * a + noise, -1.0, 1.0)


def _toy_reward(x, x_next, done):
    r = _phi(x_next) - _phi(x)
    if done:
        r = r + _phi(x_next)
    return r


HORIZON = 20


def _collect_prior(n_episodes, rng):
    buf = R.ReplayBuffer(obs_dim=1, act_dim=1, capacity=n_episodes * HORIZON, seed=int(rng.integers(1 << 30)))
    for _ in range(n_episodes):
        x = rng.uniform(-0.8, 0.8)
        for t in range(HORIZON):
            a = np.clip(-2.0 * x, -1.0, 1.0) * 0.3 + rng.normal(0.0, 0.2)
            a = np.clip(a, -1.0, 1.0)
            x_next = _toy_step(x, a, rng)
            done = t == HORIZON - 1
            r = _toy_reward(x, x_next, done)
            buf.add(np.array([x], dtype=np.float32), np.array([a], dtype=np.float32), r,
                     np.array([x_next], dtype=np.float32), False)
            x = x_next
    return buf


def _eval_policy(act_fn, n_episodes, rng):
    finals = []
    for _ in range(n_episodes):
        x = rng.uniform(-0.8, 0.8)
        for _ in range(HORIZON):
            a = act_fn(x)
            x = _toy_step(x, a, rng, noise_scale=0.0)
        finals.append(abs(x))
    return float(np.mean(finals))


def test_learns_toy_placement():
    # pin single-threaded CPU math: multi-threaded reductions reorder floating-point sums, and over
    # thousands of chained gradient updates that nondeterminism is enough to change the outcome
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_toy_placement()
    finally:
        torch.set_num_threads(prev_threads)


def _run_toy_placement():
    rng = np.random.default_rng(0)
    prior = _collect_prior(10, rng)  # ~200 transitions
    assert len(prior) == 200

    agent = R.RLPDAgent(obs_dim=1, act_dim=1, hidden=64, n_critics=5, n_target=2, lr=1e-3,
                         target_entropy_scale=0.5, init_std=0.3, seed=1)
    online = R.ReplayBuffer(obs_dim=1, act_dim=1, capacity=5000, seed=0)

    train_rng = np.random.default_rng(1)
    n_env_steps = 1500
    batch_size = 128
    utd = 10

    x = train_rng.uniform(-0.8, 0.8)
    t = 0
    for step in range(n_env_steps):
        a = agent.act(np.array([x], dtype=np.float32), deterministic=False)
        x_next = _toy_step(x, a[0], train_rng)
        done = t == HORIZON - 1
        r = _toy_reward(x, x_next, done)
        online.add(np.array([x], dtype=np.float32), np.array([a[0]], dtype=np.float32), r,
                   np.array([x_next], dtype=np.float32), False)
        x = x_next
        t += 1
        if done:
            t = 0
            x = train_rng.uniform(-0.8, 0.8)

        if len(online) >= batch_size:
            agent.update(online, prior=prior, batch_size=batch_size, utd=utd)

    eval_rng = np.random.default_rng(2)
    learned_mean = _eval_policy(lambda xx: agent.act(np.array([xx], dtype=np.float32), deterministic=True)[0],
                                 50, eval_rng)

    eval_rng2 = np.random.default_rng(2)
    zero_mean = _eval_policy(lambda xx: 0.0, 50, eval_rng2)

    assert learned_mean < 0.05
    assert learned_mean < zero_mean
