"""Backend switch on Isaac Sim (Phase I, I1.1). Runs only in the Isaac venv (.venv-isaac); skipped elsewhere.

One Isaac env per process, so every test here goes through EnvPool workers (spawned processes)."""

import importlib.util
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.isaac,
              pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="needs the Isaac venv")]


@pytest.fixture(scope="module")
def pool():
    from imitation.rollout import EnvPool
    p = EnvPool(2, {"backend": "isaac"})
    yield p
    t0 = time.time()
    p.close()
    assert time.time() - t0 < 60, "Isaac workers did not shut down"
    assert not any(proc.is_alive() for proc in p.procs)


def test_venv_has_no_mujoco():
    assert importlib.util.find_spec("mujoco") is None


def test_reset_and_steps_give_finite_139d(pool):
    from imitation.spec import ACTION_DIM, OBS_DIM
    (obs, info, meta), = pool.call([0], "reset", [(100000, None, False)])
    assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs))
    assert meta["max_steps"] == 400
    offset = meta["domain_params"]["cloth_offset_xy"]
    assert all(abs(v) <= 0.01 + 1e-9 for v in offset)
    rng = np.random.default_rng(0)
    for _ in range(20):
        (obs, r, term, trunc, info, act), = pool.call([0], "step", [rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)])
        assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs))
        if term or trunc:
            break


def test_both_workers_independent(pool):
    outs = pool.call([0, 1], "reset", [(100001, None, False), (100002, None, False)])
    a, b = outs[0][0], outs[1][0]
    assert not np.array_equal(a, b)        # different seeds, different cloth starts
