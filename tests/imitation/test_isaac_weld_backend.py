"""isaac_weld backend (Phase W2 plumbing), sim-free: eval sets and perturbations match MuJoCo, factory dispatch."""

import importlib

import numpy as np
import pytest

_CLI_ARGS = {"imitation.data.collect": [], "imitation.evaluate": ["--ckpt", "expert"],
             "imitation.dagger": ["--init", "x.pt", "--out", "o"], "imitation.check_resync": [],
             "imitation.benchmark_expert": []}


def _sig(name, n, backend):
    from imitation.seeds import eval_set
    seeds, reset_fn, perturb_fn = eval_set(name, n, backend)
    resets = None if reset_fn is None else [np.asarray(reset_fn(s)["cloth_pose"]).tolist() for s in seeds]
    perts = None
    if perturb_fn is not None:
        perts = [(lambda p: (p.t, p.k))(perturb_fn(s, np.random.default_rng(s))) for s in seeds]
    return seeds, resets, perts


@pytest.mark.parametrize("name", ["id_easy", "id_hard", "recovery"])
def test_eval_sets_match_mujoco(name):
    assert _sig(name, 24, "isaac_weld") == _sig(name, 24, "mujoco")


@pytest.mark.parametrize("name", ["id_hard", "recovery"])
def test_eval_sets_isaac_still_differs(name):
    assert _sig(name, 24, "isaac") != _sig(name, 24, "mujoco")


def test_recovery_perturbation_matches_mujoco():
    from imitation.data.collect import recovery_perturbation_for

    def draws(backend):
        fn = recovery_perturbation_for(backend, 1.0)
        return [(lambda p: (p.t, p.k))(fn(s, np.random.default_rng(s))) for s in range(30)]
    assert draws("isaac_weld") == draws("mujoco")
    assert draws("isaac") != draws("mujoco")


def test_is_isaac_truth_table():
    from imitation.tasks import is_isaac
    assert is_isaac("isaac") and is_isaac("isaac_weld")
    assert not is_isaac("mujoco") and not is_isaac("bullet")


def test_make_env_dispatch(monkeypatch):
    from imitation.tasks import half_fold
    import imitation.isaac_runtime as rt
    calls = []

    class Stop(Exception):
        pass

    def fake_base(steps=None, cameras=None, profile="lehome"):
        calls.append({"max_episode_steps": steps, "cameras": cameras, "profile": profile})
        return object()

    class FakeEnv:
        def __init__(self, **kw):
            calls[-1]["cloth_jitter"] = kw["cloth_jitter"]

    monkeypatch.setattr(rt, "make_isaac_base", fake_base)
    monkeypatch.setattr(half_fold, "HalfFoldEnv", FakeEnv)
    half_fold.make_env("isaac_weld")
    assert calls[-1] == {"max_episode_steps": 250, "cameras": None, "profile": "weld", "cloth_jitter": 0.025}
    half_fold.make_env("isaac")
    assert calls[-1] == {"max_episode_steps": 400, "cameras": None, "profile": "lehome", "cloth_jitter": 0.01}


@pytest.mark.parametrize("module", list(_CLI_ARGS))
def test_cli_parsers_accept_isaac_weld(module):
    build_parser = importlib.import_module(module).build_parser
    assert build_parser().parse_args(_CLI_ARGS[module] + ["--backend", "isaac_weld"]).backend == "isaac_weld"


def test_collect_config_records_backend():
    from imitation.data.collect import build_config, build_parser
    cfg = build_config(build_parser().parse_args(["--backend", "isaac_weld"]))
    assert cfg["backend"] == "isaac_weld" and "isaac_stack" in cfg and "isaac_weld" in cfg["teacher"]
