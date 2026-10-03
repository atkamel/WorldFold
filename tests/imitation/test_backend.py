"""Backend switch (Phase I, I1.1), sim-free parts: factory dispatch, lazy imports, the NaN guard."""

import subprocess
import sys

import numpy as np
import pytest

# imitation.tasks is imported inside the tests: cloth_fold_rl.fold_env puts mujuco/ first on sys.path, whose
# tests/ package then shadows this repo's `tests` namespace for test modules collected afterwards
# (they import `tests.imitation...`). Known defect, docs/status.md.


def test_backends_listed():
    from imitation.tasks.half_fold import BACKENDS
    assert BACKENDS == ("mujoco", "isaac")


def test_unknown_backend_rejected():
    from imitation.tasks.half_fold import make_env
    with pytest.raises(ValueError):
        make_env(backend="bullet")


def test_isaac_dict_mode_waits_for_the_rig():
    from imitation.tasks.half_fold import make_env
    with pytest.raises(NotImplementedError):
        make_env(backend="isaac", obs_mode="dict")


@pytest.mark.parametrize("module", ["imitation.rollout", "imitation.evaluate", "imitation.dagger",
                                    "imitation.train", "imitation.teachers", "imitation.isaac_runtime"])
def test_pipeline_modules_import_without_mujoco(module):
    # what the Isaac venv sees: no mujoco, no so101_nexus, no mjviser
    code = ("import sys\nfor m in ('mujoco', 'so101_nexus', 'mjviser'):\n    sys.modules[m] = None\n"
            f"import {module}\nimport imitation.teachers as t\nt.PolicyTeacher\nprint('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_finite_guard_keeps_last_finite_obs():
    from imitation.rollout import _finite_or
    last = np.zeros(3, dtype=np.float32)
    bad = np.array([0.0, np.nan, 1.0], dtype=np.float32)
    assert _finite_or(bad, last) is last
    good = np.ones(3, dtype=np.float32)
    assert _finite_or(good, last) is good
    assert _finite_or(bad, None) is bad            # nothing to fall back to at reset
    images = {"state": bad, "main": np.zeros((3, 4, 4), np.uint8)}
    assert _finite_or(images, last) is last


# ---- CLI --backend plumbing (argparse level only; nothing is simulated) ----

_CLI_ARGS = {"imitation.data.collect": [], "imitation.evaluate": ["--ckpt", "expert"],
             "imitation.dagger": ["--init", "x.pt", "--out", "o"], "imitation.check_resync": [],
             "imitation.benchmark_expert": []}


@pytest.mark.parametrize("module", list(_CLI_ARGS))
def test_cli_parsers_accept_backend(module):
    import importlib
    build_parser = importlib.import_module(module).build_parser
    assert build_parser().parse_args(_CLI_ARGS[module]).backend == "mujoco"
    assert build_parser().parse_args(_CLI_ARGS[module] + ["--backend", "isaac"]).backend == "isaac"
    with pytest.raises(SystemExit):
        build_parser().parse_args(_CLI_ARGS[module] + ["--backend", "bullet"])


def test_workers_default_follows_backend():
    from imitation.evaluate import resolve_workers
    from imitation.isaac_runtime import N_ISAAC
    assert resolve_workers("mujoco", None) == 14
    assert resolve_workers("isaac", None) == N_ISAAC
    assert resolve_workers("isaac", 5) == 5


def test_collect_config_records_isaac_stack():
    from imitation.data.collect import build_config, build_parser, parse_cameras
    cfg = build_config(build_parser().parse_args(["--backend", "isaac"]))
    assert cfg["backend"] == "isaac" and "isaac" in cfg["teacher"]
    stack = cfg["isaac_stack"]
    assert stack["isaacsim"] == "5.1.0" and len(stack["knobs_sha256"]) == 64
    int(stack["knobs_sha256"], 16)
    mj = build_config(build_parser().parse_args([]))
    assert mj["backend"] == "mujoco" and "isaac_stack" not in mj
    assert parse_cameras("main=128,left_wrist_cam=64") == {"main": 128, "left_wrist_cam": 64}
    assert parse_cameras(None) is None


@pytest.mark.parametrize("module", ["imitation.check_resync", "imitation.benchmark_expert"])
def test_isaac_expert_clis_refuse_for_now(module):
    code = f"import sys\nsys.argv = ['x', '--backend', 'isaac']\nfrom {module} import main\nmain()"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and "I2.1" in out.stderr, out.stderr[-1000:]


def test_pool_start_failure_raises_instead_of_hanging():
    # a worker that fails before its "ready" (here: an unknown backend) must surface as an error, and the
    # pool must not hang in close() (I1.1 review: the parent closes its copy of the child's pipe end)
    import time
    from imitation.rollout import EnvPool
    t0 = time.time()
    with pytest.raises(RuntimeError, match="failed to start"):
        EnvPool(1, {"backend": "bogus"})
    assert time.time() - t0 < 60