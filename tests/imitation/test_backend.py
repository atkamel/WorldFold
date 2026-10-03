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
