"""Phase F3b perturbation suite (imitation.rollout.Perturbation kinds, imitation.seeds.perturbation_fn), sim-free: the
pure transform, per-seed determinism, and the driver on the fake batch -- a knocked arm keeps its jaw command, a drop
opens only the holding jaw, noise kinds keep the teacher in control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from imitation.data.schema import ACTOR_PERTURB, ACTOR_TEACHER
from imitation.rollout import ARM_DIMS, GRIPPER_DIMS, ExpertController, Perturbation, apply_perturbation
from imitation.seeds import EVAL_SEED_BASE, PERTURB_SETS, eval_set, perturbation_fn

_spec = importlib.util.spec_from_file_location("worldfold_test_lockstep", Path(__file__).with_name("test_lockstep.py"))
_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lock)


def test_apply_perturbation_touches_only_what_it_says():
    a = np.full(12, 0.5, dtype=np.float32)
    out = apply_perturbation(a, {"mul": 1.5})
    assert np.allclose(out[ARM_DIMS], 0.75) and np.allclose(out[list(GRIPPER_DIMS)], 0.5)
    out = apply_perturbation(a, {"add": [0.8] * len(ARM_DIMS), "set": {5: 1.0}})
    assert np.allclose(out[ARM_DIMS], 1.0)            # clipped
    assert out[5] == 1.0 and out[11] == 0.5
    assert np.array_equal(a, np.full(12, 0.5, dtype=np.float32))     # input untouched


@pytest.mark.parametrize("kind", PERTURB_SETS)
def test_perturbation_is_a_function_of_the_seed(kind):
    fn = perturbation_fn(kind, "isaac_friction")
    a = fn(321, np.random.default_rng(0))
    b = fn(321, np.random.default_rng(99))             # the driver's rng does not matter
    assert a == b and a.kind == kind
    if kind == "overshoot":
        assert 1.2 <= a.gain <= 1.4
    if kind in ("knock_arm", "drop"):
        assert 35 <= a.t < 140


def test_suite_sets_are_disjoint_and_named():
    seen = set()
    for name in PERTURB_SETS:
        seeds, opts, fn = eval_set(name, 50, "isaac_friction")
        assert seeds[0] == EVAL_SEED_BASE[name] and opts is None and fn is not None
        assert not seen & set(seeds)
        seen |= set(seeds)


def _knock_arm(seed, rng):
    return Perturbation(t=2, k=3, kind="knock_arm") if seed % 2 else None


def test_knock_arm_holds_the_jaw_and_episodes_depend_only_on_seed():
    ref, _ = _lock._run(3, 1, _lock.SEEDS, ExpertController(), perturb_fn=_knock_arm)
    eps, _ = _lock._run(1, 3, _lock.SEEDS, ExpertController(), perturb_fn=_knock_arm)
    _lock._same(ref, eps)
    for e in ref:
        if e.meta["seed"] % 2 == 0:
            assert e.meta["perturb_kind"] is None
            continue
        assert e.meta["perturb_kind"] == "knock_arm"
        idx = np.flatnonzero(e.actor == ACTOR_PERTURB)
        assert list(idx) == [2, 3, 4][:len(idx)] and len(idx)
        for t in idx:                                 # the jaw command is the one executed just before
            assert np.array_equal(e.actions[t][list(GRIPPER_DIMS)], e.actions[t - 1][list(GRIPPER_DIMS)])


def test_noise_kinds_keep_the_teacher_in_control():
    fn = lambda seed, rng: Perturbation(t=0, k=0, kind="joint_noise", sigma=0.15)     # noqa: E731
    noisy, _ = _lock._run(1, 2, _lock.SEEDS[:4], ExpertController(), perturb_fn=fn)
    clean, _ = _lock._run(1, 2, _lock.SEEDS[:4], ExpertController())
    for n, c in zip(noisy, clean):
        assert (n.actor == ACTOR_TEACHER).all()
        assert not np.allclose(n.actions[0][ARM_DIMS], c.actions[0][ARM_DIMS])     # executed action was perturbed


def test_drop_opens_only_the_holding_jaw(monkeypatch):
    orig = _lock._fake.FakeSubEnv._info

    def holding_left(self, reason=None):              # the left arm holds from step 1 on
        info = orig(self, reason)
        info["grasped"] = {"left_": self.t >= 1, "right_": False}
        return info
    monkeypatch.setattr(_lock._fake.FakeSubEnv, "_info", holding_left)
    fn = lambda seed, rng: Perturbation(t=2, k=2, kind="drop")     # noqa: E731
    eps, _ = _lock._run(1, 2, [107, 108], ExpertController(), perturb_fn=fn)
    for e in eps:
        idx = np.flatnonzero(e.actor == ACTOR_PERTURB)
        assert list(idx) == [2, 3]
        assert all(e.actions[t][GRIPPER_DIMS[0]] == 1.0 for t in idx)       # left jaw forced open
        assert not any(e.actions[t][GRIPPER_DIMS[1]] == 1.0 for t in idx)   # right untouched
