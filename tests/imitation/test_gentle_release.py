"""QD (2026-10-16): the gentle-release jaw knobs (isaac_env release_open / release_open_steps / jaw_open_rate) and the
expert's phase speeds, without Isaac Sim."""
from types import SimpleNamespace

import pytest

from isaac.isaac_env import GRIPPER_OPEN, IsaacClothFoldEnv


def stub(**knobs):
    return SimpleNamespace(_gripper_closed={"left_": False}, _closed_q=0.05, grasp_mode="friction",
                           grasp_knobs=dict(knobs), _joint_targets={"left_": [0, 0, 0, 0, 0, GRIPPER_OPEN]},
                           _open_since={})


def jaw_targets(env, commands):
    out = []
    for c in commands:
        IsaacClothFoldEnv.set_gripper(env, "left_", c)
        out.append(env._joint_targets["left_"][5])
    return out


def test_defaults_open_straight_to_full():
    env = stub()
    t = jaw_targets(env, [-1.0, -1.0, 1.0, 1.0])
    assert t == [0.05, 0.05, GRIPPER_OPEN, GRIPPER_OPEN]


def test_release_opens_partly_then_fully_once_clear():
    env = stub(release_open=0.35, release_open_steps=3)
    t = jaw_targets(env, [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert t[0] == 0.05
    assert t[1:4] == [0.35, 0.35, 0.35]          # 3 steps at the partial opening
    assert t[4] == GRIPPER_OPEN                  # then full


def test_open_rate_ramps_the_partial_opening():
    env = stub(release_open=0.35, release_open_steps=10, jaw_open_rate=0.1)
    t = jaw_targets(env, [-1.0] + [1.0] * 6)
    assert t[1:5] == pytest.approx([0.15, 0.25, 0.35, 0.35])


def test_never_closed_opens_fully():
    env = stub(release_open=0.35, release_open_steps=10)
    assert jaw_targets(env, [1.0]) == [GRIPPER_OPEN]   # the approach before any grasp keeps the wide jaw


def test_expert_phase_speeds():
    from isaac.fold_expert import IsaacArmExpert
    e = SimpleNamespace(STAGE1_CARRY_SPEED=None, CARRY_SPEED=1.0, QF0_CARRY_SPEED=None, RETREAT_SPEED=0.3,
                        _stage=lambda: 1, _qf0=lambda v: None)
    assert IsaacArmExpert._speed(e, "retreat") == 0.3
    assert IsaacArmExpert._speed(e, "carry") == 1.0
    e.STAGE1_CARRY_SPEED = 0.5
    assert IsaacArmExpert._speed(e, "lift") == 0.5
    assert IsaacArmExpert._speed(e, "approach") == 1.0


def test_quarter_fold_gentle_preset_scope(monkeypatch):
    from imitation.tasks.half_fold import qf_gentle
    monkeypatch.delenv("WORLDFOLD_QF_GENTLE", raising=False)
    assert qf_gentle("quarter_fold", "isaac_friction")
    assert not qf_gentle("half_fold", "isaac_friction")          # the half fold is unchanged
    assert not qf_gentle("quarter_fold", "isaac_weld")
    monkeypatch.setenv("WORLDFOLD_QF_GENTLE", "0")
    assert not qf_gentle("quarter_fold", "isaac_friction")


def test_preset_keys_are_real_knobs():
    from isaac.fold_expert import IsaacArmExpert
    from isaac.isaac_env import FRICTION_GRASP
    from imitation.tasks.half_fold import QF_FOLD1_EXPERT, QF_FOLD1_GRASP, QF_GENTLE_EXPERT, QF_GENTLE_GRASP
    assert all(hasattr(IsaacArmExpert, k) for k in QF_GENTLE_EXPERT)
    assert set(QF_GENTLE_GRASP) <= set(FRICTION_GRASP)
    assert all(hasattr(IsaacArmExpert, k) for k in QF_FOLD1_EXPERT)       # R1
    assert set(QF_FOLD1_GRASP) <= set(FRICTION_GRASP)


def test_fold1_preset_stacks_on_gentle(monkeypatch):
    # R1 (2026-10-17): the fold-1 friction + lay-down change rides on the gentle preset; WORLDFOLD_QF_FOLD1=0 leaves
    # the QD-2 preset exactly (what isaac_qf_v1 and the n = 100 QD-2 evals were made with)
    from imitation.tasks import half_fold
    monkeypatch.delenv("WORLDFOLD_QF_GENTLE", raising=False)
    monkeypatch.delenv("WORLDFOLD_QF_FOLD1", raising=False)
    grasp, expert = half_fold.qf_preset("quarter_fold", "isaac_friction")
    assert grasp == {**half_fold.QF_GENTLE_GRASP, **half_fold.QF_FOLD1_GRASP}
    assert expert == {**half_fold.QF_GENTLE_EXPERT, **half_fold.QF_FOLD1_EXPERT}
    assert grasp["particle_friction"] == 1.0 and grasp["particle_friction_scale"] == 2.0
    assert "adhesion" not in grasp                                        # adhesion stays LeHome's 0.1
    monkeypatch.setenv("WORLDFOLD_QF_FOLD1", "0")
    assert half_fold.qf_preset("quarter_fold", "isaac_friction") == (half_fold.QF_GENTLE_GRASP, half_fold.QF_GENTLE_EXPERT)
    assert half_fold.qf_preset("half_fold", "isaac_friction") == (None, None)
    monkeypatch.setenv("WORLDFOLD_QF_GENTLE", "0")
    assert half_fold.qf_preset("quarter_fold", "isaac_friction") == (None, None)


def test_expert_takes_preset_as_instance_defaults(monkeypatch):
    from isaac.fold_expert import IsaacArmExpert
    monkeypatch.setenv("WORLDFOLD_EXPERT_PARAMS", "RETREAT_SPEED=0.7")
    base = SimpleNamespace(expert_defaults={"RETREAT_SPEED": 0.3, "SETTLE_DWELL": 6})
    env = SimpleNamespace(unwrapped=base)
    e = IsaacArmExpert.__new__(IsaacArmExpert)
    try:
        IsaacArmExpert.__init__(e, env)
    except Exception:
        pass        # the rest of __init__ needs a live env; the defaults are applied first
    assert e.SETTLE_DWELL == 6
    assert "RETREAT_SPEED" not in vars(e)              # explicit params win (they set the class attribute)
    assert IsaacArmExpert.SETTLE_DWELL == 0            # the class default is untouched


def test_batch_factory_applies_the_preset(monkeypatch):
    """The collect and the multi-env evals build envs through make_env_batch: the preset must reach that path too."""
    from imitation.tasks import half_fold
    import imitation.isaac_runtime as rt
    calls = []

    def fake_batch(n, steps, cameras=None, profile="weld", grasp_corners=None, grasp_knobs=None):
        calls.append(grasp_knobs)
        return SimpleNamespace(envs=[SimpleNamespace() for _ in range(n)])

    monkeypatch.setattr(rt, "make_isaac_batch", fake_batch)
    monkeypatch.setattr(half_fold, "HalfFoldEnv", lambda **kw: kw["base_env"])
    monkeypatch.setattr(half_fold, "_diag_start", lambda env, task, backend: env)
    monkeypatch.delenv("WORLDFOLD_QF_GENTLE", raising=False)
    monkeypatch.delenv("WORLDFOLD_QF_FOLD1", raising=False)
    batch, envs = half_fold.make_env_batch("isaac_friction", n=3, task="quarter_fold")
    grasp, expert = half_fold.qf_preset("quarter_fold", "isaac_friction")
    assert calls[-1] == grasp and grasp["particle_friction_scale"] == 2.0     # gentle + fold 1 (R1)
    assert all(e.expert_defaults == expert for e in envs)
    half_fold.make_env_batch("isaac_friction", n=2, task="half_fold")
    assert calls[-1] is None


def test_table_friction_is_a_grasp_knob():
    # R1 (2026-10-16): the table-top rigid material is a knob of its own, separate from particle_friction
    from isaac.isaac_env import FRICTION_GRASP
    assert "table_friction" in FRICTION_GRASP and "particle_friction" in FRICTION_GRASP
    assert "particle_friction_scale" in FRICTION_GRASP     # cloth-on-cloth only (PhysX particleFrictionScale)


def test_teacher_takes_overshoot_scale_from_the_preset(monkeypatch):
    # R1: ScriptedTeacher scales the stage-0 overshoot from the class attribute; the preset's instance default must
    # reach it too (the probes set QF0_OVERSHOOT_SCALE=0 through WORLDFOLD_EXPERT_PARAMS, i.e. the class attribute)
    from imitation.teachers.scripted import qf0_overshoot_scale
    cls = SimpleNamespace(QF0_OVERSHOOT_SCALE=1.0)
    monkeypatch.delenv("WORLDFOLD_EXPERT_PARAMS", raising=False)
    assert qf0_overshoot_scale(cls, SimpleNamespace()) == 1.0
    assert qf0_overshoot_scale(cls, SimpleNamespace(expert_defaults={"QF0_OVERSHOOT_SCALE": 0})) == 0
    monkeypatch.setenv("WORLDFOLD_EXPERT_PARAMS", "QF0_OVERSHOOT_SCALE=0.5")
    cls.QF0_OVERSHOOT_SCALE = 0.5                    # what the env var does to the class
    assert qf0_overshoot_scale(cls, SimpleNamespace(expert_defaults={"QF0_OVERSHOOT_SCALE": 0})) == 0.5
