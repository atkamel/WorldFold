"""isaac_anchor backend (comparison-only kinematic anchor grasp, port of feat/isaac-half-fold), sim-free."""

import importlib

import pytest

_CLI_ARGS = {"imitation.data.collect": [], "imitation.evaluate": ["--ckpt", "expert"],
             "imitation.dagger": ["--init", "x.pt", "--out", "o"], "imitation.check_resync": [],
             "imitation.benchmark_expert": []}


def test_anchor_profile_is_the_friction_profile_with_the_anchor_grasp():
    from isaac.isaac_env import PROFILES
    anchor, friction = PROFILES["anchor"], PROFILES["friction"]
    assert anchor["grasp_mode"] == "anchor"
    assert ({k: v for k, v in anchor.items() if k != "grasp_mode"}
            == {k: v for k, v in friction.items() if k != "grasp_mode"})
    assert friction["grasp_mode"] == "friction" and PROFILES["weld"]["grasp_mode"] == "weld"


def test_backend_maps_to_the_anchor_profile_and_is_not_vectorised():
    from imitation.isaac_runtime import ISAAC_PROFILES
    from imitation.tasks.half_fold import BACKENDS, GPU_BACKENDS, is_isaac, make_env_batch
    assert ISAAC_PROFILES["isaac_anchor"] == "anchor"
    assert "isaac_anchor" in BACKENDS and "isaac_anchor" not in GPU_BACKENDS and is_isaac("isaac_anchor")
    with pytest.raises(ValueError):
        make_env_batch(backend="isaac_anchor", n=2)


def test_anchor_uses_the_friction_step_cap():
    from imitation.isaac_runtime import FRICTION_MAX_STEPS
    from imitation.tasks.half_fold import _gpu_cap
    assert _gpu_cap("isaac_anchor") == FRICTION_MAX_STEPS == _gpu_cap("isaac_friction")


@pytest.mark.parametrize("module", list(_CLI_ARGS))
def test_cli_parsers_accept_isaac_anchor(module):
    build_parser = importlib.import_module(module).build_parser
    assert build_parser().parse_args(_CLI_ARGS[module] + ["--backend", "isaac_anchor"]).backend == "isaac_anchor"


def test_eval_sets_match_mujoco_for_anchor():
    from imitation.seeds import eval_set
    assert eval_set("id_hard", 8, "isaac_anchor")[0] == eval_set("id_hard", 8, "mujoco")[0]
