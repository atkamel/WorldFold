"""The half_fold / quarter_fold task choice, threaded through env builders and CLIs."""
import inspect
import json

import pytest

from cloth_fold_rl.quarter_fold_env import HALF_FOLD_MAX_STEPS, MAX_STEPS, STAGES
from imitation.tasks import half_fold
from imitation.tasks.half_fold import TASKS, make_env, task_stages


def test_task_registry():
    assert TASKS == ("half_fold", "quarter_fold")
    assert task_stages("half_fold") == STAGES[:1]
    assert task_stages("quarter_fold") == STAGES and len(STAGES) == 2
    with pytest.raises(ValueError):
        task_stages("eighth_fold")


def test_gpu_caps_by_task():
    from imitation.isaac_runtime import FRICTION_MAX_STEPS, FRICTION_QF_MAX_STEPS
    assert FRICTION_QF_MAX_STEPS == 800
    assert half_fold._gpu_cap("isaac_friction") == FRICTION_MAX_STEPS
    assert half_fold._gpu_cap("isaac_friction", "quarter_fold") == 800
    assert half_fold._gpu_cap("isaac_weld") == HALF_FOLD_MAX_STEPS
    assert half_fold._gpu_cap("isaac_weld", "quarter_fold") == MAX_STEPS


def test_make_env_passes_stages_and_cap(monkeypatch):
    seen = []
    monkeypatch.setattr(half_fold, "HalfFoldEnv", lambda **kw: seen.append(kw) or object())
    make_env("mujoco")
    make_env("mujoco", task="quarter_fold")
    make_env("mujoco", task="quarter_fold", max_episode_steps=123)
    assert seen[0] == {"stages": STAGES[:1], "carry_obs": False}    # half fold: default cap left to HalfFoldEnv
    assert seen[1]["carry_obs"]                                     # obs v2 for the quarter fold
    assert seen[1]["stages"] == STAGES and seen[1]["max_episode_steps"] == MAX_STEPS
    assert seen[2]["max_episode_steps"] == 123
    with pytest.raises(ValueError):
        make_env("mujoco", task="nope")


@pytest.mark.slow
def test_make_env_mujoco_stages_and_caps():
    half, quarter = make_env("mujoco"), make_env("mujoco", task="quarter_fold")
    assert len(half.stages) == 1 and half.unwrapped.max_episode_steps == HALF_FOLD_MAX_STEPS
    assert len(quarter.stages) == 2 and quarter.unwrapped.max_episode_steps == MAX_STEPS


def test_cli_parsers_accept_task():
    from imitation import dagger, demo, evaluate
    from imitation.data import collect
    assert collect.build_parser().parse_args([]).task == "half_fold"
    assert collect.build_parser().parse_args(["--task", "quarter_fold"]).task == "quarter_fold"
    assert evaluate.build_parser().parse_args(["--ckpt", "expert"]).task == "half_fold"
    assert evaluate.build_parser().parse_args(["--ckpt", "expert", "--task", "quarter_fold"]).task == "quarter_fold"
    assert dagger.build_parser().parse_args(
        ["--init", "a", "--out", "b", "--task", "quarter_fold"]).task == "quarter_fold"
    assert '"--task"' in inspect.getsource(demo.main)      # demo builds its parser inline


def test_collect_metadata_carries_task():
    from imitation.data import collect
    for task in TASKS:
        args = collect.build_parser().parse_args(["--task", task])
        assert collect.build_config(args)["task"] == task


def _run_eval_main(monkeypatch, argv, ran):
    from imitation import evaluate
    monkeypatch.setattr(evaluate, "evaluate", lambda ckpt, sets, *a, **kw: ran.append((tuple(sets), kw["task"])) or {})
    monkeypatch.setattr("sys.argv", ["evaluate"] + argv)
    evaluate.main()


def test_evaluate_resume_requires_matching_task(monkeypatch, tmp_path):
    out = tmp_path / "eval.json"
    prior = {"checkpoint": "expert", "replan_every": 8, "task": "half_fold",
             "results": {"id_easy": {"n": 4, "success_rate": 1.0}}}
    out.write_text(json.dumps(prior))
    base = ["--ckpt", "expert", "--sets", "id_easy", "--n", "4", "--resume", "--out", str(out)]
    ran = []
    _run_eval_main(monkeypatch, base, ran)                                  # same task: reused
    assert ran == []
    _run_eval_main(monkeypatch, base + ["--task", "quarter_fold"], ran)     # mismatch: rerun
    assert ran == [(("id_easy",), "quarter_fold")]
    assert json.loads(out.read_text())["task"] == "quarter_fold"
    out.write_text(json.dumps({k: v for k, v in prior.items() if k != "task"}))      # legacy file = half_fold
    ran.clear()
    _run_eval_main(monkeypatch, base, ran)
    assert ran == []


def test_isaac_quarter_fold_uses_the_bimanual_fold_over():
    from cloth_fold_rl.quarter_fold_env import (CLOTH_0, CLOTH_5, CLOTH_10, CLOTH_110, CLOTH_115, GRASP_CORNERS,
                                                GRASP_CORNERS_FOLD_OVER, STAGES_FOLD_OVER)
    from imitation.tasks.half_fold import task_grasp_corners
    for backend in ("isaac", "isaac_friction"):
        st = task_stages("quarter_fold", backend)
        assert st == STAGES_FOLD_OVER and st[0] == STAGES[0]
        left, right = {m.prefix: m for m in st[1].moves}["left_"], {m.prefix: m for m in st[1].moves}["right_"]
        assert (left.corners, left.goal) == ((CLOTH_0, CLOTH_10), CLOTH_110)
        assert (right.corners, right.goal) == ((CLOTH_5,), CLOTH_115)
        assert task_grasp_corners("quarter_fold", backend) == GRASP_CORNERS_FOLD_OVER
        # every corner a stage carries is one its arm may grasp
        for stage in st:
            for m in stage.moves:
                assert set(m.corners) <= set(GRASP_CORNERS_FOLD_OVER[m.prefix])
    for backend in ("mujoco", "isaac_weld"):
        assert task_stages("quarter_fold", backend) == STAGES
        assert task_grasp_corners("quarter_fold", backend) == GRASP_CORNERS
    assert task_grasp_corners("half_fold", "isaac_friction") == GRASP_CORNERS
