"""The Isaac evaluator's final-cloth scoring, its per-policy aggregation, and a full run on the scripted stand-in base
with a tiny world model and actor. Runs without Isaac Sim."""

import argparse
import json

import numpy as np
import pytest

from cloth_angles.data.state_episode import StateEpisode
from cloth_angles.model.actor_critic import Actor, feature_dim
from cloth_angles.tasks import HALF
from cloth_angles.train_imagined_actor import save_actor
from cloth_angles.train_state_predictor import save_checkpoint, train
from isaac.experts.fold_expert import IsaacArmExpert
from isaac.sim.isaac_env import CLOTH_CENTER, cloth_grid_mesh
from isaac.tests.test_world_model_play import FakeHalf, FakeIK
from isaac.tests.test_world_model_reward_check import CarryBase
from isaac.tools.grasp_metrics import wilson
from isaac.world_model import evaluate
from mujoco_sim.cloth_params import TABLE_TOP_Z


def folded_cloth():
    """The ideal half fold: every vertex north of the crease lies 6 mm above its mirror image."""
    grid = (np.array(cloth_grid_mesh()[0]) + [*CLOTH_CENTER, TABLE_TOP_Z + 0.004]).reshape(11, 11, 3)
    grid[:, 6:] = grid[:, 4::-1] + [0.0, 0.0, 0.006]
    return grid.reshape(121, 3)


def test_final_score_passes_only_through_the_gate():
    cloth = folded_cloth()
    far = [np.array([0.0, 0.0, 1.0])] * 2
    ok = evaluate.score_final(cloth, cloth, far, [1.0, 1.0])
    assert ok["passed"] and ok["gate"]["valid"] and ok["passed_valid"] and ok["mean_cm"] == 0.0
    shut = evaluate.score_final(cloth, cloth, far, [0.05, 1.0])               # a jaw still closed
    assert shut["passed"] and not shut["gate"]["valid"] and not shut["passed_valid"]
    flat = np.array(cloth_grid_mesh()[0]) + [*CLOTH_CENTER, TABLE_TOP_Z + 0.004]
    assert not evaluate.score_final(flat, flat, far, [1.0, 1.0])["passed"]   # never folded


def row(policy, seed, success, valid, passed, mean_cm):
    return {"policy": policy, "seed": seed, "success": success, "fold_score": 0.5, "gate": {"valid": valid},
            "passed_valid": passed and valid, "mean_cm": mean_cm, "edge_cm": 2 * mean_cm, "vertex_mae_mm": 0.2,
            "manipulated_vertex_mm": 1.0, "grasp_accuracy": 0.99, "control_hz": 5.0, "finish_capped": not valid,
            "finish_disp_mm": 1.0}


def test_summary_counts_with_wilson_intervals():
    rows = [row("scripted", 1, True, True, True, 0.8), row("scripted", 2, True, False, True, 0.4),
            row("scripted", 3, False, True, False, 3.0), row("clone", 1, False, False, False, 9.0)]
    scripted, clone = evaluate.summarize(rows)
    assert scripted["policy"] == "scripted" and scripted["n"] == 3 and scripted["seeds"] == [1, 2, 3]
    lo, hi = wilson(2, 3)
    assert scripted["success"] == {"k": 2, "n": 3, "wilson95": [round(lo, 1), round(hi, 1)]}
    assert scripted["gated"]["k"] == 2 and scripted["passed_valid"]["k"] == 1 and scripted["finish_capped"]["k"] == 1
    assert scripted["gated_mean_cm"] == pytest.approx(1.9)                    # seeds 1 and 3, the gated ones
    assert clone["gated_mean_cm"] is None and clone["passed_valid"]["k"] == 0


def test_run_scores_every_policy_on_the_same_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(IsaacArmExpert, "_ik", FakeIK(reach=0.30))
    base = CarryBase()
    monkeypatch.setattr(evaluate, "make_base", lambda steps, cameras=None: base)
    monkeypatch.setattr(evaluate, "ScriptedHalfFold", FakeHalf)
    rng = np.random.default_rng(0)
    eps = [StateEpisode(rng.normal(0, 0.01, (6, 121, 3)).astype(np.float32),
                        rng.normal(0, 0.01, (6, 30)).astype(np.float32),
                        rng.uniform(-1, 1, (5, 12)).astype(np.float32), {"split": "train"}) for _ in range(2)]
    model, head, _ = train(eps, HALF, steps=2, hidden_dim=16, log_every=0)
    save_checkpoint(model, head, tmp_path / "wm.pt", "state_l1", "half", 2, 0, 16)
    save_actor(tmp_path / "clone.pt", Actor(feature_dim(HALF), 12), model, HALF)
    args = argparse.Namespace(checkpoint=tmp_path / "wm.pt", actors=[tmp_path / "clone.pt"], episodes=2, seed=7,
                              out=tmp_path / "eval.json", device="cpu")
    evaluate.run(args)
    result = json.loads(args.out.read_text())
    assert [(r["policy"], r["seed"]) for r in result["rows"]] == [("scripted", 7), ("scripted", 8),
                                                                   ("clone", 7), ("clone", 8)]
    scripted, clone = result["summary"]
    assert scripted["success"]["k"] == 2 and scripted["n"] == 2                # CarryBase folds every episode
    assert {"gate", "mean_cm", "passed_valid", "vertex_mae_mm", "grasp_accuracy", "finish_steps",
            "finish_disp_mm"} <= set(result["rows"][2])
    assert scripted["finish_capped"]["k"] == 0 and clone["finish_capped"]["k"] == 0
    assert scripted["gated"]["k"] == 2 and clone["gated"]["k"] == 2           # scored after the arms let go


@pytest.fixture
def folded(monkeypatch):
    """A CarryBase fold run to its end with both arms' experts still holding (as an actor leaves them)."""
    monkeypatch.setattr(IsaacArmExpert, "_ik", FakeIK(reach=0.30))
    base = CarryBase()
    half = FakeHalf(base)
    half.reset(0)
    for _ in range(40):
        half.task.step(half.act())
        if base.t > 5 and not any(base.grasped.values()):
            break                                       # both corners placed and let go, the jaws still shut
    assert not any(base.grasped.values())
    return base, half


def test_finish_releases_from_the_current_pose_then_scores_a_still_cloth(folded):
    base, half = folded
    sites = {p: base.gripper_position(p).copy() for p in half.experts}
    cloth, prev, steps, capped, disp = evaluate.finish(base, half.experts)
    assert not capped and steps < evaluate.FINISH_CAP and disp == pytest.approx(0.0, abs=1e-9)
    for p, e in half.experts.items():
        assert e.PHASES[e.phase] == "done"
        lift = base.gripper_position(p) - sites[p]
        assert lift[2] > 0.04 and np.linalg.norm(lift[:2]) < 0.005             # straight up from where it was
        assert base.jaw_open[p] is None                                         # fully open at the top
    final = evaluate.score_final(cloth, prev, [base.gripper_position(p) for p in half.experts],
                                 [base.joint_positions(p)[5] for p in half.experts])
    assert final["gate"]["valid"] and final["gate"]["clear_cm"] >= 3.0


def test_finish_records_a_cap_hit_and_how_far_the_cloth_moved(folded, monkeypatch):
    base, half = folded
    step = base.step

    def drifting(action):
        base.positions[:, 0] += 0.002                                          # never still: 2 mm per step
        return step(action)

    monkeypatch.setattr(base, "step", drifting)
    _, _, steps, capped, disp = evaluate.finish(base, half.experts)
    assert capped and steps == evaluate.FINISH_CAP
    assert disp == pytest.approx(2.0 * evaluate.FINISH_CAP, rel=1e-6)


def test_an_arm_already_letting_go_carries_on(folded):
    _, half = folded
    e = half.experts["left_"]
    e._go("retreat")
    e.release_site = np.array([0.1, 0.2, 0.5])
    evaluate.release_here(e)
    assert e.PHASES[e.phase] == "retreat" and e.release_site.tolist() == [0.1, 0.2, 0.5]
