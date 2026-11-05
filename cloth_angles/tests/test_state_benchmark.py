"""The benchmark's windows, horizons and baselines on synthetic episodes with known motion."""

import numpy as np
import pytest

from cloth_angles.benchmark_state_predictor import HORIZONS, evaluate, summarize
from cloth_angles.data.state_episode import GRASP_INDEX, StateEpisode, StateEpisodeStore
from cloth_angles.model.state_predictor import ResidualStatePredictor
from cloth_angles.train_state_predictor import load_split


def moving_episode(kind="play", T=40):
    """Vertex 5 moves 2 mm per step and vertex 7 4 mm, in x; the left arm plays with 5 until state 25, then with 7;
    the right arm with 7 throughout, grasped at state 20 only."""
    vertices = np.zeros((T + 1, 121, 3), dtype=np.float32)
    t = np.arange(T + 1, dtype=np.float32)
    vertices[:, 5, 0], vertices[:, 7, 0] = 0.002 * t, 0.004 * t
    robot = np.zeros((T + 1, 30), dtype=np.float32)
    robot[20, 15 + GRASP_INDEX] = 1.0
    cycles = [{"arm": "left_", "start": 0, "vertex": 5}, {"arm": "right_", "start": 0, "vertex": 7},
              {"arm": "left_", "start": 25, "vertex": 7}]
    return StateEpisode(vertices, robot, np.zeros((T, 12), dtype=np.float32),
                        {"kind": kind, "split": "test", "cycles": cycles})


def persistence_model():
    model = ResidualStatePredictor(393, 12, hidden_dim=8)      # zero last layer, zero mean change: persistence
    return model.eval()


def test_windows_horizons_and_baselines():
    rows = evaluate(persistence_model(), "state_l1", [moving_episode()])
    assert [r["horizon"] for r in rows] == list(HORIZONS)
    for r in rows:
        h = r["horizon"]
        assert r["windows"] == 3 and r["grasped_windows"] == 1                    # origins 10, 20, 30
        # origins 10 and 20 score vertices (5, 7), origin 30 scores (7, 7): windows of 3, 3 and 4 mm per step
        assert r["persistence_manip_err"] == pytest.approx(0.001 * h * 10 / 3, rel=1e-5)
        assert r["persistence_vertex_mae"] == pytest.approx(0.006 * h / 363, rel=1e-5)
        assert r["const_velocity_vertex_mae"] == pytest.approx(0.0, abs=1e-7)
        assert r["vertex_mae"] == pytest.approx(r["persistence_vertex_mae"], rel=1e-5)
        assert r["manip_err"] == pytest.approx(r["persistence_manip_err"], rel=1e-5)
        assert r["grasped_persistence_manip_err"] == pytest.approx(0.003 * h, rel=1e-5)          # origin 20


def test_summary_groups_play_and_task_and_averages_seeds():
    episodes = [moving_episode("play"), moving_episode("expert", T=30)]
    rows = evaluate(persistence_model(), "state_l1", episodes)
    runs = [{"variant": "state_l1", "seed": s, "test": rows, "train": rows} for s in (0, 1)]
    summary = summarize(runs, ["state_l1"])
    test_all = {(r["group"], r["horizon"]): r for r in summary if r["split"] == "test" and r["subset"] == "all"}
    assert test_all[("play", 1)]["windows"] == 3 and test_all[("task", 1)]["windows"] == 2
    play = test_all[("play", 5)]
    assert play["state_l1_manip_err_mean"] == pytest.approx(play["persistence_manip_err"], rel=1e-5)
    assert play["state_l1_manip_err_std"] == 0.0
    grasped = [r for r in summary if r["subset"] == "grasped" and r["group"] == "task" and r["horizon"] == 1]
    assert grasped[0]["windows"] == 1                                           # task episode: origin 20


def test_load_split_takes_the_union_of_stores(tmp_path):
    for name, split in (("play", "train"), ("half", "test")):
        ep = moving_episode()
        ep.metadata["split"] = split
        StateEpisodeStore(tmp_path / name).append(ep)
    train, test = load_split([tmp_path / "play", tmp_path / "half"])
    assert len(train) == 1 and len(test) == 1
