"""Behaviour-cloning glue without Isaac: the working-bar success rule, demo episodes the dataset store accepts, merging
parallel collection versions, the student's observation history and end-of-episode scoring, and the precision batch's
recording hook."""

import numpy as np
import pytest

from imitation.data.schema import DatasetWriter, load_dataset, merge_versions, validate_episode
from isaac.tools.bc_collect import build_episode, success
from isaac.tools.bc_eval import history, score_final
from mujoco_sim.cloth_params import TABLE_TOP_Z

GOOD = {"reason": "settled", "gate": {"valid": True}, "mean": 0.015, "edge_max": 0.025}


def test_success_needs_settled_valid_and_the_bar():
    assert success(GOOD)
    assert not success(dict(GOOD, reason="truncated"))
    assert not success(dict(GOOD, gate={"valid": False}))
    assert not success(dict(GOOD, mean=0.021)) and not success(dict(GOOD, edge_max=0.031))
    assert not success(None)


def _steps(T, images=False):
    img = {"main": np.zeros((3, 8, 8), np.uint8), "left_wrist_cam": np.ones((3, 4, 4), np.uint8)}
    return [(np.full(139, k, np.float32), np.full(12, 0.5, np.float32), dict(img) if images else None)
            for k in range(T)]


def test_demo_episode_is_valid_with_images():
    ep = build_episode(_steps(5, images=True), GOOD, np.zeros(139), 7, {"mean": 0.015})
    assert validate_episode(ep, 139, 12) == []
    assert ep.terminated[-1] and not ep.truncated.any() and ep.discount[-1] == 0.0 and ep.discount[:-1].all()
    assert ep.images["main"].shape == (5, 3, 8, 8) and ep.meta["success"] and ep.meta["seed"] == 7


def test_truncated_demo_ends_truncated():
    ep = build_episode(_steps(3), dict(GOOD, reason="truncated"), np.zeros(139), 1, {})
    assert validate_episode(ep, 139, 12) == [] and ep.truncated[-1] and not ep.meta["success"]


def test_merge_lists_both_jobs_versions(tmp_path):
    for v, seed in (("c1", 1), ("c2", 2)):
        w = DatasetWriter(tmp_path, v)
        w.add(build_episode(_steps(4, images=True), GOOD, np.zeros(139), seed, {}), 139, 12)
        w.freeze()
    m = merge_versions(tmp_path, ["c1", "c2"], "all")
    assert m["n_episodes"] == 2 and m["merged_from"] == ["c1", "c2"]
    _, eps = load_dataset(tmp_path, "all", images=True)
    assert sorted(e.meta["seed"] for e in eps) == [1, 2] and eps[0].images["main"].shape[0] == 4
    with pytest.raises(ValueError):
        merge_versions(tmp_path, ["c1", "c1"], "dup")


def test_history_fills_then_rolls():
    h = []
    assert history(h, np.array([1.0]), 2).ravel().tolist() == [1.0, 1.0]
    assert history(h, np.array([2.0]), 2).ravel().tolist() == [1.0, 2.0]
    assert history(h, np.array([3.0]), 2).ravel().tolist() == [2.0, 3.0]


def _folded(shift=0.0):
    """11 x 11 grid folded in half along y (top half iy 6-10 stacked 5 mm above iy 4-0), top layer shifted in x."""
    g = np.zeros((11, 11, 3))
    for ix in range(11):
        for iy in range(11):
            src = iy if iy <= 5 else 10 - iy
            g[ix, iy] = [ix * 0.03 - 0.15 + (shift if iy > 5 else 0.0), src * 0.03, TABLE_TOP_Z + (0.005 if iy > 5 else 0)]
    return g.reshape(-1, 3)


def test_score_final_passes_a_clean_released_fold_only():
    sites, jaws = [(0.0, -0.3, TABLE_TOP_Z + 0.2), (0.1, -0.3, TABLE_TOP_Z + 0.2)], [1.0, 1.0]
    c = _folded()
    assert score_final(c, c, sites, jaws)["success"]
    off = _folded(shift=0.04)
    assert not score_final(off, off, sites, jaws)["success"]
    assert not score_final(c, c, sites, [0.05, 1.0])["success"]          # a jaw still shut


class _Sub:
    def __init__(self):
        self.submitted = []

    def step_submit(self, a):
        self.submitted.append(np.asarray(a).copy())

    def step_collect(self):
        pass

    def render_rig(self):
        return {"main": np.zeros((3, 2, 2), np.uint8)}


class _Task:
    def __init__(self):
        self.t = 0

    def _observe(self):
        return np.full(139, self.t, np.float32)

    def _expand(self, a):
        return np.r_[a, 0.0, 0.0]


class _Arm:
    PHASES = ("approach", "descend", "close", "lift", "carry", "place", "hold", "release", "retreat", "done")
    RELEASE_PHASES = ("release", "retreat", "done")

    def __init__(self, v):
        self.phase, self.v = 0, v

    def act(self):
        return np.full(6, self.v, np.float32)


def _batch():
    from isaac.rl.precision_env import PrecisionBatch
    pb = object.__new__(PrecisionBatch)
    pb.n, pb.subs, pb.tasks = 2, [_Sub(), _Sub()], [_Task(), _Task()]
    pb.experts = [{"left_": _Arm(0.1), "right_": _Arm(0.2)} for _ in range(2)]
    pb.batch = type("B", (), {"advance": lambda self: None})()
    pb.frames, pb.recorder, pb.record_images, pb.recording = None, None, False, [False, False]
    return pb


def test_recording_hook_logs_only_recording_copies_with_the_executed_action():
    pb = _batch()
    got = []
    pb.recorder = lambda i, obs, act, images: got.append((i, obs[0], act.copy(), images))
    pb.record_images = True
    pb._scripted_step()
    assert got == []                                            # nobody recording yet
    pb.recording = [True, False]
    pb._scripted_step()
    assert len(got) == 1 and got[0][0] == 0 and got[0][3]["main"].shape == (3, 2, 2)
    assert np.allclose(got[0][2], np.r_[np.full(6, 0.1), np.full(6, 0.2)])
    assert np.allclose(pb.subs[0].submitted[-1][:12], got[0][2])        # what was recorded is what was executed


def test_drive_submits_the_students_actions_clipped():
    pb = _batch()
    pb.drive([np.full(12, 2.0), np.full(12, -0.3)])
    assert np.allclose(pb.subs[0].submitted[-1][:12], 1.0) and np.allclose(pb.subs[1].submitted[-1][:12], -0.3)
