"""cloth_angles.check_fold_reward on half-fold episodes recorded through the real QuarterFoldEnv, over a scripted base
that carries both corners home and releases them (or blows up). Runs without Isaac Sim."""

import numpy as np
import pytest

from cloth_angles.check_fold_reward import check_episode, summarize
from cloth_angles.tasks import HALF
from isaac.experts.fold_expert import IsaacArmExpert
from isaac.tests.test_world_model_play import FakeHalf, FakeIK, PlayBase
from isaac.world_model import collect

MOVES = (("left_", 10, 0), ("right_", 120, 110))


class CarryBase(PlayBase):
    """Both jaws grasp at step 5, each carries its corner 3 cm per step to its goal and lets go within 5 mm; from step
    fail_at the cloth counts as blown up."""

    fail_at = None

    def reset(self, seed=None, options=None):
        out = super().reset(seed, options)
        self.start, self.grasped, self.t = self.positions.copy(), {p: False for p, _, _ in MOVES}, 0
        return out

    def step(self, action):
        out = super().step(action)
        self.t += 1
        for p, corner, goal in MOVES:
            if self.t == 5:
                self.grasped[p] = True
            if self.grasped[p]:
                d = self.start[goal] - self.positions[corner]
                self.positions[corner] += d * min(1.0, 0.03 / max(np.linalg.norm(d), 1e-9))
                self.q[p][:3] = self.positions[corner]                  # the gripper frame rides the corner
                if np.linalg.norm(self.start[goal] - self.positions[corner]) < 0.005:
                    self.grasped[p] = False
        return out

    def _failed(self):
        return self.fail_at is not None and self.t >= self.fail_at


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(IsaacArmExpert, "_ik", FakeIK(reach=0.30))
    b = CarryBase()
    b.reset()
    return b


def test_fold_reward_matches_the_env_on_a_recorded_fold(base):
    episode, _ = collect.collect_half(FakeHalf(base), "expert", seed=0)
    assert episode.metadata["termination_reason"] == "success"
    assert episode.rewards.max() > 15.0                                     # the success bonus is in the record
    row = check_episode(episode, HALF)
    assert row["max_diff"] < 1e-4 and row["terminations_agree"] and row["stages_agree"]


def test_the_unstable_rule_is_the_reported_difference(base):
    base.fail_at = 9
    episode, _ = collect.collect_half(FakeHalf(base), "expert", seed=0)
    assert episode.metadata["termination_reason"] == "unstable" and len(episode) == 9
    row = check_episode(episode, HALF)
    assert not row["terminations_agree"] and row["worst_step"] == 8
    assert row["max_diff"] == pytest.approx(5.0, abs=1e-3)                  # the env's instability penalty
    base.fail_at = None
    ok, _ = collect.collect_half(FakeHalf(base), "expert", seed=1)
    for ep, name in ((episode, "a"), (ok, "b")):
        ep.metadata["file"] = name
    summary = summarize([check_episode(ep, HALF) for ep in (episode, ok)])
    assert summary["all"]["episodes"] == 2 and summary["all"]["termination_mismatches"] == 1
    assert summary["by_reason"]["unstable"]["termination_mismatches"] == 1
    assert summary["by_reason"]["success"]["termination_mismatches"] == 0
    assert summary["by_reason"]["success"]["max_diff"] < 1e-4
