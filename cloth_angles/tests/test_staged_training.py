"""Stage-balanced training data and staged actor checkpoint compatibility."""

import torch

from cloth_angles.model.actor_critic import Actor, feature_dim
from cloth_angles.tasks import QUARTER
from scripts.train_imagined_actor import load_policy_actors, stage_data


def test_stage_data_filters_every_transition_tensor():
    stage = torch.tensor([0, 1, 0, 1, 1])
    data = tuple(torch.arange(5 * (i + 1)).reshape(5, i + 1) for i in range(6)) + (stage,)
    selected = stage_data(data, 1)
    assert all(len(x) == 3 for x in selected)
    torch.testing.assert_close(selected[6], torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(selected[0].flatten(), torch.tensor([1, 3, 4]))


def test_staged_checkpoint_loads_one_actor_per_quarter_stage():
    actors = [Actor(feature_dim(QUARTER), QUARTER.action_dim) for _ in QUARTER.stages]
    saved = {"actors": [actor.state_dict() for actor in actors]}
    loaded = load_policy_actors(saved, QUARTER)
    assert len(loaded) == 2
    features = torch.randn(1, feature_dim(QUARTER))
    assert loaded[0](features).shape == (1, QUARTER.action_dim)
    assert not loaded[0].training and not loaded[1].training


def test_legacy_checkpoint_still_loads_as_one_actor():
    actor = Actor(feature_dim(QUARTER), QUARTER.action_dim)
    loaded = load_policy_actors({"actor": actor.state_dict()}, QUARTER)
    assert len(loaded) == 1

