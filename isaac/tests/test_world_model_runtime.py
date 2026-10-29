"""Sim-free tests for fitting and using the online Isaac state predictor."""

import numpy as np

from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore
from isaac.train_world_model import train
from isaac.world_model import IsaacWorldModel, STATE_DIM


def _episode(seed, split):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(size=(5, 121, 3)).astype(np.float32) * 0.01
    robot = rng.normal(size=(5, 15)).astype(np.float32) * 0.01
    actions = rng.uniform(-1, 1, size=(4, 6)).astype(np.float32)
    return StateEpisode(vertices, robot, actions, metadata={"split": split})


def test_train_checkpoint_runs_prediction_inference(tmp_path):
    data_dir = tmp_path / "episodes"
    store = StateEpisodeStore(data_dir)
    store.append(_episode(0, "train"))
    store.append(_episode(1, "test"))
    checkpoint = tmp_path / "state_predictor.pt"

    result = train(data_dir, checkpoint, epochs=2, batch_size=4, hidden_dim=16, seed=7,
                   device_name="cpu")

    assert checkpoint.is_file()
    assert result["checkpoint"] == str(checkpoint)
    predictor = IsaacWorldModel(checkpoint, device="cpu")
    state = np.zeros(STATE_DIM, dtype=np.float32)
    predictor.reset(state)
    prediction, latency_ms = predictor.predict(np.zeros(6, dtype=np.float32))
    next_state = np.full(STATE_DIM, 0.001, dtype=np.float32)
    metrics = predictor.observe_prediction(prediction, next_state, latency_ms)
    assert prediction.shape == (STATE_DIM,)
    assert metrics["latency_ms"] >= 0
    assert metrics["vertex_mae_m"] >= 0
    assert np.array_equal(predictor.current, next_state)
