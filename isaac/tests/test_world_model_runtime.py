"""Sim-free tests for the Isaac world model's state adapter and for fitting and using the online predictor."""

import numpy as np
import torch

from cloth_angles.data.state_episode import StateEpisode
from cloth_angles.tasks import HALF, SINGLE, VERTEX_DIM
from cloth_angles.train_state_predictor import save_checkpoint, train
from isaac.world_model.adapter import IsaacWorldModel, observe_state


class ArmsBase:
    """The env readers observe_state uses, each arm's values distinct."""

    def __init__(self):
        self.cloth = np.arange(121 * 3, dtype=np.float64).reshape(121, 3) / 1000.0
        self.k = {"left_": 1.0, "right_": 2.0}
        self._joint_targets = {p: np.full(6, 0.3 * k) for p, k in self.k.items()}

    def cloth_positions(self):
        return self.cloth

    def joint_positions(self, p):
        return np.arange(6) + 10 * self.k[p]

    def joint_velocities(self, p):
        return -np.arange(6) - 10 * self.k[p]

    def gripper_position(self, p):
        return np.array([0.1, 0.2, 0.3]) * self.k[p]

    def grasp_active(self, p):
        return p == "left_"


def test_state_layout_follows_the_task_arms():
    base = ArmsBase()
    state = observe_state(base, SINGLE.arms)
    assert state.shape == (378,) and state.dtype == np.float32
    np.testing.assert_allclose(state[30:33], base.cloth[10], rtol=1e-6)     # cloth_10, not a corner slot
    np.testing.assert_allclose(state[:VERTEX_DIM].reshape(-1, 3)[10], state[30:33])
    left = SINGLE.arms[0]
    np.testing.assert_allclose(state[left.robot:left.robot + 5], np.arange(5) + 10)
    np.testing.assert_allclose(state[left.robot + 5:left.robot + 10], -np.arange(5) - 10)
    assert state[left.ctrl_index] == np.float32(0.3)
    np.testing.assert_allclose(state[left.ee_slice], [0.1, 0.2, 0.3], rtol=1e-6)
    assert state[left.grasp_index] == 1.0
    two = observe_state(base, HALF.arms)
    assert two.shape == (393,) and HALF.state_dim == 393
    np.testing.assert_array_equal(two[:378], state)                       # the left arm's block comes first
    right = HALF.arms[1]
    np.testing.assert_allclose(two[right.ee_slice], [0.2, 0.4, 0.6], rtol=1e-6)
    assert two[right.grasp_index] == 0.0 and two[right.ctrl_index] == np.float32(0.6)


def _episode(seed, split):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(size=(5, 121, 3)).astype(np.float32) * 0.01
    robot = rng.normal(size=(5, 30)).astype(np.float32) * 0.01
    actions = rng.uniform(-1, 1, size=(4, 12)).astype(np.float32)
    return StateEpisode(vertices, robot, actions, metadata={"split": split})


def test_train_checkpoint_runs_prediction_inference(tmp_path):
    model, head, loss = train([_episode(0, "train"), _episode(1, "train")], HALF, steps=2, hidden_dim=16, seed=7,
                              log_every=0)
    checkpoint = tmp_path / "state_predictor.pt"
    save_checkpoint(model, head, checkpoint, "state_l1", "half", steps=2, seed=7, hidden_dim=16)

    assert np.isfinite(loss)
    predictor = IsaacWorldModel(checkpoint, device="cpu")
    assert predictor.task is HALF
    state = np.zeros(HALF.state_dim, dtype=np.float32)
    predictor.reset(state)
    action = np.zeros(12, dtype=np.float32)
    prediction, latency_ms = predictor.predict(action)
    flags = [a.grasp_index for a in HALF.arms]
    with torch.no_grad():
        s, a = torch.as_tensor(state)[None], torch.as_tensor(action)[None]
        expected = torch.sigmoid(head(model._features(s, s, a)))[0].numpy()
    np.testing.assert_allclose(prediction[flags], expected, rtol=1e-6)           # the flags are the head's
    next_state = prediction.copy()
    next_state[30:33] += [0.003, 0.004, 0.0]             # cloth_10 lands 5 mm from its prediction, cloth_120 on it
    next_state[flags] = [float(expected[0] <= 0.5), float(expected[1] > 0.5)]    # one flag predicted right
    metrics = predictor.observe_prediction(prediction, next_state, vertices=[10, 120], latency_ms=latency_ms)
    assert prediction.shape == (HALF.state_dim,)
    assert metrics["latency_ms"] >= 0 and metrics["grasp_accuracy"] == 0.5
    np.testing.assert_allclose(metrics["manipulated_vertex_error_m"], 0.0025, rtol=1e-5)
    np.testing.assert_allclose(metrics["vertex_mae_m"], 0.007 / VERTEX_DIM, rtol=1e-4)
    assert np.array_equal(predictor.current, next_state)
