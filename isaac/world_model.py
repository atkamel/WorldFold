"""Online adapter between IsaacClothFoldEnv and the learned state predictor.

The predictor contract matches ``StateEpisode.states()``: 121 cloth vertices
(xyz) followed by the left arm's 15-value proprioceptive state. Call
``observe_transition`` once per simulator control step to make a one-step
prediction from the state available before the action, then compare it with
the observation after the action. This keeps inference on the live loop and
does not use future observations as model inputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

VERTEX_DIM = 121 * 3
ROBOT_DIM = 15
STATE_DIM = VERTEX_DIM + ROBOT_DIM
ACTION_DIM = 6


def observe_state(env) -> np.ndarray:
    """Return the same 378-value state layout used by the MuJoCo collector."""
    joints = np.asarray(env.joint_positions("left_"), dtype=np.float32)[:5]
    joint_vel = np.asarray(env.joint_velocities("left_"), dtype=np.float32)[:5]
    robot = np.concatenate((
        joints,
        joint_vel,
        np.asarray([env._joint_targets["left_"][5]], dtype=np.float32),
        np.asarray(env.gripper_position("left_"), dtype=np.float32).reshape(3),
        np.asarray([float(env.grasp_active("left_"))], dtype=np.float32),
    ))
    vertices = np.asarray(env.cloth_positions(), dtype=np.float32)
    state = np.concatenate((vertices.reshape(-1), robot)).astype(np.float32, copy=False)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"invalid Isaac world-model state: shape={state.shape}, finite={np.isfinite(state).all()}")
    return state


class IsaacWorldModel:
    """Load a trained residual predictor and score live Isaac transitions."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        # Delay torch import until after the caller has created SimulationApp;
        # importing CUDA consumers before Kit can disturb Isaac's GPU setup.
        import torch
        from cloth_angles.model.state_predictor import ResidualStatePredictor

        self.torch = torch
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        variant = saved.get("variant", "state_l1")
        if variant not in ("state_l1", "state_mse"):
            raise ValueError(f"checkpoint variant must predict full state, got {variant!r}")
        if "state_dict" not in saved:
            raise ValueError("checkpoint must contain a ResidualStatePredictor state_dict")
        if saved.get("state_dim", STATE_DIM) != STATE_DIM or saved.get("action_dim", ACTION_DIM) != ACTION_DIM:
            raise ValueError("checkpoint state/action dimensions do not match the Isaac environment")
        self.model = ResidualStatePredictor(STATE_DIM, ACTION_DIM, hidden_dim=int(saved.get("hidden_dim", 256)),
                                            loss="mse" if variant.endswith("mse") else "l1")
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()
        self.previous: np.ndarray | None = None
        self.current: np.ndarray | None = None

    def reset(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"expected state shape {(STATE_DIM,)}, got {state.shape}")
        self.previous = state.copy()
        self.current = state.copy()

    def predict(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Predict the next live state; returns prediction and inference ms."""
        if self.current is None or self.previous is None:
            raise RuntimeError("call reset(state) before prediction")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"expected action shape {(ACTION_DIM,)}, got {action.shape}")
        started = time.perf_counter()
        cur = self.torch.as_tensor(self.current, device=self.device).unsqueeze(0)
        prev = self.torch.as_tensor(self.previous, device=self.device).unsqueeze(0)
        act = self.torch.as_tensor(action, device=self.device).unsqueeze(0)
        with self.torch.inference_mode():
            prediction = self.model(cur, prev, act)[0].cpu().numpy().astype(np.float32, copy=False)
        return prediction, (time.perf_counter() - started) * 1000.0

    def observe_prediction(self, prediction: np.ndarray, next_state: np.ndarray,
                           latency_ms: float | None = None) -> dict:
        """Score a prediction made before a live action and advance the state history."""
        if self.current is None or self.previous is None:
            raise RuntimeError("call reset(state) before observing a transition")
        prediction = np.asarray(prediction, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if prediction.shape != (STATE_DIM,):
            raise ValueError(f"expected prediction shape {(STATE_DIM,)}, got {prediction.shape}")
        if next_state.shape != (STATE_DIM,):
            raise ValueError(f"expected state shape {(STATE_DIM,)}, got {next_state.shape}")
        vertex_error = np.abs(prediction[:VERTEX_DIM] - next_state[:VERTEX_DIM]).reshape(-1, 3)
        corner_error = float(np.linalg.norm(
            prediction[VERTEX_DIM:VERTEX_DIM + 3] - next_state[VERTEX_DIM:VERTEX_DIM + 3]))
        result = {"vertex_mae_m": float(vertex_error.mean()),
                  "moving_corner_error_m": corner_error}
        if latency_ms is not None:
            result["latency_ms"] = float(latency_ms)
        self.previous, self.current = self.current, next_state.copy()
        return result

    def observe_transition(self, action: np.ndarray, next_state: np.ndarray) -> dict:
        """Convenience path for callers that already have the observed next state."""
        prediction, latency_ms = self.predict(action)
        return self.observe_prediction(prediction, next_state, latency_ms)

