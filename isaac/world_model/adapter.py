"""Online adapter between IsaacClothFoldEnv and the learned state predictor.

The state follows cloth_angles.tasks: 121 cloth vertices (xyz), then per arm of the task joint positions (5), joint
velocities (5), gripper command (the jaw target, 1), end-effector position (3) and grasp flag (1). One arm gives 378
values, two give 393. Call ``predict`` before each simulator control step, from the state available before the
action, then ``observe_prediction`` with the observed next state. Inference stays on the live loop and never sees a
future observation. The predicted grasp flags are the checkpoint's grasp head's (probabilities), not the predictor's.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cloth_angles.tasks import TASKS, VERTEX_DIM


def robot_state(base, arm) -> np.ndarray:
    """float32[15]: one arm's block of the state (cloth_angles.tasks.Arm's layout)."""
    p = arm.prefix
    return np.concatenate([base.joint_positions(p)[:5], base.joint_velocities(p)[:5], [base._joint_targets[p][5]],
                           base.gripper_position(p), [float(base.grasp_active(p))]]).astype(np.float32)


def observe_state(base, arms) -> np.ndarray:
    """float32[363 + 15 * len(arms)]: the cloth vertices, then each arm's robot block."""
    state = np.concatenate([np.asarray(base.cloth_positions(), dtype=np.float32).reshape(-1),
                            *(robot_state(base, arm) for arm in arms)])
    if not np.isfinite(state).all():
        raise ValueError("non-finite Isaac world-model state")
    return state


class IsaacWorldModel:
    """Load a trained residual predictor and score live Isaac transitions."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        # Delay torch import until after the caller has created SimulationApp;
        # importing CUDA consumers before Kit can disturb Isaac's GPU setup.
        import torch
        from cloth_angles.train_state_predictor import load_checkpoint

        self.torch = torch
        self.device = torch.device(device)
        self.model, self.head, saved = load_checkpoint(checkpoint, self.device)
        if self.head is None:
            raise ValueError(f"checkpoint variant must predict full state, got {saved['variant']!r}")
        self.task = TASKS[saved["task"]]
        if saved["state_dim"] != self.task.state_dim or saved["action_dim"] != self.task.action_dim:
            raise ValueError("checkpoint state/action dimensions do not match its task")
        self.flags = [a.grasp_index for a in self.task.arms]
        self.previous: np.ndarray | None = None
        self.current: np.ndarray | None = None

    def reset(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.task.state_dim,):
            raise ValueError(f"expected state shape {(self.task.state_dim,)}, got {state.shape}")
        self.previous = state.copy()
        self.current = state.copy()

    def predict(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Predict the next live state; returns prediction and inference ms."""
        if self.current is None or self.previous is None:
            raise RuntimeError("call reset(state) before prediction")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.task.action_dim,):
            raise ValueError(f"expected action shape {(self.task.action_dim,)}, got {action.shape}")
        started = time.perf_counter()
        cur = self.torch.as_tensor(self.current, device=self.device).unsqueeze(0)
        prev = self.torch.as_tensor(self.previous, device=self.device).unsqueeze(0)
        act = self.torch.as_tensor(action, device=self.device).unsqueeze(0)
        with self.torch.inference_mode():
            prediction = self.model(cur, prev, act)[0].cpu().numpy().astype(np.float32, copy=True)
            logits = self.head(self.model._features(cur, prev, act))
            prediction[self.flags] = self.torch.sigmoid(logits)[0].cpu().numpy()
        return prediction, (time.perf_counter() - started) * 1000.0

    def observe_prediction(self, prediction: np.ndarray, next_state: np.ndarray, vertices,
                           latency_ms: float | None = None) -> dict:
        """Score a prediction made before a live action and advance the state history. vertices: the cloth vertex
        each arm is manipulating; their position errors are averaged. grasp_accuracy: the fraction of arms whose
        predicted flag (probability above 0.5) matches the observed one."""
        if self.current is None or self.previous is None:
            raise RuntimeError("call reset(state) before observing a transition")
        prediction = np.asarray(prediction, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        for name, x in (("prediction", prediction), ("state", next_state)):
            if x.shape != (self.task.state_dim,):
                raise ValueError(f"expected {name} shape {(self.task.state_dim,)}, got {x.shape}")
        predicted = prediction[:VERTEX_DIM].reshape(-1, 3)
        observed = next_state[:VERTEX_DIM].reshape(-1, 3)
        result = {"vertex_mae_m": float(np.abs(predicted - observed).mean()),
                  "manipulated_vertex_error_m": float(np.linalg.norm(predicted[vertices] - observed[vertices],
                                                                     axis=-1).mean()),
                  "grasp_accuracy": float(np.mean((prediction[self.flags] > 0.5) == (next_state[self.flags] > 0.5)))}
        if latency_ms is not None:
            result["latency_ms"] = float(latency_ms)
        self.previous, self.current = self.current, next_state.copy()
        return result

    def observe_transition(self, action: np.ndarray, next_state: np.ndarray, vertices) -> dict:
        """Convenience path for callers that already have the observed next state."""
        prediction, latency_ms = self.predict(action)
        return self.observe_prediction(prediction, next_state, vertices, latency_ms)
