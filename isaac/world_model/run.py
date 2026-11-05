"""Run a trained residual world model alongside live Isaac Sim 5.1 episodes of the half fold.

    python -m isaac.world_model.run --checkpoint outputs/cloth_angles/isaac_wm_model/state_l1_seed0.pt --episodes 1

Before each control step the model predicts the next state from the current one and the action; after the step the
prediction is scored against the observed state: vertex MAE and the error on the manipulated vertices (each arm's
half-fold corner or play-cycle vertex, averaged). The model only predicts, it never drives. The driver is the scripted
half fold (isaac/world_model/half_fold.py, default) or play (--driver play, isaac/world_model/play.py), on the
collector's env; --actions replays a collected episode_<n>.npz instead, from its seed, on the driver of its kind.
Every 25 steps and per episode it prints the errors, predictor latency and control rate; per episode also how it
ended (half fold: success and fold score) and the disturbance statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_angles.data.state_episode import StateEpisodeStore, manipulated_vertices  # noqa: E402
from cloth_angles.tasks import VERTEX_DIM  # noqa: E402
from isaac.world_model.adapter import IsaacWorldModel, observe_state  # noqa: E402
from isaac.world_model.collect import HALF_STEPS, PLAY_STEPS  # noqa: E402
from isaac.world_model.half_fold import ScriptedHalfFold  # noqa: E402
from isaac.world_model.play import (  # noqa: E402
    ARMS, PlayPolicy, disturbance_stats, expand_action, make_base, reset_play,
)


class Driver:
    """One live episode: the scripted half fold ("half"), play ("play"), or a collected episode's actions replayed
    on the driver of its kind. act() -> (action or None, each arm's manipulated vertex); step(action) -> ended."""

    def __init__(self, base, half, mode, seed, replay=None):
        self.base, self.half, self.replay, self.t = base, half, replay, 0
        self.mode = mode if replay is None else ("play" if replay.metadata["kind"] == "play" else "half")
        self.info = {}
        if self.mode == "half":
            self.info = half.reset(seed)
            self.cycles = half.cycles()
        else:
            rng, _, _ = reset_play(base, seed)
            self.play = PlayPolicy(base, rng)
            self.cycles = self.play.cycles
        if replay is not None:
            self.cycles, self.manip = replay.metadata["cycles"], manipulated_vertices(replay)

    def act(self):
        if self.replay is not None:
            if self.t >= len(self.replay):
                return None, None
            vertices = [int(v) for v in self.manip[self.t]]
            if self.mode == "play":                 # the grasp flag follows the recorded cycles, as when collected
                for arm, v in zip(ARMS, vertices):
                    self.base.weld_mask[arm.prefix] = {v}
            action = self.replay.actions[self.t]
        elif self.mode == "half":
            action, vertices = self.half.act(), [c["vertex"] for c in self.cycles]
        else:
            action = self.play.act()
            vertices = None if action is None else self.play.vertices
        self.t += 1
        return action, vertices

    def step(self, action):
        if self.mode == "half":
            _, _, term, trunc, self.info = self.half.task.step(action)
            return term or trunc
        self.base.step(expand_action(action))
        return self.base._failed()


def run_episode(base, half, world_model, mode, seed, steps, replay=None):
    driver = Driver(base, half, mode, seed, replay)
    states, actions, rows, loops = [observe_state(base, ARMS)], [], [], []
    world_model.reset(states[0])
    reason = "steps"
    for t in range(steps):
        tick = time.perf_counter()
        action, vertices = driver.act()
        if action is None:
            reason = "unreachable" if replay is None else "replay_end"
            break
        prediction, latency_ms = world_model.predict(action)     # from the state before the step
        ended = driver.step(action)
        states.append(observe_state(base, ARMS))
        actions.append(action)
        rows.append(world_model.observe_prediction(prediction, states[-1], vertices, latency_ms))
        loops.append(time.perf_counter() - tick)
        if (t + 1) % 25 == 0:
            recent = rows[-25:]
            print(f"step={t + 1} vertex_mae_mm={1000 * np.mean([r['vertex_mae_m'] for r in recent]):.3f} "
                  f"manip_mm={1000 * np.mean([r['manipulated_vertex_error_m'] for r in recent]):.2f} "
                  f"predictor_ms={np.mean([r['latency_ms'] for r in recent]):.3f} "
                  f"control_hz={len(recent) / np.sum(loops[-25:]):.2f}", flush=True)
        if ended:
            reason = driver.info.get("termination_reason") or ("truncated" if driver.mode == "half" else "failed")
            break
    states = np.stack(states)
    vertices = states[:, :VERTEX_DIM].reshape(len(states), -1, 3)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, len(ARMS) * 6)
    out = {"seed": seed, "driver": driver.mode, "steps": len(rows), "termination_reason": reason,
           "vertex_mae_mm": round(1000 * float(np.mean([r["vertex_mae_m"] for r in rows])), 4),
           "manipulated_vertex_mm": round(1000 * float(np.mean([r["manipulated_vertex_error_m"] for r in rows])), 3),
           "predictor_ms": round(float(np.mean([r["latency_ms"] for r in rows])), 3),
           "control_hz": round(len(loops) / float(np.sum(loops)), 2),
           "stats": disturbance_stats(vertices, states[:, VERTEX_DIM:], actions, driver.cycles)}
    if driver.mode == "half":
        out.update(success=bool(driver.info.get("success")), fold_score=round(float(driver.info["fold_score"]), 4))
    return out


def run(args):
    replay = StateEpisodeStore(args.actions.parent).load(args.actions) if args.actions else None
    base = make_base(HALF_STEPS)
    half = ScriptedHalfFold(base)
    world_model = IsaacWorldModel(args.checkpoint, device=args.device)     # torch after Kit is up
    if len(world_model.task.arms) != len(ARMS):
        raise ValueError(f"the checkpoint's task {world_model.task.name} is not a two-arm layout")
    steps = args.steps or (PLAY_STEPS if args.driver == "play" else HALF_STEPS)
    for i in range(args.episodes):
        seed = replay.metadata["seed"] if replay else args.seed + i
        result = run_episode(base, half, world_model, args.driver, seed, steps, replay)
        print("EPISODE", json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="a full-state predictor checkpoint")
    ap.add_argument("--driver", choices=("half", "play"), default="half")
    ap.add_argument("--actions", type=Path, help="replay this collected episode_<n>.npz instead of driving")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2_000_000, help="first episode's seed (outside the collection's)")
    ap.add_argument("--steps", type=int, help="step cap (default: the collector's for the driver)")
    ap.add_argument("--device", default="cpu", help="torch device for the predictor")
    args = ap.parse_args()
    if args.episodes < 1 or (args.steps is not None and args.steps < 1):
        ap.error("--steps and --episodes must be positive")
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)                  # Kit can hang at interpreter shutdown


if __name__ == "__main__":
    main()
