"""Collect Isaac Sim 5.1 episodes for the half fold's full-state world model.

    python -m isaac.world_model.collect --output outputs/cloth_angles/isaac_wm_a
    python -m isaac.world_model.collect --kinds play --seed-base 1100000 --output outputs/cloth_angles/isaac_wm_b

Kinds: play (isaac/world_model/play.py: both arms disturb the cloth, the bulk of the data), and the scripted half fold
(isaac/world_model/half_fold.py) as expert, expert_noisy (Gaussian noise of std 0.3 on every joint action) and
expert_release (both jaws forced open for RELEASE_HOLD steps once either arm has held its corner for a random 10 to 39
steps). One env, episodes in sequence (IsaacLab allows one scene per process), friction profile, dynamics DR on.
By default PER_KIND episodes of each kind, a fifth of them held out.

Each episode is a cloth_angles StateEpisode in the half fold's layout (393-value states, 12-dim actions) and, unless
--no-frames, a frames_<n>.npz next to it with the imitation camera rig's frames (imitation.vision.render.CAMERAS) at
every state: RGB as uint8 [T+1, 3, H, W] per camera, and the main and wrist cameras' depth as uint16 mm [T+1, H, W]
under <camera>_depth (0 where nothing was hit). Half-fold episodes also store the env's rewards, terminations and
stage per state, and goal / anchors0 in their metadata (FoldReward's other input, the start of every vertex, is the
first state). Kind k's episode i has seed seed_base + 1000 * KINDS.index(k) + i; the first --test-per-kind of each
kind are the held-out test split. Metadata: kind, seed, split, dynamics DR draw, cloth offset, each arm's cycles
(play: vertex, target, carry, release mode, shake, noise; half fold: its corner), why the episode ended, and
disturbance statistics.
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

from cloth_angles.data.state_episode import StateEpisode, StateEpisodeStore  # noqa: E402
from cloth_angles.tasks import HALF, VERTEX_DIM  # noqa: E402
from imitation.vision.render import CAMERAS  # noqa: E402
from isaac.world_model.adapter import observe_state  # noqa: E402
from isaac.world_model.half_fold import ScriptedHalfFold  # noqa: E402
from isaac.world_model.play import (  # noqa: E402
    ARMS, PlayPolicy, disturbance_stats, expand_action, make_base, reset_play,
)

KINDS = ("play", "expert", "expert_noisy", "expert_release")
PER_KIND = {"play": 100, "expert": 20, "expert_noisy": 20, "expert_release": 20}   # play is ~5/6 of the steps
PLAY_STEPS = 600       # 4 to 5 cycles per arm (100 to 150 steps each)
HALF_STEPS = 500       # the place-servo fold: ~150 steps to the carry, a window of up to ~400 with the gentle release
NOISE_STD = 0.3
RELEASE_HOLD = 15


def render(base):
    """The rig's frames at this state: RGB per camera, depth in whole mm as uint16 per depth camera."""
    shot = base.render_rig()
    for name, d in base.render_rig_depth().items():
        shot[f"{name}_depth"] = np.where(np.isfinite(d), np.round(d * 1000.0), 0.0).clip(0, 65535).astype(np.uint16)
    return shot


def _episode(base, states, actions, shots, metadata, cycles, **extra):
    states = np.stack(states)
    vertices = states[:, :VERTEX_DIM].reshape(len(states), -1, 3)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, HALF.action_dim)
    metadata.update(profile=base.profile, length=len(actions), cycles=cycles,
                    stats=disturbance_stats(vertices, states[:, VERTEX_DIM:], actions, cycles))
    episode = StateEpisode(vertices, states[:, VERTEX_DIM:], actions, metadata, **extra)
    return episode, ({name: np.stack([s[name] for s in shots]) for name in shots[0]} if shots else None)


def collect_play(base, seed, steps, frames=False):
    """One play episode -> (StateEpisode, {camera: frames} or None). Ends on a cloth blow-up
    (base._failed), when an arm has no reachable cycle left, or after steps."""
    rng, pose, info = reset_play(base, seed)
    policy = PlayPolicy(base, rng)
    states, actions, shots = [observe_state(base, ARMS)], [], ([render(base)] if frames else None)
    reason = "steps"
    for _ in range(steps):
        action = policy.act()
        if action is None:
            reason = "unreachable"
            break
        base.step(expand_action(action))
        actions.append(action)
        states.append(observe_state(base, ARMS))
        if frames:
            shots.append(render(base))
        if base._failed():
            reason = "failed"
            break
    terminated = np.zeros(len(actions), dtype=np.bool_)
    terminated[-1:] = reason == "failed"
    metadata = {"task": HALF.name, "kind": "play", "simulator": "isaacsim-5.1", "seed": seed,
                "termination_reason": reason, "cloth_pose": np.round(pose, 4).tolist(),
                "domain": info["domain_parameters"]}
    return _episode(base, states, actions, shots, metadata, policy.cycles, terminated=terminated)


def collect_half(half, kind, seed, frames=False):
    """One scripted half-fold episode of an expert kind -> (StateEpisode, frames or None). Ends when the env does."""
    base = half.task.unwrapped
    rng = np.random.default_rng(seed)
    info = half.reset(seed)
    domain, goal, anchors0 = info["domain_parameters"], HALF.goals(half.task), HALF.corner_starts(half.task)
    release_after = int(rng.integers(10, 40)) if kind == "expert_release" else None
    release_step, grasp_steps = None, 0
    states, actions, shots = [observe_state(base, ARMS)], [], ([render(base)] if frames else None)
    rewards, terminated, stages = [], [], [HALF.stage(info)]
    for t in range(base.max_episode_steps):
        action = half.act()
        if kind == "expert_noisy":
            for arm in ARMS:
                action[arm.joints] = np.clip(action[arm.joints] + rng.normal(0.0, NOISE_STD, 5), -1.0, 1.0)
        if release_after is not None and release_step is None and grasp_steps >= release_after:
            release_step = t
        if release_step is not None and t < release_step + RELEASE_HOLD:
            for arm in ARMS:
                action[arm.gripper] = 1.0
        _, reward, term, trunc, info = half.task.step(action)
        actions.append(action)
        rewards.append(reward)
        terminated.append(term)
        stages.append(HALF.stage(info))
        states.append(observe_state(base, ARMS))
        if frames:
            shots.append(render(base))
        grasp_steps += int(HALF.grasped(info))
        if term or trunc:
            break
    metadata = {"task": HALF.name, "kind": kind, "simulator": "isaacsim-5.1", "seed": seed,
                "termination_reason": info["termination_reason"] or "truncated", "success": bool(info["success"]),
                "fold_score": round(float(info["fold_score"]), 4), "grasp_steps": grasp_steps,
                "release_step": release_step, "domain": domain,
                "goal": goal.tolist(), "anchors0": anchors0.tolist()}
    return _episode(base, states, actions, shots, metadata, half.cycles(),
                    rewards=np.asarray(rewards, dtype=np.float32), terminated=np.asarray(terminated, dtype=np.bool_),
                    stage=np.asarray(stages, dtype=np.int64))


def save_episode(store, episode, frames):
    """Append the episode to the store and its frames, compressed, to frames_<n>.npz beside it -> the episode path."""
    path = store.append(episode)
    if frames is not None:
        np.savez_compressed(path.with_name(path.name.replace("episode_", "frames_")), **frames)
    return path


def run(args):
    base = make_base(HALF_STEPS, cameras=None if args.no_frames else CAMERAS)
    half = ScriptedHalfFold(base)
    store = StateEpisodeStore(args.output)
    counts = {k: args.per_kind or PER_KIND[k] for k in args.kinds}
    manifest, started, n = [], time.time(), sum(counts.values())
    for kind in args.kinds:
        for i in range(counts[kind]):
            t0 = time.time()
            seed = args.seed_base + 1000 * KINDS.index(kind) + i
            if kind == "play":
                episode, frames = collect_play(base, seed, args.play_steps, frames=not args.no_frames)
            else:
                episode, frames = collect_half(half, kind, seed, frames=not args.no_frames)
            episode.metadata["split"] = "test" if i < (args.test_per_kind or counts[kind] // 5) else "train"
            episode.metadata["file"] = save_episode(store, episode, frames).name
            manifest.append(episode.metadata)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1))
            m, wall = episode.metadata, time.time() - t0
            print(f"EPISODE {len(manifest)}/{n} kind={kind} seed={seed} split={m['split']} steps={m['length']} "
                  f"end={m['termination_reason']} wall={wall:.0f}s rate={m['length'] / wall:.2f}steps/s "
                  f"elapsed={time.time() - started:.0f}s {json.dumps(m['stats'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="a new or empty directory")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--per-kind", type=int, help="episodes of every kind (default PER_KIND)")
    ap.add_argument("--test-per-kind", type=int, help="held-out episodes of every kind (default a fifth)")
    ap.add_argument("--play-steps", type=int, default=PLAY_STEPS, help="control steps per play episode (0.05 s)")
    ap.add_argument("--seed-base", type=int, default=1_000_000)
    ap.add_argument("--no-frames", action="store_true", help="state only: no camera rig, no frames files")
    args = ap.parse_args()
    per_kind = args.per_kind or min(PER_KIND[k] for k in args.kinds)
    if not 0 < (args.test_per_kind or per_kind // 5) < per_kind <= 1000 or args.play_steps < 1:
        ap.error("need 0 < test episodes < episodes <= 1000 per kind and --play-steps >= 1")
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f"{args.output} is not empty; choose a fresh output directory")
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
