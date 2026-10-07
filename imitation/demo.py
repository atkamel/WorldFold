"""Half-fold demo: roll out a pipeline-trained checkpoint (or the expert) and render an mp4.

    python -m imitation.demo --ckpt outputs/imitation/runs/dagger_v1/best.pt --seeds 100000 100001
    python -m imitation.demo --ckpt expert --out outputs/imitation/demo/expert.mp4

Runs the policy exactly as `evaluate` does -- chunk of K actions, re-planned every
`--replan-every` steps from the last `obs_horizon` observations -- in one process,
rendering offscreen with MuJoCo's renderer. Default seeds are from the held-out
`id_easy` set, never training seeds. Writes `<out>.json` with each episode's outcome.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from imitation.seeds import EVAL_SEED_BASE
from imitation.tasks import HalfFoldEnv, is_isaac


def _overlay(frame, lines):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, 360, 14 + 18 * len(lines)], fill=(0, 0, 0, 150))
    for i, text in enumerate(lines):
        draw.text((10, 8 + 18 * i), text, fill=(235, 235, 235))
    return np.asarray(img)


def run_episode(env, act_fn, seed, renderer, cam, label, frames, hold=20):
    obs, info = env.reset(seed=seed)
    act_fn.reset(obs, seed)
    base = env.unwrapped
    for t in range(env.unwrapped.max_episode_steps):
        obs, _, term, trunc, info = env.step(act_fn(obs))
        renderer.update_scene(base.data, camera=cam)
        frames.append(_overlay(renderer.render(), [
            label, f"seed {seed}  step {t + 1}", f"fold score {info['fold_score']:.2f}",
            f"grasped L {int(info['grasped']['left_'])} R {int(info['grasped']['right_'])}"]))
        if term or trunc:
            break
    verdict = "SUCCESS" if info["success"] else f"FAIL ({info['termination_reason'] or 'truncated'})"
    frames.extend([_overlay(frames[-1], [verdict])] * hold)
    return {"seed": seed, "steps": t + 1, "success": bool(info["success"]),
            "fold_score": float(info["fold_score"]), "termination_reason": info["termination_reason"]}


class PolicyActor:
    """Chunked closed-loop control, matching `rollout.PolicyController`. A sensor-only
    (vision) checkpoint gets the robot cameras rendered each replan, with the same
    per-episode visual randomization as training."""

    def __init__(self, ckpt, replan_every, env):
        from imitation.policies.common import load_policy
        self.policy, self.replan_every = load_policy(ckpt), replan_every
        self.rig = None
        if self.policy.needs_images:
            from imitation.vision.render import CameraRig
            self.rig = CameraRig(env, dict(self.policy.cameras))   # render at the policy's sizes

    def reset(self, obs, seed):
        self.hist = deque([obs] * self.policy.obs_horizon, maxlen=self.policy.obs_horizon)
        self.queue = deque()
        if self.rig:
            self.rig.reset(seed)

    def __call__(self, obs):
        from imitation.rollout import padded_predict
        if obs is not self.hist[-1]:
            self.hist.append(obs)
        if not self.queue:
            images = [self.rig.render()] if self.rig else None
            chunk = padded_predict(self.policy, np.stack(self.hist)[None].astype(np.float32), images)[0]
            self.queue.extend(chunk[:self.replan_every])
        return self.queue.popleft()


class ExpertActor:
    def __init__(self, env):
        from imitation.teachers import ScriptedTeacher
        self.teacher = ScriptedTeacher(env)

    def reset(self, obs, seed):
        self.teacher.reset()

    def __call__(self, obs):
        return self.teacher.act()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy checkpoint, or 'expert'")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the first 3 id_easy seeds")
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--out", default="outputs/imitation/demo/half_fold.mp4")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--fps", type=int, default=20)   # control_dt = 0.05 s -> real time
    ap.add_argument("--backend", choices=("mujoco", "isaac", "isaac_weld", "isaac_friction", "isaac_anchor"), default="mujoco")
    ap.add_argument("--demo-size", type=int, default=512, help="isaac: square size of the demo camera")
    ap.add_argument("--set", default=None, choices=("id_easy", "id_hard", "recovery"),
                    help="isaac: play episodes of this eval set (its seeds unless --seeds is given, its shifted "
                         "poses and its knock), labelled in the video")
    ap.add_argument("--n", type=int, default=3, help="with --set and no --seeds: the set's first n seeds")
    args = ap.parse_args()
    args.seeds_given = args.seeds is not None
    if args.seeds is None:
        args.seeds = [EVAL_SEED_BASE["id_easy"] + i for i in range(3)]
    if is_isaac(args.backend):
        return isaac_main(args)

    import mujoco                    # MuJoCo demo only; the Isaac venv has no mujoco
    env = HalfFoldEnv()
    base = env.unwrapped
    if args.ckpt == "expert":
        act_fn, label = ExpertActor(env), "scripted expert"
    else:
        act_fn = PolicyActor(args.ckpt, args.replan_every, env)
        label = f"{'vision' if act_fn.rig else 'state'} policy {Path(args.ckpt).parent.parent.name}/{Path(args.ckpt).parent.name}"
    base.model.vis.global_.offwidth = max(base.model.vis.global_.offwidth, args.width)
    base.model.vis.global_.offheight = max(base.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(base.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    env.reset(seed=args.seeds[0])
    cam.lookat[:] = base.data.xpos[base._cloth_body_ids].mean(axis=0)
    cam.distance, cam.azimuth, cam.elevation = 0.9, 90.0, -50.0

    frames, rows = [], []
    for seed in args.seeds:
        rows.append(run_episode(env, act_fn, seed, renderer, cam, label, frames))
        print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=16)
    out.with_suffix(".json").write_text(json.dumps({"ckpt": args.ckpt, "episodes": rows}, indent=1))
    print(f"wrote {out} ({len(frames)} frames), success {sum(r['success'] for r in rows)}/{len(rows)}")


def isaac_main(args):
    """The same demo on Isaac Sim (Phase I). The frames come from a `demo` camera in the scene (the main view at
    --demo-size); a vision checkpoint gets its own rig cameras from the same dict observation, as in evaluation.
    One Isaac env per process, and Kit hangs at interpreter shutdown, so this exits through os._exit."""
    import os
    from imitation.tasks import make_env
    from imitation.teachers.scripted import ScriptedTeacher
    policy = None
    cams = {"demo": args.demo_size}
    if args.ckpt != "expert":
        from imitation.policies.common import load_policy
        policy = load_policy(args.ckpt)
        if policy.needs_images:
            cams.update(dict(policy.cameras))
    env = make_env(backend=args.backend, obs_mode="dict", cameras=cams)
    teacher = ScriptedTeacher(env) if policy is None else None
    label = ("scripted expert" if policy is None else
             f"{'vision' if policy.needs_images else 'state'} policy {Path(args.ckpt).parent.name}") + "  [Isaac Sim]"

    def frame(obs):
        return np.ascontiguousarray(np.transpose(obs["demo"], (1, 2, 0)))

    from imitation.seeds import eval_set
    reset_options = perturb_fn = None
    if args.set:                     # the evaluator's own set definition: same poses, same knock draw
        set_seeds, reset_options, perturb_fn = eval_set(args.set, args.n, args.backend)
        if not args.seeds_given:
            args.seeds = set_seeds
        label += f"  [{args.set}]"
    frames, rows = [], []
    for seed in args.seeds:
        obs, info = env.reset(seed=seed, options=reset_options(seed) if reset_options else None)
        rng = np.random.default_rng([seed, 7919])          # as rollout: the knock is drawn from this stream
        knock = perturb_fn(seed, rng) if perturb_fn else None
        if teacher:
            teacher.reset()
        hist, queue = deque([obs["state"]] * (policy.obs_horizon if policy else 1), maxlen=policy.obs_horizon
                            if policy else 1), deque()
        for t in range(env.unwrapped.max_episode_steps):
            knocked = knock is not None and knock.active(t)
            if knocked:                                       # k uniform-random actions, then the policy replans
                action = rng.uniform(-1, 1, len(env.action_space.low)).astype(np.float32)
                queue.clear()
            elif teacher:
                action = teacher.act()
            else:
                if not queue:
                    from imitation.rollout import padded_predict
                    images = [{c: obs[c] for c in dict(policy.cameras)}] if policy.needs_images else None
                    chunk = padded_predict(policy, np.stack(hist)[None].astype(np.float32), images)[0]
                    queue.extend(chunk[:args.replan_every])
                action = queue.popleft()
            obs, _, term, trunc, info = env.step(action)
            hist.append(obs["state"])
            frames.append(_overlay(frame(obs), [
                label, f"seed {seed}  step {t + 1}", f"fold score {info['fold_score']:.2f}",
                f"grasped L {int(info['grasped']['left_'])} R {int(info['grasped']['right_'])}"]
                + (["KNOCK (random actions)"] if knocked else [])))
            if term or trunc:
                break
        verdict = "SUCCESS" if info["success"] else f"FAIL ({info['termination_reason'] or 'truncated'})"
        frames.extend([_overlay(frames[-1], [verdict])] * 20)
        rows.append({"seed": seed, "steps": t + 1, "success": bool(info["success"]),
                     "fold_score": float(info["fold_score"]), "termination_reason": info["termination_reason"]})
        print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=16)
    out.with_suffix(".json").write_text(json.dumps({"ckpt": args.ckpt, "backend": args.backend, "episodes": rows},
                                                   indent=1))
    print(f"wrote {out} ({len(frames)} frames), success {sum(r['success'] for r in rows)}/{len(rows)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
