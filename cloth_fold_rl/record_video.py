"""Record a fold policy to an mp4, with a metrics overlay burned into the frames.

    python -m cloth_fold_rl.record_video --episodes 3
    python -m cloth_fold_rl.record_video --policy expert --out outputs/videos/expert.mp4
    python -m cloth_fold_rl.record_video --task quarter --policy expert \
        --episodes 1 --seed-base 100 --out outputs/videos/quarter_fold_expert.mp4
    python -m cloth_fold_rl.record_video --task quarter --policy bc \
        --checkpoint outputs/cloth_angles/quarter_policy_markov_d4/bc_staged.pt

Renders offscreen through MuJoCo's own renderer (not mjviser), so this works
headless and does not need the viewer running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from cloth_fold_rl.fold_env import make_fold_env, make_expert, SUCCESS_DIST

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(frame, lines, font, accent=(90, 220, 120)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    pad, lh = 14, font.size + 8
    box_h = pad * 2 + lh * len(lines)
    draw.rectangle([0, 0, 340, box_h], fill=(0, 0, 0, 150))
    for i, (text, hot) in enumerate(lines):
        draw.text((pad, pad + i * lh), text, font=font,
                  fill=accent if hot else (235, 235, 235))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["single", "quarter"], default="single")
    ap.add_argument("--policy", choices=["ppo", "expert", "bc"], default="ppo",
                    help="bc: a staged actor checkpoint from scripts/train_quarter_bc.py (quarter only)")
    ap.add_argument("--checkpoint", default="outputs/cloth_fold_rl/run2/best.zip")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)   # multiple of 16 for libx264
    ap.add_argument("--camera", default="main")
    ap.add_argument("--fps", type=int, default=20)   # control_dt=0.05 -> real time
    ap.add_argument("--hold-frames", type=int, default=25)  # freeze on the result
    ap.add_argument("--physical", action="store_true", help="use the physical grabber (plates, no weld) -- see physical_env.py")
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="default 200 (weld) / 250 (physical)")
    args = ap.parse_args()

    if args.task == "quarter":
        if args.physical:
            raise SystemExit("--physical records the single task")
        from cloth_fold_rl.quarter_fold_env import QuarterFoldEnv
        env = QuarterFoldEnv()
    else:
        env = make_fold_env(args.physical, max_episode_steps=args.max_episode_steps)
    base = env.unwrapped

    if args.policy == "expert":
        if args.task == "quarter":
            from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
            agent = QuarterFoldExpert(env)
            label = "quarter-fold scripted expert"
        else:
            agent = make_expert(env, args.physical)
            label = "single-fold scripted expert" + (" (physical grasp)" if args.physical else "")
        act = lambda obs: agent.act()          # noqa: E731
    elif args.policy == "bc":
        import torch
        from cloth_angles.data.fold_observation import observe_state
        from cloth_angles.model.actor_critic import policy_features
        from cloth_angles.tasks import QUARTER
        from scripts.train_imagined_actor import load_policy_actors
        if args.task != "quarter":
            raise SystemExit("--policy bc records the quarter task")
        saved = torch.load(args.checkpoint, map_location="cpu")
        actors = load_policy_actors(saved, QUARTER)
        agent = None
        label = f"imitation policy ({Path(args.checkpoint).parent.name})"

        def act(obs):
            stage = env.stage
            state = torch.as_tensor(observe_state(base, QUARTER))[None]
            goal = torch.as_tensor(QUARTER.goals(env))[None]
            with torch.no_grad():
                return actors[stage](policy_features(state, goal, saved["state_mean"], saved["state_scale"],
                                                     QUARTER, torch.tensor([stage])))[0].numpy()
    else:
        from stable_baselines3 import PPO
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        model = PPO.load(ckpt)
        agent = None
        label = f"PPO ({ckpt.name})" + (" physical grasp" if args.physical else "")
        act = lambda obs: model.predict(obs, deterministic=True)[0]   # noqa: E731

    renderer = mujoco.Renderer(base.model, height=args.height, width=args.width)
    font = load_font(20)
    frames = []
    results = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        if agent is not None:
            agent.reset()
        print(f"episode {ep}: ", end="", flush=True)

        for t in range(base.max_episode_steps):
            obs, r, term, trunc, info = env.step(act(obs))
            renderer.update_scene(base.data, camera=args.camera)
            if args.task == "quarter":
                distances = " / ".join(f"{d:.3f}" for d in info["move_distance"])
                grasped = "".join("1" if info["grasped"][p] else "0" for p in ("left_", "right_"))
                lines = [
                    (label, False),
                    (f"episode {ep + 1}/{args.episodes}   step {t + 1}", False),
                    (f"stage  {info['stage'] + 1}/2", info["stage"] == 1),
                    (f"fold_score  {info['fold_score']:.3f}", info["fold_score"] > 0.8),
                    (f"corner->goal  {distances} m", all(d < SUCCESS_DIST for d in info["move_distance"])),
                    (f"grasped L/R  {grasped}   settle {info['settle_steps']}", info["settle_steps"] > 0),
                ]
            else:
                lines = [
                    (label, False),
                    (f"episode {ep + 1}/{args.episodes}   step {t + 1}", False),
                    (f"fold_score  {info['fold_score']:.3f}", info["fold_score"] > 0.8),
                    (f"corner->goal  {info['corner_to_goal']:.3f} m "
                     f"(<{SUCCESS_DIST})", info["corner_to_goal"] < SUCCESS_DIST),
                    (f"grasped  {'YES' if info['grasped'] else 'no'}", info["grasped"]),
                ]
            frames.append(overlay(renderer.render(), lines, font))
            if term or trunc:
                break

        ok = bool(info["success"])
        results.append(ok)
        print(f"{'SUCCESS' if ok else 'failed'} "
              f"(score {info['fold_score']:.3f}, {t + 1} steps)")

        # hold the final frame so the result is readable
        tail_lines = [
            (label, False),
            (f"episode {ep + 1}/{args.episodes}", False),
            ("SUCCESS - cloth folded" if ok else "failed", ok),
            (f"fold_score  {info['fold_score']:.3f}", ok),
        ]
        if args.task == "quarter":
            tail_lines.append((f"stage  {info['stage'] + 1}/2   settle {info['settle_steps']}", ok))
        else:
            tail_lines.append((f"corner->goal  {info['corner_to_goal']:.3f} m", ok))
        tail = overlay(renderer.render(), tail_lines, font)
        frames.extend([tail] * args.hold_frames)

    out = Path(args.out or f"outputs/videos/{args.task}_fold_{args.policy}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(out, np.stack(frames), fps=args.fps, codec="libx264")

    n = sum(results)
    print(f"\n{len(frames)} frames @ {args.fps}fps "
          f"({len(frames)/args.fps:.1f}s) -> {out}")
    print(f"{n}/{len(results)} episodes succeeded")


if __name__ == "__main__":
    main()
