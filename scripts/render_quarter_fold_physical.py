"""Render an offline video of the PHYSICAL-GRASP two-arm quarter fold (no weld) --
the scoop-ramp + paddle contact grasp, not the weld cheat. Same offline pipeline
and render-quality fixes as scripts/render_quarter_fold.py (thin cosmetic cloth
radius, table-clip clamp on the LOCAL cloth qpos, free CLI camera, retreat tail);
only the env/expert and the render model's grabber plates differ.

    python scripts/render_quarter_fold_physical.py --seeds 6 --out renders/quarter_fold_physical.mp4
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mujuco"))

import imageio.v2 as imageio
import mujoco
import numpy as np
import sim_main

from cloth_fold_rl.expert import JOINT_DELTA_SCALE, solve_ik
from cloth_fold_rl.quarter_fold_physical_env import QuarterFoldPhysicalEnv, make_scoop_hook
from cloth_fold_rl.quarter_fold_physical_expert import QuarterFoldPhysicalExpert

NATIVE_FPS = 20   # 1 / control_dt
OUT_FPS = 60
RETREAT_STEPS = 35
# same reason as render_quarter_fold.py: mujoco.Renderer draws the flex's real
# collision radius as visual thickness, so render a thin-radius copy of the model
# (identical physics elsewhere) and copy live qpos into it each frame.
RENDER_CLOTH_RADIUS = 0.0015


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=str(REPO / "renders" / "quarter_fold_physical.mp4"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=1.0)
    args = ap.parse_args()

    env = QuarterFoldPhysicalEnv()
    base = env.unwrapped
    base.domain_randomization = True   # matches the expert's own validation harness
    expert = QuarterFoldPhysicalExpert(env)

    # thin-radius render copy WITH the same re-calibrated grabber plates, so the
    # ramp/paddle show and the cloth draws as thin fabric.
    orig_radius = sim_main.CLOTH_RADIUS
    sim_main.CLOTH_RADIUS = RENDER_CLOTH_RADIUS
    try:
        render_model = sim_main.compile_model(sim_main.ARM_TIMESTEP, base.grasp_corners,
                                              spec_hook=make_scoop_hook(env.scoop_mounts))
    finally:
        sim_main.CLOTH_RADIUS = orig_radius
    render_data = mujoco.MjData(render_model)
    renderer = mujoco.Renderer(render_model, height=args.height, width=args.width)
    render_cloth_body_ids = [render_model.body(f"cloth_{i}").id
                             for i in range(sim_main.CLOTH_COUNT ** 2)]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sim_main.CAMERA_TARGET
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    frame_dir = Path(tempfile.mkdtemp(prefix="quarter_fold_physical_frames_"))
    frame_idx = 0
    min_z = sim_main.TABLE_TOP_Z + RENDER_CLOTH_RADIUS + 0.0005

    def capture():
        nonlocal frame_idx
        render_data.qpos[:] = base.data.qpos
        mujoco.mj_forward(render_model, render_data)
        # cloth qpos is LOCAL/relative, so clamp the table dip by ADDING a delta
        for j, bid in zip(range(0, len(base._cloth_qpos_adr), 3), render_cloth_body_ids):
            world_z = render_data.xpos[bid][2]
            if world_z < min_z:
                render_data.qpos[base._cloth_qpos_adr[j + 2]] += min_z - world_z
        mujoco.mj_forward(render_model, render_data)
        renderer.update_scene(render_data, camera=cam)
        imageio.imwrite(frame_dir / f"frame_{frame_idx:06d}.png", renderer.render())
        frame_idx += 1

    def retreat():
        q_targets = {}
        for prefix in base.prefixes:
            site = base.data.site_xpos[base._site_id[prefix]].copy()
            sign = -1.0 if prefix == "left_" else 1.0
            target = site + np.array([sign * 0.15, -0.15, 0.2])
            q_targets[prefix], _ = solve_ik(base.model, base.data, base._site_id[prefix],
                                            base._arm_qpos_adr[prefix], base._arm_dof_adr[prefix],
                                            [base.model.joint(f"{prefix}{n}").range for n in
                                             ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                              "wrist_flex", "wrist_roll"]], target)
        for _ in range(RETREAT_STEPS):
            for prefix in base.prefixes:
                base.set_gripper(prefix, 1.0)   # open
                q_now = np.array([base.data.qpos[a] for a in base._arm_qpos_adr[prefix]])
                deltas = np.clip((q_targets[prefix] - q_now) / JOINT_DELTA_SCALE, -1.0, 1.0)
                base.apply_joint_delta(prefix, deltas)
            for _ in range(base.n_substeps):
                mujoco.mj_step(base.model, base.data)
            capture()

    seeds = args.seeds if args.seeds is not None else [args.seed_base + i for i in range(args.episodes)]
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        expert.reset()
        capture()
        for t in range(env.unwrapped.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(expert.act())
            capture()
            if terminated or truncated:
                print(f"episode seed={seed}: {info['termination_reason'] or 'truncated'}, "
                      f"success={info['success']}, fold_score={info['fold_score']:.3f}, "
                      f"move_distance={[round(x,3) for x in info['move_distance']]}, frames={frame_idx}")
                retreat()
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(NATIVE_FPS),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-r", str(OUT_FPS), "-pix_fmt", "yuv420p", str(out_path),
    ], check=True)
    shutil.rmtree(frame_dir)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
