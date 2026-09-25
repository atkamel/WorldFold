"""Render WorldFold's three teacher-facing camera views (top, left wrist, right wrist)
at the reset pose, 640x480, and a side-by-side sheet against the teacher's replay frame.

    python teacher/render_cams.py --white   (needs the cloth_fold_rl requirements; reference frame optional)
"""
import argparse, os, sys
import numpy as np, imageio, mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # WorldFold repo root
from mujuco.sim_main import ClothFoldEnv, TABLE_TOP_Z  # noqa: E402

W, H = 640, 480
TOP_FOVY = 53.0      # teacher top cam: 67 deg horizontal at 4:3 -> ~53 deg vertical
WRIST_FOVY = 41.5    # teacher wrist cams: ~54 deg horizontal -> ~41.5 deg vertical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elev", type=float, default=-75.0)
    ap.add_argument("--dist", type=float, default=0.62)
    ap.add_argument("--white", action="store_true")
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--pan", type=float, default=1.1363, help="teacher init shoulder_pan magnitude (rad)")
    ap.add_argument("--out", default=os.path.join(ROOT, "teacher", "ref_frames"))
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--ref", default="replay_0000.png")
    args = ap.parse_args()

    def spec_hook(spec):
        # bright gradient skybox so off-table views are not black void
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.95, 0.95, 0.95], rgb2=[0.75, 0.78, 0.82], width=256, height=256)

    env = ClothFoldEnv(observation_mode="state", action_mode="joint_delta", spec_hook=spec_hook)
    env.reset(seed=0)
    m, d = env.model, env.data
    # teacher init pose: every arm joint 0 except shoulder_pan = -/+pan (turned toward centre)
    for prefix, sign in (("left_", -1.0), ("right_", 1.0)):
        for jn in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
            v = sign * args.pan if jn == "shoulder_pan" else 0.0
            d.qpos[m.joint(f"{prefix}{jn}").qposadr[0]] = v
            d.ctrl[m.actuator(f"{prefix}{jn}").id] = v
    mujoco.mj_forward(m, d)
    if args.white:
        # teacher scenes: white marble table, bright room. cheap appearance match.
        m.geom_rgba[m.geom("table").id] = (0.92, 0.92, 0.90, 1.0)
        m.geom_rgba[m.geom("floor").id] = (0.85, 0.85, 0.85, 1.0)
        m.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
        m.vis.headlight.diffuse[:] = (0.45, 0.45, 0.45)

    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    print("model cameras:", cams)
    for name in cams:
        if name and "wrist_cam" in name:
            m.cam_fovy[m.camera(name).id] = WRIST_FOVY
    m.vis.global_.fovy = TOP_FOVY

    r = mujoco.Renderer(m, height=H, width=W)
    top = mujoco.MjvCamera()
    top.type = mujoco.mjtCamera.mjCAMERA_FREE
    top.lookat[:] = (0.0, 0.0, TABLE_TOP_Z)
    top.distance, top.azimuth, top.elevation = args.dist, args.az, args.elev
    r.update_scene(d, camera=top); top_img = r.render().copy()

    wrist = {}
    for name in cams:
        if name and "wrist_cam" in name:
            r.update_scene(d, camera=name); wrist[name] = r.render().copy()

    os.makedirs(args.out, exist_ok=True)
    imageio.imwrite(os.path.join(args.out, f"wf_top{args.tag}.png"), top_img)
    for name, img in wrist.items():
        imageio.imwrite(os.path.join(args.out, f"wf_{name}{args.tag}.png"), img)

    ref_path = os.path.join(args.out, args.ref)
    ref = imageio.imread(ref_path)[..., :3] if os.path.exists(ref_path) else np.zeros_like(top_img)
    row1 = np.concatenate([ref, top_img], axis=1)
    ws = list(wrist.values()) or [np.zeros_like(top_img)] * 2
    row2 = np.concatenate(ws[:2] if len(ws) >= 2 else ws + [np.zeros_like(top_img)], axis=1)
    sheet = np.concatenate([row1, row2], axis=0)
    imageio.imwrite(os.path.join(args.out, f"compare_sheet{args.tag}.png"), sheet)
    print("wrote", os.path.join(args.out, f"compare_sheet{args.tag}.png"), "cams:", list(wrist))
    env.close()


if __name__ == "__main__":
    main()
