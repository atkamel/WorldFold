"""Camera rig check on Isaac (Phase I, I1.3). Run in the Isaac venv from the repo root:

    python -u isaac/rig_check.py --out outputs/isaac/rig

Builds the half-fold env with the imitation camera rig (main 128, wrists 64), resets, steps, and writes
rig.json (frame shapes and dtypes, wrist camera pose error vs gripper link pose composed with the MJCF wrist_cam
offset, dict-mode steps/s) plus one PNG per camera. Exits through os._exit (Kit hangs at interpreter shutdown).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/isaac/rig")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from imitation.tasks import make_env
    from imitation.vision.render import CAMERAS
    env = make_env(backend="isaac", obs_mode="dict", cameras=CAMERAS)
    obs, _ = env.reset(seed=100000)
    t0 = time.time()
    for _ in range(args.steps):
        obs, *_ = env.step(np.zeros(12, dtype=np.float32))
    dict_steps_per_s = args.steps / (time.time() - t0)

    from PIL import Image
    from isaac.lab_scene import WRIST_CAM_POS, wrist_cam_quat
    base = env.unwrapped
    frames = {}
    for name, size in CAMERAS.items():
        img = obs[name]
        frames[name] = {"shape": list(img.shape), "dtype": str(img.dtype), "expected": [3, size, size],
                        "mean": float(img.mean()), "std": float(img.std())}
        Image.fromarray(np.transpose(img, (1, 2, 0))).save(out / f"{name}.png")

    poses = {}
    R_off = quat_to_matrix(wrist_cam_quat())
    for prefix in ("left_", "right_"):
        cam = base.lab.rig_cameras[f"{prefix}wrist_cam"]
        data, b, link_pos, link_R = base._gripper_link(prefix)
        want_pos = link_pos + link_R @ np.asarray(WRIST_CAM_POS)
        want_R = link_R @ R_off
        got_pos = cam.data.pos_w[0].cpu().numpy()
        got_R = quat_to_matrix(cam.data.quat_w_opengl[0].cpu().numpy())
        angle = np.degrees(np.arccos(np.clip((np.trace(want_R.T @ got_R) - 1) / 2, -1, 1)))
        poses[prefix] = {"pos_err_mm": float(1000 * np.linalg.norm(got_pos - want_pos)), "rot_err_deg": float(angle)}

    report = {"frames": frames, "wrist_pose": poses, "dict_steps_per_s": round(dict_steps_per_s, 2),
              "state_finite": bool(np.all(np.isfinite(obs["state"])))}
    (out / "rig.json").write_text(json.dumps(report, indent=1))
    print("RIG", json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
