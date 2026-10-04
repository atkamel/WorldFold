"""isaac/weld_check.py's left-arm script on MuJoCo's own weld: the reference its soft-weld bars come from (Phase W, W3).

Records the held corner's lag behind the gripper hand composed with the offset captured at engage, and the corner's
rise, for the same approach / descend / close / lift 10 cm / carry 10 cm / hold / release. Run in the MuJoCo venv
from the repo root:

    python isaac/weld_check_mujoco.py --out outputs/isaac/weld_mujoco_ref.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORNER = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/isaac/weld_mujoco_ref.json")
    args = ap.parse_args()

    from cloth_fold_rl.expert import solve_ik
    from imitation.tasks import make_env
    from mujuco.cloth_params import JOINT_DELTA_SCALE
    env = make_env("mujoco")
    b = env.unwrapped
    b.domain_randomization = False
    env.reset(seed=0, options={"cloth_pose": np.zeros(2)})
    p = "left_"
    adr, dof = b._arm_qpos_adr[p], b._arm_dof_adr[p]
    ranges = [b.model.joint(f"{p}{j}").range for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                       "wrist_roll")]
    body, hand = b._cloth_body_ids[CORNER], b.model.body(f"{p}gripper").id
    rep, state = {"phases": {}}, {"off": None}

    def go(target, grip, steps, tag):
        q, err = solve_ik(b.model, b.data, b._site_id[p], adr, dof, ranges, target, rng=np.random.default_rng(0))
        errs, zs = [], []
        for _ in range(steps):
            a = np.zeros(14, np.float32)
            a[0:5] = np.clip((q - np.array([b.data.qpos[i] for i in adr])) / JOINT_DELTA_SCALE, -1, 1)
            a[6], a[13] = grip, 1.0
            b.step(a)
            if b.grasp_active(p):
                R = b.data.xmat[hand].reshape(3, 3)
                if state["off"] is None:
                    state["off"] = R.T @ (b.data.xpos[body] - b.data.xpos[hand])
                errs.append(float(np.linalg.norm(b.data.xpos[body] - (b.data.xpos[hand] + R @ state["off"]))))
            zs.append(float(b.data.xpos[body][2]))
        rep["phases"][tag] = {"ik_err_mm": round(1000 * err, 1),
                              "max_track_mm": round(1000 * max(errs), 1) if errs else None,
                              "corner_z_start": round(zs[0], 4), "corner_z_end": round(zs[-1], 4),
                              "grasp_active": bool(b.grasp_active(p))}

    c = b.data.xpos[body].copy()
    go(c + [0, 0, 0.06], 1.0, 40, "approach")
    go(c + [0, 0, 0.005], 1.0, 30, "descend")
    go(c + [0, 0, 0.005], -1.0, 5, "close")
    go(c + [0, 0, 0.105], -1.0, 40, "lift")
    go(c + [0.10, 0, 0.105], -1.0, 40, "carry")
    go(c + [0.10, 0, 0.105], 0.0, 10, "hold_neutral")
    go(c + [0.10, 0, 0.105], 1.0, 30, "release")
    rep["max_track_mm"] = max(rep["phases"][k]["max_track_mm"] for k in ("lift", "carry", "hold_neutral"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print("MJWELD", json.dumps(rep))


if __name__ == "__main__":
    main()
