"""A recorded sim episode as a joint trajectory for the real arms (review 2026-11-10: nothing had been played on the
hardware). Reads one bc_collect episode and writes the per-step joint targets the simulator's position drives got
(measured joints + the clipped delta action * JOINT_DELTA_SCALE; the jaw's target as the observation carries it), the
measured joints for reference, and the layout the episode ran in, for hardware/so101/replay.py.

  .venv/bin/python isaac/tools/export_replay.py --version bc_smoke --seed 1000000 --out outputs/isaac/replay/s1000000.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_OBS = {"left": 0, "right": 25}       # each arm's proprio block in the 139-D observation: q 0-4, jaw target 10
ARM_ACT = {"left": 0, "right": 6}        # its 6 entries of the 12-D action


def trajectory(obs, actions, final_obs, delta_scale, bases=None, table_z=0.0):
    """obs [T, 139], actions [T, 12], final_obs [139] -> {arm: {"target": [T, 6], "q": [T + 1, 6]}} in sim radians;
    with bases {arm: xyz}, also "tip" [T + 1, 3]: the gripper site (right, forward, up from the table top) in metres
    from that arm's base, the arms facing +y."""
    allobs = np.vstack([obs, final_obs[None]])
    out = {}
    for arm, o in ARM_OBS.items():
        q = allobs[:, o:o + 5]
        jaw = allobs[:, o + 10]
        a = np.clip(actions[:, ARM_ACT[arm]:ARM_ACT[arm] + 5], -1.0, 1.0)
        target = np.column_stack([q[:-1] + a * delta_scale, jaw[1:]])       # the jaw target a step produced
        out[arm] = {"target": target, "q": np.column_stack([q, jaw])}
        if bases is not None:
            site = allobs[:, o + 11:o + 14]
            out[arm]["tip"] = np.column_stack([site[:, :2] - np.asarray(bases[arm])[:2], site[:, 2] - table_z])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="outputs/isaac/bc/datasets")
    ap.add_argument("--version", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from imitation.data.schema import load_dataset
    from mujoco_sim.cloth_params import (ARM_BASE_LEFT, ARM_BASE_QUAT, ARM_BASE_RIGHT, CLOTH_HALF, JOINT_DELTA_SCALE,
                                         TABLE_TOP_Z)
    _, episodes = load_dataset(args.root, args.version)
    ep = next(e for e in episodes if e.meta["seed"] == args.seed)
    arms = {arm: {k: np.round(v, 5).tolist() for k, v in tr.items()}
            for arm, tr in trajectory(ep.obs, ep.actions, ep.final_obs, JOINT_DELTA_SCALE,
                                       {"left": ARM_BASE_LEFT, "right": ARM_BASE_RIGHT}, TABLE_TOP_Z).items()}
    cloth_center = (0.0, -0.135)          # = isaac.sim.isaac_env.CLOTH_CENTER (importing it needs gymnasium only, kept literal)
    layout = {"frame": "table centre, z up, metres; the arms sit side by side on the -y edge facing +y",
              "arm_base": {"left": list(ARM_BASE_LEFT), "right": list(ARM_BASE_RIGHT)}, "arm_base_quat_wxyz": ARM_BASE_QUAT,
              "table_top_z": TABLE_TOP_Z, "cloth_center_xy": list(cloth_center), "cloth_side": 2 * CLOTH_HALF,
              "base_spacing": ARM_BASE_RIGHT[0] - ARM_BASE_LEFT[0],
              "base_to_cloth_centre_y": cloth_center[1] - ARM_BASE_LEFT[1]}
    doc = {"dt": 0.05, "joints": JOINTS, "arms": arms, "layout": layout,
           "source": {"dataset": args.version, "seed": args.seed, "steps": int(len(ep.obs)),
                      **{k: ep.meta.get(k) for k in ("mean", "edge_max", "grasp_knobs", "place_params")}}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc))
    print("wrote", args.out, len(ep.obs), "steps")


if __name__ == "__main__":
    main()
