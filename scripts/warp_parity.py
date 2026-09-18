"""Phase 1 check of docs/warp_port.md: replay a recorded episode through the CPU
env and the Warp sim from the same seed and compare the cloth step by step.

Three trajectories: the recording (MuJoCo 3.11, the version the world-model data
was collected on), ClothFoldEnv on the current CPU MuJoCo, and WarpClothSim.
The CPU-vs-recording gap is the yardstick: it is what a MuJoCo upgrade already
does to this cloth, so a Warp-vs-CPU gap of the same size is physics drift of a
kind the project has absorbed before.

    .venv-warp/bin/python scripts/warp_parity.py --episode outputs/cloth_angles/fold_state_v2/episode_000000.npz
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mujuco"))

from cloth_fold_rl.expert import FoldExpert  # noqa: E402
from cloth_fold_rl.fold_env import CLOTH_JITTER, SingleCornerFoldEnv  # noqa: E402
from cloth_angles.data.fold_observation import cloth_vertices  # noqa: E402
from warp_sim import WarpClothSim  # noqa: E402

REPORT_STEPS = (1, 5, 10, 25, 50, 100, 150, 200)


def expand(action6):
    a = np.zeros(14, dtype=np.float32)
    a[0:5], a[6] = action6[0:5], action6[5]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/cloth_angles/fold_state_v2/episode_000000.npz")
    ap.add_argument("--steps", type=int, default=None, help="replay only the first N actions")
    ap.add_argument("--output", default=None, help="write the per-step table as json")
    ap.add_argument("--solver", choices=["newton", "cg"], default=None, help="Warp-side solver override (CPU env keeps Newton)")
    ap.add_argument("--timestep", type=float, default=None, help="Warp-side timestep override in seconds (CPU env keeps 0.5 ms)")
    ap.add_argument("--source", choices=["recording", "expert"], default="recording",
                    help="recording: replay the npz actions open-loop in both sims. expert: FoldExpert drives the "
                         "CPU env closed-loop (so the grasp happens on this MuJoCo) and Warp replays its actions")
    args = ap.parse_args()

    rec = np.load(ROOT / args.episode)
    meta = json.loads(str(rec["metadata"]))
    actions = rec["actions"][: args.steps] if args.steps else rec["actions"]
    seed = int(meta["seed"])
    print(f"episode {args.episode}: kind={meta['kind']} seed={seed} length={len(rec['actions'])} "
          f"success={meta['success']} domain={ {k: round(v, 3) for k, v in meta['domain'].items()} }")

    # CPU env, same seed: the wrapper draws the cloth pose from default_rng(seed) and the
    # env draws the physics scales from np_random seeded the same way.
    env = SingleCornerFoldEnv(seed=seed)
    env.unwrapped.domain_randomization = True
    _, info = env.reset(seed=seed)
    base = env.unwrapped
    domain = info["domain_parameters"]
    cloth_pose = np.random.default_rng(seed).uniform(-CLOTH_JITTER, CLOTH_JITTER, size=2)
    assert np.allclose([domain[k] for k in ("cloth_mass_scale", "table_friction_scale", "cloth_damping_scale")],
                       [meta["domain"][k] for k in ("cloth_mass_scale", "table_friction_scale", "cloth_damping_scale")]), \
        "CPU reset did not reproduce the recording's physics scales"

    sim = WarpClothSim(nworld=1, **{k: v for k, v in (("solver", args.solver), ("timestep", args.timestep)) if v is not None})
    print(f"warp sim: solver={args.solver or 'newton'} timestep={(args.timestep or 5e-4) * 1e3:g} ms, {sim.n_substeps} substeps per control step")
    t0 = time.perf_counter()
    sim.reset(cloth_pose=cloth_pose[None], domain={k: [domain[k]] for k in
                                                   ("cloth_mass_scale", "table_friction_scale", "cloth_damping_scale")})
    print(f"warp reset + settle: {time.perf_counter() - t0:.1f}s on {sim.device}")

    expert = FoldExpert(env, seed=seed) if args.source == "expert" else None
    if expert is not None:
        expert.reset()
        actions = range(args.steps or base.max_episode_steps)

    def err(a, b):
        e = np.linalg.norm(a - b, axis=-1)
        return float(e.mean()), float(e.max())

    v_cpu, v_warp = cloth_vertices(base), sim.cloth_vertices()[0].cpu().numpy()
    rows = [{"step": 0, "warp_cpu": err(v_warp, v_cpu), "cpu_rec": err(v_cpu, rec["vertices"][0]),
             "z_offset": float((v_warp[:, 2] - v_cpu[:, 2]).mean()), "grasp": (False, False)}]
    print(f"step   0  warp-cpu mean/max {rows[0]['warp_cpu'][0] * 1e3:6.2f}/{rows[0]['warp_cpu'][1] * 1e3:6.2f} mm  "
          f"cpu-rec {rows[0]['cpu_rec'][0] * 1e3:6.2f}/{rows[0]['cpu_rec'][1] * 1e3:6.2f} mm  z-offset {rows[0]['z_offset'] * 1e3:5.2f} mm")
    t_cpu = t_warp = 0.0
    grasp_step = {"cpu": None, "warp": None, "rec": None}
    for t, a in enumerate(actions, start=1):
        if expert is not None:
            a = np.asarray(expert.act(), dtype=np.float32)
        t0 = time.perf_counter()
        _, _, term, trunc, info = env.step(a)
        t_cpu += time.perf_counter() - t0
        t0 = time.perf_counter()
        sim.step(torch.as_tensor(expand(a))[None])
        t_warp += time.perf_counter() - t0
        v_cpu, v_warp = cloth_vertices(base), sim.cloth_vertices()[0].cpu().numpy()
        g_cpu, g_warp = bool(info["grasped"]), bool(sim.grasp_active("left_")[0])
        for k, g in (("cpu", g_cpu), ("warp", g_warp), ("rec", t < len(rec["robot"]) and bool(rec["robot"][t][14] > 0.5))):
            if g and grasp_step[k] is None:
                grasp_step[k] = t
        row = {"step": t, "warp_cpu": err(v_warp, v_cpu), "cpu_rec": err(v_cpu, rec["vertices"][min(t, len(rec["vertices"]) - 1)]),
               "z_offset": float((v_warp[:, 2] - v_cpu[:, 2]).mean()), "grasp": (g_cpu, g_warp),
               "corner10_warp_cpu": float(np.linalg.norm(v_warp[10] - v_cpu[10])),
               "corner10_cpu_rec": float(np.linalg.norm(v_cpu[10] - rec["vertices"][min(t, len(rec["vertices"]) - 1)][10])),
               "overflow": int(sim.overflow[0])}
        rows.append(row)
        if t in REPORT_STEPS or t == len(actions) or term or trunc or (g_cpu != g_warp):
            print(f"step {t:3d}  warp-cpu mean/max {row['warp_cpu'][0] * 1e3:6.2f}/{row['warp_cpu'][1] * 1e3:6.2f} mm  "
                  f"cpu-rec {row['cpu_rec'][0] * 1e3:6.2f}/{row['cpu_rec'][1] * 1e3:6.2f} mm  z-offset {row['z_offset'] * 1e3:5.2f} mm  "
                  f"corner10 warp-cpu {row['corner10_warp_cpu'] * 1e3:6.2f} cpu-rec {row['corner10_cpu_rec'] * 1e3:6.2f} mm  "
                  f"grasp cpu={g_cpu} warp={g_warp}", flush=True)
        if row["overflow"]:
            print(f"  WARNING warp overflow flags {row['overflow']} at step {t}")
        if term or trunc:
            print(f"cpu episode ended at step {t}: {info['termination_reason']} (warp corner10-to-goal "
                  f"{float(np.linalg.norm(v_warp[10] - env._goal)) * 1e3:.1f} mm, cpu {info['corner_to_goal'] * 1e3:.1f} mm)")
            break
    print(f"grasp step: recording {grasp_step['rec']}, cpu {grasp_step['cpu']}, warp {grasp_step['warp']}")
    print(f"time per control step: cpu {t_cpu / len(rows[1:]) * 1e3:.0f} ms, warp {t_warp / len(rows[1:]) * 1e3:.0f} ms ({sim.device})")
    if args.output:
        Path(args.output).write_text(json.dumps({"episode": args.episode, "rows": rows, "grasp_step": grasp_step}))


if __name__ == "__main__":
    main()
