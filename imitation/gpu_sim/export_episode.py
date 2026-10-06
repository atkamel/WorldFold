"""Export one expert episode's physics inputs and states for GPU-simulator comparison (M5c.4).

    .venv/Scripts/python.exe -m imitation.gpu_sim.export_episode --seed 100000 --out outputs/imitation/gpu_sim/ep_100000

Runs in the pinned env (CPU MuJoCo 3.10). Writes:
  model.mjb        the compiled model *after* reset (domain randomization applied)
  episode.npz      initial qpos/qvel/act, and per control step: ctrl [T, nu], eq_active
                   [T, neq], eq_data [T, neq, 11] (weld offsets change on grasp), then the
                   CPU result after the 100 physics substeps: qpos_after [T, nq],
                   qvel_after [T, nv]; plus the fold outcome
Everything a simulator needs to replay the episode's physics without the env or the expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from imitation.tasks import HalfFoldEnv
from imitation.teachers import ScriptedTeacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--out", default="outputs/imitation/gpu_sim/ep_100000")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = HalfFoldEnv()
    teacher = ScriptedTeacher(env)
    env.reset(seed=args.seed)
    teacher.reset()
    base = env.unwrapped
    m, d = base.model, base.data
    mujoco.mj_saveModel(m, str(out / "model.mjb"), None)
    init = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(), "eq_active": d.eq_active.copy(),
            "eq_data": m.eq_data.copy()}

    rows = {k: [] for k in ("ctrl", "eq_active", "eq_data", "qpos_after", "qvel_after")}
    real_step = mujoco.mj_step
    captured = {}

    def recording_step(model, data):
        # the first substep of each control step sees the inputs the env just set
        if not captured:
            captured.update(ctrl=data.ctrl.copy(), eq_active=data.eq_active.copy(), eq_data=model.eq_data.copy())
        real_step(model, data)

    mujoco.mj_step = recording_step
    try:
        info = {}
        for t in range(base.max_episode_steps):
            captured.clear()
            _, _, term, trunc, info = env.step(teacher.act())
            for k in ("ctrl", "eq_active", "eq_data"):
                rows[k].append(captured[k])
            rows["qpos_after"].append(d.qpos.copy())
            rows["qvel_after"].append(d.qvel.copy())
            if term or trunc:
                break
    finally:
        mujoco.mj_step = real_step
    np.savez_compressed(out / "episode.npz", **{f"init_{k}": v for k, v in init.items()},
                        **{k: np.stack(v) for k, v in rows.items()})
    meta = {"seed": args.seed, "steps": len(rows["ctrl"]), "success": bool(info.get("success")),
            "fold_score": float(info.get("fold_score", 0)), "n_substeps": base.n_substeps,
            "timestep": float(m.opt.timestep), "nq": m.nq, "nv": m.nv, "nu": m.nu, "neq": m.neq,
            "nflex": m.nflex, "nflexvert": int(m.nflexvert), "nbody": m.nbody, "ngeom": m.ngeom,
            "mujoco": mujoco.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
