"""GPU-physics feasibility check with MuJoCo Warp (roadmap M5c.4). Runs in `.venv-warp`.

    .venv-warp/Scripts/python.exe imitation/gpu_sim/warp_check.py --episode outputs/imitation/gpu_sim/ep_100000

Uses only mujoco / mujoco_warp / warp / numpy (no project imports, so the pinned env is
untouched). Three questions, answered in `warp_check.json` next to the episode:
  1. load    does mujoco_warp accept the exact compiled half-fold model (cloth flex included)?
  2. match   replaying the recorded CPU inputs (ctrl, weld switches, weld offsets) control
             step by control step: how far does the GPU trajectory drift from the CPU one?
             Also re-run on the CPU in this venv as a replay-procedure sanity check (must be 0).
  3. speed   physics steps / s on the GPU for nworld = 1 .. N parallel copies, vs the CPU.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np


def cpu_replay(m, ep, n_sub):
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = ep["init_qpos"], ep["init_qvel"]
    if m.na:
        d.act[:] = ep["init_act"]
    out = []
    for t in range(len(ep["ctrl"])):
        d.ctrl[:] = ep["ctrl"][t]
        d.eq_active[:] = ep["eq_active"][t]
        m.eq_data[:] = ep["eq_data"][t]
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/imitation/gpu_sim/ep_100000")
    ap.add_argument("--nworlds", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    ap.add_argument("--speed-steps", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.episode)
    meta = json.loads((root / "meta.json").read_text())
    ep = dict(np.load(root / "episode.npz"))
    n_sub = meta["n_substeps"]
    res = {"meta": meta, "mujoco": mujoco.__version__}

    m = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    # sanity: the replay procedure itself reproduces the CPU run exactly
    cpu = cpu_replay(m, ep, n_sub)
    res["cpu_replay_max_abs_qpos_err"] = float(np.abs(cpu - ep["qpos_after"]).max())

    import warp as wp
    import mujoco_warp as mjw
    res["warp"] = wp.__version__
    try:
        m = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
        mw = mjw.put_model(m)
        res["load"] = "ok"
    except Exception as e:                                   # unsupported feature, most likely the flex
        res["load"] = f"FAILED: {type(e).__name__}: {e}"
        res["load_trace"] = traceback.format_exc()[-2000:]
        (root / "warp_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "load_trace"}, indent=1))
        return

    # 2. trajectory match, one world
    try:
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = ep["init_qpos"], ep["init_qvel"]
        dw = mjw.put_data(m, d)
        errs, t0 = [], time.perf_counter()
        for t in range(len(ep["ctrl"])):
            wp.copy(dw.ctrl, wp.array(ep["ctrl"][t][None].astype(np.float32)))
            wp.copy(dw.eq_active, wp.array(ep["eq_active"][t][None].astype(bool)))
            wp.copy(mw.eq_data, wp.array(ep["eq_data"][t].astype(np.float32), dtype=mw.eq_data.dtype))
            for _ in range(n_sub):
                mjw.step(mw, dw)
            q = dw.qpos.numpy()[0]
            errs.append(float(np.abs(q - ep["qpos_after"][t]).max()))
        wp.synchronize()
        res["match"] = {"steps": len(errs), "max_abs_qpos_err_first_step": errs[0],
                        "max_abs_qpos_err_step10": errs[min(9, len(errs) - 1)],
                        "max_abs_qpos_err_final": errs[-1], "max_abs_qpos_err_any": max(errs),
                        "first_step_over_1cm": next((i for i, e in enumerate(errs) if e > 0.01), None),
                        "seconds": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        res["match"] = f"FAILED: {type(e).__name__}: {e}"
        res["match_trace"] = traceback.format_exc()[-2000:]

    # 3. throughput: physics substeps per second, CPU vs GPU with nworld copies
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = ep["init_qpos"], ep["init_qvel"]
    t0 = time.perf_counter()
    for _ in range(args.speed_steps):
        mujoco.mj_step(m, d)
    res["cpu_steps_per_s_1thread"] = round(args.speed_steps / (time.perf_counter() - t0), 1)
    speed = {}
    for nw in args.nworlds:
        try:
            d = mujoco.MjData(m)
            d.qpos[:], d.qvel[:] = ep["init_qpos"], ep["init_qvel"]
            dw = mjw.put_data(m, d, nworld=nw)
            for _ in range(5):
                mjw.step(mw, dw)
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.speed_steps):
                mjw.step(mw, dw)
            wp.synchronize()
            dt = time.perf_counter() - t0
            speed[nw] = {"world_steps_per_s": round(nw * args.speed_steps / dt, 1),
                         "ms_per_batched_step": round(1000 * dt / args.speed_steps, 2)}
        except Exception as e:
            speed[nw] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
    res["gpu_speed"] = speed
    (root / "warp_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("trace")}, indent=1))


if __name__ == "__main__":
    main()
