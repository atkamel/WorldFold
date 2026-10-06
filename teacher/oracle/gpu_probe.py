"""GPU-pipeline diagnostic (ORACLE_GPU_PROBE=1, run with --device cuda:0). Answers two questions with numbers:
  1. Does LeHome run on IsaacLab's GPU pipeline, through reset and a garment rebuild (the earlier hang)?
  2. What does one physics step cost with the cloth's USD write-back on vs off (/physics/updateParticlesToUsd),
     and does ClothPrim (PhysX tensor read) still see the moving cloth when USD stops updating?
No optimization code: one PhysX setting and Isaac Sim's own ClothPrim view. A watchdog dumps every thread's stack
(faulthandler) and exits if no progress is made for WATCHDOG_S, so a hang shows where it is.
"""
import faulthandler
import json
import os
import sys
import time

import numpy as np

WATCHDOG_S = int(os.environ.get("ORACLE_PROBE_WATCHDOG", "300"))
USD_KEY = "/physics/updateParticlesToUsd"


def arm_watchdog(log, what):
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True, file=sys.stderr)
    log(f"PROBE step: {what} (watchdog {WATCHDOG_S}s)")


def probe(env, log, cloth_verts, set_cloth, garment):
    import carb
    import torch
    settings = carb.settings.get_settings()
    cuda = str(env.device).startswith("cuda")
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    res = dict(device=str(env.device), use_fabric=bool(env.cfg.sim.use_fabric),
               usd_setting_initial=settings.get(USD_KEY), steps={})

    def time_steps(name, n, fn):
        sync(); t = time.perf_counter()
        for _ in range(n):
            fn()
        sync(); ms = (time.perf_counter() - t) / n * 1000
        res["steps"][name] = round(ms, 2)
        log(f"PROBE_TIME {name}: {ms:.2f} ms/step over {n}")
        return ms

    phys = lambda: env.sim.step(render=False)

    arm_watchdog(log, "reset")
    env.reset()

    # ClothPrim: Isaac Sim's tensor view of the cloth (reads PhysX directly; GPU pipeline only)
    arm_watchdog(log, "ClothPrim view")
    view, read = None, None
    try:
        from isaacsim.core.prims import ClothPrim
        view = ClothPrim(prim_paths_expr=env.object.prim_path, name="probe_cloth_view")
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            view.initialize(SimulationManager.get_physics_sim_view())
        except Exception as e:
            log("PROBE ClothPrim initialize(physics_sim_view) failed, trying plain initialize():", repr(e))
            view.initialize()

        def read():
            p = view.get_world_positions()
            return p.detach().cpu().numpy()[0] if hasattr(p, "detach") else np.asarray(p)[0]
        cp0, usd0 = read(), cloth_verts(env)
        res["clothprim_points"] = int(len(cp0))
        res["clothprim_vs_usd_max_mm"] = round(float(np.abs(cp0[:len(usd0)] - usd0[:len(cp0)]).max()) * 1000, 3) \
            if len(cp0) == len(usd0) else f"count differs {len(cp0)} vs {len(usd0)}"
        log("PROBE ClothPrim ok:", res["clothprim_points"], "points; max |ClothPrim - USD| mm:", res["clothprim_vs_usd_max_mm"])
    except Exception as e:
        res["clothprim_error"] = repr(e)
        log("PROBE ClothPrim FAILED:", repr(e))

    # freshness while the cloth still falls after reset: USD write-back off -> does ClothPrim see motion, USD freeze?
    arm_watchdog(log, "freshness, USD write-back off")
    settings.set_bool(USD_KEY, False)
    usd_a = cloth_verts(env); cp_a = read() if read else None
    time_steps("physics_usd_off_falling", 60, phys)
    usd_b = cloth_verts(env); cp_b = read() if read else None
    res["usd_moved_mm_with_writeback_off"] = round(float(np.abs(usd_b - usd_a).max()) * 1000, 3)
    if cp_a is not None:
        res["clothprim_moved_mm_with_writeback_off"] = round(float(np.abs(cp_b - cp_a).max()) * 1000, 3)
    log("PROBE freshness:", {k: res.get(k) for k in ("usd_moved_mm_with_writeback_off", "clothprim_moved_mm_with_writeback_off")})

    arm_watchdog(log, "physics timing")
    settings.set_bool(USD_KEY, True)
    time_steps("physics_usd_on", 150, phys)
    settings.set_bool(USD_KEY, False)
    time_steps("physics_usd_off", 150, phys)
    if read:
        time_steps("clothprim_read_to_cpu", 50, read)

    # the full env.step (LeHome's cameras + observations as configured), holding the arms where they are
    arm_watchdog(log, "env.step timing")
    q = np.asarray(env._get_observations()["observation.state"], dtype=np.float32).ravel()[:12]
    act = torch.tensor(q, device=env.device).unsqueeze(0)
    step = lambda: env.step(act)
    settings.set_bool(USD_KEY, True)
    time_steps("env_step_usd_on", 100, step)
    settings.set_bool(USD_KEY, False)
    time_steps("env_step_usd_off", 100, step)
    settings.set_bool(USD_KEY, True)

    # the garment rebuild that hung before on cuda:0 (teleop's "real" cloth variant goes through switch_garment)
    arm_watchdog(log, "garment rebuild (set_cloth real)")
    t = time.perf_counter()
    try:
        set_cloth(env, garment, "real")
        res["garment_rebuild_s"] = round(time.perf_counter() - t, 1)
        log(f"PROBE garment rebuild ok in {res['garment_rebuild_s']} s")
        arm_watchdog(log, "after rebuild")
        env.reset()
        time_steps("physics_usd_on_after_rebuild", 60, phys)
        settings.set_bool(USD_KEY, False)
        time_steps("physics_usd_off_after_rebuild", 60, phys)
        settings.set_bool(USD_KEY, True)
    except Exception as e:
        res["garment_rebuild_error"] = repr(e)
        log("PROBE garment rebuild FAILED:", repr(e))
    faulthandler.cancel_dump_traceback_later()
    log("PROBE_RESULT " + json.dumps(res))
    return res
