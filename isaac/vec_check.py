"""Physical check of the vectorised Isaac env (milestone V): one IsaacClothFoldBatch of B sub-envs (GPU weld profile)
in one process, driven by hand (no scheduler: every sync advances the whole batch). Run in the Isaac venv from the
repo root:

    python -u isaac/vec_check.py --n 3 --out outputs/isaac/vec

Writes vec.json with:
- copies_separate: each copy's cloth (world frame) sits at its copy offset
- local_frame_matches: sub-envs reset with the same seed read the same cloth, gripper and joints in their own frames
- dr_per_copy: each copy's particle masses and material read back scaled by ITS OWN episode's DR draw; draws differ
- dr_deterministic: a sub-env's DR draw depends on its seed only (same seed on another copy -> same draw)
- reset_isolated: resetting one copy (arms + cloth) leaves the other copies' particles, masses, material and arm
  joints untouched, and through the next steps a falling cloth on another copy keeps falling (it isn't re-parsed)
- weld_per_copy: a weld pin on one copy zeroes masses only there
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="outputs/isaac/vec")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    B = args.n

    from cloth_fold_rl.quarter_fold_env import GRASP_CORNERS, GRASP_RADIUS
    from isaac.isaac_env import IsaacClothFoldBatch
    batch = IsaacClothFoldBatch(B, observation_mode="state", max_episode_steps=1000, grasp_corners=GRASP_CORNERS,
                                grasp_radius=GRASP_RADIUS, profile="weld")
    lab, envs = batch.lab, batch.envs
    rep = {"B": B, "offsets": [cp.offset.tolist() for cp in lab.copies]}
    checks = {}

    def world_mean(c):
        return lab.copies[c].cloth._cloth_prim_view.get_world_positions()[0].detach().cpu().numpy().mean(0)

    def masses(c):
        return lab.copies[c].cloth._cloth_prim_view._physics_view.get_masses().detach().cpu().numpy().copy()

    def material(c):
        m = lab.copies[c].cloth.particle_material
        return float(m.get_friction()), float(m.get_damping())

    # ---- same seed on every copy, no DR: identical local readings ----
    for e in envs:                      # without a scheduler each reset's settle steps advance every copy
        e.reset(seed=5, options={"randomization": False})
    means = [world_mean(c) for c in range(B)]
    rep["cloth_world_mean"] = [np.round(m, 4).tolist() for m in means]
    checks["copies_separate"] = all(np.linalg.norm(means[c][:2] - lab.copies[c].offset[:2]) < 0.03 for c in range(B))
    c0 = envs[0].corner_positions()
    g0 = envs[0].gripper_position("left_")
    q0 = envs[0].joint_positions("right_")
    dc = [float(np.abs(e.corner_positions() - c0).max()) for e in envs]
    dg = [float(np.abs(e.gripper_position("left_") - g0).max()) for e in envs]
    dq = [float(np.abs(e.joint_positions("right_") - q0).max()) for e in envs]
    rep["local_corner_diff_m"], rep["local_gripper_diff_m"], rep["local_joint_diff_rad"] = dc, dg, dq
    checks["local_frame_matches"] = max(dc) < 0.003 and max(dg) < 1e-3 and max(dq) < 1e-3

    # ---- per-copy DR ----
    seeds = [100 + c for c in range(B)]
    for e, s in zip(envs, seeds):
        e.reset(seed=s, options={"randomization": True})
    dr = []
    for c, e in enumerate(envs):
        p = e._domain_params
        cp = lab.copies[c]
        m_ratio = float(np.median(masses(c) / cp.rest_masses.detach().cpu().numpy()))
        f, d = material(c)
        dr.append({"params": p, "mass_ratio": m_ratio, "friction_ratio": f / cp.rest_material[0],
                   "damping_ratio": d / cp.rest_material[1]})
    rep["dr"] = dr
    checks["dr_per_copy"] = all(
        abs(r["mass_ratio"] - r["params"]["cloth_mass_scale"]) < 1e-4
        and abs(r["friction_ratio"] - r["params"]["table_friction_scale"]) < 1e-4
        and abs(r["damping_ratio"] - r["params"]["cloth_damping_scale"]) < 1e-4 for r in dr) \
        and len({round(r["params"]["cloth_mass_scale"], 6) for r in dr}) == B
    # the same seed on the last copy draws what copy 0 drew
    envs[-1].reset(seed=seeds[0], options={"randomization": True})
    checks["dr_deterministic"] = envs[-1]._domain_params == dr[0]["params"]
    rep["dr_repeat"] = envs[-1]._domain_params

    # ---- reset isolation ----
    # copy 0: cloth dropped from 5 cm, mid-fall; its arm moving toward a target
    lab.reset_cloth(np.zeros(2), 0.05, (0.0, 0.0, 0.0), 0)
    a = np.zeros(14, dtype=np.float32)
    a[0] = 1.0
    for _ in range(2):
        envs[0].step(a)
    z_before = [float(world_mean(0)[2])]
    others = [c for c in range(B) if c != 1]
    snap = {c: (envs[c].lab.particle_positions(c).copy(), masses(c), material(c), envs[c].joint_positions("left_"))
            for c in others}
    victim = 1 if B > 1 else 0
    envs[victim].reset(seed=7, options={"randomization": True})   # reset_copy + soft reset + settle steps (global)
    # the settle steps moved everyone; immediately-after-reset isolation is checked on a bare reset_copy below
    for _ in range(3):
        envs[0].step(a)
        z_before.append(float(world_mean(0)[2]))
    rep["copy0_fall_z"] = z_before
    m_after = {c: masses(c) for c in others}
    mat_after = {c: material(c) for c in others}
    checks["reset_isolated_dynamics"] = all(np.allclose(m_after[c], snap[c][1]) and mat_after[c] == snap[c][2]
                                            for c in others)
    lab.reset_cloth(np.zeros(2), 0.05, (0.0, 0.0, 0.0), 0)
    envs[0].step(a)
    z0 = float(world_mean(0)[2])
    p_others = {c: lab.particle_positions(c).copy() for c in others}
    q_others = {c: envs[c].joint_positions("left_").copy() for c in others}
    batch.on_main(lab.reset_copy, victim)
    lab.reset_cloth(np.zeros(2), 0.02, (0.0, 0.0, 0.0), victim)
    still = all(np.array_equal(lab.particle_positions(c), p_others[c])
                and np.array_equal(envs[c].joint_positions("left_"), q_others[c]) for c in others)
    zs = [z0]
    for _ in range(4):
        envs[0].step(a)
        zs.append(float(world_mean(0)[2]))
    rep["copy0_fall_after_other_reset_z"] = zs
    # a re-parse of copy 0 would put it back at its reset pose (z jump up); it must keep falling or rest
    checks["reset_isolated"] = still and all(zs[k + 1] <= zs[k] + 1e-4 for k in range(len(zs) - 1))

    # ---- weld per copy ----
    if B > 1:
        mb = {c: masses(c) for c in range(B)}
        lab.copies[victim].pins = {"left_": (np.arange(10), np.zeros((10, 3)))}
        lab.set_pinned_masses(victim)
        mv = masses(victim).reshape(-1)
        checks["weld_per_copy"] = bool(np.all(mv[:10] == 0)) and all(
            np.array_equal(masses(c), mb[c]) for c in range(B) if c != victim)
        lab.copies[victim].pins = {}
        lab.set_pinned_masses(victim)

    rep["checks"] = checks
    rep["passed"] = all(checks.values())
    (out / "vec.json").write_text(json.dumps(rep, indent=1, default=float))
    print("VEC", json.dumps(rep, default=float), flush=True)


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
