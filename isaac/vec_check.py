"""Physical check of the vectorised Isaac env (milestone V): one IsaacClothFoldBatch of B sub-envs (GPU weld profile)
in one process, driven by hand (no scheduler: every sync advances the whole batch). Run in the Isaac venv from the
repo root:

    python -u isaac/vec_check.py --n 3 --out outputs/isaac/vec

Writes vec.json with:
- copies_separate: each copy's cloth (world frame) sits at its copy offset
- local_frame_matches: sub-envs reset with the same seed read the same cloth, gripper and joints in their own frames
- dr_per_copy: each copy's particle masses and material read back scaled by ITS OWN episode's DR draw; draws differ
- dr_deterministic: a sub-env's DR draw depends on its seed only (same seed on another copy -> same draw)
- reset_isolated: resetting one copy (reset_copy + soft reset) changes nothing on two twin copies at that moment,
  and a twin's trajectory (welded corner lifted, cloth moving) with that reset mid-run equals the trajectory
  without it to within run-to-run noise
- reset_isolated_dynamics: the twins' DR'd masses and material survive it
- weld_per_copy: pinning particles on one copy changes masses only there
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
    # Copies 0 and 2 are twins: same seed and DR, the same pinned corner, the same actions, stepped by hand in
    # lockstep (step_submit / advance / step_collect), so the corner is lifted and the cloth moves. Run A leaves
    # copy 1 alone; run B resets copy 1 (reset_copy + soft reset: the per-env reset's scene writes) mid-run.
    # Isolation: copy 0's trajectory in B equals A's to within the run-to-run noise (A repeated), and the reset
    # changes nothing on the other copies at the moment it happens.
    victim, twins = 1, (0, 2)
    a = np.zeros(14, dtype=np.float32)
    a[1], a[3] = -1.0, 0.5             # shoulder lift up, wrist flex: the welded corner rises and swings

    def twin_run(reset_victim_at=None):
        for c in twins:
            envs[c].reset(seed=11, options={"randomization": True})
        for c in twins:                 # weld copy c's corner 0 patch to its left gripper wherever it is
            e = envs[c]
            site, _ = e.gripper_pose("left_")
            near = np.flatnonzero(np.linalg.norm(e._particles - e._particles[e._grid[0]], axis=1) < 0.012)
            e._pinned["left_"] = {0: (near, e._particles[near] - site)}
            e._push_pins()
        traj, still = [], None
        for k in range(8):
            if k == reset_victim_at:
                before = {c: (lab.particle_positions(c).copy(), envs[c].joint_positions("left_").copy(), masses(c))
                          for c in twins}
                batch.on_main(lab.reset_copy, victim)
                lab.reset_cloth(np.zeros(2), 0.02, (0.0, 0.0, 0.0), victim)
                still = all(np.array_equal(lab.particle_positions(c), before[c][0])
                            and np.array_equal(envs[c].joint_positions("left_"), before[c][1])
                            and np.array_equal(masses(c), before[c][2]) for c in twins)
            for c in twins:
                envs[c].step_submit(a)
            batch.advance()
            for c in twins:
                envs[c].step_collect()
            traj.append(np.concatenate([envs[0]._particles.ravel(), envs[0].joint_positions("left_")]))
        for c in twins:
            envs[c]._pinned["left_"] = {}
            envs[c]._push_pins()
        return np.stack(traj), still

    A1, _ = twin_run()
    A2, _ = twin_run()
    Bt, still = twin_run(reset_victim_at=3)
    noise = float(np.abs(A1 - A2).max())
    dev = float(np.abs(Bt - A1).max())
    moved = float(np.abs(A1[-1] - A1[0]).max())
    rep["isolation"] = {"run_to_run_noise": noise, "deviation_with_other_reset": dev, "copy0_motion": moved,
                        "unchanged_at_reset": still}
    checks["reset_isolated"] = bool(still) and moved > 0.01 and dev <= max(10 * noise, 1e-5)
    # the twins' DR'd masses and material survive the victim's reset and the steps after it
    checks["reset_isolated_dynamics"] = all(
        abs(float(np.median(masses(c) / lab.copies[c].rest_masses.detach().cpu().numpy()))
            - envs[c]._domain_params["cloth_mass_scale"]) < 1e-5
        and abs(material(c)[0] / lab.copies[c].rest_material[0] - envs[c]._domain_params["table_friction_scale"])
        < 1e-5 for c in twins)

    # ---- weld per copy: pinned masses change on that copy only ----
    # baseline first: the victim's soft reset above re-parsed its cloth (spawned masses) while its mass_scale kept the
    # last DR draw, which set_pinned_masses re-applies
    lab.set_pinned_masses(victim)
    mb = {c: masses(c) for c in range(B)}
    lab.copies[victim].pins = {"left_": (np.arange(10), np.zeros((10, 3)))}
    lab.set_pinned_masses(victim)
    mv, m0 = masses(victim).reshape(-1), mb[victim].reshape(-1)
    checks["weld_per_copy"] = bool(np.all(mv[:10] != m0[:10]) and np.array_equal(mv[10:], m0[10:])) and all(
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
