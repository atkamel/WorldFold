"""Grasp-on-demand test for the teacher (2026-10-19): since the policy cannot be prompted, the direct grasp test is to
restore the sim to a state just before a grasp and let it act. Runtime patch of lehome-challenge/scripts/utils/
remote_loop.py, applied after grasp_log_patch.py. Two parts, each off unless its env var is set:

  harvest   TEACHER_SNAPSHOT_EVERY=K TEACHER_SNAPSHOT_DIR=dir   every K policy steps save the physics state (garment
            points / velocities / pose, both arms' joints) as dir/<garment>/<garment>_<tag>_t<step>.npz; <tag> is the
            grasp logger's episode tag, so select_pregrasp.py can pair snapshots with the logged jaw closes.
  test      TEACHER_RESTORE_DIR=dir [TEACHER_RESTORE_SHIFT_CM=3 TEACHER_RESTORE_ROT_DEG=15 TEACHER_RESTORE_SETTLE=3]
            at every episode start (after his reset + stabilise) restore the next snapshot of dir/<garment>/ in exact
            mode (his restore_physics_state), optionally with the garment shifted by SHIFT in a random direction and
            rotated by +-ROT about its own centre, settle SETTLE steps, then run the policy as usual (cap with
            --max_steps). The grasp logger scores what happens. Lines: GRASPTEST_RESTORE {json}.

    python grasp_test_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-grasp-test-patch]"
p = ROOT / "lehome-challenge/scripts/utils/remote_loop.py"
s = p.read_text()
if MARK in s:
    print("already patched: grasp test")
    sys.exit(0)
assert "# [teacher-grasp-log-patch]" in s, "apply grasp_log_patch.py first"

HELPERS = '''

def _gt_snapshot(env, lg, t):
    """Harvest: save the physics state every TEACHER_SNAPSHOT_EVERY policy steps (grasp-on-demand test)."""
    import os as _os
    K = int(_os.environ.get("TEACHER_SNAPSHOT_EVERY", "0") or 0)
    d = _os.environ.get("TEACHER_SNAPSHOT_DIR")
    if not K or not d or t % K:
        return
    try:
        o = env.object
        _n = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        pos, ori = o.get_world_pose()
        pts = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float32)
        va = o._prim.GetAttribute("velocities")
        vel = np.array(va.Get(), dtype=np.float32) if va and va.HasValue() else np.zeros_like(pts)
        name = getattr(lg, "_name", None) or env.cfg.garment_name
        tag = getattr(lg, "_tag", 0)
        _os.makedirs(_os.path.join(d, name), exist_ok=True)
        np.savez(_os.path.join(d, name, f"{name}_{tag}_t{t:04d}.npz"), garment_points=pts, garment_velocities=vel,
                 garment_pos=np.asarray(_n(pos), np.float32).ravel(), garment_ori=np.asarray(_n(ori), np.float32).ravel(),
                 left_joint_pos=_n(env.left_arm.data.joint_pos)[0].astype(np.float32),
                 right_joint_pos=_n(env.right_arm.data.joint_pos)[0].astype(np.float32), t=t)
    except Exception as e:
        print(f"GRASPTEST snapshot failed: {e!r}", flush=True)


def _gt_restore(env, garment, label):
    """Test: restore the next pre-grasp snapshot for this garment (exact mode), optionally shifted / rotated."""
    import os as _os, glob as _g, math as _m
    d = _os.environ.get("TEACHER_RESTORE_DIR")
    if not d:
        return None
    try:
        files = sorted(_g.glob(_os.path.join(d, garment, "*.npz")))
        if not files:
            print(f"GRASPTEST no snapshots for {garment} in {d}", flush=True)
            return None
        k = _gt_restore.counts.get(garment, 0)
        _gt_restore.counts[garment] = k + 1
        f = files[k % len(files)]
        z = np.load(f, allow_pickle=False)
        data = {"garment_points": z["garment_points"], "garment_velocities": z["garment_velocities"],
                "garment_pos": z["garment_pos"].astype(np.float32), "garment_ori": z["garment_ori"].astype(np.float32),
                "left_joint_pos": z["left_joint_pos"], "right_joint_pos": z["right_joint_pos"]}
        shift = float(_os.environ.get("TEACHER_RESTORE_SHIFT_CM", "0") or 0) / 100.0
        rot = _m.radians(float(_os.environ.get("TEACHER_RESTORE_ROT_DEG", "0") or 0))
        yaw = 0.0
        if shift or rot:
            rng = np.random.default_rng(int(_os.environ.get("TEACHER_RESTORE_SEED", "0") or 0) + 7919 * k + sum(map(ord, garment)))
            ang = float(rng.uniform(0.0, 2.0 * _m.pi))
            data["garment_pos"] = data["garment_pos"] + np.array([shift * _m.cos(ang), shift * _m.sin(ang), 0.0], np.float32)
            yaw = float(rng.choice([-rot, rot])) if rot else 0.0
            w, x, y, zq = [float(v) for v in data["garment_ori"]]          # (w, x, y, z); q_yaw (about world z) * q
            cw, sw = _m.cos(yaw / 2.0), _m.sin(yaw / 2.0)
            data["garment_ori"] = np.array([cw * w - sw * zq, cw * x - sw * y, cw * y + sw * x, cw * zq + sw * w], np.float32)
        ok = restore_physics_state(env, data, "exact", label)
        for _ in range(int(_os.environ.get("TEACHER_RESTORE_SETTLE", "3") or 0)):
            env.sim.step(render=False)
        info = dict(garment=garment, file=_os.path.basename(f), k=k, t=int(z["t"]), shift_cm=round(shift * 100, 2),
                    rot_deg=round(_m.degrees(yaw), 2), ok=bool(ok))
        print("GRASPTEST_RESTORE " + json.dumps(info), flush=True)
        return info
    except Exception as e:
        print(f"GRASPTEST restore failed: {e!r}", flush=True)
        return None


_gt_restore.counts = {}


def _run_steps('''

edits = [
    ("\ndef _run_steps(", HELPERS),
    # episode tag at logger setup, shared by the jsonl name and the snapshot names
    ("        lg._dir, lg._name, lg._mod = _d, name, _gl\n",
     "        lg._dir, lg._name, lg._mod = _d, name, _gl\n        lg._tag = int(__import__('time').time() * 1000)\n"),
    ('        with open(f"{lg._dir}/{lg._name}_{int(_tm.time() * 1000)}.jsonl", "w") as f:\n',
     '        with open(f"{lg._dir}/{lg._name}_{getattr(lg, \'_tag\', None) or int(_tm.time() * 1000)}.jsonl", "w") as f:\n'),
    # harvest hook, right after the grasp logger's per-step call
    ("        _gl_step(_gl, env, obs, length)\n",
     "        _gl_step(_gl, env, obs, length)\n        _gt_snapshot(env, _gl, length)\n"),
    # test hook, after his reset / stabilise / settle, before the episode is announced to the server
    ("        # --- tell the server the episode is set up -------------------------\n",
     "        _gt_restore(env, current_garment, label)\n\n"
     "        # --- tell the server the episode is set up -------------------------\n"),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
    s = s.replace(old, new)
p.write_text(MARK + "\n" + s)
print("GRASP_TEST_PATCH_OK")
