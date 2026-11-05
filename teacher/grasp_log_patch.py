"""Log every grasp the teacher makes (oracle/grasp_log.py): empty / slipped / held_lifted / held_dragged / held_static,
plus how many cloth layers were taken. Runtime patch of lehome-challenge/scripts/utils/remote_loop.py::_run_steps,
active only when TEACHER_GRASP_LOG is set (a directory). ORACLE_DIR points at teacher/oracle (default
/opt/teacher/oracle). Works with or without verts_dump_patch.py / isaac_timing_patch.py applied.

    python grasp_log_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-grasp-log-patch]"
p = ROOT / "lehome-challenge/scripts/utils/remote_loop.py"
s = p.read_text()
if MARK in s:
    print("already patched: grasp log")
    sys.exit(0)

HELPERS = '''

def _gl_setup(env, policy):
    """Grasp logger for this episode, or None when TEACHER_GRASP_LOG is unset. Never raises."""
    import os as _os
    _d = _os.environ.get("TEACHER_GRASP_LOG")
    if not _d:
        return None
    try:
        _od = _os.environ.get("ORACLE_DIR", "/opt/teacher/oracle")
        if _od not in sys.path:
            sys.path.insert(0, _od)
        import grasp_log as _gl
        import fold_plans as _fp
        prim = env.object._prim
        faces = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int64)
        cnt = np.asarray(prim.GetAttribute("faceVertexCounts").Get())
        faces = faces.reshape(-1, 3) if (cnt == 3).all() else None
        v = _gl_verts(env)
        table_z = float(np.percentile(v[:, 2], 1)) - 0.002
        name = getattr(policy, "_garment_name", None) or env.cfg.garment_name
        lg = _gl.GraspLogger(lambda q, side: _fp.K.fk(q, side), faces, table_z, episode=name)
        lg._dir, lg._name, lg._mod = _d, name, _gl
        return lg
    except Exception as e:
        print(f"GRASP_LOG setup failed: {e!r}", flush=True)
        return None


def _gl_verts(env):
    """World positions of the cloth vertices via the USD points attribute (fast path of oracle_fold.cloth_verts)."""
    o = env.object
    # pose read live every call (~0.8 ms): PhysX writes the cloth back with the prim moved to the cloth's bounding-box
    # centre and the points relative to it, so the pose drifts whenever the cloth is lifted or folded (2026-10-19)
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    pos, ori = o.get_world_pose()
    sc = o.get_world_scale()
    _n = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    A = (quat_to_rot_matrix(_n(ori)) * _n(sc)[None, :]).T
    pts = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float64)
    return pts @ np.asarray(A, np.float64) + np.asarray(_n(pos), np.float64)


def _gl_step(lg, env, obs, t):
    if lg is None:
        return
    try:
        st = obs["observation.state"] if isinstance(obs, dict) else env._get_observations()["observation.state"]
        st = st.detach().cpu().numpy() if hasattr(st, "detach") else np.asarray(st)
        for r in lg.update(t, st.ravel(), _gl_verts(env)):
            print("GRASP " + json.dumps(r), flush=True)
    except Exception as e:
        print(f"GRASP_LOG step failed: {e!r}", flush=True)


def _gl_finish(lg, t, success):
    if lg is None:
        return
    try:
        import os as _os, time as _tm
        for r in lg.close_episode(t):
            print("GRASP " + json.dumps(r), flush=True)
        _os.makedirs(lg._dir, exist_ok=True)
        with open(f"{lg._dir}/{lg._name}_{int(_tm.time() * 1000)}.jsonl", "w") as f:
            for r in lg.records:
                f.write(json.dumps(dict(r, success=bool(success))) + "\\n")
        print("GRASP_EPISODE " + json.dumps(dict(lg._mod.summary(lg.records), garment=lg._name, success=bool(success))), flush=True)
    except Exception as e:
        print(f"GRASP_LOG finish failed: {e!r}", flush=True)


def _run_steps('''

edits = [
    ("\ndef _run_steps(", HELPERS),
    ("    obs = env._get_observations()\n    episode_return = 0.0\n",
     "    obs = env._get_observations()\n    _gl = _gl_setup(env, policy)\n    episode_return = 0.0\n"),
    ("        obs = env.obs_buf\n",
     "        obs = env.obs_buf\n        _gl_step(_gl, env, obs, length)\n"),
    ('    return {"return": episode_return, "length": length, "success": success}\n',
     '    _gl_finish(_gl, length, success)\n    return {"return": episode_return, "length": length, "success": success}\n'),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
    s = s.replace(old, new)
for need in ("import json", "import sys", "import numpy as np"):
    if need not in s:
        s = need + "\n" + s
p.write_text(MARK + "\n" + s)
print("GRASP_LOG_PATCH_OK")
