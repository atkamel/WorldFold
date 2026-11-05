"""Grasp-on-demand test, variant 2 (2026-10-19 evening): restore a pre-grasp state as recorded, let the policy run
TEACHER_MOVE_AT_STEP policy steps (warm-up, its action history forms on the real state), THEN move the garment by
TEACHER_MOVE_SHIFT_CM in a random direction and +-TEACHER_MOVE_ROT_DEG about its own centre, and let the policy continue.
Separates the cold-start cost from the garment move. The move is applied to the cloth vertices themselves (through the
USD points attribute, the CPU pipeline's input path); Isaac rewrites the prim pose every step, so moving only the pose
would do nothing. Applied on top of grasp_test_patch.py. Lines: GRASPTEST_MOVE {json}.

    python grasp_test_patch2.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [teacher-grasp-test-patch-2]"
p = ROOT / "lehome-challenge/scripts/utils/remote_loop.py"
s = p.read_text()
if MARK in s:
    print("already patched: grasp test 2")
    sys.exit(0)
assert "# [teacher-grasp-test-patch]" in s, "apply grasp_test_patch.py first"

HELPERS = '''

def _gt_move(env, t, label):
    """Variant 2: at policy step TEACHER_MOVE_AT_STEP, move the cloth vertices rigidly (shift + yaw about the garment's
    centre, which is the prim origin) by rewriting the USD points attribute; velocities and the arms are left alone."""
    import os as _os, math as _m
    at = int(_os.environ.get("TEACHER_MOVE_AT_STEP", "0") or 0)
    if not at or t != at:
        return
    shift = float(_os.environ.get("TEACHER_MOVE_SHIFT_CM", "0") or 0) / 100.0
    rot = _m.radians(float(_os.environ.get("TEACHER_MOVE_ROT_DEG", "0") or 0))
    if not (shift or rot):
        return
    try:
        from pxr import Vt
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        o = env.object
        _n = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        pos, ori = o.get_world_pose()
        pos = np.asarray(_n(pos), np.float64).ravel()
        A = (quat_to_rot_matrix(_n(ori)) * np.asarray(_n(o.get_world_scale()), np.float64)[None, :]).T   # world = local @ A + pos
        local = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float64)
        world = local @ A + pos
        k = _gt_move.count
        _gt_move.count += 1
        rng = np.random.default_rng(int(_os.environ.get("TEACHER_RESTORE_SEED", "0") or 0) + 7919 * k + 17)
        ang = float(rng.uniform(0.0, 2.0 * _m.pi))
        yaw = float(rng.choice([-rot, rot])) if rot else 0.0
        c, s_ = _m.cos(yaw), _m.sin(yaw)
        R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        d = np.array([shift * _m.cos(ang), shift * _m.sin(ang), 0.0])
        new_world = (world - pos) @ R.T + pos + d                      # rotate about the garment centre, then shift
        new_local = (new_world - pos) @ np.linalg.inv(A)
        o._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(new_local.astype(np.float32)))
        info = dict(t=t, k=k, shift_cm=round(shift * 100, 2), rot_deg=round(_m.degrees(yaw), 2), dir_deg=round(_m.degrees(ang), 1))
        print("GRASPTEST_MOVE " + json.dumps(info), flush=True)
    except Exception as e:
        print(f"GRASPTEST move failed: {e!r}", flush=True)


_gt_move.count = 0


def _run_steps('''

edits = [
    ("\ndef _run_steps(", HELPERS),
    ("        _gt_snapshot(env, _gl, length)\n", "        _gt_snapshot(env, _gl, length)\n        _gt_move(env, length, label)\n"),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
    s = s.replace(old, new)
p.write_text(MARK + "\n" + s)
print("GRASP_TEST_PATCH2_OK")
