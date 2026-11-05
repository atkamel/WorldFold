"""Offline checks of grasp_log.py with synthetic cloth (no Isaac):  python test_grasp_log.py"""
import numpy as np

import grasp_log as gl

TABLE = 0.52


def tip_pose(q, side):
    """Test kinematics: q[0:3] is the jaw-tip position, tip frame axes = world axes."""
    T = np.eye(4); T[:3, 3] = q[:3]
    return T


def box_center(tip):
    """World position of the pinch-box centre for a tip at `tip` (inverse of pad_mask's transform)."""
    c = np.array([(gl.BOX["x_lo"] + gl.BOX["x_hi"]) / 2, 0.0, (gl.BOX["z_lo"] + gl.BOX["z_hi"]) / 2, 1.0])
    return (tip_pose(tip, "left") @ np.linalg.inv(gl.T_LINK_TIP) @ c)[:3]


def grid(center, n=3, h=0.003, offset=0):
    """n x n patch of vertices around `center` (one cloth layer) with its triangles."""
    xs = (np.arange(n) - (n - 1) / 2) * h
    v = np.array([[center[0] + x, center[1] + y, center[2]] for y in xs for x in xs])
    f = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = offset + j * n + i
            f += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return v, np.array(f)


def state(tip_l, g_l, g_r=0.5):
    return np.r_[tip_l, 0, 0, g_l, 0, 0, 0, 0, 0, g_r]


def run(tip_path, grip_path, verts_fn, faces):
    lg = gl.GraspLogger(tip_pose, faces, TABLE)
    for t, (tip, g) in enumerate(zip(tip_path, grip_path)):
        lg.update(t, state(tip, g), verts_fn(t, tip))
    lg.close_episode(len(tip_path))
    return lg.records


def scenario(follow, n_layers=1, lift=0.05, drag=0.0, start_closed=False, reopen_early=False):
    tip0 = np.array([0.1, 0.2, 0.60])
    c0 = box_center(tip0)
    v1, f1 = grid(c0)
    verts, faces = [v1], [f1]
    if n_layers == 2:
        v2, f2 = grid(c0 + [0, 0, -0.002], offset=len(v1))
        verts.append(v2); faces.append(f2)
    v0, faces = np.concatenate(verts), np.concatenate(faces)
    g_open, g_closed = 0.5, -0.07
    grip = [g_closed] * 3 if start_closed else []
    grip += [g_open] * 3 + ([g_closed] * 3 + [g_open] * 2 if reopen_early else []) + [g_closed] * 20 + [g_open] * 3
    tips = []
    for k in range(len(grip)):
        closed_for = sum(1 for g in grip[:k + 1] if g < gl.CLOSE_RAD)
        s = min(max(closed_for - gl.SETTLE_STEPS - 1, 0) / 10, 1.0) if grip[k] < gl.CLOSE_RAD or k == len(grip) - 1 else 0
        tips.append(tip0 + [drag * s, 0, lift * s])

    def verts_fn(t, tip):
        return v0 + (tip - tip0) if follow else v0

    return run(tips, grip, verts_fn, faces)


def check(name, recs, outcome, layers=None, n=1):
    assert len(recs) == n, f"{name}: expected {n} grasp(s), got {len(recs)}: {recs}"
    if n:
        r = recs[0]
        assert r["outcome"] == outcome, f"{name}: outcome {r['outcome']} != {outcome} ({r})"
        if layers is not None:
            assert r["layers"] == layers, f"{name}: layers {r['layers']} != {layers} ({r})"
    print(f"ok  {name}: {recs[0]['outcome'] if n else 'no grasp'}" + (f", layers {recs[0]['layers']}, lift {recs[0]['max_lift_cm']} cm" if n else ""))


if __name__ == "__main__":
    check("held and lifted", scenario(follow=True), "held_lifted", layers=1)
    check("slipped (cloth stays on the table)", scenario(follow=False), "slipped")
    check("two layers taken", scenario(follow=True, n_layers=2), "held_lifted", layers=2)
    check("dragged along the table", scenario(follow=True, lift=0.0, drag=0.06), "held_dragged")
    check("held without moving", scenario(follow=True, lift=0.0), "held_static")
    rec = scenario(follow=True, start_closed=True)
    check("episode starting with closed jaws counts only the real grasp", rec, "held_lifted")
    check("reopened before settling: only the later grasp counts", scenario(follow=True, reopen_early=True), "held_lifted")
    empty = run([np.array([0.1, 0.2, 0.6])] * 15, [0.5] * 3 + [-0.07] * 10 + [0.5] * 2,
                lambda t, tip: grid(np.array([0.5, 0.5, TABLE]))[0], grid(np.zeros(3))[1])
    check("nothing between the pads", empty, "empty")
    s = gl.summary(scenario(follow=True) + scenario(follow=False) + empty)
    assert s["grasps"] == 3 and s["empty"] == 1 and s["held_share_nonempty"] == 0.5, s
    print("ok  summary:", s)
    print("ALL GRASP LOGGER TESTS PASSED")
