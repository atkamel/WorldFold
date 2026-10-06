"""Scripted ("oracle") folding inside the LeHome Isaac env: no policy, no teacher.

It reads the true cloth vertex positions, plans pick-and-place folds for the two SO-101 arms
(fold_plans.py), executes them, and scores the result with fold_metric.py. Purpose: measure what THIS
body (arms, rigid jaws, cloth physics) can do when the "brain" knows exactly where everything is.

Run from the lehome-challenge directory with its own venv (see modal_teacher.py::isaac_oracle):
    PYTHONPATH=/opt/teacher/oracle ORACLE_JOBS='[...]' ORACLE_OUT=/vol/isaac/oracle-x \
        python -u -m scripts.oracle_fold --garment_type top_long --garment_name Top_Long_Seen_0 \
        --headless --enable_cameras --device cpu

ORACLE_JOBS: list of {"garment": name, "trials": n, "params": {...}}.
"""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import json
import os
import pickle
import sys
import time
import traceback

from isaaclab.app import AppLauncher

from .utils.common import close_app, launch_app_from_args
from .utils.parser import setup_eval_parser

OUT = os.environ.get("ORACLE_OUT", "/tmp/oracle")


def log(*a):
    print("[oracle]", *a, flush=True)


def _np(x):
    import numpy as np
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def cloth_verts(env):
    """World positions (N,3) of every cloth vertex."""
    import numpy as np
    o = env.object
    try:
        # the garment's own pose (not its particles) only changes when the garment is (re)built or reset: cache it per
        # garment object (IsaacWorld.reset drops the cache). Two USD queries per call before: ~0.8 ms on an L40S host
        cache = getattr(env, "_cloth_pose_cache", None)
        if cache is None or cache[0] is not o:
            pos, ori = o.get_world_pose()
            sc = o.get_world_scale()
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            A = (quat_to_rot_matrix(_np(ori)) * _np(sc)[None, :]).T        # points @ A + pos == transform_points
            cache = env._cloth_pose_cache = (o, np.ascontiguousarray(A, dtype=np.float64), np.asarray(_np(pos), np.float64))
        # same USD attribute _get_points_pose reads, but via the buffer protocol: 0.003 ms instead of ~160 ms
        # (Isaac Sim 5.1 builds a torch tensor element by element; audit 2026-10-05)
        pts = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float64)
        return pts @ cache[1] + cache[2]
    except Exception as e:  # GPU path / API drift
        log("cloth_verts fallback:", repr(e))
        return np.asarray(o.get_current_mesh_points()[0], dtype=np.float64)


def cloth_faces(env):
    import numpy as np
    try:
        prim = env.object._prim
        idx = np.array(prim.GetAttribute("faceVertexIndices").Get())
        cnt = np.array(prim.GetAttribute("faceVertexCounts").Get())
        if (cnt == 3).all():
            return idx.reshape(-1, 3)
        log("non-triangle faces present; footprint will use vertices only")
    except Exception as e:
        log("cloth_faces failed:", repr(e))
    return None


class Runner:
    def __init__(self, env, args, params):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.env, self.args, self.p = env, args, params
        self.frames, self.rec_every, self.t = [], int(params.get("rec_every", 4)), 0
        self.track = []
        try:
            import cv2
            self.cv2 = cv2
        except Exception as e:
            log("cv2 unavailable, frames stored raw:", repr(e)); self.cv2 = None
        self.obs = env._get_observations()

    def state(self):
        s = self.np.asarray(self.obs["observation.state"], dtype=float).ravel()
        return {"left": s[0:5].copy(), "right": s[6:11].copy()}, {"left": float(s[5]), "right": float(s[11])}

    def _record(self, tag):
        np = self.np
        row = []
        for k in ("observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb"):
            im = np.asarray(self.obs[k])[..., :3]
            st = max(1, round(im.shape[0] / 240))
            row.append(np.ascontiguousarray(im[::st, ::st][:240, :320]))
        tile = np.concatenate(row, 1)
        if self.cv2 is not None:
            ok, buf = self.cv2.imencode(".jpg", tile[..., ::-1], [int(self.cv2.IMWRITE_JPEG_QUALITY), 85])
            tile = buf.tobytes() if ok else tile
        self.frames.append(dict(t=self.t, tag=tag, img=tile,
                                state=np.asarray(self.obs["observation.state"], dtype=np.float32).copy(),
                                check=np.asarray(self.obs.get("check_status", []), dtype=np.float32).copy()))

    def step(self, q, g, tag):
        np, torch = self.np, self.torch
        a = np.concatenate([q["left"], [g["left"]], q["right"], [g["right"]]]).astype(np.float32)
        self.env.step(torch.from_numpy(a).to(self.env.device).unsqueeze(0))
        self.obs = self.env._get_observations()
        self.t += 1
        if self.t % self.rec_every == 0:
            self._record(tag)
        return a

    def run(self, plan):
        """Execute a fold_plans.Plan; returns per-phase max |commanded - actual| joint error (rad)."""
        np = self.np
        err = {}
        for tk in plan.ticks:
            a = self.step(tk["q"], tk["g"], tk["tag"])
            s = np.asarray(self.obs["observation.state"], dtype=float).ravel()
            e = float(np.abs(a[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]] - s[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]]).max())
            err[tk["tag"]] = max(err.get(tk["tag"], 0.0), e)
        return {k: round(v, 3) for k, v in err.items()}


def shirt_trial(env, args, params, name, trial, out_dir):
    import numpy as np
    import fold_metric as fm
    import fold_plans as fp
    import garment_keys as gk
    from so101_kin import HOME, GRIP_CLOSED

    R = Runner(env, args, params)
    home_q = {"left": HOME["left"].copy(), "right": HOME["right"].copy()}
    home_g = {"left": GRIP_CLOSED, "right": GRIP_CLOSED}
    for _ in range(int(params.get("settle", 150))):
        R.step(home_q, home_g, "settle")
    R._record("flat")

    flat = cloth_verts(env)
    faces = cloth_faces(env)
    table_z = float(np.percentile(flat[:, 2], 1)) - 0.002
    gfile = os.path.join(os.path.dirname(os.path.abspath(fm.__file__)), "garments", f"{name}.npz")
    rest = np.load(gfile)["points"]                       # rest mesh from the garment's USD (same vertex order as the sim)
    assert len(rest) == len(flat), f"vertex count mismatch: file {len(rest)} vs sim {len(flat)}"
    keys = gk.top_keys(np.asarray(rest, float), list(env.object.check_points))
    flat_info = fm.flat_stats(flat, faces, table_z)
    kp = {s: {k: (flat[v] * 100).round(1).tolist() for k, v in keys[s].items()} for s in keys}
    log(f"{name} trial {trial}: flat {flat_info} table_z={table_z:.4f}")
    log("keypoints (cm):", json.dumps(kp))
    folds_flat = gk.top_folds(flat[:, :2], keys, sleeve=params.get("sleeve", "armpit"))

    arm_of = fp.shirt_arms(flat, keys)
    steps, snapshots = [], [dict(tag="flat", verts=flat.astype(np.float32))]
    order = params.get("order", ["sleeve_L", "sleeve_R", "bottom_up"])
    for fname in order:
        v = cloth_verts(env)
        folds_now = {f["name"]: f for f in gk.top_folds(v[:, :2], keys, sleeve=params.get("sleeve", "armpit"))}
        f = folds_now[fname]
        q_now, g_now = R.state()
        plan = fp.Plan(q_now, g_now)
        spec = fp.shirt_spec(v, keys, f, arm_of, params)
        t0 = time.time()
        errs = plan.pick_place(spec, fname, carry_ds=params.get("carry_ds", 0.0015))
        plan.home(tag=fname + ":home", sides=list(spec))
        track = R.run(plan)
        for _ in range(int(params.get("pause", 30))):
            R.step(plan.q, plan.g, fname + ":pause")
        v_after = cloth_verts(env)
        if not np.isfinite(v_after).all() or np.abs(v_after).max() > 5:
            log(f"CLOTH BLEW UP after {fname}: finite={np.isfinite(v_after).all()} max={np.nanmax(np.abs(v_after)):.2f}")
            R._record(fname + ":done")
            with open(os.path.join(out_dir, f"{name}_t{trial}.pkl"), "wb") as fh:
                pickle.dump(dict(frames=R.frames, result=dict(neat=None, error=f"cloth blew up after {fname}", steps=steps)), fh)
            raise RuntimeError(f"cloth blew up after {fname}")
        snapshots.append(dict(tag=fname, verts=v_after.astype(np.float32)))
        st = dict(fold=fname, arms=list(spec), ticks=len(plan.ticks), plan_s=round(time.time() - t0, 1),
                  grasp={a: np.asarray(sp["grasp"]).round(4).tolist() for a, sp in spec.items()},
                  place={a: np.asarray(sp["place"]).round(4).tolist() for a, sp in spec.items()},
                  ik_err_mm={f"{a}:{ph}": round(e * 1000, 1) for (a, ph), e in errs.items()}, track_err_rad=track,
                  partial=fm.score(v_after, faces, flat, [ff for n in order[:order.index(fname) + 1] for ff in folds_flat if ff["name"] == n], table_z)["neat"])
        steps.append(st); log("step", json.dumps(st))
        R._record(fname + ":done")

    q_now, g_now = R.state()
    for _ in range(int(params.get("final_settle", 90))):
        R.step(home_q, home_g, "final")
    R._record("final"); R._record("final")
    final = cloth_verts(env)
    res = fm.score(final, faces, flat, [ff for n in order for ff in folds_flat if ff["name"] == n], table_z)
    res.update(garment=name, trial=trial, steps=steps, flat=flat_info, ticks=R.t, params=params,
               his_check=np.asarray(R.obs.get("check_status", [])).round(2).tolist(),
               his_check_ratio=np.asarray(R.obs.get("check_distances", [])).round(2).tolist())
    log("RESULT " + json.dumps(res))
    base = os.path.join(out_dir, f"{name}_t{trial}")
    np.savez_compressed(base + ".npz", flat=flat.astype(np.float32), final=final.astype(np.float32),
                        faces=faces if faces is not None else np.zeros((0, 3), int),
                        snaps=np.stack([s["verts"] for s in snapshots]), snap_tags=np.array([s["tag"] for s in snapshots]),
                        keys=json.dumps(keys), folds=json.dumps(folds_flat), table_z=table_z)
    with open(base + ".pkl", "wb") as fh:
        pickle.dump(dict(frames=R.frames, result=res), fh)
    json.dump(res, open(base + ".json", "w"), indent=1)
    return res


def grasp_test(env, args, job, name, out_dir):
    """Try one grasp (the first fold's) with several parameter variants; report how much cloth each one lifts."""
    import numpy as np
    import fold_plans as fp
    import garment_keys as gk
    from so101_kin import HOME, GRIP_CLOSED
    gfile = os.path.join(os.path.dirname(os.path.abspath(fp.__file__)), "garments", f"{name}.npz")
    keys = gk.top_keys(np.load(gfile)["points"].astype(float), list(env.object.check_points))
    home_q = {"left": HOME["left"].copy(), "right": HOME["right"].copy()}
    home_g = {"left": GRIP_CLOSED, "right": GRIP_CLOSED}
    out, frames = [], []
    for i, var in enumerate(job["variants"]):
        params = dict(job.get("params", {}), **var)
        env.reset()
        R = Runner(env, args, params)
        for _ in range(int(params.get("settle", 120))):
            R.step(home_q, home_g, "settle")
        v0 = cloth_verts(env)
        fname = params.get("fold", "sleeve_L")
        f = {x["name"]: x for x in gk.top_folds(v0[:, :2], keys)}[fname]
        spec = fp.shirt_spec(v0, keys, f, fp.shirt_arms(v0, keys), params)
        q_now, g_now = R.state()
        plan = fp.Plan(q_now, g_now)
        plan.pick_place(spec, f"g{i}", only_grasp=params.get("test_lift", 0.07))
        R.run(plan)
        v1 = cloth_verts(env)
        ok = bool(np.isfinite(v1).all())
        rise = (v1[:, 2] - v0[:, 2]) if ok else np.zeros(len(v0))
        tips = {a: fp.K.tip(R.state()[0][a], a)[0] for a in spec}
        near = {a: int((np.linalg.norm(v1 - t, axis=1) < 0.03).sum()) if ok else 0 for a, t in tips.items()}
        res = dict(variant=var, finite=ok, n_lifted_3cm=int((rise > 0.03).sum()), max_rise_cm=round(float(rise.max() * 100), 1),
                   verts_near_tip=near, tip_z={a: round(float(t[2]), 3) for a, t in tips.items()})
        log("GRASP " + json.dumps(res)); out.append(res)
        R._record(f"g{i}:held"); frames += R.frames
        plan2 = fp.Plan(R.state()[0], R.state()[1]); plan2.hold(10, "open", g={a: fp.GRIP_OPEN for a in spec}); plan2.home()
        R.run(plan2)
    with open(os.path.join(out_dir, f"{name}_grasp_{job.get('tag', 'g')}.pkl"), "wb") as fh:
        pickle.dump(dict(frames=frames, result=dict(neat=None, grasp=out)), fh)
    json.dump(out, open(os.path.join(out_dir, f"{name}_grasp_{job.get('tag', 'g')}.json"), "w"), indent=1)
    return out


# ---- live mode: an outside agent sends one command at a time and gets the scene back ---------------------
def _jpg(runner, im, size=None):
    import numpy as np
    im = np.ascontiguousarray(np.asarray(im)[..., :3])
    if size and runner.cv2 is not None:
        im = runner.cv2.resize(im, size)
    if runner.cv2 is None:
        return im
    ok, buf = runner.cv2.imencode(".jpg", im[..., ::-1], [int(runner.cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes()


def live_observe(env, R, ctx, note=""):
    """Everything the agent gets back after a command (plain Python types only)."""
    import numpy as np
    import fold_plans as fp
    q, g = R.state()
    v = cloth_verts(env)
    ok = bool(np.isfinite(v).all())
    out = dict(t=R.t, note=note, cloth_ok=ok, garment=ctx["name"],
               grip_rad={a: round(g[a], 3) for a in g}, tips={}, held={},
               his_check=np.asarray(R.obs.get("check_status", [])).round(2).tolist())
    for a in ("left", "right"):
        p, ap = fp.K.tip(q[a], a)
        out["tips"][a] = dict(pos=np.round(p, 4).tolist(),
                              tilt_deg=round(float(np.degrees(np.arccos(np.clip(-ap[2], -1, 1)))), 1),
                              q=np.round(q[a], 3).tolist())
        if ok:
            near = np.linalg.norm(v - p, axis=1) < 0.03
            out["held"][a] = dict(verts_within_3cm=int(near.sum()),
                                  their_height_cm=round(float((v[near, 2].mean() - ctx["table_z"]) * 100), 1) if near.any() else None)
    if ok:
        keys = ctx["keys"]
        out["keypoints_cm"] = {s: {k: (v[i] * 100).round(1).tolist() for k, i in keys[s].items()} for s in keys}
        h = (v[:, 2] - ctx["table_z"]) * 100
        out["cloth"] = dict(x_cm=[round(float(v[:, 0].min() * 100), 1), round(float(v[:, 0].max() * 100), 1)],
                            y_cm=[round(float(v[:, 1].min() * 100), 1), round(float(v[:, 1].max() * 100), 1)],
                            h95_cm=round(float(np.percentile(h, 95)), 1), hmax_cm=round(float(h.max()), 1),
                            n_above_3cm=int((h > 3).sum()))
    if ok:
        out["map_jpg"] = map_jpg(R, ctx, v)
        out["score"] = score_now(ctx, v)
        out["done"] = list(ctx.get("done", []))
    out["top_jpg"] = _jpg(R, R.obs["observation.images.top_rgb"])
    out["left_jpg"] = _jpg(R, R.obs["observation.images.left_rgb"], (320, 240))
    out["right_jpg"] = _jpg(R, R.obs["observation.images.right_rgb"], (320, 240))
    return out


def live_start(env, args, name, params):
    import numpy as np
    import fold_metric as fm
    import garment_keys as gk
    from so101_kin import HOME, GRIP_CLOSED
    env.reset()
    R = Runner(env, args, params)
    hq = {"left": HOME["left"].copy(), "right": HOME["right"].copy()}
    hg = {"left": GRIP_CLOSED, "right": GRIP_CLOSED}
    for _ in range(int(params.get("settle", 150))):
        R.step(hq, hg, "settle")
    flat = cloth_verts(env)
    gfile = os.path.join(os.path.dirname(os.path.abspath(fm.__file__)), "garments", f"{name}.npz")
    keys = gk.top_keys(np.load(gfile)["points"].astype(float), list(env.object.check_points))
    ctx = dict(name=name, flat=flat, faces=cloth_faces(env), keys=keys,
               table_z=float(np.percentile(flat[:, 2], 1)) - 0.002,
               folds=gk.top_folds(flat[:, :2], keys), cmd={"q": hq, "g": hg})
    return R, ctx


def live_exec(env, R, ctx, cmd):
    """Execute one command. Ops: move | grip | wait | home | score."""
    import numpy as np
    import fold_metric as fm
    import fold_plans as fp
    op = cmd.get("op")
    q, _ = R.state()
    # keep commanding the last commanded gripper values (the measured ones differ while holding cloth)
    plan = fp.Plan(q, dict(ctx["cmd"]["g"]))
    info = {}
    if op == "move":
        seq = {}
        for a, spec in cmd["arms"].items():
            tgt = np.asarray(spec["pos"], float)
            cur, _ = fp.K.tip(q[a], a)
            roll = q[a][4]
            if spec.get("jaw_axis") is not None:
                qg, _ = fp._solve(a, tgt, q[a])
                ax = spec["jaw_axis"]
                roll, align = fp.K.roll_for_slide(qg, a, (ax[0], ax[1], 0), near=q[a][4])   # moving finger leads
                info[a + "_jaw_align"] = round(float(align), 2)
            via = [np.asarray(w, float) for w in spec.get("via", [])]
            qs, e = fp._path(a, [cur] + via + [tgt], q[a], roll, ds=float(cmd.get("ds", 0.002)))
            if spec.get("jaw_axis") is not None and qs:      # blend the roll change in instead of jumping
                r0 = q[a][4]
                for i, qq in enumerate(qs):
                    qq[4] = r0 + (roll - r0) * min(1.0, (i + 1) / max(1, len(qs) // 2))
            seq[a] = qs; info[a + "_ik_err_mm"] = round(e * 1000, 1)
        plan._emit(seq, cmd.get("tag", "move"))
    elif op == "grip":
        g0 = dict(plan.g); n = int(cmd.get("ramp", 8))
        for i in range(1, n + 1):
            plan.hold(1, cmd.get("tag", "grip"), g={a: g0[a] + (float(val) - g0[a]) * i / n for a, val in cmd["arms"].items()})
        plan.hold(int(cmd.get("hold", 12)), cmd.get("tag", "grip"))
    elif op == "grasp":
        arm = cmd["arm"]
        roll, align = grasp_ticks(plan, arm, cmd["target"], cmd["dir"], slide=float(cmd.get("slide", 0.03)),
                                  z_land=float(cmd.get("z_land", 0.527)), z_press=float(cmd.get("z_press", 0.519)))
        info["track_err_rad"] = R.run(plan); n0 = len(plan.ticks)
        qn, _ = R.state(); vc = cloth_verts(env)
        info.update(align=round(align, 2), held_closed=near_tip(env, qn[arm], arm, ctx["table_z"], vc)[0],
                    between_pads=between_pads(qn[arm], arm, vc))
        plan.ticks = []
        p = fp.K.tip(plan.q[arm], arm)[0]
        qs, _ = fp._path(arm, [p, (p[0], p[1], p[2] + float(cmd.get("lift", 0.05)))], plan.q[arm], roll, ds=0.0007)
        plan._emit({arm: qs}, "lift")
        R.run(plan)
        qn, _ = R.state()
        info["held_lifted"], info["lifted_height_cm"] = near_tip(env, qn[arm], arm, ctx["table_z"], None, min_h=0.02)
        info["ticks"] = n0 + len(plan.ticks)
        ctx["cmd"] = {"q": plan.q, "g": dict(plan.g)}
        return info
    elif op == "release":
        arm = cmd["arm"]
        release_ticks(plan, arm, z=cmd.get("z"), open_rad=float(cmd.get("open", 0.55)), away=float(cmd.get("away", 0.0)),
                      lift_ds=float(cmd.get("lift_ds", 0.002)), lift_to=float(cmd.get("lift_to", 0.63)))
        info["track_err_rad"] = R.run(plan); info["ticks"] = len(plan.ticks)
        qn, _ = R.state()
        info["stuck"] = near_tip(env, qn[arm], arm, ctx["table_z"], None, r=0.04, min_h=0.025)[0]
        ctx["cmd"] = {"q": plan.q, "g": dict(plan.g)}
        return info
    elif op == "fold":
        return fold_step(env, R, ctx, cmd["step"], cmd.get("params"))
    elif op == "move_corner":
        return move_corner(env, R, ctx, cmd)
    elif op == "save":
        return save_state(env, R, ctx, cmd.get("name", "s"))
    elif op == "restore":
        return restore_state(env, R, ctx, cmd.get("name", "s"))
    elif op == "wait":
        plan.hold(int(cmd.get("n", 30)), "wait")
    elif op == "home":
        plan.home(sides=cmd.get("sides", ["left", "right"]))
    elif op == "score":
        v = cloth_verts(env)
        order = cmd.get("folds", ctx.get("done") or ["sleeve_L", "sleeve_R", "bottom_up"])
        res = fm.score(v, ctx["faces"], ctx["flat"], [f for n in order for f in ctx["folds"] if f["name"] == n], ctx["table_z"])
        return dict(score=res)
    else:
        return dict(error=f"unknown op {op}")
    info["track_err_rad"] = R.run(plan)
    info["ticks"] = len(plan.ticks)
    ctx["cmd"] = {"q": plan.q, "g": dict(plan.g)}
    return info


def live(env, args):
    """File-based command loop (the Modal function bridges these files to a queue the agent uses)."""
    import fold_metric as fm
    d = os.environ["ORACLE_LIVE_DIR"]; os.makedirs(d, exist_ok=True)
    params = json.loads(os.environ.get("ORACLE_LIVE_PARAMS", "{}"))
    name = env.cfg.garment_name

    def reply(n, obj):
        tmp = os.path.join(d, f"res_{n}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        os.replace(tmp, os.path.join(d, f"res_{n}.pkl"))

    if os.environ.get("ORACLE_CLOTH"):
        set_cloth(env, name, os.environ["ORACLE_CLOTH"])
    R, ctx = live_start(env, args, name, params)
    kin_check(env)
    reply(0, dict(info=dict(started=name, flat=fm.flat_stats(ctx["flat"], ctx["faces"], ctx["table_z"]),
                            table_z=round(ctx["table_z"], 4)), obs=live_observe(env, R, ctx, "start")))
    log("LIVE ready")
    n, all_frames = 1, []
    while True:
        path = os.path.join(d, f"cmd_{n}.json")
        if not os.path.exists(path):
            time.sleep(0.05); continue
        time.sleep(0.02)
        cmd = json.load(open(path))
        log("LIVE cmd", n, json.dumps(cmd)[:300])
        try:
            if cmd.get("op") == "quit":
                reply(n, dict(info=dict(quit=True))); break
            if cmd.get("op") == "reset":
                all_frames += R.frames
                new = cmd.get("garment", name)
                if new != name:
                    env.switch_garment(new); name = new
                R, ctx = live_start(env, args, name, dict(params, **cmd.get("params", {})))
                info = dict(reset=name)
            else:
                info = live_exec(env, R, ctx, cmd)
            reply(n, dict(info=info, obs=live_observe(env, R, ctx, cmd.get("op", ""))))
        except Exception as e:
            traceback.print_exc()
            reply(n, dict(info=dict(error=repr(e))))
        n += 1
    all_frames += R.frames
    with open(os.path.join(OUT, "live_session.pkl"), "wb") as fh:
        pickle.dump(dict(frames=all_frames, result=dict(neat=None)), fh)
    log("LIVE done, frames", len(all_frames))


def _teleop_world(env):
    """The live-play world (teleop/isaac_world.IsaacWorld) with every speed fix installed, settled and checked.
    Returns (world, the cloth's current physics values for the tuning panel or None)."""
    import numpy as np
    from scripts.utils.remote_loop import capture_physics_state, restore_physics_state
    from teleop.isaac_world import IsaacWorld
    import torch
    name = env.cfg.garment_name
    # speed fixes (audit 2026-10-05); none of them changes the physics:
    # 1. every cloth-point read (ours and LeHome's success checks) via the buffer protocol: same values, ~160 ms -> ~0 ms
    try:
        from isaacsim.core.prims import SingleClothPrim

        def _fast_points(self):
            return torch.from_numpy(np.array(self._prim.GetAttribute("points").Get(), dtype=np.float32))
        SingleClothPrim._get_points_pose = _fast_points
        log("TELEOP fast cloth read installed")
    except Exception as e:
        log("TELEOP fast cloth read NOT installed:", repr(e))
    # 3. no reward computation during play (the laptop scores folds); the success-check interval is set by run_isaac.sh
    env._get_rewards = lambda: torch.zeros(1, device=env.device)
    # 4. the iterative IK compiled with numba (matches the original to 4e-11 rad). Since the exact C++ solver
    #    (so101_native_ik, loaded by IsaacWorld below) this is only the fallback for points that one cannot reach.
    try:
        import fold_plans as fp
        from so101_fastik import FastIK
        FastIK(fp.K).install()
        log("TELEOP fallback IK: numba")
    except Exception as e:
        log("TELEOP fallback IK: numpy (numba version not installed:", repr(e)[:160] + ")")
    # 5. PhysX writes its results back to USD (where we and LeHome read the cloth and the robot links). IsaacLab's headless
    #    app loads omni.physx.fabric, which turns that write-back off: without this the cloth we stream stays frozen.
    import carb
    st = carb.settings.get_settings()
    keys = ("/physics/updateToUsd", "/physics/updateParticlesToUsd", "/physics/updateVelocitiesToUsd", "/physics/fabricEnabled")
    log("TELEOP physics->USD settings before:", {k: st.get(k) for k in keys})
    st.set_bool("/physics/updateToUsd", True)
    st.set_bool("/physics/updateParticlesToUsd", True)
    if os.environ.get("ORACLE_CLOTH"):
        set_cloth(env, name, os.environ["ORACLE_CLOTH"])
    v_spawn = cloth_verts(env)
    helpers = dict(cloth_verts=cloth_verts, cloth_faces=cloth_faces, between_pads=between_pads,
                   capture=capture_physics_state, restore=lambda e, s: restore_physics_state(e, s, "exact", "oracle"),
                   set_cloth=set_cloth_tuned)
    w = IsaacWorld(env, helpers)
    log("TELEOP aim-point IK:", w.ik_backend)
    v_settled = cloth_verts(env)        # self-check: the cloth falls onto the table while the world settles
    moved = float(np.abs(v_settled - v_spawn).max()) * 1000
    log(f"TELEOP cloth self-check: moved {moved:.1f} mm while settling ({'OK' if moved > 5 else 'STALE - cloth read is frozen'}); "
        f"z range {v_settled[:, 2].min():.3f}..{v_settled[:, 2].max():.3f}")
    kin_check(env)
    try:
        start = cloth_now(env)          # live-tuning panel starts at the cloth's real values
    except Exception as e:
        log("TELEOP could not read the cloth values, panel shows defaults:", repr(e)); start = None
    return w, start


def teleop(env, args):
    """Real-time two-mouse play (teleop/): a human on the laptop drives both arms through a Modal tunnel.
    Same controller as the local MuJoCo game; grabs use the grab assist (teleop/tuning.py ASSIST)."""
    import numpy as np
    from teleop.controller import Game
    from teleop.server import TeleopServer
    name = env.cfg.garment_name
    w, start = _teleop_world(env)
    from teleop import tuning
    preset = os.environ.get("ORACLE_GRAB", "assist")
    g = Game(w, tune=dict(start or {}, **(tuning.ASSIST if preset == "assist" else tuning.MUJOCO_FEEL if preset == "mujoco" else {})))
    log(f"TELEOP grab preset: {preset}")
    state_view = os.environ.get("ORACLE_VIEW", "state") == "state"
    if not state_view and os.environ.get("ORACLE_FPV", "1") != "0":
        try:
            w.set_fpv_camera()
        except Exception as e:
            log("TELEOP FPV camera NOT set (LeHome top camera in use):", repr(e))
    nocam = os.environ.get("LEHOME_NO_CAMERAS") == "1"     # no_cameras_patch.py: no sensors, the laptop draws
    if nocam and not state_view:
        raise RuntimeError("LEHOME_NO_CAMERAS=1 needs the state view (ORACLE_VIEW=state): there is no picture to send")
    im = None if nocam else w.render()
    size = [960, 600] if im is None else [im.shape[1], im.shape[0]]
    hello = dict(backend="isaac", garment=name, dt=w.dt, tris=(w.faces.tolist() if w.faces is not None else None),
                 thickness=0.004, flat_cloth_cm=np.round(w.flat_local * 100, 2).tolist(), frame_size=size,
                 cam=w.camera_info())     # lets the laptop draw the cursor itself (no network delay on the cursor)
    if state_view:
        hello.update(w.state_hello())
    log(f"TELEOP ready: garment {name}, control dt {w.dt:.4f}s, {'no cameras' if nocam else f'frame {size[0]}x{size[1]}'}, "
        f"table z {w.table_z:.3f}, {len(w.flat_local)} cloth points")
    TeleopServer(g, w.render, hello, port=int(os.environ.get("ORACLE_TELEOP_PORT", "7777")), dt=w.dt, frame_every=1,
                 log_path=os.path.join(OUT, "teleop_demo.jsonl"), idle_s=int(os.environ.get("ORACLE_TELEOP_IDLE", "300")),
                 max_s=int(os.environ.get("ORACLE_TELEOP_MAX", "3000")), log=log,
                 wall_speed=float(os.environ.get("ORACLE_WALL_SPEED", "8")),
                 state=w.state_bytes if state_view else None,
                 profile=((300, int(os.environ["ORACLE_PROFILE"]), os.path.join(OUT, "teleop_profile.pstats"))
                          if os.environ.get("ORACLE_PROFILE") else None),
                 keepalive=_app_keepalive(), keepalive_every=int(os.environ.get("ORACLE_APP_PUMP_TICKS", "0"))).serve()


def assist_test(env, args, job):
    """Grab assist test: the same world and grab code as live play, at `n` random spots on the shirt (the same spots
    for every preset). Per grab: hover 3 cm above the spot, go down like the controller does, run w.grab, then lift
    5 cm with the jaws as the grab left them. Counts the cloth points lifted above 2 cm within 8 cm of the tip."""
    import json
    import numpy as np
    from teleop import tuning
    w, _ = _teleop_world(env)
    base = {k: v[0] for k, v in tuning.SPECS.items()}
    presets = {"mujoco_feel": tuning.MUJOCO_FEEL, "tested": {}, "assist": tuning.ASSIST}
    presets = {k: presets[k] for k in job.get("presets", list(presets))}
    snap = w.get_state()
    rng = np.random.default_rng(int(job.get("seed", 0)))
    flat = w.flat_local
    targets = []
    while len(targets) < int(job.get("n", 20)):
        p = flat[rng.integers(len(flat))].copy()
        a = "L" if p[0] < 0 else "R"
        q4, err, tilt = w.ik(a, p + np.array([0, 0, 0.03]), w.q(a), w.roll(a))
        if err < 0.002 and tilt < 40:
            targets.append((a, p))

    pump, pump_every, n_steps = _app_keepalive(), int(job.get("pump_every", 500)), [0]

    def step():
        w.step()
        n_steps[0] += 1
        if pump and pump_every and n_steps[0] % pump_every == 0:
            pump()

    def move(a, goal, ticks):
        start = w.tip(a)
        for i in range(1, ticks + 1):
            q4, err, _ = w.ik(a, start + (goal - start) * i / ticks, w.q(a), w.roll(a))
            if err < 0.004:
                w.set_arm(a, q4, w.roll(a))
            step()

    rows = []
    for name, preset in presets.items():
        w.tune = dict(base, **preset)
        ok = 0
        for i, (a, p) in enumerate(targets):
            w.set_state(snap)
            top = w.surface_under(p[:2])
            move(a, np.array([p[0], p[1], top + 0.03]), 60)
            move(a, np.array([p[0], p[1], top + w.tune["down_offset"]]), 20)
            w.grab(a)
            t = 0
            while w.busy(a) and t < 600:
                step(); t += 1
            held = int(w.held[a])
            move(a, w.tip(a) + np.array([0, 0, 0.05]), 40)
            v, tip = w.verts_local(), w.tip(a)
            lifted = int(((v[:, 2] > 0.02) & (np.hypot(v[:, 0] - tip[0], v[:, 1] - tip[1]) < 0.08)).sum())
            good = lifted >= 15
            ok += good
            row = dict(preset=name, i=i, arm=a, target_cm=np.round(p * 100, 1).tolist(), grab_ticks=t,
                       grab_s=round(t * w.dt, 2), between_pads=held, lifted=lifted, held_ok=bool(good))
            rows.append(row)
            log("ASSIST_ROW " + json.dumps(row))
        g = [r for r in rows if r["preset"] == name]
        log(f"ASSIST_SUMMARY {name}: {ok}/{len(g)} lifted the cloth; grab move {np.mean([r['grab_s'] for r in g]):.2f} s "
            f"(sim time) on average")
    log(f"ASSIST_DONE {n_steps[0]} physics steps, app loop ticked every {pump_every if pump else 'never'}")
    json.dump(rows, open(os.path.join(OUT, "assist_test.json"), "w"), indent=1)
    return rows


def _app_keepalive():
    """Without cameras IsaacLab never runs Kit's app loop (nothing renders), and Kit crashed in 3 of 5 such sessions
    (a native fault while idle or at connect/disconnect). The server calls this ~1/s while waiting and every 48 ticks
    while playing. ORACLE_APP_PUMP=0 turns it off."""
    if os.environ.get("LEHOME_NO_CAMERAS") != "1" or os.environ.get("ORACLE_APP_PUMP", "1") == "0":
        return None
    import omni.kit.app
    app = omni.kit.app.get_app()
    log("TELEOP app keepalive on (Kit app loop ~1/s while waiting for the client; during play only if ORACLE_APP_PUMP_TICKS"
        " is set: one app update costs ~0.5 s)")
    return app.update


# ---- grip assay: one standardised sleeve grasp -> lift -> hold -> carry, measured from cloth vertices -------------
CLOTH_VARIANTS = {
    "asis": {},                                                     # the challenge's settings
    "real": {"garment_config.particle_mass": "total:0.05",          # ~50 g shirt instead of ~150 kg
             "particle_material.gravity_scale": 1.0},               # normal gravity instead of x2
    "real_noadh": {"garment_config.particle_mass": "total:0.05", "particle_material.gravity_scale": 1.0,
                   "particle_material.adhesion": 0.0},              # same, no cloth-to-gripper stickiness
}


def set_cloth(env, name, variant):
    """Recreate the garment in place with particle-config overrides (no simulator restart).
    variant: a CLOTH_VARIANTS name, or a dict of overrides {"section.key": value} (live tuning)."""
    import copy
    if not hasattr(env, "_orig_particle_config"):
        env._orig_particle_config = copy.deepcopy(env.particle_config)
    pc = copy.deepcopy(env._orig_particle_config)
    npts = None
    overrides = CLOTH_VARIANTS[variant] if isinstance(variant, str) else variant
    for k, v in overrides.items():
        sec, key = k.split(".")
        if isinstance(v, str) and v.startswith("total:"):
            if npts is None:
                npts = len(env.object._prim.GetAttribute("points").Get())
            v = float(v[6:]) / npts
        pc.objects[sec][key] = v
    env.particle_config = pc
    env.switch_garment(name)
    log(f"cloth variant {variant if isinstance(variant, str) else 'tuned'}: " +
        json.dumps({k: (float(pc.objects[k.split('.')[0]][k.split('.')[1]])) for k in overrides}))


def cloth_now(env):
    """The garment's current cloth physics, in tuning.py terms (so the live-tuning panel starts truthful)."""
    from teleop.tuning import CLOTH_KEYS
    pc = env.particle_config.objects
    out = {k: float(pc[p.split(".")[0]][p.split(".")[1]]) for k, p in CLOTH_KEYS.items()}
    npts = len(env.object._prim.GetAttribute("points").Get())
    out["mass_g"] = float(pc["garment_config"]["particle_mass"]) * npts * 1000
    return out


def set_cloth_tuned(env, tuning):
    """Live tuning 'apply cloth': rebuild the current garment with the panel's cloth values."""
    npts = len(env.object._prim.GetAttribute("points").Get())
    set_cloth(env, env.cfg.garment_name, tuning.cloth_overrides(npts))


class Stepper:
    """Physics-only stepping (no camera rendering); falls back to env.step if the fast path is unavailable."""

    def __init__(self, env):
        self.env, self.fast, self.t = env, True, 0

    def step(self, a):
        import numpy as np
        import torch
        act = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.env.device).unsqueeze(0)
        if self.fast:
            try:
                e = self.env
                e._pre_physics_step(act)
                e._apply_action()
                e.scene.write_data_to_sim()
                e.sim.step(render=False)
                e.scene.update(dt=e.physics_dt)
                self.t += 1
                if self.t % 50 == 0:          # pump Kit's event loop: ~9k un-rendered steps ended in an abort (mech1 run)
                    try:
                        e.sim.render()
                    except Exception:
                        pass
                return
            except Exception as ex:
                log("fast step unavailable, using env.step:", repr(ex)); self.fast = False
        self.env.step(act); self.t += 1

    def state(self):
        import numpy as np
        e = self.env
        try:
            l = e.left_arm.data.joint_pos[0].detach().cpu().numpy(); r = e.right_arm.data.joint_pos[0].detach().cpu().numpy()
            s = np.concatenate([l[:6], r[:6]]).astype(float)
        except Exception:
            s = np.asarray(e._get_observations()["observation.state"], dtype=float).ravel()
        return {"left": s[0:5].copy(), "right": s[6:11].copy()}, {"left": float(s[5]), "right": float(s[11])}

    def run(self, ticks):
        import numpy as np
        for tk in ticks:
            self.step(np.concatenate([tk["q"]["left"], [tk["g"]["left"]], tk["q"]["right"], [tk["g"]["right"]]]))


def assay_trial(env, S, keys, speed, side="L"):
    import numpy as np
    import fold_plans as fp
    from so101_kin import HOME, GRIP_CLOSED, GRIP_OPEN
    env.reset()
    home = np.concatenate([HOME["left"], [GRIP_CLOSED], HOME["right"], [GRIP_CLOSED]])
    for i in range(110):
        S.step(home)
        if i == 49:
            v0 = cloth_verts(env)
    v1 = cloth_verts(env)
    dt = float(getattr(env, "physics_dt", 1 / 90))
    out = dict(speed_mm_step=round(speed * 1000, 2), creep_mm_s=round(float(np.linalg.norm(v1 - v0, axis=1).mean() * 1000 / (60 * dt)), 2))
    if not np.isfinite(v1).all():
        out["error"] = "nan at settle"; return out
    table_z = float(np.percentile(v1[:, 2], 1)) - 0.002
    arm = fp.shirt_arms(v1, keys)[side]
    cuff, pit = v1[keys[side]["cuff_mid"], :2], v1[keys[side]["armpit"], :2]
    axis = (pit - cuff) / np.linalg.norm(pit - cuff)
    start, target = cuff + axis * 0.01, cuff + axis * 0.04
    q, _ = S.state()
    plan = fp.Plan(q, {"left": GRIP_CLOSED, "right": GRIP_CLOSED})
    q_h, _ = fp._solve(arm, (start[0], start[1], fp.Z_HOVER), fp.SEED[arm])
    roll, _ = fp.K.roll_for_axis(q_h, arm, (axis[0], axis[1], 0), near=fp.ROLL_NEAR[arm])
    q_h, _ = fp._solve(arm, (start[0], start[1], fp.Z_HOVER), fp.SEED[arm], roll)
    plan._emit({arm: fp._joint_move(plan.q[arm], q_h)}, "approach", g={arm: GRIP_OPEN})
    qs, _ = fp._path(arm, [(start[0], start[1], fp.Z_HOVER), (start[0], start[1], 0.527), (target[0], target[1], 0.519)],
                     plan.q[arm], roll, ds=0.0015)
    plan._emit({arm: qs}, "slide")
    S.run(plan.ticks); plan.ticks = []
    qn, _ = S.state()
    out["tip_z_landed"] = round(float(fp.K.tip(qn[arm], arm)[0][2]), 4)
    for i in range(1, 17):
        plan.hold(1, "close", g={arm: GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * i / 16})
    plan.hold(25, "close")
    S.run(plan.ticks); plan.ticks = []
    qn, gn = S.state()
    tip = fp.K.tip(qn[arm], arm)[0]
    vc = cloth_verts(env)
    pinched = np.nonzero(np.linalg.norm(vc - tip, axis=1) < 0.02)[0]
    out["n_pinched"] = int(len(pinched))
    out["jaw_deg_closed"] = round(float(np.degrees(gn[arm])), 2)
    if len(pinched) == 0:
        out["error"] = "nothing pinched"; return out

    def measure(tag):
        v = cloth_verts(env)
        if not np.isfinite(v).all():
            out[tag] = "nan"; return False
        qq, gg = S.state()
        t = fp.K.tip(qq[arm], arm)[0]
        d = np.linalg.norm(v[pinched] - t, axis=1)
        out[tag] = dict(held=round(float((d < 0.03).mean()), 3), height_cm=round(float((v[pinched, 2].mean() - table_z) * 100), 1),
                        jaw_deg=round(float(np.degrees(gg[arm])), 2))
        return True

    p0 = fp.K.tip(plan.q[arm], arm)[0]
    qs, _ = fp._path(arm, [p0, (p0[0], p0[1], 0.60)], plan.q[arm], roll, ds=0.0007)
    plan._emit({arm: qs}, "lift"); S.run(plan.ticks); plan.ticks = []
    if not measure("after_lift"):
        return out
    plan.hold(135, "hold"); S.run(plan.ticks); plan.ticks = []
    if not measure("after_hold"):
        return out
    p1 = fp.K.tip(plan.q[arm], arm)[0]
    dx = 0.15 if arm == "left" else -0.15
    qs, _ = fp._path(arm, [p1, (p1[0] + dx, p1[1], p1[2])], plan.q[arm], roll, ds=speed, ds_ik=max(0.006, speed))
    plan._emit({arm: qs}, "carry"); plan.hold(30, "settle"); S.run(plan.ticks); plan.ticks = []
    if not measure("after_carry"):
        return out
    plan.hold(15, "open", g={arm: GRIP_OPEN}); plan.home(sides=[arm]); S.run(plan.ticks)
    return out


def grip_assay(env, args, job, name, out_dir):
    import numpy as np
    import garment_keys as gk
    pads = os.environ.get("ORACLE_PADS", "capsule")
    gfile = os.path.join(os.path.dirname(os.path.abspath(gk.__file__)), "garments", f"{name}.npz")
    keys = gk.top_keys(np.load(gfile)["points"].astype(float), list(env.object.check_points))
    S = Stepper(env)
    rows, t0 = [], time.time()
    budget = float(job.get("budget_s", 420))
    for rep in range(int(job.get("reps", 2))):
        for variant in job.get("cloth", ["asis", "real"]):
            if time.time() - t0 > budget:
                log(f"assay budget {budget:.0f}s reached; stopping"); break
            set_cloth(env, name, variant)
            for speed in job.get("speeds", [0.001, 0.0035, 0.010]):
                t1, s0 = time.time(), S.t
                try:
                    r = assay_trial(env, S, keys, speed)
                except Exception as e:
                    traceback.print_exc(); r = dict(error=repr(e)[:200])
                    set_cloth(env, name, variant)
                r.update(pads=pads, cloth=variant, rep=rep, wall_s=round(time.time() - t1, 1), steps=S.t - s0)
                rows.append(r); log("ASSAY_ROW " + json.dumps(r))
                if "nan" in json.dumps(r) or "error" in r:
                    set_cloth(env, name, variant)
    log(f"assay done: {len(rows)} trials in {time.time() - t0:.0f}s, fast_step={S.fast}")
    json.dump(rows, open(os.path.join(out_dir, f"assay_{pads}.json"), "w"), indent=1)
    return rows


def best_variant(rows):
    """Cloth variant and carry speed with the highest mean 'held' after the carry."""
    import numpy as np
    score = {}
    for r in rows:
        c = r.get("after_carry")
        h = c["held"] if isinstance(c, dict) else 0.0
        score.setdefault((r["cloth"], r["speed_mm_step"]), []).append(h)
    (cloth, speed), vals = max(score.items(), key=lambda kv: np.mean(kv[1]))
    return cloth, speed / 1000, float(np.mean(vals))


# ---- grasp / release primitives shared by the live tool and the mechanics assay ----------------------------------
T_LINK_TIP = None


def _t_link_tip():
    import numpy as np
    global T_LINK_TIP
    if T_LINK_TIP is None:
        c, s = np.cos(np.pi), np.sin(np.pi)
        ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
        T = np.eye(4); T[:3, 3] = [-0.0079, -0.000218, -0.0981]
        T_LINK_TIP = T @ ry
    return T_LINK_TIP


def between_pads(q_arm, side, verts):
    """Cloth vertices inside the flat-pad pinch zone (gripper-link frame box around the pad faces)."""
    import numpy as np
    import fold_plans as fp
    T = fp.K.fk(q_arm, side) @ np.linalg.inv(_t_link_tip())
    v = (np.linalg.inv(T) @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    m = (np.abs(v[:, 1]) < 0.012) & (v[:, 2] > -0.103) & (v[:, 2] < -0.075) & (v[:, 0] > -0.010) & (v[:, 0] < 0.004)
    return int(m.sum())


def grasp_ticks(plan, arm, target, direction, slide=0.03, z_land=0.527, z_press=0.519, lead="moving"):
    """Gather grasp: hover above start, descend, slide toward `target` along `direction` with the moving finger
    leading (lead="fixed" reverses it, for testing), then close. Returns (roll, signed alignment)."""
    import numpy as np
    import fold_plans as fp
    from so101_kin import GRIP_OPEN, GRIP_CLOSED
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    target = np.asarray(target, float); start = target - d * slide
    q_h, _ = fp._solve(arm, (start[0], start[1], fp.Z_HOVER), fp.SEED[arm])
    lead_dir = d if lead == "moving" else -d
    roll, align = fp.K.roll_for_slide(q_h, arm, (lead_dir[0], lead_dir[1], 0), near=fp.ROLL_NEAR[arm])
    q_h, _ = fp._solve(arm, (start[0], start[1], fp.Z_HOVER), fp.SEED[arm], roll)
    plan._emit({arm: fp._joint_move(plan.q[arm], q_h)}, "approach", g={arm: GRIP_OPEN})
    qs, _ = fp._path(arm, [(start[0], start[1], fp.Z_HOVER), (start[0], start[1], z_land), (target[0], target[1], z_press)],
                     plan.q[arm], roll, ds=0.0015)
    plan._emit({arm: qs}, "slide")
    g0 = plan.g[arm]
    for i in range(1, 17):
        plan.hold(1, "close", g={arm: g0 + (GRIP_CLOSED - g0) * i / 16})
    plan.hold(25, "close")
    return roll, align


def release_ticks(plan, arm, z=None, open_rad=0.55, away=0.0, lift_ds=0.002, lift_to=0.63):
    """Release: optionally lower to z, open to open_rad, optionally back the fixed finger away from the cloth by
    `away` (horizontal, opposite the moving-finger side), then lift to lift_to at lift_ds per tick."""
    import numpy as np
    import fold_plans as fp
    roll = plan.q[arm][4]
    p = fp.K.tip(plan.q[arm], arm)[0]
    if z is not None:
        qs, _ = fp._path(arm, [p, (p[0], p[1], z)], plan.q[arm], roll, ds=0.001); plan._emit({arm: qs}, "lower")
        p = np.array([p[0], p[1], z])
    g0 = plan.g[arm]
    for i in range(1, 9):
        plan.hold(1, "open", g={arm: g0 + (open_rad - g0) * i / 8})
    plan.hold(15, "open")
    if away > 0:
        n = fp.K.moving_side(plan.q[arm], arm)[:2]; n = n / np.linalg.norm(n)
        p2 = np.array([p[0] - n[0] * away, p[1] - n[1] * away, p[2]])
        qs, _ = fp._path(arm, [p, p2], plan.q[arm], roll, ds=0.001); plan._emit({arm: qs}, "away"); p = p2
    qs, _ = fp._path(arm, [p, (p[0], p[1], lift_to)], plan.q[arm], roll, ds=lift_ds, ds_ik=max(0.006, lift_ds))
    plan._emit({arm: qs}, "lift")


def release_multi(plan, arms, z=None, open_rad=0.55, away=0.0, lift_ds=0.002, lift_to=0.63):
    """Same tested release as release_ticks, but all arms open, back off and lift at the same time."""
    import numpy as np
    import fold_plans as fp
    roll = {a: plan.q[a][4] for a in arms}
    pos = {a: fp.K.tip(plan.q[a], a)[0] for a in arms}
    if z is not None:
        seq = {}
        for a in arms:
            seq[a], _ = fp._path(a, [pos[a], (pos[a][0], pos[a][1], z)], plan.q[a], roll[a], ds=0.001)
            pos[a] = np.array([pos[a][0], pos[a][1], z])
        plan._emit(seq, "lower")
    g0 = {a: plan.g[a] for a in arms}
    for i in range(1, 9):
        plan.hold(1, "open", g={a: g0[a] + (open_rad - g0[a]) * i / 8 for a in arms})
    plan.hold(15, "open")
    if away > 0:
        seq = {}
        for a in arms:
            n = fp.K.moving_side(plan.q[a], a)[:2]; n = n / np.linalg.norm(n)
            p2 = np.array([pos[a][0] - n[0] * away, pos[a][1] - n[1] * away, pos[a][2]])
            seq[a], _ = fp._path(a, [pos[a], p2], plan.q[a], roll[a], ds=0.001); pos[a] = p2
        plan._emit(seq, "away")
    seq = {}
    for a in arms:
        seq[a], _ = fp._path(a, [pos[a], (pos[a][0], pos[a][1], lift_to)], plan.q[a], roll[a], ds=lift_ds, ds_ik=max(0.006, lift_ds))
    plan._emit(seq, "lift")


def near_tip(env, q_arm, side, table_z, verts=None, r=0.03, min_h=0.0):
    import numpy as np
    import fold_plans as fp
    v = cloth_verts(env) if verts is None else verts
    t = fp.K.tip(q_arm, side)[0]
    m = (np.linalg.norm(v - t, axis=1) < r) & ((v[:, 2] - table_z) > min_h)
    return int(m.sum()), (round(float((v[m, 2].mean() - table_z) * 100), 1) if m.any() else None)


def obs_jaw_deg(env, side):
    import numpy as np
    try:
        s = np.asarray(env._get_observations()["observation.state"], dtype=float).ravel()
        return round(float(np.degrees(s[5 if side == "left" else 11])), 2)
    except Exception:
        return None


def release_trial(env, S, keys, table_z_ref, open_rad, away, lift_ds):
    """Grasp the left sleeve (rule-based), carry it 12 cm toward the body, then release with the given parameters."""
    import numpy as np
    import fold_plans as fp
    from so101_kin import HOME, GRIP_CLOSED
    env.reset()
    home = np.concatenate([HOME["left"], [GRIP_CLOSED], HOME["right"], [GRIP_CLOSED]])
    for _ in range(110):
        S.step(home)
    v = cloth_verts(env)
    out = dict(open=open_rad, away_cm=round(away * 100, 1), lift_mm_step=round(lift_ds * 1000, 2))
    if not np.isfinite(v).all():
        out["error"] = "nan at settle"; return out
    table_z = float(np.percentile(v[:, 2], 1)) - 0.002
    arm = fp.shirt_arms(v, keys)["L"]
    cuff, pit = v[keys["L"]["cuff_mid"], :2], v[keys["L"]["armpit"], :2]
    axis = (pit - cuff) / np.linalg.norm(pit - cuff)
    q, _ = S.state()
    plan = fp.Plan(q, {"left": GRIP_CLOSED, "right": GRIP_CLOSED})
    roll, align = grasp_ticks(plan, arm, cuff + axis * 0.04, axis)
    S.run(plan.ticks); plan.ticks = []
    qn, gn = S.state()
    vc = cloth_verts(env)
    out.update(align=round(align, 2), held_closed=near_tip(env, qn[arm], arm, table_z, vc)[0],
               between_pads=between_pads(qn[arm], arm, vc), jaw_deg_art=round(float(np.degrees(gn[arm])), 2), jaw_deg_obs=obs_jaw_deg(env, arm))
    p = fp.K.tip(plan.q[arm], arm)[0]
    qs, _ = fp._path(arm, [p, (p[0], p[1], 0.60)], plan.q[arm], roll, ds=0.0007); plan._emit({arm: qs}, "lift")
    dx = 0.12 if arm == "left" else -0.12
    qs, _ = fp._path(arm, [(p[0], p[1], 0.60), (p[0] + dx, p[1], 0.60), (p[0] + dx, p[1], 0.545)], plan.q[arm], roll, ds=0.0035)
    plan._emit({arm: qs}, "carry"); S.run(plan.ticks); plan.ticks = []
    qn, _ = S.state()
    vb = cloth_verts(env)
    tip = fp.K.tip(qn[arm], arm)[0]
    near = np.nonzero(np.linalg.norm(vb - tip, axis=1) < 0.03)[0]
    out["held_before_release"] = int(len(near))
    if len(near) == 0:
        out["error"] = "dropped before release"; return out
    release_ticks(plan, arm, z=0.536, open_rad=open_rad, away=away, lift_ds=lift_ds)
    S.run(plan.ticks); plan.ticks = []
    qn, _ = S.state()
    va = cloth_verts(env)
    if not np.isfinite(va).all():
        out["error"] = "nan at release"; return out
    out["stuck"] = near_tip(env, qn[arm], arm, table_z, va, r=0.04, min_h=0.025)[0]
    out["drag_cm"] = round(float(np.linalg.norm(va[near, :2] - vb[near, :2], axis=1).mean() * 100), 2)
    out["lifted_cloth_max_cm"] = round(float((va[:, 2].max() - table_z) * 100), 1)
    plan.home(sides=[arm]); S.run(plan.ticks)
    return out


def mechanics_assay(env, args, job, name, out_dir):
    """Release factorial (cloth variant x open angle x back-away x lift speed) on the left sleeve."""
    import itertools
    import numpy as np
    import garment_keys as gk
    gfile = os.path.join(os.path.dirname(os.path.abspath(gk.__file__)), "garments", f"{name}.npz")
    keys = gk.top_keys(np.load(gfile)["points"].astype(float), list(env.object.check_points))
    S = Stepper(env)
    rows, t0 = [], time.time()
    for variant in job.get("cloth", ["real", "real_noadh"]):
        set_cloth(env, name, variant)
        for rep in range(int(job.get("reps", 2))):
            for open_rad, away, lift_ds in itertools.product(job.get("opens", [0.55, 1.4]), job.get("aways", [0.0, 0.02]),
                                                              job.get("lift_dss", [0.0007, 0.004])):
                if time.time() - t0 > float(job.get("budget_s", 420)):
                    break
                t1 = time.time()
                try:
                    r = release_trial(env, S, keys, None, open_rad, away, lift_ds)
                except Exception as e:
                    traceback.print_exc(); r = dict(error=repr(e)[:200])
                r.update(cloth=variant, rep=rep, wall_s=round(time.time() - t1, 1))
                rows.append(r); log("MECH_ROW " + json.dumps(r))
                if "error" in r:
                    set_cloth(env, name, variant)
    json.dump(rows, open(os.path.join(out_dir, f"mech_release_{name}.json"), "w"), indent=1)
    log(f"release assay: {len(rows)} trials in {time.time() - t0:.0f}s")
    return rows


def grasp_rule_check(env, args, job, name, out_dir):
    """Right-sleeve grasp of the given shirt with the moving finger leading vs the fixed finger leading."""
    import numpy as np
    import fold_plans as fp
    import garment_keys as gk
    from so101_kin import HOME, GRIP_CLOSED
    gfile = os.path.join(os.path.dirname(os.path.abspath(gk.__file__)), "garments", f"{name}.npz")
    keys = gk.top_keys(np.load(gfile)["points"].astype(float), list(env.object.check_points))
    set_cloth(env, name, job.get("cloth_variant", "real"))
    S = Stepper(env); rows = []
    for rep in range(int(job.get("reps", 3))):
        for lead in ("moving", "fixed"):
            try:
                env.reset()
                home = np.concatenate([HOME["left"], [GRIP_CLOSED], HOME["right"], [GRIP_CLOSED]])
                for _ in range(110):
                    S.step(home)
                v = cloth_verts(env); table_z = float(np.percentile(v[:, 2], 1)) - 0.002
                side = job.get("side", "R"); arm = fp.shirt_arms(v, keys)[side]
                cuff, pit = v[keys[side]["cuff_mid"], :2], v[keys[side]["armpit"], :2]
                axis = (pit - cuff) / np.linalg.norm(pit - cuff)
                q, _ = S.state()
                plan = fp.Plan(q, {"left": GRIP_CLOSED, "right": GRIP_CLOSED})
                roll, align = grasp_ticks(plan, arm, cuff + axis * 0.04, axis, lead=lead)
                S.run(plan.ticks); plan.ticks = []
                qn, _ = S.state(); vc = cloth_verts(env)
                r = dict(lead=lead, rep=rep, align=round(align, 2), held_closed=near_tip(env, qn[arm], arm, table_z, vc)[0],
                         between_pads=between_pads(qn[arm], arm, vc))
                p = fp.K.tip(plan.q[arm], arm)[0]
                qs, _ = fp._path(arm, [p, (p[0], p[1], 0.58)], plan.q[arm], roll, ds=0.0007); plan._emit({arm: qs}, "lift")
                S.run(plan.ticks); plan.ticks = []
                qn, _ = S.state()
                r["held_lifted"], r["lifted_height_cm"] = near_tip(env, qn[arm], arm, table_z, None, r=0.03, min_h=0.02)
                plan.ticks = []; plan.hold(10, "open", g={arm: 0.55}); plan.home(sides=[arm]); S.run(plan.ticks)
            except Exception as e:
                traceback.print_exc(); r = dict(lead=lead, rep=rep, error=repr(e)[:200]); set_cloth(env, name, job.get("cloth_variant", "real"))
            r["garment"] = name; rows.append(r); log("RULE_ROW " + json.dumps(r))
    json.dump(rows, open(os.path.join(out_dir, f"mech_rule_{name}.json"), "w"), indent=1)
    return rows


# ---- agent toolset: whole fold steps from the full cloth state, corner moves, save/restore, top-down map -------------
FOLD_DEFAULTS = dict(inset=0.04, slide=0.03, z_land=0.527, z_press=0.519, lift=0.073, carry_ds=0.0035, arc_h=0.03,
                     z_place=0.552, release_z=0.545, open=1.4, away=0.02, lift_ds=0.0007, inset_x=0.03, inset_y=0.02,
                     overshoot=0.0, dir=None)


def _unit(v):
    import numpy as np
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _reflect(pt, fold):
    import numpy as np
    p = np.asarray(fold["p"], float); d = _unit(fold["d"])
    rel = np.asarray(pt, float) - p
    a = rel @ d
    return p + a * d - (rel - a * d)


def map_jpg(R, ctx, verts, size=360):
    """Top-down map drawn from every cloth vertex (world x right, y up; arms at the bottom). Colour = height above
    the table (blue low -> red high), green = ideal outline for the steps done so far, black = full-plan target."""
    import numpy as np
    import fold_metric as fm
    x0, x1, y0, y1 = -0.30, 0.30, -0.25, 0.25
    W = size; H = int(size * (y1 - y0) / (x1 - x0))
    img = np.full((H, W, 3), 255, np.uint8)
    for gx in np.arange(-0.30, 0.31, 0.05):
        img[:, int((gx - x0) / (x1 - x0) * (W - 1))] = 225
    for gy in np.arange(-0.25, 0.26, 0.05):
        img[int((y1 - gy) / (y1 - y0) * (H - 1)), :] = 225
    pix = lambda xy: (np.clip(((y1 - xy[:, 1]) / (y1 - y0) * (H - 1)).astype(int), 0, H - 1),
                      np.clip(((xy[:, 0] - x0) / (x1 - x0) * (W - 1)).astype(int), 0, W - 1))
    h = np.clip((verts[:, 2] - ctx["table_z"]) / 0.04, 0, 1)
    order = np.argsort(verts[:, 2])
    r, c = pix(verts[order, :2]); hh = h[order]
    col = np.stack([255 * hh, 80 + 0 * hh, 255 * (1 - hh)], 1).astype(np.uint8)
    for dr in (0, 1):
        for dc in (0, 1):
            img[np.clip(r + dr, 0, H - 1), np.clip(c + dc, 0, W - 1)] = col

    def outline(xy, color):
        m = np.zeros((H, W), bool); rr, cc = pix(xy); m[rr, cc] = True
        e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        img[e] = color
    done = [f for n in ctx.get("done", []) for f in ctx["folds"] if f["name"] == n]
    if done:
        outline(fm.ideal_fold(ctx["flat"][:, :2], done)[0], (0, 170, 0))
    outline(fm.ideal_fold(ctx["flat"][:, :2], ctx["folds"])[0], (0, 0, 0))
    for s, colr in (("L", (255, 140, 0)), ("R", (160, 0, 200))):
        for k, i in ctx["keys"][s].items():
            if k in ("cuff_mid", "armpit", "shoulder", "hem_out"):
                rr, cc = pix(verts[[i], :2]); img[max(rr[0] - 2, 0):rr[0] + 3, max(cc[0] - 2, 0):cc[0] + 3] = colr
    return _jpg(R, img)


def score_now(ctx, verts, steps=None):
    import fold_metric as fm
    steps = ctx.get("done", []) if steps is None else steps
    folds = [f for n in steps for f in ctx["folds"] if f["name"] == n]
    if not folds:
        return None
    s = fm.score(verts, ctx["faces"], ctx["flat"], folds, ctx["table_z"])
    return {k: s[k] for k in ("neat", "neat_area", "s_area", "s_rect", "s_vertex", "s_mirror", "s_height", "s_footprint", "vertex_err_cm", "mirror_err_cm",
                              "h95_cm", "area_ratio", "rectangularity", "size_cm", "target_size_cm")}


def fold_step(env, R, ctx, step, params):
    """One whole fold step (sleeve_L / sleeve_R / bottom_up) from the current full cloth state, with checks."""
    import numpy as np
    import fold_plans as fp
    import garment_keys as gk
    p = dict(FOLD_DEFAULTS, **(params or {}))
    v = cloth_verts(env); keys = ctx["keys"]
    f = {x["name"]: x for x in gk.top_folds(v[:, :2], keys)}.get(step)
    arm_of = fp.shirt_arms(ctx["flat"], keys)
    jobs = []
    if step.startswith("sleeve_"):
        s = step[-1]
        cuff, pit = v[keys[s]["cuff_mid"], :2], v[keys[s]["armpit"], :2]
        axis = _unit(pit - cuff)
        d = _unit(p["dir"]) if p["dir"] is not None else axis
        target = cuff + axis * p["inset"]
        jobs.append((arm_of[s], target, d, _reflect(target, f) + d * p["overshoot"]))
    elif step == "half":
        # current packet: its bottom edge (lowest y) is folded up onto its top edge, both arms at the bottom corners
        xy = v[:, :2]; y0, y1 = np.percentile(xy[:, 1], 1), np.percentile(xy[:, 1], 99); ym = (y0 + y1) / 2
        low = xy[xy[:, 1] < y0 + 0.02]
        xl, xr = np.percentile(low[:, 0], 2), np.percentile(low[:, 0], 98)
        for arm, x, inward in (("left", xl, np.array([1.0, 0.0])), ("right", xr, np.array([-1.0, 0.0]))):
            target = np.array([x, y0]) + inward * p["inset_x"] + np.array([0.0, p["inset_y"]])
            jobs.append((arm, target, inward, np.array([target[0], 2 * ym - target[1] + p["overshoot"]])))
    else:
        hl, hr = v[keys["L"]["hem_out"], :2], v[keys["R"]["hem_out"], :2]
        across = _unit(hr - hl); up = _unit(np.asarray(f["p"]) - (hl + hr) / 2)
        for s, h, inward in (("L", hl, across), ("R", hr, -across)):
            target = h + inward * p["inset_x"] + up * p["inset_y"]
            jobs.append((arm_of[s], target, inward, _reflect(target, f) + up * p["overshoot"]))
    q, _ = R.state()
    plan = fp.Plan(q, dict(ctx["cmd"]["g"]))
    info = dict(step=step, arms={a: dict(target=np.round(t, 4).tolist(), dir=np.round(d, 3).tolist(), place=np.round(pl, 4).tolist())
                                 for a, t, d, pl in jobs})
    s0 = score_now(ctx, v, ctx.get("done", []) + [step])
    rolls = {}
    for a, t, d, _ in jobs:                       # grasps one arm after the other; the first keeps holding
        rolls[a], al = grasp_ticks(plan, a, t, d, slide=p["slide"], z_land=p["z_land"], z_press=p["z_press"])
        info["arms"][a]["align"] = round(al, 2)
    R.run(plan); plan.ticks = []
    qn, _ = R.state(); vc = cloth_verts(env)
    for a, *_ in jobs:
        info["arms"][a]["held_closed"] = near_tip(env, qn[a], a, ctx["table_z"], vc)[0]
    seq = {}
    for a, t, d, pl in jobs:                      # lift + arc carry, both arms together
        p0 = fp.K.tip(plan.q[a], a)[0]
        zl = p["z_press"] + p["lift"]
        pts = [p0, (p0[0], p0[1], zl)] + fp._arc(p0[:2], pl, zl, p["z_place"], p["arc_h"])
        seq[a], _ = fp._path(a, pts, plan.q[a], rolls[a], ds=p["carry_ds"], ds_ik=max(0.006, p["carry_ds"]))
    plan._emit(seq, "carry"); plan.hold(15, "settle")
    R.run(plan); plan.ticks = []
    qn, _ = R.state(); vb = cloth_verts(env)
    for a, *_ in jobs:
        info["arms"][a]["held_at_place"] = near_tip(env, qn[a], a, ctx["table_z"], vb)[0]
    release_multi(plan, [a for a, *_ in jobs], z=p["release_z"], open_rad=p["open"], away=p["away"], lift_ds=p["lift_ds"])
    R.run(plan); plan.ticks = []
    qn, _ = R.state()
    for a, *_ in jobs:
        info["arms"][a]["stuck"] = near_tip(env, qn[a], a, ctx["table_z"], None, r=0.04, min_h=0.025)[0]
    plan.home(sides=[a for a, *_ in jobs]); plan.hold(30, "settle")
    R.run(plan)
    ctx["cmd"] = {"q": plan.q, "g": dict(plan.g)}
    if step not in ctx.setdefault("done", []):
        ctx["done"].append(step)
    va = cloth_verts(env)
    info["score_step"] = score_now(ctx, va)
    info["score_before"] = s0["neat"] if s0 else None
    return info


def move_corner(env, R, ctx, cmd):
    """Generic tested pick-and-place: grab vertex (or key point) with a gather slide along `dir`, put it at `to`."""
    import numpy as np
    import fold_plans as fp
    p = dict(FOLD_DEFAULTS, **cmd.get("params", {}))
    v = cloth_verts(env)
    if "key" in cmd:
        s, k = cmd["key"].split("."); idx = ctx["keys"][s][k]
    else:
        idx = int(cmd["vertex"])
    a = cmd["arm"]; d = _unit(cmd["dir"])
    target = v[idx, :2] + d * float(cmd.get("inset", 0.02)); dest = np.asarray(cmd["to"], float)
    q, _ = R.state(); plan = fp.Plan(q, dict(ctx["cmd"]["g"]))
    roll, al = grasp_ticks(plan, a, target, d, slide=p["slide"], z_land=p["z_land"], z_press=p["z_press"])
    R.run(plan); plan.ticks = []
    qn, _ = R.state(); info = dict(align=round(al, 2), held_closed=near_tip(env, qn[a], a, ctx["table_z"])[0])
    p0 = fp.K.tip(plan.q[a], a)[0]; zl = p["z_press"] + float(cmd.get("lift", 0.04))
    pts = [p0, (p0[0], p0[1], zl)] + fp._arc(p0[:2], dest, zl, p["z_place"], p["arc_h"])
    qs, _ = fp._path(a, pts, plan.q[a], roll, ds=p["carry_ds"], ds_ik=max(0.006, p["carry_ds"]))
    plan._emit({a: qs}, "carry"); plan.hold(15, "settle")
    release_ticks(plan, a, z=p["release_z"], open_rad=p["open"], away=p["away"], lift_ds=p["lift_ds"])
    R.run(plan); plan.ticks = []
    qn, _ = R.state(); info["stuck"] = near_tip(env, qn[a], a, ctx["table_z"], None, r=0.04, min_h=0.025)[0]
    plan.home(sides=[a]); plan.hold(20, "settle"); R.run(plan)
    ctx["cmd"] = {"q": plan.q, "g": dict(plan.g)}
    info["score"] = score_now(ctx, cloth_verts(env))
    return info


def save_state(env, R, ctx, name):
    from scripts.utils.remote_loop import capture_physics_state
    ctx.setdefault("saves", {})[name] = dict(state=capture_physics_state(env), cmd=ctx["cmd"], done=list(ctx.get("done", [])))
    return dict(saved=name, slots=list(ctx["saves"]))


def restore_state(env, R, ctx, name):
    import numpy as np
    from scripts.utils.remote_loop import restore_physics_state
    sv = ctx["saves"][name]
    ok = restore_physics_state(env, sv["state"], "exact", "oracle")
    ctx["cmd"] = sv["cmd"]; ctx["done"] = list(sv["done"])
    q, g = sv["cmd"]["q"], sv["cmd"]["g"]
    for _ in range(10):
        R.step({"left": q["left"], "right": q["right"]}, g, "restore")
    return dict(restored=name, ok=bool(ok), done=ctx["done"])


def kin_check(env):
    """Log the simulator's link positions next to our URDF forward kinematics (sanity check of so101_kin)."""
    import numpy as np
    from so101_kin import SO101
    k = SO101()
    obs = env._get_observations()
    s = np.asarray(obs["observation.state"], float).ravel()
    for side, arm, sl in (("left", env.left_arm, slice(0, 5)), ("right", env.right_arm, slice(6, 11))):
        try:
            names = list(arm.body_names)
            pos = _np(arm.data.body_pos_w)[0]
            tip, appr = k.tip(s[sl], side)
            log(f"kin {side}: q={s[sl].round(3).tolist()} fk_tip={tip.round(4).tolist()} approach={appr.round(2).tolist()}")
            log(f"kin {side}: sim bodies " + json.dumps({n: pos[i].round(4).tolist() for i, n in enumerate(names)}))
            log(f"kin {side}: joint names {list(arm.joint_names)}")
        except Exception as e:
            log("kin_check failed:", repr(e))


def run(args, simulation_app):
    import gymnasium as gym
    import numpy as np
    import lehome.tasks.bedroom  # noqa: F401  (registers the task)
    from isaaclab_tasks.utils import parse_env_cfg
    from .utils.evaluation import apply_camera_overrides

    sys.path.insert(0, os.environ.get("ORACLE_DIR", "/opt/teacher/oracle"))
    jobs = json.loads(os.environ["ORACLE_JOBS"])
    os.makedirs(OUT, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = os.environ.get("ORACLE_FABRIC", "0") == "1"
    probe_mode = os.environ.get("ORACLE_GPU_PROBE") == "1"
    if probe_mode:                       # GPU-pipeline diagnostic (gpu_probe.py): stack dump + exit on a hang
        import gpu_probe
        gpu_probe.arm_watchdog(log, f"env creation on {args.device}, use_fabric={env_cfg.sim.use_fabric}")
    env_cfg.use_random_seed = False
    env_cfg.seed = args.seed
    env_cfg.random_seed = args.seed
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    apply_camera_overrides(env_cfg, args)
    env_cfg.garment_name = jobs[0]["garment"]
    if os.environ.get("ORACLE_LIVE_DIR") or os.environ.get("ORACLE_TELEOP"):
        env_cfg.episode_length_s = 36000.0      # no automatic reset during a live / teleop session
    if os.environ.get("ORACLE_TELEOP") and os.environ.get("ORACLE_TELEOP_FAST", "1") != "0":
        # live play only (the recorded data comes from replays with the normal settings): fast render preset and
        # tiny wrist cameras, so each step renders ~one camera's worth of pixels instead of three 640x480 'quality' ones
        env_cfg.sim.render.rendering_mode = "performance"
        for cam in (env_cfg.left_wrist, env_cfg.right_wrist):
            cam.width, cam.height = 64, 48
        if os.environ.get("ORACLE_FPV", "1") != "0":
            import math
            vw, vh = (int(x) for x in os.environ.get("ORACLE_VIEW_SIZE", "960x600").split("x"))
            tc = env_cfg.top_camera
            tc.width, tc.height = vw, vh
            vap = tc.spawn.horizontal_aperture * vh / vw
            tc.spawn.focal_length = (vap / 2) / math.tan(math.radians(65 / 2))   # MuJoCo game: fovy 65
        k = int(os.environ.get("ORACLE_STEPS_PER_TICK", "1"))
        state_view = os.environ.get("ORACLE_VIEW", "state") == "state"
        # state view: the laptop draws, Isaac renders only every 48 steps (Kit aborted once after ~7k unrendered steps)
        env_cfg.sim.render_interval = 48 if state_view else k
        log(f"teleop: {k} physics steps per tick, Isaac renders every {env_cfg.sim.render_interval} steps"
            f" ({'laptop draws from state' if state_view else 'JPEG stream'})")
        log("teleop fast view: rendering_mode=performance, wrist cameras 64x48")
    env_cfg.garment_version = "Release"
    if os.environ.get("ORACLE_PADS") == "flat":
        import pads as padmod
        src = env_cfg.left_robot.spawn.usd_path
        dst = os.path.join(os.path.dirname(src), "so101_flatpads.usd")
        padmod.make(src, dst)
        env_cfg.left_robot.spawn.usd_path = dst
        env_cfg.right_robot.spawn.usd_path = dst
        log("robot pads: flat ->", dst)
    else:
        log("robot pads: original capsules")
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()
    log(f"env ready in {time.time() - t0:.1f}s; device={env.device} dt={env.cfg.sim.dt} max_ep_len={env.max_episode_length}")
    if probe_mode:
        try:
            gpu_probe.probe(env, log, cloth_verts, set_cloth, jobs[0]["garment"])
        finally:
            env.close()
        return
    if os.environ.get("ORACLE_TELEOP"):
        try:
            teleop(env, args)
        finally:
            env.close()
        return
    if os.environ.get("ORACLE_LIVE_DIR"):
        try:
            live(env, args)
        finally:
            env.close()
        return
    results, current = [], jobs[0]["garment"]
    try:
        for job in jobs:
            name = job["garment"]
            if name != current:
                env.switch_garment(name); current = name
            if job.get("mode") in ("mechanics", "grasp_rule"):
                try:
                    fn = mechanics_assay if job["mode"] == "mechanics" else grasp_rule_check
                    results.append(dict(garment=name, mode=job["mode"], n=len(fn(env, args, job, name, OUT))))
                except Exception as e:
                    log("MECH FAILED", name, repr(e)); traceback.print_exc()
                continue
            if job.get("mode") == "grip_assay":
                try:
                    rows = grip_assay(env, args, job, name, OUT)
                    env._assay_rows = rows
                    results.append(dict(garment=name, assay=len(rows)))
                except Exception as e:
                    log("ASSAY FAILED", name, repr(e)); traceback.print_exc()
                continue
            if job.get("mode") == "fold_best":
                try:
                    params = dict(job.get("params", {}))
                    cloth, speed, held = best_variant(getattr(env, "_assay_rows", []) or [dict(cloth="asis", speed_mm_step=1.5)])
                    params["carry_ds"] = float(min(max(speed, 0.0015), 0.004))
                    log(f"FOLD with cloth={cloth} carry_ds={params['carry_ds']} (assay held {held:.2f})")
                    set_cloth(env, name, cloth)
                    r = shirt_trial(env, args, params, name, f"{os.environ.get('ORACLE_PADS', 'capsule')}_{cloth}", OUT)
                    results.append(dict(garment=name, trial=r["trial"], neat=r["neat"], cloth=cloth))
                except Exception as e:
                    log("FOLD FAILED", name, repr(e)); traceback.print_exc()
                continue
            if job.get("mode") == "assist_test":
                try:
                    results.append(dict(garment=name, assist=len(assist_test(env, args, job))))
                except Exception as e:
                    log("ASSIST TEST FAILED", repr(e)); traceback.print_exc()
                continue
            if job.get("mode") == "grasp_test":
                try:
                    env.reset()
                    results.append(dict(garment=name, grasp=grasp_test(env, args, job, name, OUT)))
                except Exception as e:
                    log("GRASP TEST FAILED", name, repr(e)); traceback.print_exc()
                continue
            for trial in range(int(job.get("trials", 1))):
                try:
                    env.cfg.seed = env.cfg.random_seed = int(job.get("seed", 42)) + trial
                    env.reset()
                    if not results:
                        kin_check(env)
                    r = shirt_trial(env, args, job.get("params", {}), name, f"{job.get('tag', 'a')}{trial}", OUT)
                    results.append(dict(garment=name, trial=r["trial"], neat=r["neat"]))
                except Exception as e:
                    log("TRIAL FAILED", name, trial, repr(e)); traceback.print_exc()
                    results.append(dict(garment=name, trial=trial, error=repr(e)))
    finally:
        json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
        log("SUMMARY " + json.dumps(results))
        env.close()


def main():
    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = launch_app_from_args(args)
    try:
        if getattr(args, "headless", False):
            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
        run(args, simulation_app)
    except BaseException as e:      # BaseException: a sys.exit() deep in env creation otherwise closes the app silently
        log("FATAL", repr(e)); traceback.print_exc()
    finally:
        close_app(simulation_app)


if __name__ == "__main__":
    main()
