"""Unit checks for the Anchor-Lab fit (python anchor_tests.py). Writes results/anchor_tests.json."""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import anchor_data as AD  # noqa: E402
import anchor_fit as AF  # noqa: E402

NV_CAM_USD = os.path.join(AD.SRC, "robot_assets", "so101_camera_mass_23g_at_camera.usd")


def t_schedule():
    """A step at 1.000 s with a 30 ms delay: steps fully before 1.030 s hold the old value, steps fully after hold
    the new one, the straddling step is the time-weighted blend."""
    t_cmd = np.array([0.0, 1.0, 5.0])
    qc = np.array([[0.0], [1.0], [1.0]])
    dt = 0.004
    U = AF.event_targets(t_cmd, qc, 0.030, 0.0, 400, dt)[:, 0]
    edge = 1.030 / dt                     # 257.5 -> step 257 covers [1.028, 1.032): half old, half new
    ok = (np.all(U[:257] == 0.0) and abs(U[257] - 0.5) < 1e-9 and np.all(np.abs(U[258:] - 1.0) < 1e-12))
    nat = AF.native_targets(t_cmd, qc, 0.0, 400, dt)[:, 0]
    ok_nat = abs(nat[0]) < 1e-12 and abs(nat[5] - 0.02) < 1e-12   # 2nd control step = interp at 0.02 s
    return {"ok": bool(ok and ok_nat), "edge_step": edge, "U_256_257_258": U[256:259].tolist()}


def t_masses():
    import mujoco
    o = AF.opts(camera_g=AF.CAM_NVIDIA_G)
    s = AF.get_pos_sim(o, AF.base_params())
    m = s.model
    total = float(m.body_mass.sum())
    expect_total = 0.632006 + 0.0231
    gid, cid = m.body("gripper").id, m.body("camera_mount").id
    mg, mc = m.body_mass[gid], m.body_mass[cid]
    com = (mg * m.body_ipos[gid] + mc * m.body_ipos[cid]) / (mg + mc)
    nv_com = None
    nv_mass = None
    try:
        from pxr import Usd, UsdPhysics
        st = Usd.Stage.Open(NV_CAM_USD)
        p = UsdPhysics.MassAPI(st.GetPrimAtPath("/so101_new_calib/gripper"))
        nv_mass = float(p.GetMassAttr().Get())
        nv_com = np.array(p.GetCenterOfMassAttr().Get(), float)
    except Exception as e:   # pragma: no cover
        nv_com = repr(e)
    ok_com = isinstance(nv_com, np.ndarray) and float(np.max(np.abs(com - nv_com))) < 1e-4
    ok = abs(total - expect_total) < 1e-5 and ok_com and abs((mg + mc) - nv_mass) < 1e-6
    _ = mujoco
    return {"ok": bool(ok), "total_kg": total, "expect_kg": expect_total, "gripper_plus_cam_kg": float(mg + mc),
            "nvidia_gripper_kg": nv_mass, "com": com.tolist(),
            "nvidia_com": nv_com.tolist() if isinstance(nv_com, np.ndarray) else nv_com,
            "com_maxdiff_m": float(np.max(np.abs(com - nv_com))) if isinstance(nv_com, np.ndarray) else None}


def t_json_roundtrip():
    """fitted_json -> rebuild a sim from the JSON -> model arrays equal the params."""
    o = AF.opts(friction="F1_imp990")
    p = AF.base_params()
    js = AF.fitted_json(o, p, 0.031, "test")
    p2 = {k: [js["joints"][j][k] for j in AF.JOINTS] for k in p}
    s = AF.get_pos_sim(o, p2)
    m = s.model
    back = {"kp": m.actuator_gainprm[s.aid, 0], "kv": -m.actuator_biasprm[s.aid, 2],
            "damping": m.dof_damping[s.vadr], "frictionloss": m.dof_frictionloss[s.vadr],
            "armature": m.dof_armature[s.vadr], "forcerange": m.actuator_forcerange[s.aid, 1]}
    err = max(float(np.max(np.abs(np.asarray(p[k]) - back[k]))) for k in p)
    ok_imp = np.allclose(m.dof_solimp[s.vadr, :2], 0.99)
    return {"ok": bool(err < 1e-12 and ok_imp and js["sim"]["solimpfriction"][0] == 0.99), "max_abs_err": err}


def t_jaw_map():
    x = np.array([0.0, 0.5, 1.0])
    th = AD.jaw_to_rad(x)
    ok = np.allclose(np.degrees(th), [-10.0, -10.0 + 63.5449, 117.0898], atol=1e-3) and \
        np.allclose(AD.rad_to_jaw(th), x)
    return {"ok": bool(ok), "deg": np.degrees(th).tolist(), "span_deg": AD.JAW_SPAN_DEG}


if __name__ == "__main__":
    res = {}
    for name, fn in (("schedule", t_schedule), ("masses_com", t_masses), ("json_roundtrip", t_json_roundtrip),
                     ("jaw_map", t_jaw_map)):
        res[name] = fn()
        print(f"{name:15s} {'PASS' if res[name]['ok'] else 'FAIL'}  {json.dumps(res[name])[:300]}")
    with open(os.path.join(AF.RESULTS, "anchor_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    sys.exit(0 if all(r["ok"] for r in res.values()) else 1)
