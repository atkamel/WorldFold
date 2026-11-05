"""Isaac side of the Dream-1 grasp test (dream_test/server.py drives it), active only when DREAM_SESSION=1.

  evaluation.py   DREAM_CAMS=1: the top camera moved to Dream's station view (behind the arms, 30 deg off vertical,
                  the YAM station's pose scaled by our arm spacing 0.46/0.61) and all three cameras at the YAM
                  cameras' 78 deg horizontal field of view (dream_test/README.md)
  remote_loop.py  per episode: POST /dream_episode with the garment's start shape and check points (the server saves
                  them for scoring) and gets back whether this episode has the red block; per step: the physics dt
                  and the block's position ride along with the observation; n_steps > 1 runs the extra physics
                  steps without rendering (the cameras only need the last one), so a Dream row (1/20 s) costs one
                  render instead of 4.5

Run after grasp_log_patch.py (it anchors on that patch's _gl_setup line):
    python dream_isaac_patch.py /opt/lehome_solution
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/lehome_solution")
MARK = "# [dream-test-patch]"
CH = ROOT / "lehome-challenge"

# ---- cameras -------------------------------------------------------------------------------------------------------
p = CH / "scripts/utils/evaluation.py"
s = p.read_text()
if MARK not in s:
    old = "    apply_camera_overrides(env_cfg, args)\n"
    assert s.count(old) == 1, "evaluation.py: apply_camera_overrides call not found"
    s = s.replace(old, old + '''    try:  ''' + MARK + '''
        import os as _os_dc
        if _os_dc.environ.get("DREAM_CAMS") == "1":
            # pose: dream_test/README.md "Camera"; parent frame = the right robot's base (LeHome's own top camera mount)
            env_cfg.top_camera.offset.pos = (0.244, 0.1413, 0.7446)
            env_cfg.top_camera.offset.rot = (0.0, 0.0, 0.25882, 0.96593)
            env_cfg.top_camera.offset.convention = "opengl"
            env_cfg.top_camera.spawn.focal_length = 23.35      # 78.4 deg over the 38.11 mm aperture (YAM fx 392 @ 640)
            for _w in (env_cfg.left_wrist, env_cfg.right_wrist):
                _w.spawn.focal_length = 22.48                  # 78.6 deg over 36.83 mm (YAM wrist fx 391 @ 640)
            for _c in (env_cfg.top_camera, env_cfg.left_wrist, env_cfg.right_wrist):
                _c.width, _c.height = 640, 480
            env_cfg.top_camera.data_types = ["rgb"]
            print("[dream] cameras: station view, 78 deg, 640x480", flush=True)
    except Exception as _e_dc:
        print(f"[dream] cameras NOT changed: {_e_dc!r}", flush=True)
''')
    p.write_text(MARK + "\n" + s)
    print("patched:", p.name)
else:
    print("already patched:", p.name)

# ---- episode loop --------------------------------------------------------------------------------------------------
HELPERS = '''

_DREAM_BLOCK = {}


def _dream_block(env, place, xy=None, z=None):
    """The red 4 cm block (USD rigid body, created on first use): placed at (xy, z), or parked kinematic off the
    table. Returns its prim path or None."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    stage = env.sim.stage if hasattr(env.sim, "stage") else __import__("omni.usd").usd.get_context().get_stage()
    path = "/World/DreamBlock"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        if not place:
            return None
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(0.04)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.08, 0.08)])
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(0.05)
        mat = UsdShade.Material.Define(stage, path + "_mat")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.GetStaticFrictionAttr().Set(1.0)
        pm.GetDynamicFrictionAttr().Set(1.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        _DREAM_BLOCK["xf"] = UsdGeom.XformCommonAPI(prim)
    rb = UsdPhysics.RigidBodyAPI(prim)
    if place:
        rb.GetKinematicEnabledAttr().Set(False)
        _DREAM_BLOCK["xf"].SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), float(z)))
    else:
        rb.GetKinematicEnabledAttr().Set(True)
        _DREAM_BLOCK["xf"].SetTranslate(Gf.Vec3d(2.0, 2.0, 0.02))
    _DREAM_BLOCK["xf"].SetRotate(Gf.Vec3f(0, 0, 0))
    rb.GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    _DREAM_BLOCK["prim"] = prim
    return path


def _dream_begin(env, policy):
    """Dream episode start (DREAM_SESSION=1): send the garment's start shape, place or park the block."""
    import os as _os
    if _os.environ.get("DREAM_SESSION") != "1":
        return None
    st = {"dt": float(getattr(env, "physics_dt", 1.0 / 90.0)), "block": False, "fast": True}
    try:
        o = env.object
        v = _gl_verts(env)
        local = np.array(o._prim.GetAttribute("points").Get(), dtype=np.float32)
        table_z = float(np.percentile(v[:, 2], 1)) - 0.002
        cfg = policy.post("/dream_episode", {
            "garment": getattr(policy, "_garment_name", None) or env.cfg.garment_name,
            "check_points": list(map(int, getattr(o, "check_points", []) or [])),
            "verts": np.round(v, 4).tolist(), "local": np.round(local, 4).tolist(),
            "table_z": table_z, "physics_dt": st["dt"]})
        if cfg.get("block"):
            c = v.mean(0)
            _dream_block(env, True, xy=c[:2], z=float(v[:, 2].max()) + 0.03)
            st["block"] = True
        else:
            _dream_block(env, False)
    except Exception as e:
        print(f"[dream] episode start failed: {e!r}", flush=True)
    return st


def _dream_obs(st, env, payload):
    if st is None:
        return payload
    payload["dream_physics_dt"] = st["dt"]
    if st["block"] and "prim" in _DREAM_BLOCK:
        try:
            from pxr import UsdGeom
            t = UsdGeom.Xformable(_DREAM_BLOCK["prim"]).ComputeLocalToWorldTransform(0).ExtractTranslation()
            payload["dream_block_pos"] = [float(t[0]), float(t[1]), float(t[2])]
        except Exception as e:
            print(f"[dream] block read failed: {e!r}", flush=True)
    return payload


def _dream_step(st, env, action, n):
    """n physics steps holding one action; only the last one renders (DirectRLEnv.step's inner loop otherwise)."""
    if st is not None and st["fast"] and n > 1:
        try:
            for _ in range(n - 1):
                env._pre_physics_step(action)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            n = 1
        except Exception as e:
            st["fast"] = False
            print(f"[dream] fast stepping off, rendering every step: {e!r}", flush=True)
    for _ in range(n):
        env.step(action)


def _run_steps('''

p = CH / "scripts/utils/remote_loop.py"
s = p.read_text()
if MARK not in s:
    assert "_gl_setup(env, policy)" in s, "apply grasp_log_patch.py first"
    edits = [
        ("\ndef _run_steps(", HELPERS),
        ("    _gl = _gl_setup(env, policy)\n", "    _gl = _gl_setup(env, policy)\n    _dream = _dream_begin(env, policy)\n"),
        ('        resp = policy.post("/infer", policy.serialize_observation(obs))\n',
         '        resp = policy.post("/infer", _dream_obs(_dream, env, policy.serialize_observation(obs)))\n'),
        ('        for _ in range(int(resp.get("n_steps", 1))):\n            env.step(action)\n',
         '        _dream_step(_dream, env, action, int(resp.get("n_steps", 1)))\n'),
    ]
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"expected 1 match, found {n} for:\n{old[:120]}"
        s = s.replace(old, new)
    p.write_text(MARK + "\n" + s)
    print("patched:", p.name)
else:
    print("already patched:", p.name)
# ---- optional start pose (DREAM_START_POSE=ready): both hands 8 cm over the garment's near half, jaws down, open ----
MARK2 = "# [dream-test-patch-pose]"
POSE = '''

def _dream_pose(st, env):
    """DREAM_START_POSE=ready: put both arms in the ready pose (solved with oracle/so101_kin.py: tips at x = -/+0.12,
    y = -0.02, z = 0.60, 8 cm over the table, jaws down, gripper 0.55 rad open) and hold it 30 steps."""
    import os as _os
    if st is None or _os.environ.get("DREAM_START_POSE") != "ready":
        return False
    import torch as _t
    q = [0.564, 0.515, -0.671, 1.658, 0.0, 0.55, -0.407, 0.337, -0.398, 1.601, 0.0, 0.55]
    dev = env.left_arm.data.joint_pos.device
    for arm, j in ((env.left_arm, q[:6]), (env.right_arm, q[6:])):
        t = _t.tensor([j], dtype=_t.float32, device=dev)
        arm.set_joint_position_target(t)
        arm.write_joint_state_to_sim(t, t * 0)
    a = _t.tensor([q], dtype=_t.float32, device=dev)
    for _ in range(30):
        env.step(a)
    return True


def _run_steps('''
p = CH / "scripts/utils/remote_loop.py"
s = p.read_text()
if MARK2 not in s:
    for old, new in (("\ndef _run_steps(", POSE),
                     ("    _dream = _dream_begin(env, policy)\n",
                      "    _dream = _dream_begin(env, policy)\n    if _dream_pose(_dream, env):\n        obs = env.obs_buf\n")):
        assert s.count(old) == 1, f"pose patch anchor: {old[:60]}"
        s = s.replace(old, new)
    p.write_text(MARK2 + "\n" + s)
    print("patched: remote_loop.py (start pose)")
else:
    print("already patched: start pose")
print("DREAM_ISAAC_PATCH_OK")
