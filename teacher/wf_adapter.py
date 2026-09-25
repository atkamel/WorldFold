"""WorldFold <-> lehome_sim (pi0.5) adapter.

Teacher contract (from the runbook):
  obs:  observation.images.{top_rgb,left_rgb,right_rgb} 640x480x3 uint8, raw;
        observation.state = 12 floats, ABSOLUTE joint rad,
        [L pan, lift, elbow, wflex, wroll, gripper, R same]; garment_type_id int.
  out:  actions (E, 12) absolute joint targets (rad) at 30 Hz; next_initial_actions for inpainting.

WorldFold ClothFoldEnv (joint_delta mode, 14-dim action):
  a[0:5] left joint deltas (x JOINT_DELTA_SCALE rad/step), a[6] left gripper cmd (<-0.3 close, >0.3 open),
  a[7:12] right joint deltas, a[13] right gripper cmd; a[5], a[12] unused.

Mapping: delta = clip((target - qpos) / JOINT_DELTA_SCALE, -1, 1); gripper cmd = +1 if target > GRIP_THR else -1.
Control at ~30 Hz (control_dt = 67 * 0.5 ms). Same joint order and calibration (limits match to the degree).
"""
from __future__ import annotations

import base64, json, os, time
import numpy as np
import mujoco

from mujuco.sim_main import ClothFoldEnv, ARM_JOINTS, JOINT_DELTA_SCALE, TABLE_TOP_Z  # noqa: E402

W, H = 640, 480
TOP_FOVY, WRIST_FOVY = 53.0, 41.5
TOP_VIEW = dict(lookat=(0.0, 0.0, TABLE_TOP_Z), distance=0.70, azimuth=90.0, elevation=-68.0)
CONTROL_DT = 67 * 0.0005          # 29.85 Hz, nearest integer substep count to 1/30
GRIP_THR = 0.2                    # rad; his data: gripper ~-0.15 closed .. ~0.5 open (rarely 1.2)
# his start pose, identical across all 250 episodes checked (dataset first frames), rad:
HIS_START = (-1.24, -1.69, 1.49, 1.05, -0.08, -0.01,  1.24, -1.69, 1.49, 1.05, -0.08, -0.01)
WRIST_CAM_POS, WRIST_CAM_EULER_X = (0.0, 0.12, -0.02), -0.8   # matched visually to his wrist frames
TABLE_HALF = 0.70                 # his arms sit over a large table; ours were off a 0.6 m one
GARMENT_TYPES = ("top_long", "top_short", "pant_long", "pant_short")
PREFIXES = ("left_", "right_")


def _spec_hook(spec):
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.95, 0.95, 0.95], rgb2=[0.75, 0.78, 0.82], width=256, height=256)


TEACHER_MOUNT_GAP = 0.075   # arm base line sits this far behind the cloth's near edge (worked in pilot 2)


def _is_roy_sim(SM):
    import inspect
    return "grasp_corners" not in inspect.signature(SM.ClothFoldEnv.__init__).parameters


def make_env(max_episode_steps=600, white=True, cloth_spacing=None, cloth_rgba=None):
    """cloth_spacing: vertex gap (m). None keeps the sim's own cloth (main 0.30 m, Roy's 0.22 m).
    Roy's stats-worldfold sim: file untouched; only its arm mount is moved to the teacher's layout
    (both arms side by side behind the cloth, facing it), its own cloth/welds/half-fold goal kept."""
    import mujuco.sim_main as SM
    if cloth_spacing is not None:
        SM.CLOTH_SPACING = float(cloth_spacing)      # read at compile time by build_cloth_xml
    if _is_roy_sim(SM):
        half = (SM.CLOTH_COUNT - 1) * SM.CLOTH_SPACING / 2
        y = -(half + TEACHER_MOUNT_GAP); z = SM.TABLE_TOP_Z + 0.06
        SM.ARM_BASE_LEFT, SM.ARM_BASE_RIGHT = (-0.22, y, z), (0.22, y, z)
        SM.ARM_QUAT_LEFT = SM.ARM_QUAT_RIGHT = [0.70710678, 0.0, 0.0, 0.70710678]   # both face +y
        env = SM.ClothFoldEnv(observation_mode="state", action_mode="joint_delta",
                              control_dt=CONTROL_DT, max_episode_steps=max_episode_steps)
        env._roy = True
        m = env.model
        tid = m.geom("table").id
        m.geom_size[tid][0] = m.geom_size[tid][1] = TABLE_HALF
        return _finish_env(env, white, cloth_rgba)
    n = SM.CLOTH_COUNT * SM.CLOTH_COUNT
    # grasp proxy: a closing gripper welds ANY cloth vertex within grasp_radius (teacher can grab anywhere)
    anyv = {"left_": tuple(range(n)), "right_": tuple(range(n))}
    env = ClothFoldEnv(observation_mode="state", action_mode="joint_delta",
                       control_dt=CONTROL_DT, max_episode_steps=max_episode_steps, spec_hook=_spec_hook,
                       grasp_corners=anyv, grasp_radius=0.04)
    env._roy = False
    m = env.model
    tid = m.geom("table").id
    m.geom_size[tid][0] = m.geom_size[tid][1] = TABLE_HALF
    return _finish_env(env, white, cloth_rgba)


def _finish_env(env, white, cloth_rgba):
    m = env.model
    # his arms are all yellow; WorldFold paints left blue / right yellow for humans
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if bn.startswith(PREFIXES):
            m.geom_rgba[g] = (0.93, 0.85, 0.25, 1.0)
    for p in PREFIXES:
        cid = m.camera(f"{p}wrist_cam").id
        m.cam_pos[cid] = WRIST_CAM_POS
        a = WRIST_CAM_EULER_X
        m.cam_quat[cid] = (np.cos(a / 2), np.sin(a / 2), 0.0, 0.0)
    env._grip_target = {p: None for p in PREFIXES}
    _orig_set_gripper = env.set_gripper

    def _set_gripper(prefix, command):
        _orig_set_gripper(prefix, command)
        tgt = env._grip_target.get(prefix)
        if tgt is not None:
            lo, hi = env.model.actuator_ctrlrange[env._gripper_act[prefix]]
            env.data.ctrl[env._gripper_act[prefix]] = float(np.clip(tgt, lo, hi))
    env.set_gripper = _set_gripper
    if cloth_rgba is not None:
        m = env.model
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in set(env._cloth_body_ids):
                m.geom_rgba[g] = cloth_rgba
        for f in range(m.nflex):
            m.flex_rgba[f] = cloth_rgba
    m = env.model
    if white:
        m.geom_rgba[m.geom("table").id] = (0.92, 0.92, 0.90, 1.0)
        m.geom_rgba[m.geom("floor").id] = (0.85, 0.85, 0.85, 1.0)
        m.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
        m.vis.headlight.diffuse[:] = (0.45, 0.45, 0.45)
    for p in PREFIXES:
        m.cam_fovy[m.camera(f"{p}wrist_cam").id] = WRIST_FOVY
    m.vis.global_.fovy = TOP_FOVY
    return env


class Cameras:
    def __init__(self, env):
        self.env = env
        self.r = mujoco.Renderer(env.model, height=H, width=W)
        self.top = mujoco.MjvCamera()
        self.top.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.top.lookat[:] = TOP_VIEW["lookat"]
        self.top.distance, self.top.azimuth, self.top.elevation = (
            TOP_VIEW["distance"], TOP_VIEW["azimuth"], TOP_VIEW["elevation"])

    def render(self):
        d = self.env.data
        self.r.update_scene(d, camera=self.top); top = np.ascontiguousarray(np.rot90(self.r.render(), 2))
        self.r.update_scene(d, camera="left_wrist_cam"); left = self.r.render().copy()
        self.r.update_scene(d, camera="right_wrist_cam"); right = self.r.render().copy()
        return {"top_rgb": top, "left_rgb": left, "right_rgb": right}


def set_teacher_init_pose(env, q12=HIS_START):
    m, d = env.model, env.data
    for i, p in enumerate(PREFIXES):
        for k, jn in enumerate(ARM_JOINTS):
            v = q12[6 * i + k]
            d.qpos[m.joint(f"{p}{jn}").qposadr[0]] = v
            d.qvel[m.joint(f"{p}{jn}").dofadr[0]] = 0.0
            d.ctrl[m.actuator(f"{p}{jn}").id] = v
        d.qpos[m.joint(f"{p}gripper").qposadr[0]] = q12[6 * i + 5]
        d.ctrl[env._gripper_act[p]] = -0.1
        env._gripper_closed[p] = True
        if hasattr(env, '_grip_target'):
            env._grip_target[p] = None
    mujoco.mj_forward(m, d)


def grasp_on(env, prefix):
    if hasattr(env, "grasp_active"):
        return env.grasp_active(prefix)
    return bool(env.data.eq_active[env._weld_id[prefix]])     # Roy's sim: one weld per arm


def state12(env):
    m, d = env.model, env.data
    out = []
    for p in PREFIXES:
        out += [float(d.qpos[a]) for a in env._arm_qpos_adr[p]]
        out.append(float(d.qpos[m.joint(f"{p}gripper").qposadr[0]]))
    return np.asarray(out, dtype=np.float32)


def teacher_to_env_action(env, target12):
    """Absolute 12-dim joint target (rad) -> 14-dim joint_delta action. Returns (action, clipped_fraction)."""
    a = np.zeros(14, dtype=np.float32)
    n_clip = 0
    for i, p in enumerate(PREFIXES):
        base, tgt = 7 * i, np.asarray(target12[6 * i:6 * i + 6], dtype=np.float64)
        for k in range(5):
            q = env.data.qpos[env._arm_qpos_adr[p][k]]
            raw = (tgt[k] - q) / JOINT_DELTA_SCALE
            a[base + k] = np.clip(raw, -1.0, 1.0)
            n_clip += abs(raw) > 1.0
        a[base + 6] = 1.0 if tgt[5] > GRIP_THR else -1.0
        env._grip_target[p] = float(tgt[5])
    return a, n_clip / 10.0


# ---- teacher policies ------------------------------------------------------
def _enc(arr):
    arr = np.ascontiguousarray(arr)
    return {"base64": base64.b64encode(arr.tobytes()).decode(), "shape": list(arr.shape), "dtype": str(arr.dtype)}


class StubTeacher:
    """Local dry-run stand-in: drifts the arms slowly toward the cloth, closes grippers late. No server."""
    def __init__(self, E=5):
        self.E = E; self.calls = 0

    def __call__(self, imgs, state, garment_id, cfg, initial_actions):
        self.calls += 1
        t = np.tile(state, (self.E, 1)).astype(np.float32)
        t[:, 1] -= 0.02; t[:, 7] -= 0.02        # shoulder_lift down a little each call
        t[:, 5] = t[:, 11] = 1.0 if self.calls < 20 else 0.0
        return {"actions": t, "next_initial_actions": None, "garment_type_pred": 1}


class WsTeacher:
    """Synchronous websocket client for scripts/serve.py (JSON + base64 numpy)."""
    def __init__(self, uri="ws://localhost:8000", timeout=600):
        from websockets.sync.client import connect
        self.ws = connect(uri, max_size=100 * 1024 * 1024, open_timeout=60)
        self.timeout = timeout
        self.ws.send(json.dumps({"type": "ping"})); assert json.loads(self.ws.recv(timeout))["status"] == "ok"

    def __call__(self, imgs, state, garment_id, cfg, initial_actions):
        msg = {"type": "infer_chunk", "session_id": "wf", "garment_type_id": int(garment_id),
               "observation.images.top_rgb": _enc(imgs["top_rgb"]),
               "observation.images.left_rgb": _enc(imgs["left_rgb"]),
               "observation.images.right_rgb": _enc(imgs["right_rgb"]),
               "observation.state": [float(x) for x in state], "inference_config": cfg}
        if initial_actions is not None:
            msg["initial_actions"] = initial_actions
        self.ws.send(json.dumps(msg))
        r = json.loads(self.ws.recv(self.timeout))
        if "error" in r:
            raise RuntimeError(r["error"])
        r["actions"] = np.asarray(r["actions"], dtype=np.float32)
        return r

    def close(self):
        self.ws.close()


# ---- episode ----------------------------------------------------------------
def run_episode(env, cams, teacher, inference_cfg_all, seed=0, max_steps=600,
                garment_id=None, video_every=1, log=print, cloth_shift_y=0.0):
    """One closed-loop episode. Returns a dict of arrays (obs/action pairs keyed by env step) + summary.
    cloth_shift_y < 0 moves the cloth toward the arms (his garments sit ~5 cm from the gripper tips)."""
    opts = {"task": 0}                     # "fold" (Roy's half-fold goal is the same for every task id)
    if cloth_shift_y:
        opts["cloth_pose"] = (0.0, cloth_shift_y)
    env.reset(seed=seed, options=opts)
    set_teacher_init_pose(env)
    # warm-up call (like his eval worker): predict garment type from the first frame
    imgs = cams.render(); s = state12(env)
    if garment_id is None:
        r0 = teacher(imgs, s, 0, inference_cfg_all[GARMENT_TYPES[0]], None)
        garment_id = int(r0.get("garment_type_pred", 0))
    garment = GARMENT_TYPES[garment_id]
    cfg = dict(inference_cfg_all[garment]); cfg.pop("k_execute", None); cfg.pop("num_steps", None)
    log(f"episode seed={seed} garment={garment} cfg={cfg}")

    states, targets, env_actions, fold, clipf, call_idx = [], [], [], [], [], []
    grasp, flat_obs = [], []
    chunks, chunk_step, call_times = [], [], []
    frames = []
    initial = None
    step = 0
    info = {}
    while step < max_steps:
        imgs = cams.render(); s = state12(env)
        t0 = time.perf_counter()
        r = teacher(imgs, s, garment_id, cfg, initial)
        call_times.append(time.perf_counter() - t0)
        acts = np.asarray(r["actions"], dtype=np.float32)      # (E, 12)
        initial = r.get("next_initial_actions")
        chunks.append(acts); chunk_step.append(step)
        for a12 in acts:
            a14, cf = teacher_to_env_action(env, a12)
            o = env._get_obs()
            flat_obs.append(np.concatenate([o[k] for k in ("proprio", "cloth_state", "task")]).astype(np.float32))
            states.append(state12(env)); targets.append(a12); env_actions.append(a14); clipf.append(cf)
            call_idx.append(len(chunks) - 1)
            _, _, term, trunc, info = env.step(a14)
            fold.append(env._fold_score())
            grasp.append((grasp_on(env, "left_"), grasp_on(env, "right_")))
            if video_every and step % video_every == 0:
                frames.append(cams.render()["top_rgb"])
            step += 1
            if term or trunc or step >= max_steps:
                break
        if term or trunc:
            break
    out = {
        "seed": seed, "garment": garment, "garment_id": garment_id, "steps": step,
        "success": bool(info.get("success", False)), "fold_score_final": float(fold[-1]) if fold else 0.0,
        "fold_score_max": float(max(fold)) if fold else 0.0, "n_calls": len(chunks),
        "call_time_median_s": float(np.median(call_times)) if call_times else 0.0,
        "clip_fraction_mean": float(np.mean(clipf)) if clipf else 0.0,
        "left_grasp_frac": float(np.mean([g[0] for g in grasp])) if grasp else 0.0,
        "right_grasp_frac": float(np.mean([g[1] for g in grasp])) if grasp else 0.0,
        "term_reason": str(info.get("termination_reason")),
    }
    arrays = {
        "states": np.asarray(states, np.float32), "targets": np.asarray(targets, np.float32),
        "env_actions": np.asarray(env_actions, np.float32), "fold_score": np.asarray(fold, np.float32),
        "clip_fraction": np.asarray(clipf, np.float32), "call_idx": np.asarray(call_idx, np.int32),
        "chunks": np.asarray(chunks, np.float32), "chunk_step": np.asarray(chunk_step, np.int32),
        "call_time_s": np.asarray(call_times, np.float32),
        "grasp": np.asarray(grasp, np.bool_),
        "obs": np.asarray(flat_obs, np.float32),
    }
    return out, arrays, frames


def save_episode(out_dir, tag, summary, arrays, frames, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{tag}.npz"); tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, summary=json.dumps(summary), **arrays); os.replace(tmp, npz)
    mp4 = None
    if frames:
        import imageio
        mp4 = os.path.join(out_dir, f"{tag}.mp4"); tmpv = os.path.join(out_dir, f"{tag}.tmp.mp4")
        imageio.mimwrite(tmpv, frames, fps=fps, macro_block_size=1); os.replace(tmpv, mp4)
    return npz, mp4
