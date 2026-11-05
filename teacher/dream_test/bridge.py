"""Dream-1 on our two SO-101 arms in LeHome's Isaac scene: the unit conversions and the Dream client loop.

LeHome's state and action are 12 joint angles in radians, [left 5 arm joints, left gripper, right 5, right gripper].
Dream's SO-101 model (`so101_faas` in dream_ik) is the same joint chain as our so101_new_calib URDF (joint origins
identical, checked 2026-10-19), so the arm joints pass through as they are. The gripper is LeHome's -10..100 deg
(LeRobot's 0..100 range mapped linearly), Dream's grip is 0 closed .. 1 open.

The public Dream-1 models were trained on YAM arms, so on SO-101 the client sends no joint state (the model acts on the
three cameras and the prompt) and solves the hand poses it answers to our joints with Dream's own IK. Rows come at
20 Hz (48 = 2.4 s). The sim holds physics while Dream answers, so every row of a chunk is played (sync) and the next
request names the last row played as the commanded pose: the client then skips no rows for the round trip.
"""
import base64
import math
import time

import numpy as np

GRIP_LO = math.radians(-10.0)        # LeHome gripper joint at LeRobot 0 (closed)
GRIP_HI = math.radians(100.0)        # ... at LeRobot 100 (open)
ROW_HZ = 20.0                        # Dream's row rate (its YAM sim example's action_hz)
SIZE = (320, 176)                    # what Dream-1 reads, width x height
CHECKPOINT = "dreamlabs/dream-1-beta"

# the body as Dream's client reads it: our 12-wide state/action, the three cameras in Dream's roles
EMBODIMENT = {
    # the client adds each arm's tool pose and grip (Dream's FK of our joints), so they are declared parts too
    "state": {"left_arm": [0, 5], "left_gripper": [5, 6], "left_eef_xyz": [6, 9], "left_eef_rot6": [9, 15],
              "left_grip": [15, 16], "right_arm": [16, 21], "right_gripper": [21, 22], "right_eef_xyz": [22, 25],
              "right_eef_rot6": [25, 31], "right_grip": [31, 32]},
    "action": {"left_arm": [0, 5], "left_gripper": [5, 6], "right_arm": [6, 11], "right_gripper": [11, 12]},
    "video": {r: f"observation.images.{r}" for r in ("overhead", "wrist_l", "wrist_r")},
    "head_camera": "overhead",
}
# LeHome observation key -> Dream camera role
CAMERAS = {"observation.images.top_rgb": "overhead", "observation.images.left_rgb": "wrist_l",
           "observation.images.right_rgb": "wrist_r"}


try:                                  # our SO-101 kinematics (oracle/so101_kin.py), for the per-call diagnosis only
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.environ.get("ORACLE_DIR", _os.path.join(_os.path.dirname(__file__), "..", "oracle")))
    import so101_kin as _sk
    _KIN = _sk.SO101()
except Exception:
    _KIN = None


def rad_to_grip(q):
    return float(np.clip((q - GRIP_LO) / (GRIP_HI - GRIP_LO), 0.0, 1.0))


def grip_to_rad(g):
    return float(GRIP_LO + np.clip(g, 0.0, 1.0) * (GRIP_HI - GRIP_LO))


def decode_image(entry):
    """RemotePolicy.serialize_observation's image entry {base64, shape, dtype} -> uint8 HxWx3."""
    a = np.frombuffer(base64.b64decode(entry["base64"]), dtype=np.dtype(entry["dtype"])).reshape(entry["shape"])
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return a[..., :3]


def to_dream_frame(rgb):
    """640x480 render -> rows 60:420 (640x360, the YAM station's crop) -> 320x176 RGB, as Dream's frames are made."""
    import cv2
    h = rgb.shape[0]
    if h * 16 > rgb.shape[1] * 9 + 8:                        # taller than 16:9: centre crop to 16:9
        keep = int(round(rgb.shape[1] * 9 / 16))
        top = (h - keep) // 2
        rgb = rgb[top:top + keep]
    return cv2.resize(np.ascontiguousarray(rgb), SIZE, interpolation=cv2.INTER_AREA)


def jpeg_b64(rgb):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def dream_state(state12, prompt):
    s = np.asarray(state12, dtype=np.float64).ravel()
    return {
        "state.left_arm": s[0:5].astype(np.float32), "state.left_grip": np.array([rad_to_grip(s[5])], np.float32),
        "state.right_arm": s[6:11].astype(np.float32), "state.right_grip": np.array([rad_to_grip(s[11])], np.float32),
        "prompt": prompt,
    }


def rows_to_actions(q_l, q_r):
    """Dream's q_l_chunk / q_r_chunk ([H, 5 joints + grip]) -> [H, 12] LeHome joint targets."""
    q_l, q_r = np.asarray(q_l, np.float64), np.asarray(q_r, np.float64)
    n = min(len(q_l), len(q_r))
    out = np.zeros((n, 12))
    out[:, 0:5], out[:, 6:11] = q_l[:n, :5], q_r[:n, :5]
    out[:, 5] = [grip_to_rad(g) for g in q_l[:n, 5]]
    out[:, 11] = [grip_to_rad(g) for g in q_r[:n, 5]]
    return out


class RowClock:
    """Physics steps per Dream row so the arms move in real time: row k ends at (k+1)/ROW_HZ seconds of physics."""

    def __init__(self, dt):
        self.dt, self.k = float(dt), 0

    def next(self):
        per = 1.0 / (ROW_HZ * self.dt)
        n = int(round((self.k + 1) * per)) - int(round(self.k * per))
        self.k += 1
        return max(1, n)


def make_config(checkpoint=CHECKPOINT):
    return {"checkpoint": checkpoint, "timeout_ms": 20000, "action_hz": ROW_HZ, "strict": True,
            "ik": {"arm": "so101_faas", "head": "eef"}, "embodiment": EMBODIMENT}


class DreamBridge:
    """One sim's Dream session: begin(prompt) per episode, observe() every row, next_action() for the row to play."""

    def __init__(self, client, log=print):
        self.client, self.log = client, log
        self.queue, self.last, self.prompt, self.calls = [], None, None, []

    def begin(self, prompt):
        self.client.reset_episode()
        self.queue, self.last, self.prompt, self.calls = [], None, prompt, []

    def observe(self, images, state12, t):
        """images: {LeHome key: uint8 HxWx3}; returns the 320x176 frames (for the episode video)."""
        small = {CAMERAS[k]: to_dream_frame(v) for k, v in images.items() if k in CAMERAS}
        frames = {f"video.{r}": jpeg_b64(small[r]) for r in ("overhead", "wrist_l", "wrist_r")}
        self.client.set_observation(frames, dream_state(state12, self.prompt), timestamp=float(t))
        return small

    def next_action(self):
        """The next [12] LeHome joint target; asks Dream for a new chunk once the last one has been played whole."""
        if not self.queue:
            self._fetch()
        action, self.last = self.queue.pop(0)          # last: the raw (left row, right row) as Dream's client wrote it
        return action

    def _fetch(self):
        import dream
        commanded = None
        if self.last is not None:
            l, r = self.last
            commanded = {"q_l_chunk": l, "q_r_chunk": r}
        for attempt in range(60):
            t0 = time.perf_counter()
            try:
                act = self.client.get_action(commanded=commanded)
                break
            except dream.DreamSessionExpired:        # (a DreamServerError subclass: must come first)
                # the sim plays a 2.4 s chunk in ~13 s of wall time, longer than the gateway keeps an idle session;
                # each chunk is computed from the frames sent with it, so ask again on the new session (already adopted)
                self.sessions_replaced = getattr(self, "sessions_replaced", 0) + 1
                if attempt >= 3:
                    raise
            except dream.DreamServerError as e:      # rate_limited / pool_warming: retryable, wait as told
                if not e.retryable or attempt == 59:
                    raise
                wait = e.retry_after_s or 1.0
                self.log(f"[dream] {e.reason}: retry in {wait:.1f} s")
                time.sleep(wait)
            except (dream.DreamTimeout, dream.DreamProtocolError) as e:
                # the gateway drops an idle keep-alive connection while the sim plays a chunk (SSL EOF, 2026-10-20);
                # the request never reached the model, so sending it again is safe
                if attempt >= 3:
                    raise
                self.log(f"[dream] {type(e).__name__}, retrying on a fresh connection")
                time.sleep(0.5)
        timing = dict(self.client.last_timing or {})
        self.calls.append({"rtt_ms": round((time.perf_counter() - t0) * 1000, 1),
                           "serve_infer_ms": timing.get("serve_infer_ms"), "network_ms": timing.get("network_ms")})
        q_l, q_r = np.asarray(act["q_l_chunk"], np.float64), np.asarray(act["q_r_chunk"], np.float64)
        try:      # diagnosis: how far Dream's hand targets travel vs how far the solved joints move the hand
            per = act.get("per_key") or {}
            for side, q in (("left", q_l), ("right", q_r)):
                x = np.asarray(per.get(f"action.{side}_eef_xyz"), np.float64)
                d = {"tgt_first": np.round(x[0], 3).tolist(), "tgt_path_cm": round(float(
                    np.linalg.norm(np.diff(x, axis=0), axis=1).sum() * 100), 1),
                    "tgt_net_cm": round(float(np.linalg.norm(x[-1] - x[0]) * 100), 1),
                    "q_net_rad": round(float(np.abs(q[-1, :5] - q[0, :5]).max()), 3),
                    "grip": [round(float(q[0, 5]), 2), round(float(q[-1, 5]), 2)]}
                if _KIN is not None:
                    d["hand_net_cm"] = round(float(np.linalg.norm(_KIN.fk_base(q[-1, :5])[:3, 3]
                                                                  - _KIN.fk_base(q[0, :5])[:3, 3]) * 100), 1)
                self.calls[-1][side] = d
        except Exception as e:
            self.calls[-1]["diag_error"] = repr(e)[:120]
        acts = rows_to_actions(q_l, q_r)
        self.queue = [(acts[i], (q_l[i], q_r[i])) for i in range(len(acts))]
