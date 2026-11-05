"""LeRobot SO-101 follower joint values  ->  SO-101 sim joint angles (radians).

Sim convention = TheRobotStudio `Simulation/SO101/so101_new_calib.{xml,urdf}`; MuJoCo Menagerie
`robotstudio_so101/so101.xml` has identical kinematics (checked: 0.0000 mm body-pose difference over
200 random configurations, check_sim_conventions.py [1]).

Evidence for every constant below is in NOTES_units.md (file:line quotes). Short version:

* LeRobot >= the 2025-06-05 hardware-API redesign (commit e23b41e7, all tagged releases v0.3.2..v0.6.1,
  feature names "shoulder_pan.pos", ...):
    - DEGREES mode   : v = (tick - (range_min+range_max)/2) * 360/4095          (not clamped)
    - RANGE_M100_100 : v = (clamp(tick) - range_min)/(range_max - range_min)*200 - 100   (clamped)
    - RANGE_0_100    : v = (clamp(tick) - range_min)/(range_max - range_min)*100         (gripper, always)
  Both arm modes put 0 at the MIDDLE of the calibrated range; M100 only differs by the per-joint
  scale span/200. drive_mode is always 0 for SO arms (no software sign flip).
  `use_degrees` default: False up to v0.4.3, True from v0.4.4 (commit 6600b60e, 2026-02-13).
  wrist_roll range: swept like other joints up to v0.4.2; forced to 0..4095 from v0.4.3
  (commit ccfd609e, 2026-01-08) so its zero is then the half-turn homing pose.
* LeRobot before the redesign (pyproject 0.1.0, feature names "main_shoulder_pan", ...):
  degrees = (tick*sign + homing_offset)/2048*180 with zero = "straight horizontal position with gripper
  upwards and closed". FK shows TRS old_calib -> new_calib is  new = old + [0, +90, -90, 0, +2.789, 0] deg
  (all signs +1). That the old LeRobot zero/signs equal TRS old_calib is UNVERIFIED.
* Signs LeRobot -> new_calib: +1 for all joints. Evidence: LeRobot's own kinematics feeds
  np.deg2rad(degrees) straight into so101_new_calib.urdf (src/lerobot/model/kinematics.py:85-90);
  TRS PR #106 "fixed direction" and renamed joints to LeRobot names; LeIsaac maps leader M100 values
  onto the USD limits with no flips; and the gravity-sign test on a real episode agrees for
  shoulder_lift / elbow_flex / wrist_flex in 89-100 % of quasi-static frames (check_sim_conventions.py [6]).
  shoulder_pan and wrist_roll signs are supported by the code evidence only (no gravity load).
* Offsets: default 0 for all joints. Known candidates (cannot be resolved without a real arm):
  elbow_flex 0 or -5 deg (TRS limits were [-100,+90] before PR #106 "shift joint(3) elbow flex by five
  degrees"); wrist_roll 0 or +2.789 deg (gripper is upright at sim +2.789 deg = sim range middle).
  A constant offset applied to both action and state only changes gravity/contact geometry, so it is a
  second-order effect on tracking RMSE; fit it in sysid if needed.
* Gripper: LeRobot 0 = range_min = closed, 100 = range_max = open (TRS README "Gripper Note"; LeRobot
  examples/isaac_teleop_to_so101/common.py "100 = open"). In the sim the jaws are closed at the LOWER
  limit (Menagerie jaw primitives: 1.7 mm gap at -10 deg, 14.8 mm at 0 deg). Default map is linear onto
  the full sim range [-10, 100] deg (APPROXIMATION: real open span per arm is unknown; one real arm
  measured 125.8 deg of calibrated span from quantization).
* Frame alignment (all versions): row t holds state read BEFORE action t is sent (new API) or ~ms after
  it was written (old API). Either way action t can only show up in state t+1 onward.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
ARM_JOINTS = JOINT_NAMES[:5]

TICKS_PER_REV = 4096                  # lerobot motors/feetech/tables.py:190  "sts3215": 4096
LEROBOT_MAX_RES = TICKS_PER_REV - 1   # motors_bus.py:876  max_res = resolution - 1
LEROBOT_DEG_PER_TICK = 360.0 / LEROBOT_MAX_RES   # DEGREES mode scale used by LeRobot (0.0879121 deg)
TRUE_DEG_PER_TICK = 360.0 / TICKS_PER_REV        # physical, if the encoder is 4096/rev (0.0878906 deg)
OLD_API_DEG_PER_TICK = 180.0 / 2048              # pre-redesign apply_calibration: /(resolution//2)*180

# Sim joint limits (deg), TRS so101_new_calib.xml == Menagerie so101.xml (rounded)
SIM_RANGE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),               # +-1.91986 rad
    "shoulder_lift": (-100.0, 100.0),              # +-1.74533 rad
    "elbow_flex": (-96.8296, 96.8296),             # +-1.69 rad ("5-degree calibration offset" comment)
    "wrist_flex": (-95.0, 95.0),                   # +-1.65806 rad
    "wrist_roll": (-157.2113, 162.7887),           # -2.74385 .. 2.84121 rad (middle = +2.789 deg)
    "gripper": (-10.0, 100.0),                     # -0.174533 .. 1.74533 rad
}
SIM_SPAN_DEG = {j: hi - lo for j, (lo, hi) in SIM_RANGE_DEG.items()}

# LeRobot -> sim sign and default offset (deg). See module docstring.
DEFAULT_SIGNS = {j: 1.0 for j in JOINT_NAMES}
DEFAULT_OFFSETS_DEG = {j: 0.0 for j in JOINT_NAMES}
OFFSET_CANDIDATES_DEG = {"elbow_flex": (0.0, -5.0), "wrist_roll": (0.0, 2.789)}

# FK-derived (derive_old_to_new_calib.py, residual 0.0000 mm): q_new = q_old + this (signs all +1)
OLD_CALIB_TO_NEW_CALIB_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": 90.0, "elbow_flex": -90.0,
    "wrist_flex": 0.0, "wrist_roll": 2.789, "gripper": 0.0,
}

SIM_GRIPPER_CLOSED_DEG = -10.0   # sim lower limit; Menagerie jaw primitives 1.7 mm apart here
SIM_GRIPPER_OPEN_DEG = 100.0     # sim upper limit


# ----------------------------------------------------------------------------------------------
# Exact re-implementation of LeRobot's per-motor normalisation (motors_bus.py:854-913, main 53feaef)
# ----------------------------------------------------------------------------------------------
def normalize_ticks(ticks, mode: str, range_min: int, range_max: int, drive_mode: int = 0, max_res: int = LEROBOT_MAX_RES):
    """Raw Present_Position ticks (after Homing_Offset) -> LeRobot value. mode: 'degrees'|'m100'|'0_100'."""
    t = np.asarray(ticks, dtype=np.float64)
    if range_max == range_min:
        raise ValueError("range_min == range_max")
    b = np.clip(t, range_min, range_max)
    if mode == "m100":
        n = (b - range_min) / (range_max - range_min) * 200 - 100
        return -n if drive_mode else n
    if mode == "0_100":
        n = (b - range_min) / (range_max - range_min) * 100
        return 100 - n if drive_mode else n
    if mode == "degrees":  # NOTE: not clamped, drive_mode ignored (as in LeRobot)
        mid = (range_min + range_max) / 2
        return (t - mid) * 360 / max_res
    raise ValueError(mode)


def unnormalize_values(values, mode: str, range_min: int, range_max: int, drive_mode: int = 0, max_res: int = LEROBOT_MAX_RES):
    """LeRobot value -> Goal_Position ticks exactly as LeRobot writes them (int() truncation, motors_bus.py:883-913).
    The servo additionally holds Min/Max_Position_Limit = range_min/range_max (feetech.py:272-273)."""
    v = np.asarray(values, dtype=np.float64)
    if mode == "m100":
        v = -v if drive_mode else v
        b = np.clip(v, -100.0, 100.0)
        return np.trunc((b + 100) / 200 * (range_max - range_min) + range_min).astype(np.int64)
    if mode == "0_100":
        v = 100 - v if drive_mode else v
        b = np.clip(v, 0.0, 100.0)
        return np.trunc(b / 100 * (range_max - range_min) + range_min).astype(np.int64)
    if mode == "degrees":
        mid = (range_min + range_max) / 2
        return np.trunc(v * max_res / 360 + mid).astype(np.int64)
    raise ValueError(mode)


# ----------------------------------------------------------------------------------------------
# Recover the per-joint scale from the data itself (works because Present_Position is an integer tick)
# ----------------------------------------------------------------------------------------------
def estimate_tick_step(x: np.ndarray, min_diff: float = 1e-3) -> tuple[float, int, float]:
    """Return (step, n_intervals_used, rel_rms_residual) of the lattice the values lie on.

    For observation.state: DEGREES -> 360/4095 = 0.087912; M100 -> 200/(range_max-range_min);
    0_100 -> 100/(range_max-range_min); old API degrees -> 180/2048 = 0.087891.
    Needs adjacent ticks to occur in the data (true for any moving joint). Not valid for `action`
    columns: those carry the LEADER's calibration (teleop) or policy outputs.
    """
    u = np.unique(np.asarray(x, dtype=np.float64).ravel())
    d = np.diff(u)
    d = d[d > min_diff]
    if d.size == 0:
        return float("nan"), 0, float("nan")
    s0 = d.min()
    small = d[d < 20.5 * s0]
    n = np.round(small / s0)
    step = small.sum() / n.sum()
    for _ in range(3):  # refine
        n = np.round(small / step)
        step = small.sum() / n.sum()
    resid = small - n * step
    return float(step), int(n.sum()), float(np.sqrt(np.mean(resid**2)) / step)


def infer_units(state: np.ndarray, joint_names: Sequence[str] = JOINT_NAMES, tol: float = 2e-3) -> dict:
    """Guess 'degrees' vs 'm100' for the arm joints of an observation.state array (T x 6) and return
    the per-joint physical span (true degrees) implied by the quantisation step.

    Degrees mode gives the same step 360/4095 on every arm joint. M100 gives 200/span_ticks, which
    differs per joint. Gripper is always 0_100 (step = 100/span_ticks).
    """
    state = np.asarray(state, dtype=np.float64)
    out = {"steps": {}, "span_ticks": {}, "spans_deg": {}, "resid": {}}
    deg_like = []
    for i, j in enumerate(joint_names):
        step, n, r = estimate_tick_step(state[:, i])
        out["steps"][j], out["resid"][j] = step, r
        if j == "gripper":
            out["span_ticks"][j] = 100.0 / step
        else:
            deg_like.append(abs(step / LEROBOT_DEG_PER_TICK - 1) < tol or abs(step / OLD_API_DEG_PER_TICK - 1) < tol)
            out["span_ticks"][j] = 200.0 / step
        out["spans_deg"][j] = out["span_ticks"][j] * TRUE_DEG_PER_TICK
    out["units"] = "degrees" if all(deg_like) else "m100"
    if out["units"] == "degrees":
        for j in joint_names:
            if j != "gripper":
                out["span_ticks"][j] = float("nan")  # span not recoverable in degrees mode
                out["spans_deg"][j] = float("nan")
    return out


# ----------------------------------------------------------------------------------------------
# Main conversion
# ----------------------------------------------------------------------------------------------
def _default_spans_deg(wrist_roll_full_turn: bool) -> dict[str, float]:
    sp = dict(SIM_SPAN_DEG)
    if wrist_roll_full_turn:
        sp["wrist_roll"] = LEROBOT_MAX_RES * TRUE_DEG_PER_TICK  # range 0..4095 forced (>= v0.4.3)
    return sp


def gripper_to_sim_deg(v, mode: str = "linear_sim", gripper_span_deg: float | None = None):
    """LeRobot gripper value (0 = closed = range_min, 100 = open = range_max) -> sim jaw angle (deg).

    mode 'linear_sim' (default): -10 + 1.10*v, i.e. calibrated range -> full sim range. APPROXIMATION:
                                 assumes the real calibrated sweep equals the sim's 110 deg.
    mode 'span'                : -10 + v/100*gripper_span_deg (span from infer_units); may exceed +100.
    mode 'identity'            : v deg (what LeRobot kinematics.py does implicitly; puts closed at 0 deg,
                                 which is ~15 mm open in the sim, so not recommended).
    """
    v = np.asarray(v, dtype=np.float64)
    if mode == "linear_sim":
        return SIM_GRIPPER_CLOSED_DEG + v / 100.0 * (SIM_GRIPPER_OPEN_DEG - SIM_GRIPPER_CLOSED_DEG)
    if mode == "span":
        if gripper_span_deg is None:
            raise ValueError("gripper_span_deg required for mode='span'")
        return SIM_GRIPPER_CLOSED_DEG + v / 100.0 * gripper_span_deg
    if mode == "identity":
        return v
    raise ValueError(mode)


def to_radians(
    values,
    units: str = "degrees",
    joint_names: Sequence[str] = JOINT_NAMES,
    spans_deg: dict[str, float] | None = None,
    *,
    gripper_mode: str = "linear_sim",
    gripper_span_deg: float | None = None,
    offsets_deg: dict[str, float] | None = None,
    signs: dict[str, float] | None = None,
    exact_tick_scale: bool = True,
    wrist_roll_full_turn: bool = True,
    clip_to_sim: bool = False,
) -> np.ndarray:
    """Convert LeRobot SO-101 follower values (action or observation.state) to sim radians.

    values      : array (..., len(joint_names)) in LeRobot units.
    units       : 'degrees'     -> use_degrees=True (default from lerobot v0.4.4)
                  'm100'        -> use_degrees=False (default up to v0.4.3); needs spans_deg
                  'old_degrees' -> pre-2025-06-05 API ("main_*" feature names); adds OLD_CALIB_TO_NEW_CALIB_DEG
    spans_deg   : m100 only. Per-joint physical span (range_max-range_min)*360/4096 of the FOLLOWER
                  calibration. Get it from the calibration json, or from infer_units(observation.state).
                  Use the follower spans for the action columns too: the follower un-normalises the
                  action with its own calibration (motors_bus.py:896-899). If None, the sim joint spans are
                  used (LeIsaac-style assumption; wrist_roll = 359.9 deg if wrist_roll_full_turn).
    exact_tick_scale : degrees mode only; multiply by 4095/4096 to undo LeRobot's 360/4095 tick scale
                  (0.024 % correction, i.e. 0.024 deg at 100 deg).
    offsets_deg / signs : override DEFAULT_OFFSETS_DEG / DEFAULT_SIGNS (all 0 / +1).
    clip_to_sim : clip the result to the sim joint limits.
    Returns radians, same shape, in the so101_new_calib convention.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape[-1] != len(joint_names):
        raise ValueError(f"last dim {v.shape[-1]} != {len(joint_names)} joints")
    offs = dict(DEFAULT_OFFSETS_DEG)
    offs.update(offsets_deg or {})
    sg = dict(DEFAULT_SIGNS)
    sg.update(signs or {})
    spans = _default_spans_deg(wrist_roll_full_turn)
    if spans_deg:
        spans.update({k: float(x) for k, x in spans_deg.items() if x is not None and np.isfinite(x)})

    out = np.empty_like(v)
    for i, j in enumerate(joint_names):
        x = v[..., i]
        if j == "gripper":
            g = gripper_to_sim_deg(x, gripper_mode, gripper_span_deg)
            deg = sg[j] * g + offs[j]
        else:
            if units == "degrees":
                phys = x * (LEROBOT_MAX_RES / TICKS_PER_REV if exact_tick_scale else 1.0)
            elif units == "m100":
                phys = x / 200.0 * spans[j]
            elif units == "old_degrees":
                phys = x + OLD_CALIB_TO_NEW_CALIB_DEG[j]
            else:
                raise ValueError(units)
            deg = sg[j] * phys + offs[j]
        if clip_to_sim:
            lo, hi = SIM_RANGE_DEG[j]
            deg = np.clip(deg, lo, hi)
        out[..., i] = np.deg2rad(deg)
    return out


def from_radians(
    q,
    units: str = "degrees",
    joint_names: Sequence[str] = JOINT_NAMES,
    spans_deg: dict[str, float] | None = None,
    *,
    gripper_mode: str = "linear_sim",
    gripper_span_deg: float | None = None,
    offsets_deg: dict[str, float] | None = None,
    signs: dict[str, float] | None = None,
    exact_tick_scale: bool = True,
    wrist_roll_full_turn: bool = True,
) -> np.ndarray:
    """Inverse of to_radians (without clipping)."""
    q = np.rad2deg(np.asarray(q, dtype=np.float64))
    offs = dict(DEFAULT_OFFSETS_DEG)
    offs.update(offsets_deg or {})
    sg = dict(DEFAULT_SIGNS)
    sg.update(signs or {})
    spans = _default_spans_deg(wrist_roll_full_turn)
    if spans_deg:
        spans.update({k: float(x) for k, x in spans_deg.items() if x is not None and np.isfinite(x)})
    out = np.empty_like(q)
    for i, j in enumerate(joint_names):
        x = (q[..., i] - offs[j]) / sg[j]
        if j == "gripper":
            if gripper_mode == "linear_sim":
                out[..., i] = (x - SIM_GRIPPER_CLOSED_DEG) / (SIM_GRIPPER_OPEN_DEG - SIM_GRIPPER_CLOSED_DEG) * 100
            elif gripper_mode == "span":
                out[..., i] = (x - SIM_GRIPPER_CLOSED_DEG) / gripper_span_deg * 100
            else:
                out[..., i] = x
        elif units == "degrees":
            out[..., i] = x / (LEROBOT_MAX_RES / TICKS_PER_REV if exact_tick_scale else 1.0)
        elif units == "m100":
            out[..., i] = x / spans[j] * 200.0
        elif units == "old_degrees":
            out[..., i] = x - OLD_CALIB_TO_NEW_CALIB_DEG[j]
        else:
            raise ValueError(units)
    return out


# ----------------------------------------------------------------------------------------------
# Version -> units, and action/state frame alignment
# ----------------------------------------------------------------------------------------------
def units_from_metadata(feature_names: Iterable[str] | None = None, lerobot_version: str | None = None,
                        codebase_version: str | None = None) -> dict:
    """Best guess of units from dataset metadata. Always confirm with infer_units(observation.state).

    - names like 'main_shoulder_pan'  -> pre-redesign API ('old_degrees'; codebase v2.0/v2.1)
    - names like 'shoulder_pan.pos'   -> new API; 'm100' if lerobot < 0.4.4 else 'degrees' (default only;
      users could override use_degrees). codebase_version 'v2.1' -> lerobot 0.3.x (m100 default),
      'v3.0' -> lerobot >= 0.4.0 (m100 default up to 0.4.3, degrees from 0.4.4).
    """
    names = list(feature_names or [])
    if any(n.startswith(("main_", "left_", "right_")) and not n.endswith(".pos") for n in names):
        return {"units": "old_degrees", "api": "old", "confidence": "names"}
    guess = {"api": "new" if any(n.endswith(".pos") for n in names) else "unknown"}
    if lerobot_version:
        parts = tuple(int(p) for p in lerobot_version.lstrip("v").split(".")[:3])
        guess["units"] = "degrees" if parts >= (0, 4, 4) else "m100"
        guess["wrist_roll_full_turn"] = parts >= (0, 4, 3)
        guess["confidence"] = "version default"
    elif codebase_version == "v2.1":
        guess.update(units="m100", wrist_roll_full_turn=False, confidence="codebase v2.1 -> lerobot 0.3.x default")
    elif codebase_version == "v3.0":
        guess.update(units="ambiguous: m100 (<=0.4.3) or degrees (>=0.4.4)", confidence="check quantisation")
    else:
        guess.update(units="unknown", confidence="none")
    return guess


def frame_alignment(api: str = "new") -> dict:
    """What one dataset row means in time. See NOTES_units.md section 'Frame alignment'."""
    if api == "new":
        return {
            "row_t_state": "Present_Position read at the start of control tick t (lerobot_record.py:308)",
            "row_t_action": "teleop/policy action computed after that read, sent at ~same tick (lerobot_record.py:320-362)",
            "recorded_action": "pre-clip action_values (v0.4.0+, lerobot_record.py:367); post-clip sent_action in v0.3.x (record.py:274)",
            "earliest_effect": "state[t+1]",
            "pairing_for_sysid": "drive sim with action[t] held for one frame (zero-order hold) and compare to state[t+1]",
        }
    if api == "old":
        return {
            "row_t_state": "Present_Position read a few ms AFTER Goal_Position was written (manipulator.py:478-490 @b536f47e)",
            "row_t_action": "leader position written as goal in the same teleop_step (post max_relative_target clip)",
            "earliest_effect": "state[t+1] (state[t] can contain at most ~ms of response)",
            "pairing_for_sysid": "same as new API",
        }
    raise ValueError(api)


def align_action_state(action: np.ndarray, state: np.ndarray, lag_frames: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Pair action[t] with state[t+lag]. lag_frames=1 is the minimum causal lag; servo/bus delay is extra
    and should be identified (e.g. as a sim actuator delay), not assumed."""
    if lag_frames < 0:
        raise ValueError("lag_frames must be >= 0")
    if lag_frames == 0:
        return action, state
    return action[:-lag_frames], state[lag_frames:]


# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    # 1. zero = middle of calibrated range, both arm modes
    rmin, rmax = 812, 3205
    mid = (rmin + rmax) / 2
    check("DEGREES: value at mid tick == 0", abs(normalize_ticks(mid, "degrees", rmin, rmax)) < 1e-12)
    check("M100: value at mid tick == 0", abs(normalize_ticks(mid, "m100", rmin, rmax)) < 1e-12)
    check("M100: range ends -> -100/+100",
          np.allclose(normalize_ticks([rmin, rmax], "m100", rmin, rmax), [-100, 100]))
    check("DEGREES not clamped (tick beyond range_max gives >half span)",
          normalize_ticks(rmax + 100, "degrees", rmin, rmax) > (rmax - rmin) / 2 * LEROBOT_DEG_PER_TICK)

    # 2. M100 -> radians with the true span == degrees-mode -> radians (same ticks)
    ticks = rng.integers(rmin, rmax + 1, 1000)
    span_deg = (rmax - rmin) * TRUE_DEG_PER_TICK
    v_deg = normalize_ticks(ticks, "degrees", rmin, rmax)
    v_m100 = normalize_ticks(ticks, "m100", rmin, rmax)
    r_deg = to_radians(np.c_[v_deg, np.zeros((1000, 5))], "degrees")[:, 0]
    r_m100 = to_radians(np.c_[v_m100, np.zeros((1000, 5))], "m100", spans_deg={"shoulder_pan": span_deg})[:, 0]
    check("M100 with true span == DEGREES (max diff < 1e-9 rad)", np.abs(r_deg - r_m100).max() < 1e-9)
    check("radians == (tick-mid)*2pi/4096", np.allclose(r_deg, (ticks - mid) * 2 * np.pi / 4096))

    # 3. unnormalize round trip within one tick (LeRobot int() truncation)
    back = unnormalize_values(v_deg, "degrees", rmin, rmax)
    check("DEGREES round trip within 1 tick", np.abs(back - ticks).max() <= 1)

    # 4. quantisation recovers the span (float32 storage, like parquet)
    vals32 = v_m100.astype(np.float32)
    step, n, res = estimate_tick_step(vals32)
    check(f"step estimate -> span_ticks {200/step:.2f} vs true {rmax-rmin}", abs(200 / step - (rmax - rmin)) < 0.5)
    st = np.c_[normalize_ticks(ticks, "degrees", rmin, rmax)[:, None].repeat(5, 1),
               normalize_ticks(ticks, "0_100", rmin, rmax)].astype(np.float32)
    check("infer_units on synthetic DEGREES data -> 'degrees'", infer_units(st)["units"] == "degrees")
    st2 = np.c_[np.stack([normalize_ticks(ticks, "m100", rmin + 10 * k, rmax - 7 * k) for k in range(5)], 1),
                normalize_ticks(ticks, "0_100", rmin, rmax)].astype(np.float32)
    inf2 = infer_units(st2)
    check("infer_units on synthetic M100 data -> 'm100'", inf2["units"] == "m100")
    check("infer_units recovers M100 spans",
          all(abs(inf2["span_ticks"][j] - (rmax - 7 * k - rmin - 10 * k)) < 0.5 for k, j in enumerate(ARM_JOINTS)))

    # 5. gripper endpoints and inverse
    g = to_radians(np.array([[0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 100.0]]))[:, 5]
    check("gripper 0 -> -10 deg, 100 -> +100 deg (linear_sim)", np.allclose(np.rad2deg(g), [-10, 100]))
    x = rng.uniform(-90, 90, (50, 6))
    x[:, 5] = rng.uniform(0, 100, 50)
    for u in ("degrees", "m100", "old_degrees"):
        check(f"from_radians(to_radians(x)) == x  [{u}]", np.allclose(from_radians(to_radians(x, u), u), x))

    # 6. alignment helper
    a, s = align_action_state(np.arange(10)[:, None], np.arange(10)[:, None], 1)
    check("align_action_state pairs a[t] with s[t+1]", a[0, 0] == 0 and s[0, 0] == 1 and len(a) == 9)

    # 7. optional: real episode
    if len(sys.argv) > 1:
        import pandas as pd

        df = pd.read_parquet(sys.argv[1])
        S = np.stack(df["observation.state"].values)
        blocks = [(0, 6)] + ([(18, 24)] if S.shape[1] >= 24 else [])
        for b0, b1 in blocks:
            inf = infer_units(S[:, b0:b1])
            print(f"state[{b0}:{b1}] -> units={inf['units']}")
            for j in JOINT_NAMES:
                print(f"   {j:13s} step {inf['steps'][j]:.5f}  span {inf['span_ticks'][j]:7.1f} ticks = "
                      f"{inf['spans_deg'][j]:6.1f} deg (sim span {SIM_SPAN_DEG[j]:6.1f})  lattice rel-resid {inf['resid'][j]:.1e}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
