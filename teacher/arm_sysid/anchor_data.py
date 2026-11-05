"""NVIDIA Anchor-Lab SO-101 test motions (HF nvidia/Anchor-Lab, data/so101_arm_50motion, Apache-2.0) -> per-motion
arrays in the MuJoCo joint convention, plus the FIT / VAL / HELD-OUT split and the seeded window list.

Source files: the 50 parquet files already downloaded at HF revision 647edd57 into the session scratchpad; they are
copied to data/anchor_lab/raw/ and checked against the sha256 that huggingface_hub stored next to the download.

Facts used (planning analysis 2026-10-15, re-checked by `python anchor_data.py check`):
  * long format: time_ns, time_utc, experiment, field ("<joint>/<feature>"), value
  * two clocks: all 6 command_q share one (17-50 Hz, jittery), all state fields share another (~475 Hz median,
    ~1 % gaps of 28-31 ms)
  * arm actual_q / command_q in rad on a 2*pi/4096 grid; jaw in 0..1 on a 1/1446 grid -> 127.09 deg span.
    The jaw zero is not observable (no gravity/contact on it); theta = -10 deg + 127.09 deg * x (CAD lower limit).
  * dq (multiples of pi/4) and tau_abs (scale unrelated to load) are not used.
Joint map (dataset -> model): rotation->shoulder_pan, pitch->shoulder_lift, elbow->elbow_flex,
wrist_pitch->wrist_flex, wrist_roll->wrist_roll, jaw->gripper.

    python anchor_data.py build     # copy + verify + cache npz + split/windows json
    python anchor_data.py check     # unit / timing / sign checks -> results/anchor_data_checks.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = (r"C:/Users/valen/AppData/Local/Temp/claude/C--Users-valen-Documents-vla-distill/"
       r"c1e14d5c-c49d-4f52-b957-d99b0501209c/scratchpad/gripper_nv/dl/anchor")
SRC_DATA = os.path.join(SRC, "data", "so101_arm_50motion")
SRC_META = os.path.join(SRC, ".cache", "huggingface", "download", "data", "so101_arm_50motion")
HF_REVISION = "647edd5787cd764cdc041103ad282dc59214d919"
ROOT = os.path.join(HERE, "data", "anchor_lab")
RAW = os.path.join(ROOT, "raw")
CACHE = os.path.join(ROOT, "cache")
SPLIT_JSON = os.path.join(ROOT, "split_anchor.json")
RESULTS = os.path.join(HERE, "results")
PREFIX = "so101-sysid-50motion-"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DS_JOINTS = ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw"]
ARM = 5
TICK = 2 * math.pi / 4096
JAW_TICKS = 1446
JAW_SPAN_DEG = 360.0 * JAW_TICKS / 4096          # 127.09 deg
JAW_ZERO_DEG = -10.0                              # x = 0 placed at the CAD lower limit (assumption)

VAL = ["oscillation-med-freq", "step-response", "reaching-points", "single-joint-wrist-roll", "velocity-ramp",
       "lissajous-pattern"]
LONG_DYNAMIC = ["prbs-excitation", "chirp-sweep", "static-holding", "repetitive-motion"]
WIN_SCORED = 6.0
WIN_WARM = 0.5
SEED = 20261015


def jaw_to_rad(x):
    return np.radians(JAW_ZERO_DEG + JAW_SPAN_DEG * np.asarray(x, float))


def rad_to_jaw(theta):
    return (np.degrees(np.asarray(theta, float)) - JAW_ZERO_DEG) / JAW_SPAN_DEG


def motion_names(split=None):
    names = sorted(os.path.basename(p)[len(PREFIX):-len(".parquet")]
                   for p in glob.glob(os.path.join(RAW, PREFIX + "*.parquet")))
    if split is None:
        return names
    return [n for n in names if n.startswith(split + "-")]


def short(name):
    return name.split("-", 1)[1]


# ------------------------------------------------------------------------------------------------ copy + verify
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_and_verify():
    os.makedirs(RAW, exist_ok=True)
    rows = []
    for src in sorted(glob.glob(os.path.join(SRC_DATA, PREFIX + "*.parquet"))):
        base = os.path.basename(src)
        meta = os.path.join(SRC_META, base + ".metadata")
        with open(meta, encoding="utf-8") as fh:
            lines = fh.read().split()
        rev, expect = lines[0], lines[1]
        dst = os.path.join(RAW, base)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst)
        got = _sha256(dst)
        rows.append({"file": base, "bytes": os.path.getsize(dst), "sha256": got, "ok": got == expect,
                     "revision": rev})
    bad = [r["file"] for r in rows if not r["ok"] or r["revision"] != HF_REVISION]
    if len(rows) != 50 or bad:
        raise RuntimeError(f"{len(rows)} files, bad: {bad}")
    with open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"hf_dataset": "nvidia/Anchor-Lab", "revision": HF_REVISION, "files": rows}, fh, indent=1)
    print(f"copied + verified {len(rows)} files ({sum(r['bytes'] for r in rows) / 1e6:.1f} MB)")


# ------------------------------------------------------------------------------------------------ parse
def parse_motion(name):
    """-> dict of arrays. Times in s relative to t0 = first sample of either stream (int64 arithmetic first)."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, PREFIX + name + ".parquet"), columns=["time_ns", "field", "value"])
    piv = {}
    for fld, g in df.groupby("field", sort=False):
        piv[str(fld)] = (g["time_ns"].to_numpy(np.int64), g["value"].to_numpy(np.float64))
    ts = piv["rotation/actual_q"][0]
    tc = piv["rotation/command_q"][0]
    for j in DS_JOINTS:
        for feat in ("actual_q", "present_load_raw"):
            assert np.array_equal(piv[f"{j}/{feat}"][0], ts), (name, j, feat)
        assert np.array_equal(piv[f"{j}/command_q"][0], tc), (name, j)

    def on_state_clock(fld):
        # diagnostics only; some diagnostic fields (e.g. jaw/temperature) have fewer samples than the state clock
        t, v = piv[fld]
        if np.array_equal(t, ts):
            return v
        k = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(t) - 1)
        return v[k]

    t0 = min(int(ts[0]), int(tc[0]))
    q = np.stack([piv[f"{j}/actual_q"][1] for j in DS_JOINTS], 1)
    qc = np.stack([piv[f"{j}/command_q"][1] for j in DS_JOINTS], 1)
    out = {
        "t_state": (ts - t0).astype(np.float64) * 1e-9,
        "t_cmd": (tc - t0).astype(np.float64) * 1e-9,
        "q_raw": q, "qc_raw": qc,
        "load": np.stack([piv[f"{j}/present_load_raw"][1] for j in DS_JOINTS], 1),
        "temperature": np.stack([on_state_clock(f"{j}/temperature") for j in DS_JOINTS], 1),
        "voltage": np.stack([on_state_clock(f"{j}/voltage") for j in DS_JOINTS], 1),
        "is_moving": np.stack([on_state_clock(f"{j}/is_moving") for j in DS_JOINTS], 1),
        "t0_ns": np.int64(t0),
    }
    q = q.copy()
    qc = qc.copy()
    q[:, 5] = jaw_to_rad(q[:, 5])
    qc[:, 5] = jaw_to_rad(qc[:, 5])
    out["q"] = q          # model convention, rad (jaw converted)
    out["qc"] = qc
    return out


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    for n in motion_names():
        p = os.path.join(CACHE, n + ".npz")
        if os.path.exists(p):
            continue
        np.savez_compressed(p, **parse_motion(n))
        print("cached", n)


_MEM = {}


def load(name):
    if name not in _MEM:
        with np.load(os.path.join(CACHE, name + ".npz")) as z:
            _MEM[name] = {k: z[k] for k in z.files}
    return _MEM[name]


# ------------------------------------------------------------------------------------------------ split + windows
def build_split():
    names = motion_names()
    held = [n for n in names if n.startswith("heldout-")]
    train = [n for n in names if n.startswith("train-")]
    val = ["train-" + v for v in VAL]
    assert all(v in train for v in val), val
    fit = [n for n in train if n not in val]
    rng = np.random.default_rng(SEED)
    windows = {}
    L = WIN_WARM + WIN_SCORED
    info = {}
    for n in fit + val:
        d = load(n)
        t_lo = max(d["t_state"][0], d["t_cmd"][0])
        t_hi = d["t_state"][-1]
        D = t_hi - t_lo
        parts = 6 if short(n) in LONG_DYNAMIC else 4
        k = int(min(parts, math.floor(D / L)))
        P = D / k
        ws = []
        for i in range(k):
            a = t_lo + i * P
            ws.append(round(float(a + rng.uniform(0.0, P - L)), 4))
        windows[n] = ws
        info[n] = {"duration_s": round(float(D), 2), "windows": k}
    jaw_active = []
    for n in fit + val:
        x = rad_to_jaw(load(n)["qc"][:, 5])
        if np.ptp(x) > 0.05:
            jaw_active.append(n)
    out = {"seed": SEED, "warm_s": WIN_WARM, "scored_s": WIN_SCORED, "fit": fit, "val": val, "heldout": held,
           "windows": windows, "jaw_active": jaw_active, "durations": info,
           "durations_heldout": {n: round(float(load(n)["t_state"][-1] - load(n)["t_state"][0]), 2) for n in held}}
    with open(SPLIT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    nw = sum(len(v) for k, v in windows.items() if k in fit)
    print(f"split: fit {len(fit)} motions / {nw} windows, val {len(val)}, heldout {len(held)}; "
          f"jaw-active {len(jaw_active)}")
    return out


def split():
    with open(SPLIT_JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------ checks
def run_checks():
    import mujoco
    sys.path.insert(0, HERE)
    import models
    names = motion_names()
    res = {"grid": {}, "timing": {}, "gaps": {}, "voltage": {}, "temperature_median": {}}
    worst_arm, worst_jaw = 0.0, 0.0
    # measured arm angles sit on the 2*pi/4096 grid with a fixed per-joint phase (0 for pan/lift, half a tick for
    # elbow/wrist_flex/wrist_roll); commands are continuous (generator output), so only actual_q is checked
    phase = np.array([0.0, 0.0, 0.5, 0.5, 0.5])
    for n in names:
        d = load(n)
        r = d["q_raw"][:, :ARM] / TICK - phase
        worst_arm = max(worst_arm, float(np.max(np.abs(r - np.round(r)))))
        r = d["q_raw"][:, 5] * JAW_TICKS
        worst_jaw = max(worst_jaw, float(np.max(np.abs(r - np.round(r)))))
        dt = np.diff(d["t_state"])
        dc = np.diff(d["t_cmd"])
        res["timing"][n] = {"state_median_ms": round(1e3 * float(np.median(dt)), 3),
                            "state_mean_hz": round(1.0 / float(np.mean(dt)), 1),
                            "cmd_median_ms": round(1e3 * float(np.median(dc)), 2),
                            "cmd_mean_hz": round(1.0 / float(np.mean(dc)), 1),
                            "monotonic": bool(np.all(dt > 0) and np.all(dc > 0)),
                            "duration_s": round(float(d["t_state"][-1] - d["t_state"][0]), 2)}
        res["gaps"][n] = {"n_gt_10ms": int((dt > 0.010).sum()), "max_ms": round(1e3 * float(dt.max()), 1),
                          "p99_ms": round(1e3 * float(np.percentile(dt, 99)), 2)}
        res["voltage"][n] = [round(float(np.median(d["voltage"][:, j])), 2) for j in range(6)]
        res["temperature_median"][n] = [float(np.median(d["temperature"][:, j])) for j in range(6)]
    res["grid"] = {"arm_max_residual_ticks": worst_arm, "jaw_max_residual_1446": worst_jaw}

    # gravity-sign test: at loaded holds, cmd - q must have the sign of the gravity torque the model predicts
    m = mujoco.MjModel.from_xml_path(models.MENAGERIE_XML)
    dd = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    vadr = [m.jnt_dofadr[m.joint(j).id] for j in JOINTS]
    agree = {j: [0, 0] for j in (1, 2, 3)}
    for n in names:
        d = load(n)
        tc, qc = d["t_cmd"], d["qc"]
        k = np.clip(np.searchsorted(tc, d["t_state"], side="right") - 1, 0, len(tc) - 1)
        err = qc[k] - d["q"]
        moving = np.abs(np.gradient(d["q"], d["t_state"], axis=0)) > np.radians(2.0)
        sel = np.where(~moving.any(1))[0][::25]
        for i in sel:
            dd.qpos[qadr] = d["q"][i]
            dd.qvel[:] = 0
            mujoco.mj_forward(m, dd)
            g = dd.qfrc_bias[vadr]
            for j in (1, 2, 3):
                if abs(err[i, j]) > 3 * TICK and abs(g[j]) > 0.05 and abs(d["load"][i, j]) > 100:
                    agree[j][1] += 1
                    agree[j][0] += int(np.sign(err[i, j]) == np.sign(g[j]))
    res["gravity_sign"] = {JOINTS[j]: {"agree": a, "n": b, "frac": round(a / b, 3) if b else None}
                           for j, (a, b) in agree.items()}

    # step-onset latency on PRBS: command step -> first measured move in the step direction (> 3 ticks)
    d = load("train-prbs-excitation")
    lat = []
    for j in range(ARM):
        c = d["qc"][:, j]
        steps = np.where(np.abs(np.diff(c)) > np.radians(1.0))[0] + 1
        for s in steps:
            tcmd = d["t_cmd"][s]
            i0 = np.searchsorted(d["t_state"], tcmd)
            if i0 < 2 or i0 + 200 >= len(d["t_state"]):
                continue
            base = d["q"][i0 - 1, j]
            sgn = np.sign(c[s] - c[s - 1])
            mv = np.where(sgn * (d["q"][i0:i0 + 200, j] - base) > 3 * TICK)[0]
            if len(mv):
                lat.append(d["t_state"][i0 + mv[0]] - tcmd)
    lat = np.array(lat) * 1e3
    res["prbs_step_onset_ms"] = {"n": int(len(lat)), "median": round(float(np.median(lat)), 1),
                                 "iqr": [round(float(x), 1) for x in np.percentile(lat, [25, 75])]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "anchor_data_checks.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    tot = sum(v["duration_s"] for v in res["timing"].values())
    print(f"grid residual arm {worst_arm:.2e} ticks, jaw {worst_jaw:.2e}; total {tot / 60:.1f} min")
    print("gravity sign:", res["gravity_sign"])
    print("prbs onset ms:", res["prbs_step_onset_ms"])
    print("all monotonic:", all(v["monotonic"] for v in res["timing"].values()))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        copy_and_verify()
        build_cache()
        build_split()
    elif cmd == "check":
        run_checks()
    else:
        raise SystemExit(__doc__)
