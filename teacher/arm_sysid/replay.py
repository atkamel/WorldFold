"""Open-loop windowed replay of REAL SO-101 LeRobot recordings through the models.py variants.

First honest real-vs-sim number: per-joint RMS error (deg) between the real measured joint position
(observation.state) and the simulated joint position when the sim is driven open-loop by the recorded
commands (action), for each model variant, plus no-physics baselines.

Method (all choices are written into results/baseline.json["method"] too)
--------------------------------------------------------------------------
Data      data/manifest.json. Six single-arm datasets whose arm joints are in LeRobot DEGREES mode
          (verified by encoder-tick quantisation) are the PRIMARY set; their mean is the headline.
          NeuroTo bimanual ep0 (RANGE_M100_100 mode) is SUPPLEMENTARY: its per-joint follower span is
          recovered exactly from the state's tick lattice (lerobot_units.infer_units) and the same
          follower span converts the action (stock LeRobot unnormalisation; UNVERIFIED for that custom
          recorder). It has one episode, so it is split by time blocks, and it is not in the headline mean.
          Chaenn episodes 87-106 are excluded (10-45 % of their action frames are off the encoder grid,
          i.e. probably policy/intervention rollouts, not leader-arm readings).
Units     lerobot_units.to_radians with defaults: signs +1, offsets 0, gripper 'linear_sim'
          (0..100 -> -10..+100 deg, an APPROXIMATION). Errors are reported in sim-convention degrees.
Alignment Row t = (state read BEFORE action t is sent, action t)  [NOTES_units.md section 7].
          So action[t] is the command applied during (t, t+1] and the first state it can affect is
          state[t+1]. The sim is driven with action[t] (zero-order hold for 1/30 s) and its qpos at the
          end of that interval is compared with state[t+1].
Split     episode_index % 4 == 3 -> TEST, others TRAIN (NeuroTo: 4 contiguous time blocks, block 3 = TEST).
          Saved to split.json. All reported numbers are TEST only. TRAIN is used only to pick the
          integer delay of the 'best_pure_delay' baseline.
Windows   Non-overlapping windows of W s (W = 4 and 1), tiled from frame 0 of every TEST episode;
          a trailing remainder shorter than W is dropped. Per window: reset the sim to the real state
          at t0 (velocity = backward finite difference (s[t0]-s[t0-1])*30, 0 at t0 = 0), feed
          action[t0 .. t0+N-1] open loop, compare with state[t0+1 .. t0+N].
Limits    'sim'  = models exactly as built by models.py (joint limits and ctrlrange as in Menagerie).
                   The reset state is clipped into the sim joint range.
          'wide' = same, but the 5 arm joints' range limits and the position actuators' ctrlrange are
                   switched off at runtime (mjModel.jnt_limited / actuator_ctrllimited = 0). Real arms
                   sit beyond the sim range in 17-44 % of frames (mostly shoulder_lift at the folded rest
                   pose and wrist_flex), which is a calibration/range mismatch rather than actuator
                   dynamics; 'wide' removes it. Nothing else changes.
Baselines (no physics) perfect_tracking: state[t] := action[t-1].
          best_pure_delay: state[t] := action[t-d], integer d in 0..15 chosen per dataset on TRAIN
          (minimises the mean over arm joints of pooled RMSE). Uses the recorded action history, so it
          does not depend on W.
          hold_start: state[t] := state[t0] (how much the arm moves within a window).
Metrics   per dataset x model x joint: pooled RMSE over all TEST window frames, and the median over
          windows of the per-window RMSE. Arm mean = mean over the 5 arm joints (gripper reported
          separately). Headline = mean over the 6 primary datasets (equal weight per dataset).
Subsets   all windows; no_grasp (no frame where gripper state - gripper action > 5 units for >= 10
          consecutive frames, i.e. jaw stalled on an object); in_range (real state never > 1 deg outside
          the sim joint range); moving (commanded excursion > 5 deg on some arm joint). No table-contact
          detection is done (table height per dataset is unknown).

Run:  python replay.py               (full run, writes split.json and results/)
      python replay.py --quick       (few windows per dataset, smoke test)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lerobot_units as lu  # noqa: E402

MANIFEST = os.path.join(HERE, "data", "manifest.json")
RESULTS = os.path.join(HERE, "results")
SPLIT_JSON = os.path.join(HERE, "split.json")

FPS = 30
JOINTS = list(lu.JOINT_NAMES)
ARM = 5
SIM_MODELS = ("A_stock", "B_nvidia", "C_bam_m1", "D_bam_m6")
BASELINES = ("perfect_tracking", "best_pure_delay", "hold_start")
ALL_MODELS = SIM_MODELS + BASELINES
TEST_MOD, TEST_REM = 4, 3
CHAENN_MAX_EP = 86
GRASP_THRESH = 5.0        # gripper units (0..100)
GRASP_MIN_FRAMES = 10     # 0.33 s
OOR_MARGIN_DEG = 1.0
MOVING_DEG = 5.0
MAX_DELAY = 15
SUBSETS = ("all", "no_grasp", "in_range", "moving")


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def _grasp_mask(g_state, g_action):
    """Jaw stalled on an object: state - action > GRASP_THRESH for >= GRASP_MIN_FRAMES consecutive frames."""
    raw = (np.asarray(g_state) - np.asarray(g_action)) > GRASP_THRESH
    out = np.zeros_like(raw)
    i, n = 0, len(raw)
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            if j - i >= GRASP_MIN_FRAMES:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _episode(A_raw, S_raw, units, spans=None):
    A = lu.to_radians(A_raw, units, spans_deg=spans)
    S = lu.to_radians(S_raw, units, spans_deg=spans)
    return {"A": A, "S": S, "grasp": _grasp_mask(S_raw[:, 5], A_raw[:, 5]), "T": len(S)}


def load_datasets(manifest_path=MANIFEST):
    man = json.load(open(manifest_path))
    dsets = []
    for e in man:
        if e["repo_id"].startswith("NeuroTo"):
            dsets += _load_neuroto(e)
            continue
        df = pd.concat([pd.read_parquet(os.path.join(e["local_dir"], f)) for f in e["data_files"]])
        df = df.sort_values(["episode_index", "frame_index"])
        S_all = np.stack(df["observation.state"].values).astype(np.float64)
        inf = lu.infer_units(S_all)
        if inf["units"] != "degrees":
            raise RuntimeError(f"{e['repo_id']}: expected degrees, infer_units says {inf['units']}")
        name = e["repo_id"].split("/")[0]
        excluded = []
        eps = {}
        for ep, g in df.groupby("episode_index"):
            ep = int(ep)
            if name == "Chaenn" and ep > CHAENN_MAX_EP:
                excluded.append(ep)
                continue
            A_raw = np.stack(g["action"].values).astype(np.float64)
            S_raw = np.stack(g["observation.state"].values).astype(np.float64)
            eps[ep] = _episode(A_raw, S_raw, "degrees")
        test = sorted(k for k in eps if k % TEST_MOD == TEST_REM)
        train = sorted(k for k in eps if k % TEST_MOD != TEST_REM)
        dsets.append({
            "name": name, "repo_id": e["repo_id"], "revision": e["revision"], "units": "degrees",
            "primary": True, "episodes": eps, "test": test, "train": train, "excluded": excluded,
            "split_unit": "episode", "split_rule": f"episode_index % {TEST_MOD} == {TEST_REM} -> TEST",
        })
    return dsets


def _load_neuroto(e):
    df = pd.read_parquet(os.path.join(e["local_dir"], e["data_files"][0])).sort_values("frame_index")
    S36 = np.stack(df["observation.state"].values).astype(np.float64)
    A24 = np.stack(df["action"].values).astype(np.float64)
    out = []
    for arm, s0, a0 in (("right", 0, 0), ("left", 18, 12)):
        S_raw, A_raw = S36[:, s0:s0 + 6], A24[:, a0:a0 + 6]
        inf = lu.infer_units(S_raw)
        if inf["units"] != "m100":
            raise RuntimeError(f"NeuroTo {arm}: expected m100, got {inf['units']}")
        spans = {j: inf["spans_deg"][j] for j in lu.ARM_JOINTS}
        T = len(S_raw)
        edges = np.linspace(0, T, 5).astype(int)
        eps = {}
        blocks = {}
        for b in range(4):
            sl = slice(edges[b], edges[b + 1])
            eps[b] = _episode(A_raw[sl], S_raw[sl], "m100", spans)
            blocks[b] = [int(edges[b]), int(edges[b + 1])]
        out.append({
            "name": f"NeuroTo_{arm}", "repo_id": "NeuroTo/bimanual-so101-fold-tshirt ep0 (" + arm + " arm)",
            "revision": e["revision"], "units": "m100", "primary": False,
            "spans_ticks": {j: round(inf["span_ticks"][j], 2) for j in JOINTS},
            "spans_deg": {j: round(float(inf["spans_deg"][j]), 3) for j in JOINTS},
            "episodes": eps, "test": [3], "train": [0, 1, 2], "excluded": [],
            "split_unit": "time_block", "blocks_frames": blocks,
            "split_rule": "single episode cut into 4 contiguous time blocks [start,end) frames; block 3 -> TEST",
        })
    return out


# ----------------------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------------------
def delayed(A, d):
    idx = np.maximum(np.arange(len(A)) - d, 0)
    return A[idx]


def fit_best_delay(ds):
    """Integer delay d (frames) minimising mean-over-arm-joints pooled RMSE of state[t] vs action[t-d], TRAIN only."""
    sse = np.zeros((MAX_DELAY + 1, 6))
    n = 0
    for ep in ds["train"]:
        e = ds["episodes"][ep]
        if e["T"] <= MAX_DELAY + 1:
            continue
        S = e["S"][MAX_DELAY:]
        for d in range(MAX_DELAY + 1):
            P = delayed(e["A"], d)[MAX_DELAY:]
            sse[d] += (np.degrees(P - S) ** 2).sum(0)
        n += len(S)
    rmse = np.sqrt(sse / max(n, 1))
    obj = rmse[:, :ARM].mean(1)
    d_best = int(np.argmin(obj))
    per_joint = {j: int(np.argmin(rmse[:, i])) for i, j in enumerate(JOINTS)}
    return {"d_frames": d_best, "d_ms": round(1000 * d_best / FPS, 1), "train_frames": int(n),
            "train_arm_rmse_by_d": [round(float(x), 3) for x in obj], "per_joint_best_d": per_joint}


# ----------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------
def make_windows(ds, W):
    N = int(round(W * FPS))
    wins = []
    for ep in ds["test"]:
        T = ds["episodes"][ep]["T"]
        t0 = 0
        while t0 + N <= T - 1:
            wins.append((ep, t0))
            t0 += N
    return wins


# ----------------------------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------------------------
_SIMS = {}


def _get_sims(limits, dt):
    key = (limits, dt)
    if key in _SIMS:
        return _SIMS[key]
    import models  # noqa: E402

    sims = {}
    for v in SIM_MODELS:
        s = models.make_sim(v, dt=dt)
        s._range_deg = np.degrees(np.array([s.model.jnt_range[s.model.joint(j).id] for j in JOINTS]))
        if limits == "wide":
            for k, j in enumerate(JOINTS[:ARM]):
                s.model.jnt_limited[s.model.joint(j).id] = 0
                s.model.actuator_ctrllimited[s.aid[k]] = 0
        sims[v] = s
    _SIMS[key] = sims
    return sims


def simulate_window(sim, q0, v0, cmd, limits):
    import mujoco

    lo, hi = np.radians(sim._range_deg[:, 0]), np.radians(sim._range_deg[:, 1])
    q0 = np.array(q0, dtype=float)
    v0 = np.array(v0, dtype=float)
    if limits == "sim":
        qc = np.clip(q0, lo, hi)
        v0[qc != q0] = 0.0
        q0 = qc
    sim.reset(q0)
    sim.data.qvel[sim.vadr] = v0
    mujoco.mj_forward(sim.model, sim.data)
    if sim.bank is not None:
        sim.bank.reset()
    N = len(cmd)
    pred = np.empty((N, 6))
    for k in range(N):
        sim.set_target(cmd[k])
        sim.advance(1.0 / FPS)
        pred[k] = sim.qpos()
    return pred


def run_job(job):
    """job = dict(name, W, limits, dt, delay_d, windows=[(ep,t0)], eps={ep: {A,S,grasp,T}})"""
    W, limits, dt, d = job["W"], job["limits"], job["dt"], job["delay_d"]
    sims = _get_sims(limits, dt)
    rng_deg = sims["A_stock"]._range_deg
    N = int(round(W * FPS))
    cpu = {v: 0.0 for v in SIM_MODELS}
    recs = []
    for ep, t0 in job["windows"]:
        e = job["eps"][ep]
        A, S, grasp = e["A"], e["S"], e["grasp"]
        real = S[t0 + 1:t0 + N + 1]
        cmd = A[t0:t0 + N]
        real_deg = np.degrees(real)
        v0 = (S[t0] - S[t0 - 1]) * FPS if t0 > 0 else np.zeros(6)
        preds = {}
        for v in SIM_MODELS:
            c0 = time.process_time()
            preds[v] = simulate_window(sims[v], S[t0], v0, cmd, limits)
            cpu[v] += time.process_time() - c0
        preds["perfect_tracking"] = A[t0:t0 + N]
        preds["best_pure_delay"] = delayed(A, d)[t0 + 1:t0 + N + 1]
        preds["hold_start"] = np.repeat(S[t0][None], N, 0)
        rec = {
            "dataset": job["name"], "W": W, "limits": limits, "episode": int(ep), "t0": int(t0), "n": N,
            "grasp": bool(grasp[t0 + 1:t0 + N + 1].any()),
            "oor": bool(((real_deg[:, :ARM] < rng_deg[:ARM, 0] - OOR_MARGIN_DEG) |
                         (real_deg[:, :ARM] > rng_deg[:ARM, 1] + OOR_MARGIN_DEG)).any()),
            "cmd_excursion_deg": float(np.degrees(cmd[:, :ARM].max(0) - cmd[:, :ARM].min(0)).max()),
        }
        rec["moving"] = rec["cmd_excursion_deg"] > MOVING_DEG
        for m, P in preds.items():
            err = np.degrees(P) - real_deg
            if not np.all(np.isfinite(err)):
                rec["nonfinite_" + m] = True
                err = np.nan_to_num(err, nan=1e3, posinf=1e3, neginf=-1e3)
            sse = (err ** 2).sum(0)
            for i, j in enumerate(JOINTS):
                rec[f"sse|{m}|{j}"] = float(sse[i])
                rec[f"rmse|{m}|{j}"] = float(math.sqrt(sse[i] / N))
        recs.append(rec)
    return {"recs": recs, "cpu": cpu, "sim_seconds": len(job["windows"]) * N / FPS}


# ----------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------
def _subset_mask(df, subset):
    if subset == "all":
        return np.ones(len(df), bool)
    if subset == "no_grasp":
        return ~df["grasp"].values
    if subset == "in_range":
        return ~df["oor"].values
    if subset == "moving":
        return df["moving"].values
    raise ValueError(subset)


def aggregate(df, dsets):
    """-> nested dict [limits][W][subset][dataset][model] = {joint: {rmse, median}, arm_mean_*, n_windows}"""
    out = {}
    for (limits, W), g0 in df.groupby(["limits", "W"]):
        for subset in SUBSETS:
            g1 = g0[_subset_mask(g0, subset)]
            for ds in dsets:
                g = g1[g1["dataset"] == ds["name"]]
                for m in ALL_MODELS:
                    cell = {"n_windows": int(len(g)), "n_frames": int(g["n"].sum()) if len(g) else 0}
                    for j in JOINTS:
                        if len(g):
                            cell[j] = {
                                "rmse": float(math.sqrt(g[f"sse|{m}|{j}"].sum() / g["n"].sum())),
                                "median_window_rmse": float(np.median(g[f"rmse|{m}|{j}"])),
                            }
                        else:
                            cell[j] = {"rmse": None, "median_window_rmse": None}
                    if len(g):
                        cell["arm_mean_rmse"] = float(np.mean([cell[j]["rmse"] for j in JOINTS[:ARM]]))
                        cell["arm_mean_median_window_rmse"] = float(
                            np.mean([cell[j]["median_window_rmse"] for j in JOINTS[:ARM]]))
                    else:
                        cell["arm_mean_rmse"] = cell["arm_mean_median_window_rmse"] = None
                    out.setdefault(limits, {}).setdefault(str(W), {}).setdefault(subset, {}) \
                        .setdefault(ds["name"], {})[m] = cell
            # headline over primary datasets
            prim = [ds["name"] for ds in dsets if ds["primary"]]
            sub = out[limits][str(W)][subset]
            ov = {}
            for m in ALL_MODELS:
                vals = [sub[n][m] for n in prim if sub[n][m]["n_windows"] > 0]
                o = {"n_datasets": len(vals), "n_windows": int(sum(v["n_windows"] for v in vals))}
                for j in JOINTS:
                    o[j] = {"rmse": float(np.mean([v[j]["rmse"] for v in vals])) if vals else None,
                            "median_window_rmse": float(np.mean([v[j]["median_window_rmse"] for v in vals])) if vals else None}
                o["arm_mean_rmse"] = float(np.mean([v["arm_mean_rmse"] for v in vals])) if vals else None
                o["arm_mean_median_window_rmse"] = float(np.mean([v["arm_mean_median_window_rmse"] for v in vals])) if vals else None
                ov[m] = o
            sub["OVERALL_primary_mean"] = ov
    return out


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def write_table(agg, dsets, meta, path):
    L = []
    L.append("# SO-101 real-vs-sim open-loop replay: first baseline (TEST episodes only)")
    L.append("")
    L.append(f"Generated by `replay.py` on {meta['generated']}. dt = {meta['dt']} s. All numbers in degrees "
             "(sim joint convention). Cell = pooled RMSE over all TEST window frames (median of per-window RMSE "
             "in parentheses). Arm mean = mean of the 5 arm joints; gripper is separate and uses the APPROXIMATE "
             "linear 0..100 -> -10..100 deg map with contacts off, so treat it as indicative only.")
    L.append("")
    L.append("Primary datasets (LeRobot DEGREES units): " + ", ".join(d["repo_id"] for d in dsets if d["primary"])
             + ". OVERALL = equal-weight mean over these 6. NeuroTo (M100 units, cloth folding, 1 episode, time-block "
             "split) is supplementary and not in OVERALL.")
    L.append("")
    L.append("Models: A_stock = Menagerie position servo kp 998; B_nvidia = NVIDIA Isaac SysID values; "
             "C_bam_m1 / D_bam_m6 = Rhoban BAM STS3215 7.4 V models M1 / M6 at firmware P=16. "
             "Baselines (no physics): perfect_tracking = state[t] := action[t-1]; best_pure_delay = state[t] := "
             "action[t-d] with d fit per dataset on TRAIN; hold_start = state stays at the window's first real state.")
    L.append("")

    # summary
    L.append("## 1. Headline: OVERALL arm-mean RMSE (6 primary datasets)")
    L.append("")
    cfgs = [(lim, W) for W in meta["windows"] for lim in meta["limits"]]
    hdr = "| model | " + " | ".join(f"W={W}s, limits={lim}" for lim, W in cfgs) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (len(cfgs) + 1))
    for m in ALL_MODELS:
        row = [m]
        for lim, W in cfgs:
            o = agg[lim][str(W)]["all"]["OVERALL_primary_mean"][m]
            row.append(f"{_f(o['arm_mean_rmse'])} ({_f(o['arm_mean_median_window_rmse'])})")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Same, by window subset (W=4 s). no_grasp = windows with no gripper stall on an object; in_range = real "
             "state never >1 deg outside the sim joint range; moving = commanded excursion > 5 deg.")
    L.append("")
    subs = [s for s in SUBSETS if s != "all"]
    W0 = meta["windows"][0]
    hdr = "| model | " + " | ".join(f"{s}, limits={lim}" for s in subs for lim in meta["limits"]) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (len(subs) * len(meta["limits"]) + 1))
    for m in ALL_MODELS:
        row = [m]
        for s in subs:
            for lim in meta["limits"]:
                o = agg[lim][str(W0)][s]["OVERALL_primary_mean"][m]
                row.append(f"{_f(o['arm_mean_rmse'])} ({_f(o['arm_mean_median_window_rmse'])})")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    nwin = {s: agg[meta["limits"][0]][str(W0)][s]["OVERALL_primary_mean"]["A_stock"]["n_windows"] for s in SUBSETS}
    L.append("Windows (W=4 s, primary): " + ", ".join(f"{s} {n}" for s, n in nwin.items()))
    L.append("")

    # per-joint overall
    L.append("## 2. OVERALL per joint (6 primary datasets, all windows)")
    for lim, W in cfgs:
        L.append("")
        L.append(f"### W={W} s, limits={lim}")
        L.append("")
        L.append("| model | " + " | ".join(JOINTS) + " | arm mean |")
        L.append("|" + "---|" * 8)
        for m in ALL_MODELS:
            o = agg[lim][str(W)]["all"]["OVERALL_primary_mean"][m]
            L.append("| " + m + " | " + " | ".join(
                f"{_f(o[j]['rmse'])} ({_f(o[j]['median_window_rmse'])})" for j in JOINTS)
                + f" | {_f(o['arm_mean_rmse'])} ({_f(o['arm_mean_median_window_rmse'])}) |")

    # per dataset per joint, primary config and the others
    L.append("")
    L.append("## 3. Per dataset x model x joint")
    for lim, W in cfgs:
        L.append("")
        L.append(f"### W={W} s, limits={lim}, all TEST windows")
        L.append("")
        L.append("| dataset | model | " + " | ".join(JOINTS) + " | arm mean | windows |")
        L.append("|" + "---|" * 10)
        for ds in dsets:
            for m in ALL_MODELS:
                c = agg[lim][str(W)]["all"][ds["name"]][m]
                L.append(f"| {ds['name']} | {m} | " + " | ".join(
                    f"{_f(c[j]['rmse'])} ({_f(c[j]['median_window_rmse'])})" for j in JOINTS)
                    + f" | {_f(c['arm_mean_rmse'])} ({_f(c['arm_mean_median_window_rmse'])}) | {c['n_windows']} |")

    # dataset facts
    L.append("")
    L.append("## 4. Dataset facts")
    L.append("")
    L.append("| dataset | units | TRAIN eps | TEST eps | best delay d (TRAIN) | per-joint best d | TEST frames | "
             "frames in W=4 windows | grasp frac (W=4 windows) | out-of-range frac (W=4 windows) |")
    L.append("|" + "---|" * 10)
    for ds in dsets:
        f = meta["dataset_facts"][ds["name"]]
        L.append(f"| {ds['name']} | {ds['units']} | {len(ds['train'])} | {len(ds['test'])} | {f['delay']['d_frames']} "
                 f"({f['delay']['d_ms']} ms) | {f['delay']['per_joint_best_d']} | {f['test_frames']} | "
                 f"{f['frames_in_windows'].get(str(W0), '-')} | {_f(f.get('grasp_window_frac'))} | {_f(f.get('oor_window_frac'))} |")
    L.append("")

    # runtime
    L.append("## 5. Runtime (CPU time inside the sim loop, time.process_time, per process)")
    L.append("")
    L.append("| model | CPU s total | simulated s | CPU s per simulated s | real-time factor |")
    L.append("|---|---|---|---|---|")
    for v in SIM_MODELS:
        r = meta["runtime"][v]
        L.append(f"| {v} | {r['cpu_s']:.1f} | {r['sim_s']:.0f} | {r['cpu_per_sim_s']:.4f} | {1 / r['cpu_per_sim_s']:.1f}x |")
    L.append("")
    L.append(f"Total wall-clock {meta['wall_s']:.0f} s with {meta['procs']} worker processes on this laptop (no GPU).")
    L.append("")
    L.append("## 6. Verdict (computed)")
    L.append("")
    for line in meta["verdict"]:
        L.append("- " + line)
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def verdict_lines(agg, meta):
    out = []
    for W in meta["windows"]:
        for lim in meta["limits"]:
            ov = agg[lim][str(W)]["all"]["OVERALL_primary_mean"]
            bpd, pt = ov["best_pure_delay"]["arm_mean_rmse"], ov["perfect_tracking"]["arm_mean_rmse"]
            best_sim = min(SIM_MODELS, key=lambda m: ov[m]["arm_mean_rmse"])
            ranks = sorted(ALL_MODELS, key=lambda m: ov[m]["arm_mean_rmse"])
            beats_bpd = [m for m in SIM_MODELS if ov[m]["arm_mean_rmse"] < bpd]
            beats_pt = [m for m in SIM_MODELS if ov[m]["arm_mean_rmse"] < pt]
            # per-dataset wins vs best_pure_delay
            wins = {}
            for m in SIM_MODELS:
                n = 0
                for ds in meta["primary"]:
                    c = agg[lim][str(W)]["all"][ds]
                    n += c[m]["arm_mean_rmse"] < c["best_pure_delay"]["arm_mean_rmse"]
                wins[m] = n
            out.append(
                f"W={W} s, limits={lim}: ranking by OVERALL arm-mean pooled RMSE = "
                + ", ".join(f"{m} {ov[m]['arm_mean_rmse']:.2f}" for m in ranks)
                + f". Best physics model: {best_sim}. Physics models better than perfect_tracking: "
                + (", ".join(beats_pt) if beats_pt else "NONE")
                + "; better than best_pure_delay: " + (", ".join(beats_bpd) if beats_bpd else "NONE")
                + ". Datasets (of 6) where each beats best_pure_delay: "
                + ", ".join(f"{m} {n}" for m, n in wins.items()) + ".")
    return out


# ----------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------
def plot_example(dsets, df, dt, path, prefer="dobri420"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = 4
    g = df[(df["W"] == W) & (df["limits"] == "sim") & (df["dataset"] == prefer) & (~df["oor"]) & (~df["grasp"])]
    if not len(g):
        g = df[(df["W"] == W) & (df["limits"] == "sim") & (df["dataset"] == prefer)]
    # most commanded motion (summed over the arm joints) among in-range, grasp-free TEST windows:
    # shows dynamics, not just holding
    ds = next(d for d in dsets if d["name"] == prefer)

    def _total_exc(row):
        c = ds["episodes"][int(row["episode"])]["A"][int(row["t0"]):int(row["t0"]) + int(row["n"]), :ARM]
        return float(np.degrees(c.max(0) - c.min(0)).sum())

    g = g.assign(total_exc=g.apply(_total_exc, axis=1))
    r = g.sort_values("total_exc", ascending=False).iloc[0]
    e = ds["episodes"][int(r["episode"])]
    t0, N = int(r["t0"]), int(r["n"])
    sims = _get_sims("sim", dt)
    S, A = e["S"], e["A"]
    v0 = (S[t0] - S[t0 - 1]) * FPS if t0 > 0 else np.zeros(6)
    preds = {v: simulate_window(sims[v], S[t0], v0, A[t0:t0 + N], "sim") for v in SIM_MODELS}
    t_state = np.arange(t0, t0 + N + 1) / FPS
    t_cmd = np.arange(t0, t0 + N) / FPS

    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"A_stock": "#2a78d6", "B_nvidia": "#eb6834", "C_bam_m1": "#1baf7a", "D_bam_m6": "#eda100"}
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, facecolor=surf)
    for i, j in enumerate(JOINTS[:ARM]):
        ax = axes[i]
        ax.set_facecolor(surf)
        ax.step(t_cmd, np.degrees(A[t0:t0 + N, i]), where="post", color=ink2, lw=1.2, ls="--",
                label="action (command, held 1 frame)")
        for v in SIM_MODELS:
            ax.plot(t_state, np.degrees(np.r_[S[t0, i], preds[v][:, i]]), color=colors[v], lw=1.6, label=v)
        ax.plot(t_state, np.degrees(S[t0:t0 + N + 1, i]), color=ink, lw=2.2, label="real observation.state")
        rm = {v: math.sqrt(np.mean((np.degrees(preds[v][:, i] - S[t0 + 1:t0 + N + 1, i])) ** 2)) for v in SIM_MODELS}
        ax.set_ylabel(f"{j}\n[deg]", color=ink2, fontsize=9)
        ax.set_title("window RMSE vs real [deg]:  " + "   ".join(f"{v} {rm[v]:.2f}" for v in SIM_MODELS),
                     loc="right", fontsize=8, color=ink2, pad=3)
        ax.grid(True, color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    axes[-1].set_xlabel("time in episode [s]", color=ink2)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{ds['repo_id']}  TEST episode {int(r['episode'])}, window t0 = {t0 / FPS:.2f} s, W = {W} s, "
                 f"sim joint limits, open-loop from the real state", color=ink, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=130, facecolor=surf)
    plt.close(fig)
    return {"dataset": ds["repo_id"], "episode": int(r["episode"]), "t0_frame": t0, "W": W,
            "cmd_excursion_deg": float(r["cmd_excursion_deg"]),
            "window_rmse_deg": {v: {j: float(math.sqrt(np.mean((np.degrees(preds[v][:, i] - S[t0 + 1:t0 + N + 1, i])) ** 2)))
                                    for i, j in enumerate(JOINTS)} for v in SIM_MODELS}}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="+", default=[4, 1])
    ap.add_argument("--limits", nargs="+", default=["sim", "wide"], choices=["sim", "wide"])
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--procs", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--chunk", type=int, default=40, help="windows per worker job")
    ap.add_argument("--quick", action="store_true", help="3 windows per dataset (smoke test, no files written)")
    ap.add_argument("--plot-only", action="store_true", help="redraw episode_example.png from results/windows.csv")
    args = ap.parse_args()
    args.windows = [int(w) if float(w).is_integer() else w for w in args.windows]

    t_wall = time.perf_counter()
    dsets = load_datasets()
    if args.plot_only:
        df = pd.read_csv(os.path.join(RESULTS, "windows.csv"))
        ex = plot_example(dsets, df, args.dt, os.path.join(RESULTS, "episode_example.png"))
        bj = os.path.join(RESULTS, "baseline.json")
        with open(bj) as fh:
            full = json.load(fh)
        full["meta"]["example_plot"] = ex
        with open(bj, "w") as fh:
            json.dump(full, fh, indent=1)
        print("example window:", {k: ex[k] for k in ("dataset", "episode", "t0_frame")})
        return
    print(f"loaded {len(dsets)} datasets in {time.perf_counter() - t_wall:.1f} s")

    facts = {}
    for ds in dsets:
        dl = fit_best_delay(ds)
        ds["delay"] = dl
        facts[ds["name"]] = {"delay": dl, "test_frames": int(sum(ds["episodes"][k]["T"] for k in ds["test"])),
                             "train_frames": int(sum(ds["episodes"][k]["T"] for k in ds["train"])),
                             "frames_in_windows": {}}
        print(f"  {ds['name']:16s} train {len(ds['train']):3d} eps, test {len(ds['test']):3d} eps, "
              f"best delay d={dl['d_frames']} ({dl['d_ms']} ms), per-joint {dl['per_joint_best_d']}")

    if not args.quick:
        split = {}
        for ds in dsets:
            s = {"repo_id": ds["repo_id"], "revision": ds["revision"], "unit": ds["split_unit"],
                 "rule": ds["split_rule"], "train": ds["train"], "test": ds["test"], "excluded": ds["excluded"]}
            if ds["split_unit"] == "episode":
                s["excluded_reason"] = ("Chaenn eps 87-106: 10-45 % of action frames off the encoder grid "
                                        "(probably policy/intervention rollouts)") if ds["excluded"] else None
            else:
                s["blocks_frames"] = ds["blocks_frames"]
            split[ds["name"]] = s
        with open(SPLIT_JSON, "w") as fh:
            json.dump(split, fh, indent=1)

    jobs = []
    for W in args.windows:
        for ds in dsets:
            wins = make_windows(ds, W)
            facts[ds["name"]]["frames_in_windows"][str(W)] = int(len(wins) * round(W * FPS))
            if args.quick:
                wins = wins[:3]
            eps = {ep: ds["episodes"][ep] for ep in ds["test"]}
            for limits in args.limits:
                for i in range(0, len(wins), args.chunk):
                    jobs.append({"name": ds["name"], "W": W, "limits": limits, "dt": args.dt,
                                 "delay_d": ds["delay"]["d_frames"], "windows": wins[i:i + args.chunk],
                                 "eps": {ep: eps[ep] for ep in {w[0] for w in wins[i:i + args.chunk]}}})
    jobs.sort(key=lambda j: -len(j["windows"]) * j["W"])
    print(f"{len(jobs)} jobs on {args.procs} processes")
    recs, cpu, sim_s = [], {v: 0.0 for v in SIM_MODELS}, 0.0
    t_sim = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs) as pool:
        for k, res in enumerate(pool.imap_unordered(run_job, jobs)):
            recs += res["recs"]
            for v in SIM_MODELS:
                cpu[v] += res["cpu"][v]
            sim_s += res["sim_seconds"]
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"  {k + 1}/{len(jobs)} jobs, {time.perf_counter() - t_sim:.0f} s")
    df = pd.DataFrame(recs).sort_values(["limits", "W", "dataset", "episode", "t0"]).reset_index(drop=True)
    wall = time.perf_counter() - t_wall

    for ds in dsets:
        g = df[(df["dataset"] == ds["name"]) & (df["W"] == args.windows[0]) & (df["limits"] == args.limits[0])]
        facts[ds["name"]]["grasp_window_frac"] = float(g["grasp"].mean()) if len(g) else None
        facts[ds["name"]]["oor_window_frac"] = float(g["oor"].mean()) if len(g) else None
        facts[ds["name"]]["moving_window_frac"] = float(g["moving"].mean()) if len(g) else None
        if ds["units"] == "m100":
            facts[ds["name"]]["spans_deg"] = ds["spans_deg"]
            facts[ds["name"]]["spans_ticks"] = ds["spans_ticks"]
    nonfinite = [c for c in df.columns if c.startswith("nonfinite_")]

    agg = aggregate(df, dsets)
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "dt": args.dt, "windows": args.windows, "limits": args.limits,
        "procs": args.procs, "wall_s": wall, "primary": [d["name"] for d in dsets if d["primary"]],
        "dataset_facts": facts,
        "runtime": {v: {"cpu_s": cpu[v], "sim_s": sim_s, "cpu_per_sim_s": cpu[v] / max(sim_s, 1e-9)} for v in SIM_MODELS},
        "nonfinite_columns": nonfinite,
    }
    meta["verdict"] = verdict_lines(agg, meta)
    for line in meta["verdict"]:
        print(line)
    if args.quick:
        print(json.dumps(agg["sim"][str(args.windows[0])]["all"]["OVERALL_primary_mean"]["A_stock"], indent=None)[:600])
        return

    os.makedirs(RESULTS, exist_ok=True)
    ex = plot_example(dsets, df, args.dt, os.path.join(RESULTS, "episode_example.png"))
    meta["example_plot"] = ex
    df.to_csv(os.path.join(RESULTS, "windows.csv"), index=False)
    method = {k: v for k, v in [
        ("alignment", "sim driven by action[t] (zero-order hold 1/30 s), its qpos compared with state[t+1]"),
        ("reset", "sim reset to real state[t0]; qvel = (state[t0]-state[t0-1])*30 (0 at t0=0); 'sim' limits clip q0 into the joint range"),
        ("split", f"episode_index % {TEST_MOD} == {TEST_REM} -> TEST (NeuroTo: time block 3 of 4)"),
        ("windows", "non-overlapping, tiled from frame 0 of each TEST episode, trailing remainder < W dropped"),
        ("units", "lerobot_units.to_radians defaults: signs +1, offsets 0, gripper linear_sim (approx.)"),
        ("baselines", "perfect_tracking: action[t-1]; best_pure_delay: action[t-d], d in 0..15 fit on TRAIN per dataset; hold_start: state[t0]"),
        ("grasp_heuristic", f"gripper state - action > {GRASP_THRESH} units for >= {GRASP_MIN_FRAMES} consecutive frames"),
        ("oor", f"real state > {OOR_MARGIN_DEG} deg outside the Menagerie joint range on an arm joint"),
        ("moving", f"commanded excursion within the window > {MOVING_DEG} deg on some arm joint"),
        ("table_contact", "not detected (table height unknown per dataset); not excluded"),
        ("headline", "equal-weight mean over the 6 primary (DEGREES) datasets; NeuroTo supplementary"),
    ]}
    with open(os.path.join(RESULTS, "baseline.json"), "w") as fh:
        json.dump({"method": method, "meta": meta, "results": agg}, fh, indent=1)
    write_table(agg, dsets, meta, os.path.join(RESULTS, "baseline_table.md"))
    print(f"done in {time.perf_counter() - t_wall:.0f} s; wrote results/baseline.json, baseline_table.md, "
          f"windows.csv, episode_example.png, split.json")


if __name__ == "__main__":
    main()
