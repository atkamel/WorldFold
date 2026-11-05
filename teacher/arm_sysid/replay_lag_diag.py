"""Diagnostic for replay.py: WHY do the models score as they do?

For the in-range, grasp-free W=4 s TEST windows of the 6 primary datasets (selected from
results/windows.csv), re-simulate every variant exactly as replay.py does (limits='sim', same reset and
alignment) and measure, for the real arm and for each model:

1. effective lag: integer d (frames, 0..10) minimising the RMS of (y[t] - action[t-d]) after removing
   each window's mean difference (so a constant sag offset does not bias the lag), per joint;
2. hold offset: median of (y[t] - action[t-1]) over frames whose command has stayed within 0.2 deg for the
   previous 15 frames (0.5 s), per joint (gravity sag + deadband + friction stick, signed, deg);
3. hold |offset| median (deg).

Writes results/lag_diag.json and prints a table. Run after replay.py.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay as rp  # noqa: E402

MAXD = 10
HOLD_FRAMES = 15
HOLD_TOL_DEG = 0.2


def job(args):
    name, eps, wins, dt = args
    sims = rp._get_sims("sim", dt)
    N = 4 * rp.FPS
    who = ("real",) + rp.SIM_MODELS
    sse = {w: np.zeros((MAXD + 1, 6)) for w in who}
    hold = {w: [] for w in who}
    nfr = 0
    for ep, t0 in wins:
        e = eps[ep]
        A, S = e["A"], e["S"]
        if t0 < MAXD:
            continue
        v0 = (S[t0] - S[t0 - 1]) * rp.FPS if t0 > 0 else np.zeros(6)
        ys = {"real": S[t0 + 1:t0 + N + 1]}
        for v in rp.SIM_MODELS:
            ys[v] = rp.simulate_window(sims[v], S[t0], v0, A[t0:t0 + N], "sim")
        t = np.arange(t0 + 1, t0 + N + 1)
        Ad = np.degrees(A)
        # command held steady for HOLD_FRAMES frames before (and including) t-1
        steady = np.ones((N, 6), bool)
        for k in range(HOLD_FRAMES):
            steady &= np.abs(Ad[t - 1 - k] - Ad[t - 1]) < HOLD_TOL_DEG
        for w, y in ys.items():
            yd = np.degrees(y)
            for d in range(MAXD + 1):
                diff = yd - Ad[t - d]
                diff = diff - diff.mean(0)
                sse[w][d] += (diff ** 2).sum(0)
            off = yd - Ad[t - 1]
            hold[w].append(np.where(steady, off, np.nan))
        nfr += N
    out = {"n_windows": len(wins), "n_frames": nfr}
    for w in who:
        H = np.concatenate(hold[w]) if hold[w] else np.full((0, 6), np.nan)
        out[w] = {
            "lag_frames": [int(np.argmin(sse[w][:, i])) for i in range(6)],
            "hold_offset_median_deg": [float(np.nanmedian(H[:, i])) if np.isfinite(H[:, i]).any() else None for i in range(6)],
            "hold_abs_offset_median_deg": [float(np.nanmedian(np.abs(H[:, i]))) if np.isfinite(H[:, i]).any() else None for i in range(6)],
            "hold_frames": [int(np.isfinite(H[:, i]).sum()) for i in range(6)],
        }
    return name, out


def main():
    df = pd.read_csv(os.path.join(rp.RESULTS, "windows.csv"))
    df = df[(df["W"] == 4) & (df["limits"] == "sim") & (~df["oor"]) & (~df["grasp"])]
    dsets = [d for d in rp.load_datasets() if d["primary"]]
    jobs = []
    for ds in dsets:
        g = df[df["dataset"] == ds["name"]]
        wins = [(int(a), int(b)) for a, b in zip(g["episode"], g["t0"])]
        jobs.append((ds["name"], {ep: ds["episodes"][ep] for ep in ds["test"]}, wins, 0.002))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = dict(pool.map(job, jobs))
    who = ("real",) + rp.SIM_MODELS
    J = rp.JOINTS
    print("effective lag [frames @30 fps] per joint (pan, lift, elbow, wflex, wroll, grip)")
    for name, r in res.items():
        print(f"  {name:16s} n_win {r['n_windows']:3d}  " + "  ".join(f"{w}={r[w]['lag_frames'][:5]}" for w in who))
    print("hold offset median (y - command) [deg], shoulder_lift / elbow_flex / wrist_flex")
    for name, r in res.items():
        print(f"  {name:16s} " + "  ".join(
            f"{w}=" + "/".join("-" if x is None else f"{x:+.2f}" for x in r[w]['hold_offset_median_deg'][1:4]) for w in who))
    # pooled summary: median across datasets
    summ = {}
    for w in who:
        summ[w] = {
            "lag_frames_median_over_datasets": {J[i]: float(np.median([res[n][w]["lag_frames"][i] for n in res])) for i in range(5)},
            "hold_offset_median_over_datasets_deg": {J[i]: float(np.median([res[n][w]["hold_offset_median_deg"][i] for n in res
                                                                             if res[n][w]["hold_offset_median_deg"][i] is not None]))
                                                     for i in range(5)},
            "hold_abs_offset_median_over_datasets_deg": {J[i]: float(np.median([res[n][w]["hold_abs_offset_median_deg"][i] for n in res
                                                                                 if res[n][w]["hold_abs_offset_median_deg"][i] is not None]))
                                                         for i in range(5)},
        }
    print("SUMMARY (median over 6 datasets), arm joints [pan, lift, elbow, wrist_flex, wrist_roll]:")
    for w in who:
        print(f"  {w:9s} lag " + str(list(summ[w]["lag_frames_median_over_datasets"].values()))
              + "  hold offset " + str([round(v, 2) for v in summ[w]["hold_offset_median_over_datasets_deg"].values()])
              + "  |hold offset| " + str([round(v, 2) for v in summ[w]["hold_abs_offset_median_over_datasets_deg"].values()]))
    with open(os.path.join(rp.RESULTS, "lag_diag.json"), "w") as fh:
        json.dump({"method": __doc__, "per_dataset": res, "summary": summ}, fh, indent=1)
    write_md_section(res, summ)


def write_md_section(res, summ):
    """Replace/append section 7 of results/baseline_table.md with this diagnostic."""
    who = ("real",) + rp.SIM_MODELS
    L = ["## 7. Diagnostic: effective lag and hold offset (replay_lag_diag.py)", "",
         "In-range, grasp-free W=4 s TEST windows of the 6 primary datasets ("
         + ", ".join(f"{n} {r['n_windows']}" for n, r in res.items()) + " windows), limits=sim. "
         "Lag = integer frame shift d minimising the window-demeaned RMS of y[t] - action[t-d] (30 fps, so 1 frame = 33 ms). "
         f"Hold offset = median of y[t] - action[t-1] over frames whose command stayed within {HOLD_TOL_DEG} deg for "
         f"{HOLD_FRAMES} frames (gravity sag + deadband + stick). Values = median over the 6 datasets.", "",
         "| source | lag pan | lag lift | lag elbow | lag wflex | lag wroll | hold pan | hold lift | hold elbow | hold wflex | hold wroll |",
         "|" + "---|" * 11]
    for w in who:
        lag = list(summ[w]["lag_frames_median_over_datasets"].values())
        off = list(summ[w]["hold_offset_median_over_datasets_deg"].values())
        L.append(f"| {w} | " + " | ".join(f"{x:g}" for x in lag) + " | " + " | ".join(f"{x:+.2f}" for x in off) + " |")
    L.append("")
    path = os.path.join(rp.RESULTS, "baseline_table.md")
    with open(path, encoding="utf-8") as fh:
        md = fh.read()
    i = md.find("## 7. Diagnostic")
    if i >= 0:
        md = md[:i]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md.rstrip("\n") + "\n\n" + "\n".join(L))


if __name__ == "__main__":
    main()
