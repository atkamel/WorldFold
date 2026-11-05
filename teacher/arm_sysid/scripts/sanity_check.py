"""Sanity-check downloaded LeRobot SO-101 datasets: columns, alignment, ranges,
unit fingerprint (encoder quantization), and raw action->state lag.
Usage: python sanity_check.py <dataset_dir> [<dataset_dir> ...]  -> prints + writes sanity.json in each dir
"""
import sys, os, json, glob
import numpy as np, pandas as pd

TICK_DEG = 360.0 / 4095.0  # LeRobot DEGREES normalization: (ticks-mid)*360/4095 (motors_bus.py)

def load(d):
    files = sorted(glob.glob(os.path.join(d, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df, files

def quant_R(v, step):
    """circular concentration of v/step mod 1 (1 = all values on one lattice)."""
    ph = np.exp(2j * np.pi * (v / step))
    return float(np.abs(ph.mean()))

def best_lag(a, s, maxlag=15):
    """lag k (frames) minimizing RMS(s[t] - a[t-k])."""
    best = None
    for k in range(0, maxlag + 1):
        e = s[k:] - a[: len(a) - k]
        r = float(np.sqrt(np.mean(e ** 2)))
        if best is None or r < best[1]: best = (k, r)
    return best

def check(d):
    info = json.load(open(os.path.join(d, "meta", "info.json")))
    fps = info["fps"]
    df, files = load(d)
    out = dict(dir=d, files=len(files), rows=len(df), columns=list(df.columns), fps=fps)
    A = np.stack(df["action"].to_numpy()).astype(np.float64)
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    out["action_shape"] = list(A.shape); out["state_shape"] = list(S.shape)
    out["nan_action"] = int(np.isnan(A).sum()); out["nan_state"] = int(np.isnan(S).sum())
    names = info["features"]["action"]["names"]; snames = info["features"]["observation.state"]["names"]
    out["action_names"] = names; out["state_names"] = snames
    out["names_match"] = names == snames
    # episode structure
    eps = df["episode_index"].unique(); out["n_episodes"] = int(len(eps)); out["episodes"] = [int(eps.min()), int(eps.max())]
    bad_fi, bad_ts, dts = 0, 0, []
    lens = []
    for e, g in df.groupby("episode_index"):
        fi = g["frame_index"].to_numpy(); ts = g["timestamp"].to_numpy()
        lens.append(len(g))
        if not np.array_equal(fi, np.arange(len(g))): bad_fi += 1
        dt = np.diff(ts); dts.append(dt)
        if np.abs(dt - 1.0 / fps).max() > 0.25 / fps: bad_ts += 1
    dts = np.concatenate(dts)
    out["episode_len_min_med_max"] = [int(min(lens)), int(np.median(lens)), int(max(lens))]
    out["episodes_bad_frame_index"] = bad_fi; out["episodes_bad_timestamp"] = bad_ts
    out["dt_mean_min_max"] = [float(dts.mean()), float(dts.min()), float(dts.max())]
    # ranges
    out["action_min"] = A.min(0).round(2).tolist(); out["action_max"] = A.max(0).round(2).tolist()
    out["state_min"] = S.min(0).round(2).tolist(); out["state_max"] = S.max(0).round(2).tolist()
    # unit fingerprints
    out["state_quantR_deg_tick"] = [round(quant_R(S[:, j], TICK_DEG), 3) for j in range(S.shape[1])]
    out["action_quantR_deg_tick"] = [round(quant_R(A[:, j], TICK_DEG), 3) for j in range(A.shape[1])]
    out["state_frac_mode_deg_tick"] = [round(float(np.median((S[:, j] / TICK_DEG) % 1.0)), 3) for j in range(S.shape[1])]
    # smallest positive step between distinct state values per joint
    steps = []
    for j in range(S.shape[1]):
        u = np.unique(np.round(S[:, j], 4)); dd = np.diff(u); dd = dd[dd > 1e-4]
        steps.append(round(float(np.percentile(dd, 5)) if len(dd) else float("nan"), 4))
    out["state_min_step_p5"] = steps
    out["frac_state_eq_action"] = float(np.mean(np.all(np.isclose(A, S, atol=1e-6), axis=1)))
    # raw tracking error per joint (no sim), lag 0 and best lag, computed per episode then pooled
    r0 = np.zeros(A.shape[1]); n = 0; lagres = [[] for _ in range(A.shape[1])]
    for e, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(float); s = np.stack(g["observation.state"].to_numpy()).astype(float)
        r0 += ((s - a) ** 2).sum(0); n += len(a)
        for j in range(a.shape[1]): lagres[j].append(best_lag(a[:, j], s[:, j])[0])
    out["rms_state_minus_action_lag0"] = np.sqrt(r0 / n).round(2).tolist()
    out["median_best_lag_frames"] = [int(np.median(l)) for l in lagres]
    out["mean_abs_state_minus_action"] = np.abs(S - A).mean(0).round(2).tolist()
    out["median_state_minus_action"] = np.median(S - A, 0).round(2).tolist()
    # motion size per joint (std of action) to judge 'large free-space motion'
    out["action_std"] = A.std(0).round(1).tolist()
    return out

if __name__ == "__main__":
    for d in sys.argv[1:]:
        o = check(d)
        json.dump(o, open(os.path.join(d, "sanity.json"), "w"), indent=1)
        print("\n=====", os.path.basename(os.path.normpath(d)))
        for k, v in o.items():
            if k in ("dir", "columns"): v = v if k == "dir" else ",".join(v)
            print(f"  {k}: {v}")
