"""Contact parameters for the cloth sim: SO-101 gripper squeeze from public LeRobot cloth logs, plus jaw/cloth friction.

Part 1 (measured, public logs). Every LeRobot SO-101 dataset logs `action` (leader = command) and `observation.state`
(follower = measured), gripper in 0..100 (0 = closed). While the jaws hold cloth, the follower stalls above the command
by a steady error e = measured - commanded; with empty jaws e is small (jaw-on-jaw).
  deg/unit  0.088 deg per encoder tick / smallest nonzero step of the measured gripper value
  hold      >= 1 s where the command sits in the bottom 3 units of that episode's command range and stays within a
            0.5 unit band; e = median(measured - commanded) over the hold minus its first 0.25 s (jaw settling)
  clusters  per dataset/gripper, 1D 2-means on hold e (Lloyd, init at the min/max); low cluster = empty closure,
            high cluster = cloth. Offset = cloth p50 - empty p50 in degrees (cancels each dataset's calibration zero)
  pull-out  cloth hold in which the measured value drops by > 1 unit within 0.5 s (command is stable by construction)

Part 2 (derived): static squeeze from the error with the BAM STS3215 12 V M6 model (bam/params/feetech_sts3215_12V/m6.json,
values copied below) and LeRobot's SO-101 gripper settings (P_Coefficient 16, Max_Torque_Limit 500 = 50 % drive cap):
  drive    d   = clip(err_rad * P * 0.166 * error_gain_ratio, 0, 0.5)               (bam/feetech/actuator.py)
  motor    tm  = kt / R * 12 V * d                                                    (bam/actuator.py, dq = 0)
  squeeze  S   = tm - f(tm, te = -S), the jaw stalled while closing, so friction opposes the motor (lower band edge).
           M6 friction at rest (bam/model.py, stribeck coeff = 1, motor > 0, external = -S < 0, |S| < tm):
             f = fb + fs + (S*lfe + tm*lfm) + (S*lfe_s + tm*lfm_s) + lfe_q * S^2
           so lfe_q*S^2 + (1 + lfe + lfe_s)*S - (tm*(1 - lfm - lfm_s) - fb - fs) = 0, positive root, S = 0 if none
  tip      F   = S / lever (0.080 m main; 0.060 and 0.090 reported)
  M1 (constant 0.2 N.m Coulomb, m1.json) is reported alongside as a friction-model sensitivity.
  Overload_Torque 25: after 2 s above 25 % drive the servo falls to the 20 % protective level; reported as a variant.

Stages:
  fetch    download meta/info.json + data/*.parquet (never videos) into --root, one folder per repo
  run      analyse --root, write results/contact/contact_params.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "contact", "contact_params.json")

DATASETS = [
    "jhimmens/linique-v2-fold-pickup",
    "open-cloth/so101-full-fold-merged",
    "skylarklerobot/two-towel-merged",
    "Janmeier820/align_fabric_dataset_combined",
    "gaspardthrl/walleed_fold_combined_with_vincent_random",
    "CoRL2026-CSI/SO101-DualArm-teleop_fold_towel_100epi",
    "Sengankou/so101_fold_towel",
    "villekuosmanen/armnetbench_fold_tea_towel",
    "KirillZubarev/T-shirt_dataset_20260830_154211",
    "Zekai-Chen/bi_so101_fold_towel_merged_r1",
]
MAX_EPISODES = {"Zekai-Chen/bi_so101_fold_towel_merged_r1": 150}  # large set: first ~150 episodes only
MAX_FILES = {"Zekai-Chen/bi_so101_fold_towel_merged_r1": 1}

ENC_DEG = 0.088      # one STS3215 encoder tick, deg
CLOSED_BAND = 3.0    # units above the episode's minimum command
STABLE = 0.5         # units, max - min of the command over a hold
HOLD_MIN_S = 1.0
SETTLE_S = 0.25
PULL_DROP = 1.0      # units
PULL_WIN_S = 0.5
VALLEY_MAX = 0.8   # clean 2-cluster split: smoothed density at the split at least 20 % below the smaller peak

# BAM M6, feetech_sts3215_12V/m6.json
M6 = dict(kt=2.261727383018318, R=4.158130587333767, gain=1.0737235827288294, fb=0.14906617022857369,
          fs=3.607359260500659e-08, lfm=2.2651030687538048e-08, lfe=0.15470384321887548,
          lfm_s=0.14451270849433837, lfe_s=0.2571230219372956, lfe_q=0.0006893797992811467)
# BAM M1, feetech_sts3215_12V/m1.json (constant Coulomb friction)
M1 = dict(kt=2.2261708626710552, R=4.061801280762969, gain=1.0438063094224284, fb=0.19999999999986195)
VIN, KP, ERR_GAIN, DRIVE_CAP, PROTECT_DRIVE, OVERLOAD_DRIVE, OVERLOAD_S = 12.0, 16, 0.166, 0.5, 0.20, 0.25, 2.0
LEVERS = (0.060, 0.080, 0.090)
MAP_DEG = (0.5, 1, 2, 3, 5, 7.5, 10)


def sig(x, n=3):
    """Round to n significant figures (recursively through lists/dicts)."""
    if isinstance(x, dict):
        return {k: sig(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{n}g}") if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def pct(x, qs=(10, 50, 90)):
    x = np.asarray(x, float)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {f"p{q}": None for q in qs}


# ----------------------------------------------------------------------------------------------- fetch

def folder(root, repo):
    return os.path.join(root, repo.replace("/", "__"))


def fetch(root):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    for repo in DATASETS:
        try:
            hf_hub_download(repo, "meta/info.json", repo_type="dataset", local_dir=folder(root, repo))
            files = sorted(f.path for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="data",
                                                              recursive=True) if f.path.endswith(".parquet"))
            files = [f for f in files if os.path.basename(f).startswith("file-")][:MAX_FILES.get(repo, 10**6)]
            for f in files:
                hf_hub_download(repo, f, repo_type="dataset", local_dir=folder(root, repo))
            print(f"{repo}: {len(files)} parquet files")
        except Exception as e:  # rate limit / missing repo: skip, reported in the output
            print(f"skip {repo}: {type(e).__name__}")


# ----------------------------------------------------------------------------------------------- load

def load(root, repo):
    """fps, {gripper name: (command, measured)}, episode index per frame."""
    import pyarrow.parquet as pq
    d = folder(root, repo)
    info = json.load(open(os.path.join(d, "meta", "info.json")))
    files = sorted(glob.glob(os.path.join(d, "data", "chunk-*", "file-*.parquet")))
    tabs = [pq.read_table(f, columns=["action", "observation.state", "episode_index"]) for f in files]
    act = np.concatenate([np.array(t.column("action").to_pylist(), float) for t in tabs])
    st = np.concatenate([np.array(t.column("observation.state").to_pylist(), float) for t in tabs])
    ep = np.concatenate([t.column("episode_index").to_numpy() for t in tabs])
    keep = np.isin(ep, np.unique(ep)[:MAX_EPISODES.get(repo, 10**9)])
    a_names = info["features"]["action"]["names"]
    s_names = info["features"]["observation.state"]["names"]
    grips = {}
    for i, n in enumerate(a_names):
        if "gripper" in n:
            grips[n.replace(".pos", "")] = (act[keep, i], st[keep, s_names.index(n)])
    return float(info["fps"]), grips, ep[keep]


# ----------------------------------------------------------------------------------------------- part 1

def deg_per_unit(meas, ep):
    """0.088 deg / smallest nonzero step of the measured value (steps taken within episodes)."""
    same = ep[1:] == ep[:-1]
    dm = np.abs(np.diff(meas))[same]
    dm = dm[dm > 1e-4]
    step = float(dm.min())
    lattice = float(np.mean(np.abs(dm / step - np.round(dm / step)) < 0.05))  # share of steps on the tick lattice
    return ENC_DEG / step, step, lattice


def stable_runs(idx, cmd):
    """Split consecutive frame indices into stretches whose command stays inside a STABLE band."""
    runs, s = [], 0
    lo = hi = cmd[idx[0]]
    for k in range(1, len(idx)):
        c = cmd[idx[k]]
        lo2, hi2 = min(lo, c), max(hi, c)
        if hi2 - lo2 >= STABLE:
            runs.append(idx[s:k])
            s, lo2, hi2 = k, c, c
        lo, hi = lo2, hi2
    runs.append(idx[s:])
    return runs


def settled(m, nset):
    """First index from which the measured value stays inside a STABLE band for SETTLE_S (jaw done closing)."""
    for k in range(0, len(m) - nset):
        w = m[k:k + nset]
        if w.max() - w.min() < STABLE:
            return k
    return len(m)


def holds(cmd, meas, ep, fps):
    """Closed, stable command stretches, measured part counted only after the jaw settled, >= HOLD_MIN_S.
    Returns list of dicts: e (median measured - commanded), dur (s), drop (largest measured fall within
    PULL_WIN_S, units), low (min measured - commanded after that fall)."""
    out = []
    nmin, nset, nwin = int(round(HOLD_MIN_S * fps)), int(round(SETTLE_S * fps)), int(round(PULL_WIN_S * fps))
    for e_id in np.unique(ep):
        ix = np.flatnonzero(ep == e_id)
        c = cmd[ix]
        closed = c <= c.min() + CLOSED_BAND
        edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))  # consecutive closed frames
        for a, b in zip(edges[::2], edges[1::2]):
            for run in stable_runs(np.arange(a, b), c):
                g = ix[run]
                g = g[settled(meas[g], nset):]
                if len(g) < nmin:
                    continue
                m, d = meas[g], meas[g] - cmd[g]
                drops = [m[:-k] - m[k:] for k in range(1, min(nwin, len(m) - 1) + 1)]
                best = max(range(len(drops)), key=lambda j: drops[j].max())
                i = int(np.argmax(drops[best]))
                out.append(dict(e=float(np.median(d)), dur=len(g) / fps, drop=float(drops[best][i]),
                                low=float(d[i + best + 1:].min())))
    return out


def valley_ratio(x, t, bw=0.15):
    """Smoothed density at the split / the smaller of the two cluster peaks (< 0.5 = clean bimodal split)."""
    grid = np.linspace(x.min() - 1, x.max() + 1, 600)
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1)
    lo, hi = dens[grid <= t].max(), dens[grid > t].max()
    return float(dens[np.argmin(np.abs(grid - t))] / min(lo, hi))


def two_means(x, iters=100):
    """1D 2-means; returns threshold between the clusters."""
    x = np.sort(np.asarray(x, float))
    c0, c1 = x[0], x[-1]
    for _ in range(iters):
        t = 0.5 * (c0 + c1)
        lo, hi = x[x <= t], x[x > t]
        if not len(lo) or not len(hi):
            break
        n0, n1 = lo.mean(), hi.mean()
        if n0 == c0 and n1 == c1:
            break
        c0, c1 = n0, n1
    return 0.5 * (c0 + c1)


# ----------------------------------------------------------------------------------------------- part 2

def drive(err_deg, gain):
    return np.clip(np.deg2rad(err_deg) * KP * ERR_GAIN * gain, 0.0, DRIVE_CAP)


def squeeze_m6(tm):
    """Static squeeze torque (N.m) at the lower edge of the M6 friction band, solved self-consistently in S."""
    p = M6
    a = p["lfe_q"]
    b = 1.0 + p["lfe"] + p["lfe_s"]
    c = p["fb"] + p["fs"] - tm * (1.0 - p["lfm"] - p["lfm_s"])
    s = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return np.maximum(s, 0.0)


def squeeze(err_deg, model="m6", d=None):
    p = M6 if model == "m6" else M1
    d = drive(err_deg, p["gain"]) if d is None else d
    tm = p["kt"] / p["R"] * VIN * d
    return squeeze_m6(tm) if model == "m6" else np.maximum(tm - p["fb"], 0.0)


def force(err_deg, lever=0.080, model="m6", protected=None):
    """Tip force (N). protected: bool array, holds whose drive falls to PROTECT_DRIVE (> OVERLOAD_S above 25 %)."""
    p = M6 if model == "m6" else M1
    d = drive(np.asarray(err_deg, float), p["gain"])
    if protected is not None:
        d = np.where(protected & (d > OVERLOAD_DRIVE), PROTECT_DRIVE, d)
    return squeeze(err_deg, model, d) / lever


def map_table():
    rows = {}
    for e in MAP_DEG:
        d = float(drive(e, M6["gain"]))
        rows[f"{e:g}"] = dict(drive=d, motor_Nm=M6["kt"] / M6["R"] * VIN * d, squeeze_Nm=float(squeeze(e)),
                              **{f"N_{int(L * 1000)}mm": float(force(e, L)) for L in LEVERS},
                              N_80mm_m1=float(force(e, 0.080, "m1")))
    return rows


# ----------------------------------------------------------------------------------------------- run

def analyse(root):
    per, stop, pull, pooled, skipped = {}, {}, {}, [], []
    for repo in DATASETS:
        try:
            fps, grips, ep = load(root, repo)
        except Exception as ex:
            skipped.append(f"{repo}: {type(ex).__name__}")
            continue
        for g, (cmd, meas) in grips.items():
            key = f"{repo}:{g}" if len(grips) > 1 else repo
            dpu, step, lattice = deg_per_unit(meas, ep)
            hs = holds(cmd, meas, ep, fps)
            e = np.array([h["e"] for h in hs])
            t = two_means(e)
            vr = valley_ratio(e, t)
            emp, clo = e <= t, e > t
            cl = [h for h, k in zip(hs, clo) if k]
            e_deg = e[clo] * dpu
            prot = np.array([h["dur"] > OVERLOAD_S for h in cl])
            pulled = np.array([h["drop"] > PULL_DROP for h in cl])
            to_empty = np.array([h["drop"] > PULL_DROP and h["low"] <= t for h in cl])
            f80 = force(e_deg)
            off = (np.median(e[clo]) - np.median(e[emp])) * dpu
            per[key] = dict(
                fps=fps, episodes=int(len(np.unique(ep))), step_units=step, deg_per_unit=dpu,
                step_lattice_share=lattice, holds=len(hs), empty_n=int(emp.sum()), cloth_n=int(clo.sum()),
                split_units=t, valley_ratio=vr, clean_split=bool(vr < VALLEY_MAX),
                empty_p50_units=float(np.median(e[emp])), empty_p50_deg=float(np.median(e[emp]) * dpu),
                cloth_units=pct(e[clo]), cloth_deg=pct(e_deg), cloth_minus_empty_deg=off,
                N_80mm=pct(f80), N_60mm=pct(force(e_deg, 0.060)), N_90mm=pct(force(e_deg, 0.090)),
                N_80mm_overload_protected=pct(force(e_deg, protected=prot)),
                cloth_holds_over_2s_and_25pct=int(np.sum(prot & (drive(e_deg, M6["gain"]) > OVERLOAD_DRIVE))),
                pullout_rate=float(pulled.mean()), pullout_to_empty_rate=float(to_empty.mean()))
            stop[key] = dict(deg=off, clean_split=per[key]["clean_split"])
            pull[key] = dict(rate=float(pulled.mean()), to_empty_rate=float(to_empty.mean()), cloth_holds=len(cl),
                             pulled=int(pulled.sum()), pulled_to_empty=int(to_empty.sum()))
            pooled.append(dict(e_deg=e_deg, prot=prot, clean=per[key]["clean_split"]))
            print(f"{key:66s} dpu {dpu:.3f} holds {len(hs):4d} emp {emp.sum():4d} cloth {clo.sum():4d} "
                  f"split {t:5.2f}u vr {vr:.2f} e_emp {np.median(e[emp]):5.2f}u cloth p10/50/90 "
                  f"{np.percentile(e[clo], 10):5.2f}/{np.median(e[clo]):5.2f}/{np.percentile(e[clo], 90):5.2f}u "
                  f"off {off:5.2f}deg N80 p50 {np.median(f80):5.2f} pull {pulled.mean():.2f}/{to_empty.mean():.2f}")
    return per, stop, pull, pooled, skipped


MU = {
    "jaw_pla_cloth": {"cotton": [0.07, 0.25], "cotton_central": [0.10, 0.15], "polyester": [0.06, 0.20],
                      "jersey": [0.08, 0.28], "terry": [0.10, 0.35],
                      "tag": "derived: pressure law n~0.80 anchored on Rasheed CVPR2020 PET film, Hermann 2004 PMMA, "
                             "Tribology Letters 2021 polycarbonate; ceiling from Aliouche 2000 Al/PMMA"},
    "cloth_table": {"range": [0.34, 0.73],
                    "tag": "measured incline, Rasheed CVPR2020 (hard smooth surfaces, self-weight pressure)"},
    "cloth_cloth": {"range": [0.62, 0.97], "tag": "measured sled, JTATM 2015 / Aliouche 2000"},
}


def build(per, stop, pull, pooled, skipped):
    def cat(k, clean_only):
        return np.concatenate([p[k] for p in pooled if p["clean"] or not clean_only])
    pooled_e, pooled_prot = cat("e_deg", True), cat("prot", True)
    all_e = cat("e_deg", False)
    pooled_N = pct(force(pooled_e))
    stops = np.array([v["deg"] for v in stop.values() if v["clean_split"]])
    all_pull = sum(p["pulled"] for p in pull.values())
    all_empty = sum(p["pulled_to_empty"] for p in pull.values())
    all_cloth = sum(p["cloth_holds"] for p in pull.values())
    return sig({
        "squeeze": {
            "model": "BAM M6 STS3215 12V, P16, 50% cap",
            "formula": "d = clip(err_rad*16*0.166*1.0737, 0, 0.5); tm = kt/R*12V*d; S solves "
                       "lfe_q*S^2 + (1+lfe+lfe_s)*S = tm*(1-lfm-lfm_s) - fb - fs (M6 friction at rest, lower band "
                       "edge, external = -S); F = S/lever",
            "map_deg_to_N_at_80mm": {k: v["N_80mm"] for k, v in map_table().items()},
            "map_table": map_table(),
            "per_dataset": per,
            "pooled_over": "cloth holds of the clean-split datasets/grippers (clean_split true)",
            "pooled_cloth_holds": len(pooled_e),
            "pooled_err_deg": pct(pooled_e),
            "pooled_N": pooled_N,
            "pooled_N_60mm": pct(force(pooled_e, 0.060)),
            "pooled_N_90mm": pct(force(pooled_e, 0.090)),
            "pooled_N_overload_protected": pct(force(pooled_e, protected=pooled_prot)),
            "pooled_N_m1_friction": pct(force(pooled_e, model="m1")),
            "pooled_all_datasets": {"cloth_holds": len(all_e), "err_deg": pct(all_e), "N": pct(force(all_e))},
            "rl_range_N": [pooled_N["p10"], pooled_N["p90"]],
            "rl_range_rule": "pooled cloth-hold p10..p90 at 80 mm lever, M6, 50% cap, clean-split datasets",
            "tag": "derived (public logs + bench servo model)",
            "caveats": [
                "dataset arms' servo voltage is unknown; a 7.4 V STS3215 gives about half the torque per degree, so "
                "the same error on a 7.4 V arm means roughly half these forces; our arm is 12 V",
                "e = measured - commanded taken as the servo's position error; any calibration offset between leader "
                "command and follower target (the empty-closure error) is inside e",
                "static lower friction-band edge (jaw stalled while closing); squeeze can sit anywhere up to "
                "tm + f, M1 constant-friction numbers given as the sensitivity",
                "Overload_Torque 25: holds > 2 s above 25 % drive fall to 20 % drive; protected variant reported",
                "lever 80 mm assumed contact point; 60/90 mm reported",
                "cloth/empty split by 1D 2-means per dataset/gripper; splits with valley_ratio >= 0.8 (no dip between "
                "the clusters, KDE bandwidth 0.15 unit) are reported but left out of the main pooled numbers",
                "holds where the operator did not fully close the leader still count as closed if within 3 units "
                "of the episode minimum",
            ],
        },
        "stop_angle": {
            "per_dataset_cloth_minus_empty_deg": stop,
            "pooled_p10_p50_p90_deg": [float(np.percentile(stops, q)) for q in (10, 50, 90)],
            "pooled_over": "one offset per clean-split dataset/gripper",
            "tag": "measured (public logs)",
        },
        "pullout_rate": {"per_dataset": pull, "pooled": all_pull / all_cloth if all_cloth else None,
                         "pooled_to_empty": all_empty / all_cloth if all_cloth else None,
                         "pooled_cloth_holds": all_cloth,
                         "rule": "measured drops > 1 unit within 0.5 s during a settled cloth hold, command stable; "
                                 "to_empty: and the error afterwards falls to the empty-cluster side of the split",
                         "tag": "measured (public logs)"},
        "mu": MU,
        "skipped": skipped,
        "settings": dict(enc_deg=ENC_DEG, closed_band_units=CLOSED_BAND, stable_units=STABLE, hold_min_s=HOLD_MIN_S,
                         settle_s=SETTLE_S, valley_max=VALLEY_MAX, pull_drop_units=PULL_DROP, pull_window_s=PULL_WIN_S,
                         max_episodes=MAX_EPISODES),
        "sources": {
            "datasets": [f"https://huggingface.co/datasets/{r}" for r in DATASETS],
            "bam": "https://github.com/Rhoban/bam (params/feetech_sts3215_12V/m6.json, m1.json; bam/model.py, "
                   "bam/feetech/actuator.py)",
            "lerobot_so101_gripper": "https://github.com/huggingface/lerobot (robots/so_follower/so_follower.py: "
                                     "P_Coefficient 16, Max_Torque_Limit 500, Overload_Torque 25)",
            "mu": ["Rasheed et al. CVPR 2020 (cloth friction, incline)", "Hermann 2004 (PMMA/fabric)",
                   "Tribology Letters 2021 (polycarbonate/fabric)", "Aliouche & Viallier 2000 (fabric friction)",
                   "JTATM 2015 (fabric-fabric sled)"],
        },
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "run"])
    ap.add_argument("--root", required=True, help="folder with one sub-folder per downloaded dataset")
    args = ap.parse_args()
    if args.stage == "fetch":
        fetch(args.root)
        return
    out = build(*analyse(args.root))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("pooled N (80 mm):", out["squeeze"]["pooled_N"], "rl_range", out["squeeze"]["rl_range_N"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
