"""Write results/cloth_table.md from the cached stage outputs (python cloth_fit.py --stage report)."""
from __future__ import annotations

import json
import os

import numpy as np

import coltraro_data as CD

HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILE = os.environ.get("CLOTH_SYSID_PROFILE", "paper")
RES = os.path.join(HERE, "results") if _PROFILE == "paper" else os.path.join(HERE, "results", _PROFILE)


def _j(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _fmt(x, n=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def write():
    L = ["# Cloth step 1: clothilde-sim vs Coltraro free-hanging mocap", ""]
    proto, fit, pre, fin, flo = (_j("protocol.json"), _j("fitted_rep1.json"), _j("prereg.json"), _j("final.json"),
                                 _j("floors.json"))
    p26 = _j("protocol_clothilde2026.json")
    L += ["Protocol: paper (Coltraro AMM 2022 / 2024, MMT 2025): two upper corners scripted along their recorded paths, "
          "marker grid refined x2 (A2 7x9, A3 5x7), implicit Euler, one step per frame at dt = 0.01 s (recording "
          "linearly resampled), projection tolerance 0.1 %, bending 0, inextensible, only alpha (damping) and delta "
          "(virtual mass) free. clothilde-sim commit 383987e.", ""]
    if proto:
        vs = proto["variant_scores"]
        L += ["## 1. Protocol check: the paper's own (alpha, delta) on the 32 train recordings", "",
              "| Metric variant | median ours / paper | median abs log ratio |", "|---|---|---|"]
        for v, s in vs.items():
            L.append(f"| {v} | {_fmt(s['median_ratio'])} | {_fmt(s['median_abs_log_ratio'])} |")
        L += ["", f"- Chosen metric (pre-registered rule): **{proto['chosen_variant']}**.",
              f"- Weight-free spatial std, ours / paper, median: **{_fmt(proto['std_median_ratio'])}** "
              f"(pre-registered limit 1.25): harness check **{'PASS' if proto['harness_valid'] else 'FAIL'}**.",
              f"- Speed: {_fmt(proto['wall_per_sim_s_median'])} s wall per simulated second (one core).", ""]
    if p26:
        L += [f"- Same params under clothilde's 2026 defaults (trapezoidal, 10 sub-steps, 0.75 % tol): "
              f"metric ratio {_fmt(p26['ratio_chosen'])}, std ratio {_fmt(p26['std_median_ratio'])}, "
              f"{p26['n_nan']} runs blew up.", ""]
    if fit:
        s = fit["summary"]
        L += ["## 2. Reproduction: per-recording in-sample sweep on the train session (paper sec. 5.4)", "",
              "| Group | ours | paper, same recordings |", "|---|---|---|"]
        for k in ["polyester", "wool", "denim", "cotton", "A2", "A3", "shake", "twist", "slow", "fast", "global"]:
            L.append(f"| {k} | {_fmt(s.get(k))} | {_fmt(s['paper'].get(k))} |")
        L += ["", f"Pass check (global <= 1.25 x paper and A2 within 25 % of paper): **{'PASS' if s['pass'] else 'FAIL'}**.", ""]
        rms = [v["rms_mm"] for v in fit["fitted"].values()]
        L += [f"Plain marker RMS at the fitted params: median {_fmt(float(np.median(rms)), 1)} mm, "
              f"range {_fmt(min(rms), 1)}-{_fmt(max(rms), 1)} mm.", ""]
    if fin:
        L += ["## 3. Held-out session (*_hands files = paper repetition I), scored once", "",
              f"Frozen params sha256 `{pre['sha256'][:16]}…` at {pre['frozen_at']}.", "",
              "| Model | marker RMS mm (median) | mean | paper metric (mean) | runs |", "|---|---|---|---|---|"]
        by = {}
        for r in fin["runs"]:
            by.setdefault(r.get("tag"), []).append(r)
        var = fit["variant"] if fit else "lit_marker"
        for tag in ["rigid", "defaults", "apriori_II", "ours_rep1", "paper_insample_held", "ours_insample_II"]:
            rs = [r for r in by.get(tag, []) if "error" not in r]
            if not rs:
                continue
            mm = [r["rms_mm"] for r in rs if np.isfinite(r["rms_mm"])]
            pm = [r.get(var) for r in rs if r.get(var) is not None]
            L.append(f"| {tag} | {_fmt(float(np.median(mm)), 1)} | {_fmt(float(np.mean(mm)), 1)} | "
                     f"{_fmt(float(np.mean(pm)) if pm else None)} | {len(rs)} |")
        L.append("")
    if flo:
        rr = flo["real_repeat_selfcollision"]
        L += ["## 4. Floors", "",
              f"- Mocap still-marker jitter (first 0.25 s, upper bound): median {_fmt(flo['marker_still_jitter_mm']['median'], 2)} mm.",
              f"- Real repeat (Self-collisions rep1-3, hand-released; symmetric marker-set Chamfer): start states differ by {_fmt(rr['start_rms_mm_median'], 1)} mm "
              f"Chamfer, end states by {_fmt(rr['end_rms_mm_median'], 1)} mm (median over {len(rr['per_case'])} cases).", ""]
        runs = flo["sim_runs"]
        L += ["| Recording | base mm | chaos spread mm (10 nudges, max-min) | sub-steps 20 | sub-steps 5 | tol/2 | refine 3 | refine 4 |",
              "|---|---|---|---|---|---|---|---|"]
        for name in sorted({r["name"] for r in runs if "name" in r}):
            rs = {r["tag"]: r for r in runs if r.get("name") == name and "error" not in r}
            ch = [rs[k]["rms_mm"] for k in rs if k.startswith("chaos")]
            L.append(f"| {name} | {_fmt(rs.get('base', {}).get('rms_mm'), 2)} | {_fmt(max(ch) - min(ch), 3) if ch else '–'} | "
                     + " | ".join(_fmt(rs.get(k, {}).get("rms_mm"), 2) for k in ["sub20", "sub5", "tol_half", "refine3", "refine4"]) + " |")
        L.append("")
    with open(os.path.join(RES, "cloth_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
