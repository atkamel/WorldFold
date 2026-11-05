"""Report for fit.py (step 4): results/fit_table.md, results/fit_params_<name>.json, so101_fitted.json,
results/fit_example.png. Reads results/fit_cache/*.json written by fit.py. The model written to
so101_fitted.json is chosen on TRAIN (all TRAIN windows, W=4), never on TEST."""
from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import fit  # noqa: E402

C = fit.CACHE
R = fit.RESULTS
JOINTS = fit.JOINTS
ARM = fit.ARM
PRIMARY = fit.PRIMARY


def _l(name):
    p = os.path.join(C, name)
    return fit.jload(p) if os.path.exists(p) else None


def f2(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


LABEL = {
    "baseline:perfect_tracking": "perfect_tracking (state[t] = action[t-1], clipped to the stops)",
    "baseline:best_pure_delay": "best_pure_delay (fractional delay fit on TRAIN, clipped to the stops)",
    "baseline:best_pure_delay_unclipped": "best_pure_delay, unclipped",
    "baseline:perfect_tracking_unclipped": "perfect_tracking, unclipped",
    "before:A_stock": "A_stock (Menagerie)",
    "before:B_nvidia": "B_nvidia (NVIDIA SysID)",
    "before:C_bam_m1": "C_bam_m1 (BAM M1, 7.4V set, vin 7.4)",
    "before:D_bam_m6": "D_bam_m6 (BAM M6, 7.4V set, vin 7.4)",
    "before:D_bam_m6_vin5": "BAM M6, 7.4V set, vin 5.0",
    "before:M6_12Vset_vin12": "BAM M6, 12V set (PR#20), vin 12",
    "after:fit1": "FIT 1: B_nvidia + 25 per-joint multipliers + delta",
    "after:fit1@shared_delta": "FIT 1 params at the shared before-fit delta",
    "after:fit1_refit": "FIT 1 re-polished after offset refinement",
    "after:fit1db": "FIT 1 + firmware deadband (5 params)",
    "after:fit2_7.4V": "FIT 2: BAM M6 7.4V set, vin + per-joint friction/gain scales + delta",
    "after:fit2_12V": "FIT 2: BAM M6 12V set, vin + per-joint friction/gain scales + delta",
}


def row_keys(final):
    order = ["baseline:perfect_tracking", "baseline:best_pure_delay"]
    order += [k for k in final if k.startswith("before:")]
    order += [k for k in final if k.startswith("before_owndelta:")]
    order += [k for k in final if k.startswith("after:")]
    order += ["baseline:best_pure_delay_unclipped", "baseline:perfect_tracking_unclipped"]
    return [k for k in order if k in final]


def label(k):
    if k.startswith("before_owndelta:"):
        b = "before:" + k.split(":", 1)[1]
        return LABEL.get(b, b) + " at its own TRAIN-best delta"
    return LABEL.get(k, k)


def group(k):
    if k.startswith("baseline"):
        return "baseline (no physics)"
    if k.startswith("before_owndelta"):
        return "before fit, own delta"
    if k.startswith("before"):
        return "before fit, shared delta"
    return "after fit"


def cell(final, k, split, sub="OVERALL"):
    try:
        return final[k][split][sub]["arm"]
    except (KeyError, TypeError):
        return None


def best_by_train(final, prefix="after:"):
    cands = [(cell(final, k, "train4"), k) for k in final if k.startswith(prefix) and "@shared" not in k
             and cell(final, k, "train4") is not None]
    return min(cands)[1] if cands else None


# ------------------------------------------------------------------------------------------------
# parameter files
# ------------------------------------------------------------------------------------------------
def pos_params_json(name, f, prep, final, key):
    base = f["base"]
    p = f["params"]
    joints = {}
    for j, jn in enumerate(JOINTS):
        row = {}
        for k, unit in (("kp", "N*m/rad"), ("kv", "N*m*s/rad"), ("damping", "N*m*s/rad"), ("frictionloss", "N*m"),
                        ("armature", "kg*m^2"), ("forcerange", "N*m")):
            row[k] = {"fitted": p[k][j], "start_B_nvidia": base[k][j],
                      "multiplier": (p[k][j] / base[k][j]) if base[k][j] else None, "unit": unit}
            if k == "forcerange":
                row[k]["note"] = "not fitted (kept at NVIDIA 3.35)"
            if jn == "gripper":
                row[k]["note"] = "gripper not fitted (contacts off, jaw-stall frames dominate); B_nvidia value"
        if f.get("deadband_deg") is not None:
            row["deadband"] = {"fitted_deg": f["deadband_deg"][j], "unit": "deg",
                               "law": "position target per physics step: q + sign(e)*max(|e|-db, 0), e = target - q"}
        joints[jn] = row
    out = {
        "name": name, "kind": "MuJoCo position actuator (PD) + joint damping/frictionloss/armature",
        "start": "B_nvidia (NVIDIA Isaac Sim 6.1 so101_new_calib SysID, converted to SI per rad in models.py)",
        "fitted_on": "TRAIN episodes (episode_index % 4 != 3) of the 6 primary datasets; CMA-ES (cma 4.5.0) on a fixed "
                     f"random subset of {fit.N_FIT_WIN} W=4 s TRAIN windows per dataset; equal-weight mean of per-dataset "
                     "arm-mean pooled RMSE",
        "cma": {"objective_train_fit_subset_deg": f["objective_train_fit_subset"],
                "start_objective_deg": f["start_objective"], "generations": len(f["history"]), "stop": f["stop"]},
        "joints": joints,
        "nuisance_per_dataset": {dn: {
            "delta_s": f["deltas_s"][dn],
            "offset_deg": dict(zip(JOINTS, f.get("offsets_deg", {}).get(dn, prep["datasets"][dn]["fk_offsets_deg"]))),
            "joint_lo_deg_data_coords": prep["datasets"][dn]["lo_deg"],
            "joint_hi_deg_data_coords": prep["datasets"][dn]["hi_deg"],
        } for dn in PRIMARY},
        "nuisance_meaning": {
            "delta_s": "sub-frame transport delay: action[t] reaches the servo delta s after state[t] is read "
                       "(LeRobot loop: read follower, cameras, leader, then send). Fitted jointly with the model on TRAIN; "
                       "trades off against the model's own lag (not separately identifiable from these data).",
            "offset_deg": "constant zero offset added to LeRobot degrees (state and action) -> sim joint angle; "
                          "shoulder_lift/elbow/wrist_flex/wrist_roll; FK flat-table test + TRAIN tracking (see fit_table.md)",
            "joint_lo/hi": "real hard stops detected on TRAIN states (sim joint limit + target clamp); non-stop ends "
                           "are 10 deg beyond CAD and the observed extreme (never binding)",
        },
        "results_deg": {s: final.get(key, {}).get(s) for s in ("train4", "test4", "test1")},
    }
    return out


def bam_params_json(name, run, prep, final, key, offsets):
    cfg = run["cfg"]
    import models
    pset_dir = "feetech_sts3215_7_4V" if cfg["set"] == "7.4V" else "feetech_sts3215_12V"
    path = models.bam_params_path("m6", pset_dir)
    base = json.load(open(path))
    kt, Rm, eg, pwm = base["kt"], base["R"], 0.166, 0.97
    joints = {}
    for j, jn in enumerate(JOINTS):
        egr = base["error_gain_ratio"] * cfg["egr_scale"][j]
        joints[jn] = {
            "error_gain_ratio": {"fitted": egr, "start": base["error_gain_ratio"], "multiplier": cfg["egr_scale"][j]},
            "friction_scale": {"fitted": cfg["fric_scale"][j], "start": 1.0,
                               "meaning": "multiplies BAM M6's whole friction budget (frictionloss + viscous) for this joint"},
            "equiv_position_kp_Nm_per_rad": eg * egr * 16 * cfg["vin"] * kt / Rm,
            "stall_torque_Nm": pwm * cfg["vin"] * kt / Rm,
            "backemf_damping_Nms_per_rad": kt * kt / Rm,
            "friction_base_Nm": base["friction_base"] * cfg["fric_scale"][j],
            "friction_viscous_Nms_per_rad": base["friction_viscous"] * cfg["fric_scale"][j],
        }
    return {
        "name": name, "kind": "Rhoban BAM STS3215 M6 servo model (BAM's own code, per physics step)",
        "bam_params_file": path, "bam_params_start": base, "firmware_kp": 16,
        "vin": {"fitted": cfg["vin"], "start": run["x0"][0]},
        "joints": joints,
        "unmodelled": "BAM was identified with firmware D=0; LeRobot writes D_Coefficient=32. Acceleration=254 "
                      "also not modelled.",
        "cma": {"objective_train_fit_subset_deg": run["objective_train_fit_subset"], "generations": len(run["history"]),
                "stop": run["stop"]},
        "nuisance_per_dataset": {dn: {"delta_s": run["deltas_s"][dn], "offset_deg": dict(zip(JOINTS, offsets[dn])),
                                      "joint_lo_deg_data_coords": prep["datasets"][dn]["lo_deg"],
                                      "joint_hi_deg_data_coords": prep["datasets"][dn]["hi_deg"]} for dn in PRIMARY},
        "results_deg": {s: final.get(key, {}).get(s) for s in ("train4", "test4", "test1")},
    }


# ------------------------------------------------------------------------------------------------
# domain randomisation
# ------------------------------------------------------------------------------------------------
def dr_ranges(spread, fbest, prep, offsets):
    if spread is None:
        return None
    per = spread["per_arm"]
    out = {}
    for name in fit.SPREAD_NAMES:
        vals = [per[dn]["multipliers_vs_pooled"][name] for dn in PRIMARY]
        if name == "delta":
            out[name] = {"per_arm_ms": [round(1000 * v, 1) for v in vals],
                         "recommended_ms": [0.0, round(max(33.3, 1000 * max(vals)), 1)],
                         "why": "delta is the loop latency between reading the state and the servo receiving the command; "
                                "the per-arm fits span this range and a deployed policy adds its own inference time"}
            continue
        lo, hi = min(vals), max(vals)
        rec = [round(min(lo, 0.8), 3), round(max(hi, 1.2), 3)]
        out[name] = {"per_arm_multipliers": [round(v, 3) for v in vals], "spread": [round(lo, 3), round(hi, 3)],
                     "recommended_multiplier": rec,
                     "why": ("per-arm spread is wider than +-20 %" if (lo < 0.8 or hi > 1.2) else
                             "per-arm spread is inside +-20 %, so the OmniReset-style +-20 % floor is used")}
    offs = np.array([offsets[dn] for dn in PRIMARY])
    out["joint_zero_offset_deg"] = {JOINTS[j]: {"per_arm": offs[:, j].round(2).tolist(),
                                                "recommended_abs_deg": round(max(1.0, float(np.abs(offs[:, j]).max())), 2)}
                                    for j in fit.OFFSET_JOINTS}
    return out


# ------------------------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------------------------
def plot_example(prep, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = fit.datasets()["dobri420"]
    wins = fit.make_windows(ds, "test", 4)
    N = 120
    # rule fixed before looking at any model error: grasp-free TEST window of dobri420 with the largest summed
    # commanded excursion over the 5 arm joints
    best, score = None, -1
    for ep, t0 in wins:
        e = ds["episodes"][ep]
        if e["grasp"][t0 + 1:t0 + N + 1].any():
            continue
        c = e["A"][t0:t0 + N, :ARM]
        sc = float(np.degrees(c.max(0) - c.min(0)).sum())
        if sc > score:
            best, score = (ep, t0), sc
    ep, t0 = best
    items = [("dobri420", ep, t0, 4)]
    preds = {}
    res = fit.run_many([rows[k] for k in rows], items, chunk=1, want_pred=True)
    for k, r in zip(rows, res):
        preds[k] = np.array(r[0]["pred_deg"])
    e = ds["episodes"][ep]
    S, A = np.degrees(e["S"]), np.degrees(e["A"])
    ts = t0 - fit.K_WARM
    t_obs = (np.arange(fit.K_WARM + N) + ts + 1) / fit.FPS
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eda100", "#eb6834", "#1baf7a", "#8a5cd6"]
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 12.5), sharex=True, facecolor=surf)
    for i, jn in enumerate(JOINTS[:ARM]):
        ax = axes[i]
        ax.set_facecolor(surf)
        ax.axvspan(ts / fit.FPS, t0 / fit.FPS, color=grid, alpha=0.6, lw=0)
        ax.step(np.arange(ts, t0 + N) / fit.FPS, A[ts:t0 + N, i], where="post", color=ink2, lw=1.1, ls="--",
                label="action (leader command)")
        ax.plot(np.arange(ts, t0 + N + 1) / fit.FPS, S[ts:t0 + N + 1, i], color=ink, lw=2.3, label="real observation.state")
        txt = []
        for c, (k, P) in zip(colors, preds.items()):
            ax.plot(t_obs, P[:, i], color=c, lw=1.5, label=k)
            err = P[fit.K_WARM:, i] - S[t0 + 1:t0 + N + 1, i]
            txt.append(f"{k.split(':')[-1]} {math.sqrt(np.mean(err ** 2)):.2f}")
        ax.set_ylabel(f"{jn}\n[deg]", color=ink2, fontsize=9)
        ax.set_title("window RMSE [deg]: " + "   ".join(txt), loc="right", fontsize=7.5, color=ink2, pad=3)
        ax.grid(True, color=grid, lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    axes[-1].set_xlabel("time in episode [s]  (grey band = 8-frame warm-up, not scored)", color=ink2)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"dobri420/pick-cube-so101 TEST episode {ep}, scored frames {t0 + 1}-{t0 + N}, W = 4 s, new protocol "
                 f"(open loop from the real state at frame {ts})", color=ink, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=125, facecolor=surf)
    plt.close(fig)
    return {"dataset": "dobri420", "episode": int(ep), "t0": int(t0), "selection": "grasp-free TEST window with the "
            "largest summed commanded excursion (chosen without looking at model errors)", "rows": list(rows)}


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def main():
    prep = _l("prep.json")
    before = _l("before.json")
    final = _l("final.json")
    f1 = _l("fit1.json")
    f1r = _l("fit1_refit.json")
    f1db = _l("fit1db.json")
    f2 = _l("fit2.json")
    offj = _l("offsets.json")
    spread = _l("spread.json")
    old = json.load(open(os.path.join(R, "baseline.json")))["results"]
    offsets = fit.fitted_offsets(prep)
    keys = row_keys(final)

    # ---------- parameter files
    written = []
    for name, f, key in (("fit1", f1, "after:fit1"), ("fit1_refit", f1r, "after:fit1_refit"), ("fit1db", f1db, "after:fit1db")):
        if f is not None:
            js = pos_params_json(name, f, prep, final, key)
            fit.jdump(js, os.path.join(R, f"fit_params_{name}.json"))
            written.append(f"fit_params_{name}.json")
    if f2 is not None:
        for pset, run in f2["runs"].items():
            nm = f"fit2_{pset}"
            fit.jdump(bam_params_json(nm, run, prep, final, f"after:{nm}", f2["offsets_deg"]),
                      os.path.join(R, f"fit_params_{nm}.json"))
            written.append(f"fit_params_{nm}.json")

    best_after = best_by_train(final)
    best_pos = best_by_train({k: v for k, v in final.items() if k in ("after:fit1", "after:fit1_refit", "after:fit1db")})
    fpos = {"after:fit1": f1, "after:fit1_refit": f1r, "after:fit1db": f1db}[best_pos]
    dr = dr_ranges(spread, fpos, prep, offsets)

    # ---------- standalone parameter file (PD model; chosen on TRAIN)
    sel_note = (f"Chosen on TRAIN (all TRAIN W=4 windows, OVERALL arm-mean): best position-actuator fit = {best_pos} "
                f"({f2(cell(final, best_pos, 'train4'), 3)} deg); best fit of any kind = {best_after} "
                f"({f2(cell(final, best_after, 'train4'), 3)} deg).")
    jp = {}
    for j, jn in enumerate(JOINTS):
        p = fpos["params"]
        jp[jn] = {"kp_Nm_per_rad": p["kp"][j], "kv_Nms_per_rad": p["kv"][j], "damping_Nms_per_rad": p["damping"][j],
                  "frictionloss_Nm": p["frictionloss"][j], "armature_kgm2": p["armature"][j],
                  "forcerange_Nm": p["forcerange"][j],
                  "deadband_rad": (math.radians(fpos["deadband_deg"][j]) if fpos.get("deadband_deg") else 0.0),
                  "fitted": jn != "gripper"}
    stops = {}
    for j in range(ARM):
        for end in ("lo", "hi"):
            v = [prep["datasets"][dn]["ranges"][JOINTS[j]][end]["limit_deg"] for dn in PRIMARY
                 if prep["datasets"][dn]["ranges"][JOINTS[j]][end]["stop"]]
            if v:
                stops.setdefault(JOINTS[j], {})[end] = {"observed_real_stop_deg_per_arm": v,
                                                        "cad_deg": float(fit.CAD_DEG[j, 0 if end == 'lo' else 1])}
    delta_vals = list(fpos["deltas_s"].values())
    sofit = {
        "name": "SO-101 follower arm, position-servo model fitted to real LeRobot data (step 4)",
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "selection": sel_note,
        "geometry_and_inertia": "MuJoCo Menagerie robotstudio_so101 (commit 62ab00f) = TheRobotStudio so101_new_calib "
                                "kinematics and CAD masses; unchanged by the fit",
        "actuator_law": "per joint: tau = clip(kp*(u - q) - kv*qdot, -forcerange, +forcerange) (MuJoCo position actuator "
                        "with forcelimited), plus joint viscous damping, dry friction 'frictionloss' and armature. "
                        "u = q + sign(e)*max(|e| - deadband, 0) with e = target - q if deadband_rad > 0 (applied every physics "
                        "step; no native deadband in MuJoCo/Newton). Targets are clamped to the joint range.",
        "units": "SI, radians. Joint angle convention = so101_new_calib (LeRobot degrees mode maps with sign +1, offset ~0).",
        "joints": jp,
        "command_delay_s": {
            "value": float(np.median(delta_vals)), "per_dataset_fit": fpos["deltas_s"],
            "meaning": "time between the policy/recorder reading the joint state and the servo receiving the new target, "
                       "within a 30 Hz control period; model it as a pure transport delay on the target (the fitted "
                       "actuator already contains the servo's own lag). Not separately identifiable from the actuator lag "
                       "in these data: keep kp/kv and this delay together.",
        },
        "control_rate_hz": 30, "physics": {"dt_s": fit.DT, "integrator": "implicitfast", "noslip_iterations": 1,
                                           "contacts_in_fit": "disabled (free-space replay)", "gravity": "on"},
        "joint_range_guidance": {
            "rule": "Use the CAD ranges, except where real arms reach a hard stop beyond them: set the sim limit (and the "
                    "target clamp) at the real stop. Stops differ per arm (each arm's LeRobot zero is the middle of its "
                    "own calibration sweep), so randomise them within the observed spread.",
            "cad_deg": {JOINTS[j]: fit.CAD_DEG[j].tolist() for j in range(6)},
            "observed_real_stops": stops,
        },
        "newton_or_usd_mapping": {
            "UsdPhysics drive:angular:physics:stiffness [N*m/deg]": "kp_Nm_per_rad * pi/180",
            "UsdPhysics drive:angular:physics:damping [N*m*s/deg]": "kv_Nms_per_rad * pi/180",
            "drive:angular:physics:maxForce [N*m]": "forcerange_Nm",
            "newton:damping [N*m*s/deg]": "damping_Nms_per_rad * pi/180",
            "newton:friction [N*m]": "frictionloss_Nm",
            "newton:armature [kg*m^2]": "armature_kgm2",
            "source": "inverse of the conversion VERIFIED in models.py NVIDIA_CONVERSION (OpenUSD DriveAPI + NewtonJointAPI docs, "
                      "Newton v1.6.1 import_usd.py); Newton SolverMuJoCo builds gainprm [kp], biasprm [0,-kp,-kd]",
        },
        "domain_randomization": dr,
        "error_vs_real_deg": {k: final.get(best_pos, {}).get(k, {}).get("OVERALL") for k in ("train4", "test4", "test1")},
        "error_vs_real_verified_subset_deg": {k: final.get(best_pos, {}).get(k, {}).get("VERIFIED") for k in ("test4", "test1")},
        "caveats": [
            "Fitted on slow teleoperation (leader-arm) motion of 6 arms at 30 Hz; no chirp/high-frequency excitation.",
            "Frame period 1/30 s assumed (LeRobot timestamps are synthetic).",
            "Gripper parameters are NOT fitted (contacts off in the replay, jaw stalls on objects dominate its error).",
            "LeRobot firmware settings P=16, D=32, Acceleration=254 are what these arms ran; the PD model absorbs them.",
            "kv and joint damping are nearly degenerate (both viscous); only their sum is well determined.",
        ],
    }
    fit.jdump(sofit, os.path.join(HERE, "so101_fitted.json"))

    # ---------- plot
    rows_all = fit.final_models(prep)
    plot_rows = {}
    for k in ("before:B_nvidia", "before:D_bam_m6", best_pos, "after:fit2_7.4V" if f2 and "7.4V" in f2["runs"] else None,
              "after:fit2_12V" if f2 and "12V" in f2["runs"] else None):
        if k and k in rows_all:
            plot_rows[k] = rows_all[k]
    ex = plot_example(prep, plot_rows, os.path.join(R, "fit_example.png"))

    # ---------- table
    L = []
    L.append("# SO-101 arm sysid, step 4: fit to real TRAIN data, held-out TEST error")
    L.append("")
    L.append(f"Generated by `fit.py` + `fit_report.py` on {time.strftime('%Y-%m-%d %H:%M')}. All numbers are joint-angle RMSE "
             "in degrees between the simulated and the real measured joint position (observation.state), sim driven open "
             "loop by the recorded leader commands (action). Cell = equal-weight mean over datasets of each dataset's "
             "pooled RMSE; 'arm' = mean of the 5 arm joints. 6 ds = dobri420, HashtagRobotics, DecisionFacts, Chaenn "
             "(eps 0-86), dharak22, vasco281204. verified 3 = dobri420, HashtagRobotics, DecisionFacts (card states real "
             "SO-101 teleoperation). TEST = episode_index % 4 == 3, never used for any fit or selection.")
    L.append("")
    L.append("## New evaluation protocol (all rows below)")
    L.append("")
    L.append("- **Real hard stops (fix 1).** Per dataset and arm joint, a stop is detected on TRAIN states where the robust "
             "extreme is within 15 deg of / beyond the CAD limit and >= max(30, 0.2 %) of frames sit within 0.5 deg of it. "
             "The stop is the sim joint limit and clamps the commanded target (servo position limit); other ends are never "
             "binding. Mid-range obstructions (e.g. DecisionFacts elbow held at -59 deg while the leader goes to -64, i.e. "
             "table contact) are deliberately not turned into limits.")
    L.append("- **Sub-frame delay delta (fix 2).** action[t] takes effect delta after state[t] is read (target box-filtered "
             "per 2 ms physics step, so the error is continuous in delta). Before the fit: one delta per dataset, shared "
             "by all models, from a 0-30 ms grid on TRAIN (mean of the B, C, D curves each normalised by its own minimum; "
             "A_stock left out because it has no servo lag and its error falls monotonically with delta). Rows 'own delta' "
             "use each model's TRAIN-best grid value. Fitted models fit delta per dataset jointly (0-33.3 ms).")
    L.append("- **Offsets (fix 4).** q_sim = q_lerobot + offset for lift/elbow/wrist_flex/wrist_roll. Before the fit: "
             "minimum-norm offsets that make the pick-moment jaw height flat vs reach (FK flat-table test, TRAIN picks). "
             "After: refined per dataset on TRAIN tracking within +-5 deg while keeping the FK test inside its 2-sigma band.")
    L.append(f"- **Warm start (fix 5).** Each window starts {fit.K_WARM} frames early from the real state and is driven by "
             "the recorded actions through the warm-up; only the W s after it are scored. Windows tile from frame 8.")
    L.append("- Unchanged: dt 2 ms, implicitfast, contacts off, gravity on, units/sign conventions of replay.py, 30 Hz "
             "frame period ASSUMED (timestamps are synthetic; no time-scale was fitted).")
    L.append("")

    def table(splits, subs, rows):
        hdr = "| group | model | " + " | ".join(f"{s} {u}" for s, u in zip(splits, subs)) + " |"
        out = [hdr, "|" + "---|" * (len(splits) + 2)]
        for k in rows:
            vals = [f2(cell(final, k, s, u)) for s, u in zip(splits, subs)]
            out.append(f"| {group(k)} | {label(k)} | " + " | ".join(vals) + " |")
        return out

    L.append("## 1. Headline: arm-mean RMSE [deg]")
    L.append("")
    L.append("TRAIN = all TRAIN windows (W=4 s); the fit itself saw only a fixed subset of 20 windows per dataset.")
    L.append("")
    nice = {"train4": "TRAIN W=4", "test4": "TEST W=4", "test1": "TEST W=1"}
    sp = ["train4", "test4", "test4", "test1", "test1"]
    su = ["OVERALL", "OVERALL", "VERIFIED", "OVERALL", "VERIFIED"]
    hdr = "| group | model | " + " | ".join(f"{nice[s]} ({'6 ds' if u == 'OVERALL' else 'verified 3'})" for s, u in zip(sp, su)) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (len(sp) + 2))
    for k in keys:
        vals = [f2(cell(final, k, s, u)) for s, u in zip(sp, su)]
        L.append(f"| {group(k)} | {label(k)} | " + " | ".join(vals) + " |")
    L.append("")
    # old protocol reference
    L.append("Old protocol for reference (results/baseline_table.md, TEST W=4, 6 ds, no warm start, delta 0, zero offsets; "
             "windows tiled from frame 0): " + ", ".join(
                 f"{m} {old['sim']['4']['all']['OVERALL_primary_mean'][m]['arm_mean_rmse']:.2f} (CAD limits) / "
                 f"{old['wide']['4']['all']['OVERALL_primary_mean'][m]['arm_mean_rmse']:.2f} (no limits)"
                 for m in ("A_stock", "B_nvidia", "C_bam_m1", "D_bam_m6", "perfect_tracking", "best_pure_delay")) + ".")
    L.append("")

    # per joint
    sel = ["baseline:perfect_tracking", "baseline:best_pure_delay", "before:B_nvidia", "before:D_bam_m6",
           "before:D_bam_m6_vin5", "before:M6_12Vset_vin12"] + [k for k in keys if k.startswith("after:")]
    sel = [k for k in sel if k in final]
    for split, sub in (("test4", "OVERALL"), ("test4", "VERIFIED"), ("test1", "OVERALL"), ("train4", "OVERALL")):
        L.append(f"## 2. Per joint, {nice[split]}, {'6 datasets' if sub == 'OVERALL' else 'verified 3'} [deg]")
        L.append("")
        L.append("| model | " + " | ".join(JOINTS) + " | arm |")
        L.append("|" + "---|" * 8)
        for k in sel:
            try:
                jv = final[k][split][sub]["joint"]
                L.append(f"| {label(k)} | " + " | ".join(f2(x) for x in jv) + f" | {f2(final[k][split][sub]['arm'])} |")
            except KeyError:
                pass
        L.append("")
    L.append("The gripper column is indicative only: approximate 0..100 -> -10..100 deg map, contacts off, not fitted.")
    L.append("")

    # per dataset
    L.append("## 3. Per dataset, TEST W=4, arm mean [deg]")
    L.append("")
    L.append("| model | " + " | ".join(PRIMARY) + " |")
    L.append("|" + "---|" * (len(PRIMARY) + 1))
    for k in keys:
        try:
            L.append(f"| {label(k)} | " + " | ".join(f2(final[k]['test4']['per_ds'][dn]['arm']) for dn in PRIMARY) + " |")
        except KeyError:
            pass
    nwin = {dn: final["before:B_nvidia"]["test4"]["per_ds"][dn]["n_windows"] for dn in PRIMARY}
    L.append("")
    L.append("TEST windows W=4: " + ", ".join(f"{dn} {n}" for dn, n in nwin.items()) +
             f"; W=1: " + ", ".join(f"{dn} {final['before:B_nvidia']['test1']['per_ds'][dn]['n_windows']}" for dn in PRIMARY) + ".")
    L.append("")

    # nuisances
    L.append("## 4. Per-dataset nuisances (all fitted on TRAIN)")
    L.append("")
    L.append("### 4a. Detected real hard stops (data coordinates, deg) and share of TEST frames at them")
    L.append("")
    share = final.get("_limit_share_test_w4", {})
    L.append("| dataset | stops (joint: end = limit) | TEST frames within 0.5 deg of a stop | TEST frames > 1 deg beyond CAD |")
    L.append("|---|---|---|---|")
    for dn in PRIMARY:
        r = prep["datasets"][dn]["ranges"]
        st = "; ".join(f"{j} {e} {r[j][e]['limit_deg']:.1f}" for j in r for e in ("lo", "hi") if r[j][e]["stop"])
        sh = share.get(dn, {})
        L.append(f"| {dn} | {st} | {f2(100 * sh.get('frac_at_detected_stop', float('nan')), 1)} % | "
                 f"{f2(100 * sh.get('frac_beyond_cad_1deg', float('nan')), 1)} % |")
    L.append("")
    L.append("### 4b. Sub-frame delay delta [ms]")
    L.append("")
    cols = [("shared (before)", before["shared_delta_s"])]
    for nm, f in (("FIT 1", f1), ("FIT 1 refit", f1r), ("FIT 1 + deadband", f1db)):
        if f:
            cols.append((nm, f["deltas_s"]))
    if f2:
        for pset, run in f2["runs"].items():
            cols.append((f"FIT 2 {pset}", run["deltas_s"]))
    L.append("| dataset | " + " | ".join(c[0] for c in cols) + " |")
    L.append("|" + "---|" * (len(cols) + 1))
    for dn in PRIMARY:
        L.append(f"| {dn} | " + " | ".join(f"{1000 * c[1][dn]:.1f}" for c in cols) + " |")
    L.append("")
    L.append("TRAIN-subset arm-mean RMSE vs delta (before fit, OVERALL): " + "; ".join(
        f"{m}: " + " ".join(f"{d}ms {before['curves_train_fit_subset'][m][str(d)]['OVERALL']:.3f}"
                            for d in before["delta_grid_ms"]) for m in before["curves_train_fit_subset"]) + ".")
    L.append("")
    L.append("### 4c. Zero offsets [deg] (lift, elbow, wrist_flex, roll) and the FK flat-table test")
    L.append("")
    L.append("| dataset | TRAIN picks | FK usable | FK slope z-vs-reach at offset 0 (se) | z median at 0 [cm] | FK-only offsets | "
             "refined offsets | TRAIN-subset gain vs 0 offsets [deg] | RMSE change for +-3 deg (lift / elbow / wflex / roll) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for dn in PRIMARY:
        p = prep["datasets"][dn]
        fi = p["fk_identity"]
        o = offj.get(dn) if offj else None
        prof = " / ".join(f"{v['minus3']:+.3f},{v['plus3']:+.3f}" for v in o["profile_pm3deg"].values()) if o else "-"
        gain = f2(o["objective_zero_offsets"] - o["objective"], 3) if o else "-"
        L.append(f"| {dn} | {p['n_picks']} | {p['fk_usable']} | "
                 f"{f2(fi.get('slope'), 3)} ({f2(fi.get('slope_se'), 3)}) | {f2(fi.get('z_med_cm'))} | "
                 f"{np.round(np.array(p['fk_offsets_deg'])[1:5], 2).tolist()} | "
                 f"{np.round(np.array(offsets[dn])[1:5], 2).tolist()} | {gain} | {prof} |")
    L.append("")
    L.append("FK usable = >= 20 picks, reach p10-p90 spread >= 3 cm and residual < 3 cm (dobri420 picks span only ~2 cm of "
             "reach, DecisionFacts nests cups at varying heights, HashtagRobotics has almost no grasp-stall events). "
             "Offsets of datasets without a usable FK test are constrained only by tracking and the +-5 deg bound; small "
             "profile changes mean that offset is not identified by these data.")
    L.append("")
    L.append("### 4d. Best pure delay (fractional, TRAIN) [frames at 30 Hz]")
    L.append("")
    L.append(", ".join(f"{dn} {prep['datasets'][dn]['best_pure_delay_frames']['clipped']:.2f}" for dn in PRIMARY))
    L.append("")

    # fitted params summary
    L.append("## 5. Fitted parameters")
    L.append("")
    for nm, f in (("FIT 1", f1), ("FIT 1 refit", f1r), ("FIT 1 + deadband", f1db)):
        if not f:
            continue
        L.append(f"### {nm} (CMA-ES {len(f['history'])} generations, popsize 16; TRAIN-subset objective "
                 f"{f['start_objective']:.3f} -> {f['objective_train_fit_subset']:.3f})")
        L.append("")
        L.append("| joint | kp [N*m/rad] | kv [N*m*s/rad] | damping [N*m*s/rad] | frictionloss [N*m] | armature [kg*m^2] |"
                 + (" deadband [deg] |" if f.get("deadband_deg") else ""))
        L.append("|---|---|---|---|---|---|" + ("---|" if f.get("deadband_deg") else ""))
        for j in range(ARM):
            cells = []
            for k in fit.POS_KEYS:
                v, b = f["params"][k][j], f["base"][k][j]
                cells.append(f"{v:.4g} (x{v / b:.2f})")
            if f.get("deadband_deg"):
                cells.append(f"{f['deadband_deg'][j]:.3f}")
            L.append(f"| {JOINTS[j]} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("(x = multiplier on the B_nvidia start value. forcerange kept at 3.35 N*m. Gripper unchanged.)")
        L.append("")
    if f2:
        L.append("### FIT 2 (BAM M6, BAM's own code; CMA-ES popsize 12)")
        L.append("")
        L.append("Start candidates on the TRAIN BAM subset (shared delta): " +
                 ", ".join(f"{k} {v:.3f}" for k, v in f2["candidates_train_fit_subset"].items()) + ".")
        L.append("")
        L.append("| run | vin [V] | friction scale (pan..roll) | error_gain_ratio scale (pan..roll) | generations | TRAIN-subset objective |")
        L.append("|---|---|---|---|---|---|")
        for pset, run in f2["runs"].items():
            c = run["cfg"]
            L.append(f"| {pset} set | {c['vin']:.2f} | {[round(x, 2) for x in c['fric_scale'][:ARM]]} | "
                     f"{[round(x, 2) for x in c['egr_scale'][:ARM]]} | {len(run['history'])} | {run['objective_train_fit_subset']:.3f} |")
        L.append("")
        L.append("BAM was identified with firmware D=0; LeRobot writes D=32 and Acceleration=254 (not modelled). "
                 "vin < 7.1 V or > 7.3 V for the 7.4V set is outside BAM's measured supply range (extrapolation).")
        L.append("")

    # DR
    L.append("## 6. Recommended domain randomisation for RL")
    L.append("")
    if dr:
        L.append(f"From per-arm refits around {spread['source']} (each of the 6 arms re-fits per-joint kp and frictionloss, "
                 "one kv+damping and one armature multiplier and its delta on its own TRAIN subset). Recommended range = the "
                 "per-arm spread, but never narrower than +-20 % (OmniReset uses +-20 % around identified values).")
        L.append("")
        L.append("| parameter | per-arm multipliers (dobri, Hashtag, DecisionF, Chaenn, dharak, vasco) | recommended multiplier range | why |")
        L.append("|---|---|---|---|")
        for k, v in dr.items():
            if k in ("delta", "joint_zero_offset_deg"):
                continue
            L.append(f"| {k} | {v['per_arm_multipliers']} | {v['recommended_multiplier']} | {v['why']} |")
        L.append(f"| delta [ms] | {dr['delta']['per_arm_ms']} | {dr['delta']['recommended_ms']} | {dr['delta']['why']} |")
        for j, v in dr["joint_zero_offset_deg"].items():
            L.append(f"| zero offset {j} [deg] | {v['per_arm']} | +-{v['recommended_abs_deg']} | per-arm calibration "
                     "(LeRobot zero = middle of each arm's sweep) |")
        L.append("")
        L.append("Joint limits: randomise the hard stops within the observed per-arm values in so101_fitted.json "
                 "(e.g. shoulder_lift lower stop -102.5 to -104.8 deg vs CAD -100).")
        L.append("")
    L.append("## 7. Example (results/fit_example.png)")
    L.append("")
    L.append(f"{ex['dataset']} TEST episode {ex['episode']}, frames {ex['t0'] + 1}-{ex['t0'] + 120}; {ex['selection']}. "
             f"Curves: {', '.join(ex['rows'])}.")
    L.append("")
    L.append("## 8. Files")
    L.append("")
    L.append("- fit.py (protocol + fits), fit_report.py (this report), results/fit_log.txt (progress log), "
             "results/fit_cache/*.json (all intermediate results), " + ", ".join(f"results/{w}" for w in written) +
             ", so101_fitted.json, results/fit_example.png.")
    L.append("")
    with open(os.path.join(R, "fit_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    fit.log("report written: results/fit_table.md, so101_fitted.json, fit_example.png, " + ", ".join(written))
    if fit._POOL is not None:
        fit._POOL.close()
        fit._POOL.join()


if __name__ == "__main__":
    main()
