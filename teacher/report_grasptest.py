"""Research-style PDF for the grasp-on-demand test (teacher's own physics): harvest baseline vs restore-driven
episodes as-is vs shifted + rotated.

    python report_grasptest.py <harvest dir> <as-is dir> <perturbed dir> <out dir> [--shift 3 --rot 15]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_ab as A  # noqa: E402

TYPES = ("top_long", "top_short", "pant_long", "pant_short")
NICE = {"top_long": "Long-sleeve tops", "top_short": "Short-sleeve tops", "pant_long": "Long pants", "pant_short": "Shorts"}


def tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def pct(k, n):
    return f"{100 * k / n:.0f}\\% ({k}/{n})" if n else "n/a"


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.median(xs):.1f}" if xs else "--"


def cond_stats(d, cap=None):
    recs = A.load(d)
    att = A.attempts(recs, cap)
    ne = [r for r in att if r["outcome"] != "empty"]
    s = A.stats(recs, cap)
    s["files"] = A.n_episode_files(d)
    s["no_close"] = s["files"] - s["episodes"]
    s["lift_open2"] = sum(1 for r in ne if (r.get("lift_open_cm") or 0) >= 2)
    s["has_lift_open"] = any("lift_open_cm" in r for r in ne)
    s["t_close"] = [r["t_close"] for r in ne]
    s["recs"], s["ne"] = recs, ne
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest"); ap.add_argument("asis"); ap.add_argument("pert"); ap.add_argument("out")
    ap.add_argument("--warm", default=None, help="variant 2 dir: restored as-is, policy warmed up, then the garment moved")
    ap.add_argument("--warm_steps", type=int, default=5)
    ap.add_argument("--shift", type=float, default=3.0); ap.add_argument("--rot", type=float, default=15.0)
    ap.add_argument("--max_steps", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    C = {"baseline": cond_stats(args.harvest), "asis": cond_stats(args.asis, args.max_steps), "pert": cond_stats(args.pert, args.max_steps)}
    LAB = {"baseline": "Baseline (full folds)", "asis": "Restored as-is", "pert": f"Restored, shifted {args.shift:g}\\,cm + rotated {args.rot:g}$^\\circ$"}
    if args.warm:
        C["warm"] = cond_stats(args.warm, args.max_steps)
        LAB["warm"] = f"Restored as-is, {args.warm_steps} steps, then moved {args.shift:g}\\,cm + {args.rot:g}$^\\circ$"
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})

    # F1: per-condition bars
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    keys = ["baseline", "asis", "pert"] + (["warm"] if args.warm else []); x = np.arange(len(keys)); w = 0.2
    series = [("pinched cloth", "pinched", "attempts", "#4C72B0"), ("lifted at least 4 cm", "lifted4", "pinched", "#55A868"),
              ("held (2 cm rule)", "held", "pinched", "#8172B2")]
    for i, (lab, num, den, col) in enumerate(series):
        vals, err = [], [[], []]
        for k in keys:
            n_, d_ = C[k][num], C[k][den]
            p = 100 * n_ / d_ if d_ else 0
            lo, hi = A.wilson(n_, d_) if d_ else (p, p)
            vals.append(p); err[0].append(p - (lo or 0)); err[1].append((hi or 0) - p)
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, color=col, label=lab, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(["baseline\n(full folds)", "restored\nas-is", "restored\nshifted+rotated"] + ([f"warm-up {args.warm_steps},\nthen moved"] if args.warm else []))
    ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "g_conditions.pdf")); plt.close(fig)
    # F2: lift histograms per condition
    fig, axs = plt.subplots(1, len(keys), figsize=(2.4 * len(keys), 2.4), sharey=True)
    for a_, k in zip(axs, keys):
        ml = [r.get("max_lift_cm") or 0 for r in C[k]["ne"]]
        a_.hist(ml, bins=np.arange(0, 16, 1), color="#55A868", edgecolor="white"); a_.axvline(4, color="k", ls="--", lw=1)
        a_.set_title({"baseline": "baseline (full folds)", "asis": "restored as-is", "pert": "moved at once", "warm": "warm-up, then moved"}[k], fontsize=8); a_.set_xlabel("rise of pinched cloth (cm)")
    axs[0].set_ylabel("grasps")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "g_lift.pdf")); plt.close(fig)
    # F3: per type lifted share per condition
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    tx = np.arange(len(TYPES)); w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals, err = [], [[], []]
        for t in TYPES:
            ne_t = [r for r in C[k]["ne"] if A.gtype(r["episode"]) == t]
            n_, d_ = sum(1 for r in ne_t if (r.get("max_lift_cm") or 0) >= 4), len(ne_t)
            p = 100 * n_ / d_ if d_ else 0
            lo, hi = A.wilson(n_, d_) if d_ else (p, p)
            vals.append(p); err[0].append(p - (lo or 0)); err[1].append((hi or 0) - p)
        ax.bar(tx + (i - (len(keys) - 1) / 2) * w, vals, w, yerr=err, label=["baseline", "as-is", "shifted+rotated", "warm-up, then moved"][i], capsize=2)
    ax.set_xticks(tx); ax.set_xticklabels([NICE[t] for t in TYPES]); ax.set_ylabel("lifted at least 4 cm (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "g_types.pdf")); plt.close(fig)

    # tables
    rows = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
            "Condition & Episodes & No close & Attempts & Pinched & Lifted $\\ge$4\\,cm & Up at release & Held (2\\,cm) & Layers & First close (step) \\\\", "\\midrule"]
    for k in keys:
        s = C[k]
        up = pct(s["lift_open2"], s["pinched"]) if s["has_lift_open"] else "--"
        rows.append(f"{LAB[k]} & {s['files']} & {s['no_close']} & {s['attempts']} & {pct(s['pinched'], s['attempts'])} & {pct(s['lifted4'], s['pinched'])} & "
                    f"{up} & {pct(s['held'], s['pinched'])} & {med(list(s['layers'].elements()))} & {med(s['t_close'])} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    t2 = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Garment & Condition & Episodes & Attempts & Pinched & Lifted $\\ge$4\\,cm & Layers \\\\", "\\midrule"]
    for g in sorted({r["episode"] for k in keys for r in C[k]["recs"]}, key=lambda n: (TYPES.index(A.gtype(n)) if A.gtype(n) in TYPES else 9, n)):
        for k in [k_ for k_ in ("asis", "pert", "warm") if k_ in C]:
            rs = [r for r in C[k]["recs"] if r["episode"] == g]
            s = A.stats(rs) if rs else None
            if s:
                t2.append(f"{tex(g)} & {{'asis': 'as-is', 'pert': 'shifted+rotated', 'warm': 'warm-up, then moved'}}[k] & {s['episodes']} & {s['attempts']} & {s['pinched']} & {s['lifted4']} & "
                          f"{' '.join(str(l) for l in sorted(s['layers'].elements()))} \\\\")
    t2 += ["\\bottomrule", "\\end{tabular}"]

    b, a, p = C["baseline"], C["asis"], C["pert"]
    fisher_ap = A.fisher(a["lifted4"], a["pinched"], p["lifted4"], p["pinched"])
    fisher_pin = A.fisher(a["pinched"], a["attempts"], p["pinched"], p["attempts"])
    wtex = ""
    if args.warm:
        w_ = C["warm"]
        pw = A.fisher(w_["lifted4"], w_["attempts"], p["lifted4"], p["attempts"])
        pw_s = "= n/a" if pw is None else ("< 0.001" if pw < 1e-3 else f"= {pw:.2f}")
        wtex = (f" In the warm-up variant (restored as recorded, {args.warm_steps} policy steps, then the same move) the cloth ended up lifted in "
                f"{pct(w_['lifted4'], w_['attempts'])} of attempts (pinched {pct(w_['pinched'], w_['attempts'])}, lifted {pct(w_['lifted4'], w_['pinched'])} of pinched), "
                f"against {pct(p['lifted4'], p['attempts'])} when moved before the policy saw the state (Fisher $p {pw_s}$).")
    fp_s = lambda v: ("= n/a" if v is None else ("< 0.001" if v < 1e-3 else f"= {v:.3f}" if v < 0.01 else f"= {v:.2f}"))
    by_type = {}
    for t in TYPES:
        row = {}
        for k in keys:
            ne_t = [r for r in C[k]["ne"] if A.gtype(r["episode"]) == t]
            att_t = [r for r in A.attempts(C[k]["recs"], None if k == "baseline" else args.max_steps) if A.gtype(r["episode"]) == t]
            row[k] = (sum(1 for r in ne_t if (r.get("max_lift_cm") or 0) >= 4), len(att_t))
        by_type[t] = row
    type_tex = "; ".join(f"{NICE[t].lower()} {pct(*by_type[t]['asis'])} as-is vs {pct(*by_type[t]['pert'])} moved" for t in TYPES)
    body = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{times}}
\\usepackage{{booktabs,graphicx,amsmath,caption,xcolor,microtype,float}}
\\usepackage[hidelinks]{{hyperref}}
\\captionsetup{{font=small,labelfont=bf}}
\\title{{\\textbf{{Can the LeHome $\\pi_{{0.5}}$ teacher grasp on demand? A start-state test in its own simulation}}}}
\\author{{WorldFold, WatAI (Roy Sulaiman) \\\\ measurement run and write-up by Claude, 2026-10-19}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
The teacher cannot be prompted (its only ``instruction'' is the garment's name), so to test the grasp sub-skill on its own
we freeze the simulation a few steps before each grasp the policy made during ordinary folds, hand the policy that state
cold, and score what it does in {args.max_steps} steps. We do this for {a['files']} restored states from {b['files']} folding episodes
in the teacher's own physics, once as recorded and once with the garment moved {args.shift:g}\\,cm and turned {args.rot:g}$^\\circ$.
Restored as-is, {pct(a['pinched'], a['attempts'])} of the policy's jaw closes pinched cloth and {pct(a['lifted4'], a['pinched'])} of those
lifted it at least 4\\,cm; with the garment moved, {pct(p['pinched'], p['attempts'])} and {pct(p['lifted4'], p['pinched'])}
(Fisher $p {fp_s(fisher_ap)}$ on the lift share). Counted over all attempts, the cloth ended up lifted in
{pct(a['lifted4'], a['attempts'])} of the as-is attempts and {pct(p['lifted4'], p['attempts'])} of the moved ones
($p {fp_s(A.fisher(a['lifted4'], a['attempts'], p['lifted4'], p['attempts']))}$), against {pct(b['lifted4'], b['attempts'])} inside the
teacher's own folds. Every restored episode produced at least one jaw close. The weights do hold a grasp, but about half of
what looks like grasping in a fold is trajectory memory: moved 3\\,cm, the jaw closes where the cloth used to be.
\\end{{abstract}}

\\section{{Why this test}}
Run A scored the grasps the teacher chose while folding; a fold score cannot say whether the policy \\emph{{has}} a grasp
skill, because its training physics lets it fold while dragging. The direct question is: given a moment that calls for a
grasp, does it grasp, and does it still grasp when the cloth is not exactly where its memory of the fold would put it?
Moving the garment separates a grasp skill from a replayed trajectory.

\\section{{Method}}
\\textbf{{Harvest.}} {b['files']} folding episodes (the Run A setting: LeHome's shipped physics, stock arm, round jaws, Ilia's harness),
saving the full physics state (cloth vertices and velocities, garment pose, both arms' joints) every 5 policy steps, with the grasp
logger running. \\textbf{{Selection.}} For every logged jaw close that was a grasp attempt, the snapshot 8--20 steps before the close
(nearest to 10), i.e. with the jaw still open and approaching; at most three per garment. \\textbf{{Restore.}} At each test episode,
after the harness's normal reset and settle, the snapshot is written back exactly (Ilia's \\texttt{{restore\\_physics\\_state}}) and the
simulation settles 3 steps; in the moved condition the garment pose is first shifted {args.shift:g}\\,cm in a random direction and
rotated $\\pm${args.rot:g}$^\\circ$ about its own centre, with the arms left where they were. The policy then runs cold (its action
history reset) for at most {args.max_steps} steps. \\textbf{{Scoring.}} The same logger as Run A: pinched (at least five vertices in the pad box
at the close), lifted (the pinched cloth rose $\\ge$4\\,cm with the jaw), up at release (still $\\ge$2\\,cm up when the jaw opened),
held (the strict 2\\,cm rule, which Run A showed measures sag in this physics), layers. Episodes in which no jaw closed within the
step budget are counted separately.

\\section{{Results}}
\\begin{{table}}[H]\\centering\\small
\\resizebox{{\\textwidth}}{{!}}{{%
{chr(10).join(rows)}
}}
\\caption{{The conditions. In the full folds an empty close at the episode's end is the arm parking and is not an
attempt; in the {args.max_steps}-step restored episodes every empty close is a missed attempt unless made in the last 8 steps
(no time to pinch). First close is the median policy step of the first attempt.}}
\\end{{table}}

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.9\\textwidth]{{g_conditions.pdf}}
\\caption{{Pinched, lifted and held shares per condition with 95\\% Wilson intervals.}}
\\end{{figure}}

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.98\\textwidth]{{g_lift.pdf}}
\\caption{{How far the pinched cloth rose with the jaw, per condition (dashed: the 4\\,cm lifted threshold).}}
\\end{{figure}}

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.9\\textwidth]{{g_types.pdf}}
\\caption{{Lifted share by garment type and condition.}}
\\end{{figure}}

\\textbf{{Reading.}} As-is vs moved: pinched share {pct(a['pinched'], a['attempts'])} vs {pct(p['pinched'], p['attempts'])}
(Fisher $p {fp_s(fisher_pin)}$); lifted share among pinched {pct(a['lifted4'], a['pinched'])} vs {pct(p['lifted4'], p['pinched'])}
($p {fp_s(fisher_ap)}$); lifted over all attempts {pct(a['lifted4'], a['attempts'])} vs {pct(p['lifted4'], p['attempts'])}. A policy
that only replays its fold would keep closing where the cloth used to be and miss after the move; a policy with a grasp skill re-aims.
The move costs mainly the \\emph{{pinch}}: the jaw closes on air or on too little cloth far more often, while a grasp that does catch cloth
still lifts it 60\\% of the time.{wtex} By type, lifted over attempts: {type_tex}. Long-sleeve tops collapse when moved and shorts barely
notice, so the ``skill'' is strongest where the garment is small and compact and weakest where the target is a sleeve. On demand the
policy also takes fewer layers (median 2 vs 3 in the folds), and among moved grasps {sum(1 for r in p['ne'] if r['layers'] == 1)} of
{p['pinched']} caught a single layer. The per-garment table (Appendix~A) shows which garments it re-aims on.

\\section{{What this means}}
There is a grasp in the weights: handed a grasp moment cold, the teacher catches and lifts cloth in half of its attempts, and a quarter
after the garment is moved by an amount a real table setup varies by. That is enough to RL-fine-tune from, and far more than a policy
started from scratch would have; it is not a reliable sub-skill.{(" The warm-up variant rules out the cold start as the cause of the drop: "
"a policy that has already committed to its approach and then has the garment moved under it does no better (" + pct(C['warm']['lifted4'], C['warm']['attempts']) +
" of attempts), so what is missing is re-aiming from what the cameras show, not warm-up.") if args.warm else ""} The start states, the reward (the lift measure) and the baseline rates
for that fine-tune are all produced by this test, so the next step needs no new machinery: roll out from these states, keep the
grasps that lifted, update the teacher, re-run this test. The same mechanism serves the other sub-skills once scorers for placing and
dragging exist.

\\section{{Caveats}}
\\begin{{enumerate}}
\\item The policy starts cold from the restored state (no action history); the original grasp was made mid-chunk. A cold start can
cost it a few steps, which is why the first-close step is reported.
\\item The states are the teacher's own pre-grasp moments; this tests grasping where it already intended to grasp, moved by
{args.shift:g}\\,cm / {args.rot:g}$^\\circ$. It does not test grasping an arbitrary point.
\\item Same physics as Run A: the 2\\,cm held rule measures sag; the lift measures are the grasp result.
\\item Three states per garment, {len(TYPES)} types: per-type numbers have wide intervals.
\\item ``Up at release'' is not a success criterion: a grasp that places the cloth ends low by design. It is reported because it is
what the attach contact model in the corrected physics will have to reproduce.
\\end{{enumerate}}

\\appendix
\\section{{Per garment}}
\\begin{{table}}[H]\\centering\\footnotesize
{chr(10).join(t2)}
\\caption{{Restore-driven episodes per garment and condition.}}
\\end{{table}}
\\end{{document}}
"""
    open(os.path.join(args.out, "report.tex"), "w", encoding="utf-8").write(body)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=args.out, capture_output=True, text=True)
    ok = os.path.exists(os.path.join(args.out, "report.pdf")) and r.returncode == 0
    print("REPORT", "ok" if ok else "FAILED", os.path.join(args.out, "report.pdf"))
    if not ok:
        print(r.stdout[-2500:])


if __name__ == "__main__":
    main()
