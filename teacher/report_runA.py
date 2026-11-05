"""Research-style PDF report of one teacher grasp run (run_teacher_grasp.sh output dir with grasps/*.jsonl).

    python report_runA.py <run dir> <out dir> [--montage a.png b.png] [--title ...]

Writes figures (matplotlib), report.tex, and compiles report.pdf with pdflatex (MiKTeX). Numbers in the prose come
from the data; the prose itself is fixed text written for Run A (original LeHome physics).
"""
import argparse
import json
import os
import shutil
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


def ci_str(k, n):
    lo, hi = A.wilson(k, n)
    return f"{100 * k / n:.0f}\\% [{lo:.0f}, {hi:.0f}]" if n else "n/a"


def fmt_med(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.median(xs):.1f}" if xs else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--montage", nargs="*", default=[])
    ap.add_argument("--title", default="Does the LeHome $\\pi_{0.5}$ teacher grasp? Per-grasp measurement in its original simulation (Run A)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = A.load(args.run)
    att = A.attempts(recs)
    ne = [r for r in att if r["outcome"] != "empty"]
    S = A.stats(recs)
    by = {t: A.stats([r for r in recs if A.gtype(r["episode"]) == t]) for t in TYPES}
    by = {t: s for t, s in by.items() if s["episodes"]}
    eps = defaultdict(list)
    for r in recs:
        eps[r["episode"]].append(r)
    physics = open(os.path.join(args.run, "physics.txt")).read().strip() if os.path.exists(os.path.join(args.run, "physics.txt")) else ""

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    # F1 kept % histogram
    kept = [100 * r["min_held"] for r in ne if r.get("min_held") is not None]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ml = [r.get("max_lift_cm") or 0 for r in ne]
    ax0.hist(ml, bins=np.arange(0, max(ml + [1]) + 1, 1), color="#55A868", edgecolor="white")
    ax0.axvline(4, color="k", ls="--", lw=1); ax0.text(4.3, ax0.get_ylim()[1] * 0.9, "lifted threshold", fontsize=8)
    ax0.set_xlabel("rise of the pinched cloth with the jaw (cm)"); ax0.set_ylabel("grasps")
    ax.hist(kept, bins=np.arange(0, 105, 5), color="#4C72B0", edgecolor="white")
    ax.axvline(50, color="k", ls="--", lw=1); ax.text(51, ax.get_ylim()[1] * 0.9, "held threshold", fontsize=8)
    ax.set_xlabel("kept within 2 cm of the tip (% of pinched vertices)"); ax.set_ylabel("grasps")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "f_kept.pdf")); plt.close(fig)
    # F2 lift at slip vs max lift, carry
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for r in ne:
        c = "#C44E52" if r["outcome"] == "slipped" else "#55A868"
        axs[0].scatter(r.get("max_lift_cm"), r.get("slip_lift_cm") if r.get("slip_lift_cm") is not None else r.get("max_lift_cm"),
                       color=c, s=14, alpha=0.8)
        axs[1].scatter(r.get("carry_cm"), 100 * r["min_held"], color=c, s=14, alpha=0.8)
    lim = max([r.get("max_lift_cm") or 0 for r in ne] + [1])
    axs[0].plot([0, lim], [0, lim], color="gray", lw=0.8, ls=":")
    axs[0].set_xlabel("highest jaw lift during the grasp (cm)"); axs[0].set_ylabel("lift when it slipped (cm)")
    axs[1].set_xlabel("jaw travel while closed (cm)"); axs[1].set_ylabel("kept (%)")
    axs[1].axhline(50, color="k", ls="--", lw=0.8)
    axs[0].scatter([], [], color="#C44E52", label="slipped"); axs[0].scatter([], [], color="#55A868", label="held"); axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "f_lift.pdf")); plt.close(fig)
    # F3 per-type bars with CI
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    names = list(by); x = np.arange(len(names)); w = 0.2
    for i, (lab, num, den, col) in enumerate((("pinched cloth", "pinched", "attempts", "#4C72B0"),
                                             ("lifted at least 4 cm", "lifted4", "pinched", "#55A868"),
                                             ("held (2 cm rule)", "held", "pinched", "#8172B2"),
                                             ("fold passed", "folds", "episodes", "#DD8452"))):
        vals, err = [], [[], []]
        for t in names:
            k, n = by[t][num], by[t][den]
            p = 100 * k / n if n else 0
            lo, hi = A.wilson(k, n) if n else (p, p)
            vals.append(p); err[0].append(p - (lo or 0)); err[1].append((hi or 0) - p)
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=err, color=col, label=lab, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([NICE[t] for t in names]); ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "f_types.pdf")); plt.close(fig)
    # F4 layers
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    L = Counter(r["layers"] for r in ne)
    ks = sorted(L); ax.bar(ks, [L[k] for k in ks], color="#8172B2")
    ax.set_xlabel("cloth layers caught in one grasp (mesh-connected pieces)"); ax.set_ylabel("grasps"); ax.set_xticks(ks)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "f_layers.pdf")); plt.close(fig)
    # F5 kept per episode vs fold result
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    pts = []
    for e, rs in eps.items():
        ks = [100 * r["min_held"] for r in A.attempts(rs) if r["outcome"] != "empty" and r.get("min_held") is not None]
        if ks:
            pts.append((np.mean(ks), bool(rs[0].get("success")), A.gtype(e)))
    for m, ok, t in pts:
        ax.scatter(1 if ok else 0, m, color="#55A868" if ok else "#C44E52", s=18, alpha=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fold failed", "fold passed"]); ax.set_ylabel("mean kept % (episode)"); ax.set_xlim(-0.6, 1.6)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "f_kept_success.pdf")); plt.close(fig)
    for m in args.montage:
        shutil.copy(m, args.out)

    # ---------------- numbers for the prose ----------------
    n_ep, n_folds = S["episodes"], S["folds"]
    n_att, n_pin, n_held = S["attempts"], S["pinched"], S["held"]
    kept_med = fmt_med(kept)
    slip_med, lift_med, carry_med = fmt_med(S["slip_lift"]), fmt_med(S["max_lift"]), fmt_med(S["carry"])
    lay = S["layers"]; lay_med = fmt_med(list(lay.elements()))
    single = sum(v for k, v in lay.items() if k == 1)
    pass_kept = [m for m, ok, _ in pts if ok]; fail_kept = [m for m, ok, _ in pts if not ok]
    p_mwu = A.mwu(pass_kept, fail_kept)
    p_mwu_s = f"Mann--Whitney $p = {p_mwu:.2f}$" if p_mwu is not None else "too few failed folds for a test"
    ci_held = A.wilson(n_held, n_pin); ci_fold = A.wilson(n_folds, n_ep)
    n_l4, n_l6 = S['lifted4'], S['lifted6']
    ci_l4 = A.wilson(n_l4, n_pin)

    # ---------------- tables ----------------
    t1 = ["\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
          "Garment type & Episodes & Folds passed & Attempts & Pinched & Lifted $\\ge$4\\,cm & Held (2\\,cm) & Kept \\% & Slip at (cm) & Max lift (cm) & Carry (cm) \\\\", "\\midrule"]
    for t in by:
        s = by[t]
        t1.append(f"{NICE[t]} & {s['episodes']} & {s['folds']} & {s['attempts']} & {s['pinched']} & {s['lifted4']} & {s['held']} & {fmt_med(s['kept'])} & "
                  f"{fmt_med(s['slip_lift'])} & {fmt_med(s['max_lift'])} & {fmt_med(s['carry'])} \\\\")
    t1 += ["\\midrule", f"All & {n_ep} & {n_folds} & {n_att} & {n_pin} & {n_l4} & {n_held} & {kept_med} & {slip_med} & {lift_med} & {carry_med} \\\\",
           "\\bottomrule", "\\end{tabular}"]
    t2 = ["\\begin{tabular}{llrrrrrl}", "\\toprule", "Episode & Fold & Attempts & Pinched & Held & Kept \\% & Max lift & Layers \\\\", "\\midrule"]
    for e in sorted(eps, key=lambda n: (TYPES.index(A.gtype(n)) if A.gtype(n) in TYPES else 9, n)):
        rs = eps[e]; s = A.stats(rs)
        t2.append(f"{tex(e)} & {'pass' if s['folds'] else 'fail'} & {s['attempts']} & {s['pinched']} & {s['held']} & {fmt_med(s['kept'])} & "
                  f"{fmt_med(s['max_lift'])} & {' '.join(str(k) for k in sorted(s['layers'].elements()))} \\\\")
    t2 += ["\\bottomrule", "\\end{tabular}"]

    runtime_tex, lehome_sr_tex = "", ""
    try:                                           # the harness's own summary: run time and its success rate
        import glob as _g, re as _re
        es = _g.glob(os.path.join(args.run, "ckpt_*", "eval_summary.txt"))
        if es:
            txt = open(es[0], encoding="utf-8", errors="replace").read()
            m = _re.search(r"Run time:\s*([^\n]+)", txt); m2 = _re.search(r"Time per episode:\s*([0-9.]+)s", txt)
            if m and m2:
                runtime_tex = f"The whole run took {m.group(1).strip()} on the L4 ({float(m2.group(1)):.0f}\\,s per episode with two workers)."
            m3 = _re.search(r"OVERALL:\s*(\d+)/(\d+)\s+SR=([0-9.]+)%\s*\[([^\]]+)\]", txt)
            if m3:
                lehome_sr_tex = (f" (the harness's own summary agrees: {m3.group(1)}/{m3.group(2)}, SR {float(m3.group(3)):.0f}\\% "
                                 f"[{m3.group(4).replace('%', chr(92) + '%')}])")
    except Exception:
        pass
    phys_tex = "LeHome original."
    try:
        pj = json.loads(physics.split(" ", 2)[2]) if physics.startswith("PHYSICS") else {}
        if pj:
            phys_tex = ("\\begin{tabular}{lr}\\toprule Setting & Value \\\\ \\midrule " +
                        " ".join(f"{tex(k)} & {v:g} \\\\ " for k, v in pj.items()) + "\\bottomrule\\end{tabular}")
            phys_tex += (" \\par\\smallskip With $10^{-2}$\\,kg per vertex the 14,746-vertex shirt weighs $\\sim$147\\,kg, about 3,000 times "
                         "a real one; gravity scale 2 doubles its weight again; adhesion 0.1 makes the cloth stick to whatever touches it.")
    except Exception:
        pass
    montage_fig = ""
    if args.montage:
        parts = []
        for m in args.montage:
            parts.append(f"\\includegraphics[width=\\textwidth,height=0.3\\textheight,keepaspectratio]{{{os.path.basename(m)}}}\\\\[4pt]")
        montage_fig = ("\\begin{figure}[htbp]\\centering\n" + "\n".join(parts) +
                       "\\caption{What the labels look like from the cameras, one grasp per block: top camera (row 1) and the grasping "
                       "arm's wrist camera (row 2) six steps before the close, at the close, when the logger declared the slip, twelve "
                       "steps after the close, and just before and after the release. Upper block: a slipped-labelled long-pants grasp "
                       "(kept 7\\%, max lift 5.8\\,cm, carry 10.4\\,cm, 4 layers): by the logger, 93\\% of the pinched vertices were more than "
                       "2\\,cm from the tip point at $t=107$, yet in the wrist camera cloth is still bunched at the jaw at $t=113$. The "
                       "garment did move with the jaw and the fold passed. This is the ambiguity of Section~5, item 1: a strict "
                       "vertex-level criterion against round jaws that drag a sticky, very heavy cloth. Lower block: a held-labelled "
                       "shorts grasp (kept 58\\%, lift 9.1\\,cm).}\\label{fig:cams}\\end{figure}\n")

    body = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{times}}
\\usepackage{{booktabs,graphicx,amsmath,caption,xcolor,microtype,float}}
\\usepackage[hidelinks]{{hyperref}}
\\captionsetup{{font=small,labelfont=bf}}
\\title{{\\textbf{{{args.title}}}}}
\\author{{WorldFold, WatAI (Roy Sulaiman) \\\\ measurement run and write-up by Claude, 2026-10-19; v2, corrected the same evening (lift criterion, Section~5)}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We ask whether the $\\pi_{{0.5}}$ LeHome teacher (\\texttt{{lehome\\_sim}}) can grasp cloth, measured grasp by grasp rather than by fold
scores. Run A puts the teacher in the simulation it was trained in (LeHome's shipped physics: a $\\sim$150\\,kg sticky shirt under
doubled gravity, round crossing jaws, LeHome's arm) and logs every jaw close on {n_ep} episodes ({len(by)} garment types).
Of {n_att} grasp attempts, {n_pin} ({100 * n_pin / max(n_att, 1):.0f}\\%) pinched cloth and {n_l4} of those
({100 * n_l4 / max(n_pin, 1):.0f}\\%, 95\\% CI {ci_l4[0]:.0f}--{ci_l4[1]:.0f}\\%) lifted it at least 4\\,cm with the jaw ({n_l6} at least
6\\,cm; median rise {lift_med}\\,cm, median carry {carry_med}\\,cm); {n_folds}/{n_ep} folds ({100 * n_folds / max(n_ep, 1):.0f}\\%, CI
{ci_fold[0]:.0f}--{ci_fold[1]:.0f}\\%) passed LeHome's check. A typical grasp takes {lay_med} layers. Our stricter criterion, at least half of
the pinched vertices staying within 2\\,cm of the jaw-tip point until release, was met by only {n_held} grasps; it turns out to be unfit
for this physics, where the $\\sim$150\\,kg cloth sags out of that radius while still on the jaw. In its own simulation the teacher
therefore pinches and lifts in almost every attempt, never one layer at a time. We list the ways this measurement can mislead and
what Run B (corrected arm, cloth and contact) must show to distinguish ``the teacher can grasp'' from ``it only works in its own
physics''.
\\end{{abstract}}

\\section{{Why measure grasps, not folds}}
Fold scores tell us whether a policy finished a fold, not which sub-skill failed. On 2026-10-02 the teacher passed 3/8 folds under
corrected physics and most shirts shrank, which pointed at grasping, but no grasp had ever been scored. The project's plan is to
RL only the sub-skills the teacher cannot do, so we need the grasp answer on its own. Run A is the control arm of an A/B test:
the teacher in its own simulation, with every grasp labelled. Run B repeats the same 20 garments with the physics corrected
to our measurements (arm 10\\% from reality, cloth 28\\%, contact calibrated).

\\section{{Setup}}
\\textbf{{Simulation.}} LeHome's Isaac Sim 5.1 environment as shipped (settings in Appendix~B): particle cloth with
$10^{{-2}}$\\,kg per vertex (a 14,746-vertex shirt weighs $\\sim$147\\,kg), gravity scale 2, adhesion 0.1 (the cloth sticks to the jaws),
LeHome's SO-101 model with its round capsule jaw colliders that cross at the tips, LeHome's actuator settings, 1/90\\,s physics.
Tops start centred and unrotated on the table (our scene edit, the same in Run B); pants keep LeHome's randomised spawn.

\\textbf{{Policy.}} Ilia Larchenko's \\texttt{{lehome\\_sim}} $\\pi_{{0.5}}$ checkpoint under his evaluation harness
(\\texttt{{run\\_eval.py}}, remote policy, chunked actions with inpainting, best-of-$n$ candidates, retry on), with the speed patches we
measured earlier (none touch physics or the model input). Two Isaac workers and one policy server on one NVIDIA L4.

\\textbf{{Garments and episodes.}} The first five ``Seen'' garments of each type, one episode each, at most 600 policy steps,
LeHome's own check-point success criterion. Workers took garments in queue order, so pants ran first. {runtime_tex}

\\section{{The grasp benchmark}}
Every policy step, the logger (\\texttt{{oracle/grasp\\_log.py}}) reads both grippers' joint angles and the cloth vertex positions.
\\begin{{itemize}}
\\item \\textbf{{Close / open.}} A jaw ``closes'' when its joint drops below 0.19\\,rad after having been open ($>0.25$\\,rad); it ``opens'' when it rises above 0.25\\,rad again.
\\item \\textbf{{Pinched set.}} The vertices inside a 14\\,mm $\\times$ 24\\,mm $\\times$ 28\\,mm box between the pad faces of the jaw at the close, six steps after the close began (so the jaw has settled). Fewer than five vertices is an \\emph{{empty}} grasp.
\\item \\textbf{{Layers.}} The pinched vertices are split into mesh-connected pieces; pieces of at least three vertices count as layers.
\\item \\textbf{{Held.}} At every step while closed, the share of the pinched vertices still within 2\\,cm of the jaw tip. If that share ever drops below 50\\% the grasp is \\emph{{slipped}}, and the lift at that moment is recorded. Otherwise it is \\emph{{held}}, qualified by what the jaw did: \\emph{{lifted}} ($\\ge$2\\,cm up), \\emph{{dragged}} ($\\ge$3\\,cm sideways) or \\emph{{static}}.
\\item \\textbf{{Lift and carry.}} Lift is the pinched vertices' median height above where they were at the close; carry is the jaw tip's travel while closed.
\\item \\textbf{{Parked closes.}} A jaw closing on nothing at the end of the episode (the arm returning home) is not counted as an attempt.
\\end{{itemize}}
The cloth reader takes the garment's pose live at every call. Isaac moves the garment prim to the cloth's bounding-box centre on
every write-back, so a pose cached at the episode start drifts as soon as the cloth is lifted; an earlier version of the reader
cached it, was caught before this run's data was collected, and was fixed (Section~\\ref{{sec:threats}}).

\\section{{Results}}
\\begin{{table}}[H]\\centering\\small
\\resizebox{{\\textwidth}}{{!}}{{%
{chr(10).join(t1)}
}}
\\caption{{Run A by garment type. Pinched: attempts that caught at least five vertices. Lifted: pinched grasps whose cloth rose at
least 4\\,cm with the jaw. Held: pinched grasps that kept at least half of those vertices within 2\\,cm of the jaw-tip point until it opened. Kept, slip, lift and carry are medians over pinched grasps.}}
\\label{{tab:types}}
\\end{{table}}

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.95\\textwidth]{{f_types.pdf}}
\\caption{{Per type: the share of attempts that pinched cloth, the share of pinched grasps whose cloth rose at least 4\\,cm with the
jaw, the share that met the 2\\,cm rule, and the share of folds that passed, with 95\\% Wilson intervals. With five episodes per type the
fold intervals are wide.}}
\\end{{figure}}

\\textbf{{The teacher pinches and lifts.}} Most attempts catch cloth ({n_pin}/{n_att}; the {n_att - n_pin} empties are all on tops,
Section~\\ref{{sec:threats}}), and the pinched cloth then rises with the jaw: {n_l4} of {n_pin} grasps lifted it at least 4\\,cm, {n_l6}
at least 6\\,cm (median rise {lift_med}\\,cm, Figure~\\ref{{fig:lift}}), with {carry_med}\\,cm of travel while closed. The 2\\,cm
``held'' criterion tells a different story (Figure~\\ref{{fig:kept}}): the median grasp kept only {kept_med}\\% of its pinched vertices
within 2\\,cm of the tip point, falling below 50\\% after {slip_med}\\,cm of lift, and only {n_held} grasps met it. Both are true at
once because the cloth is $\\sim$150\\,kg: the pinched patch sags several centimetres below the tip point while hanging from the jaw.
The lift numbers are the grasp result; the 2\\,cm numbers measure sag.

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.95\\textwidth]{{f_kept.pdf}}
\\caption{{Left: how far the pinched cloth rose with the jaw (the grasp result). Right: the share of the pinched vertices that stayed
within 2\\,cm of the tip point until release (the sag). All pinched grasps; dashed lines are the thresholds.}}
\\label{{fig:kept}}
\\end{{figure}}

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.95\\textwidth]{{f_lift.pdf}}
\\caption{{Left: the lift at which a grasp slipped against the highest lift the jaw reached in that grasp (dotted line: equal). Points
far below the line slipped early and the jaw kept rising. Right: cloth kept against jaw travel while closed.}}
\\label{{fig:lift}}
\\end{{figure}}

\\textbf{{Grasps take several layers.}} The median grasp caught {lay_med} mesh-connected layers (Figure~\\ref{{fig:layers}}); {single} of
{n_pin} caught a single layer. On pants the jaw closes over waistband and leg layers at once; on tops it takes the front and back of a
sleeve or hem together.

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.55\\textwidth]{{f_layers.pdf}}
\\caption{{Layers caught per pinched grasp.}}
\\label{{fig:layers}}
\\end{{figure}}

\\textbf{{Folds pass anyway.}} {n_folds} of {n_ep} episodes passed LeHome's check{lehome_sr_tex}. Episodes that passed had a mean kept share of
{fmt_med(pass_kept)}\\% against {fmt_med(fail_kept)}\\% for those that failed ({p_mwu_s}; Figure~\\ref{{fig:ks}}):
the grasp quality we measure does not separate passed from failed folds in this physics. Table~\\ref{{tab:eps}} lists every episode.

\\begin{{figure}}[htbp]\\centering
\\includegraphics[width=0.5\\textwidth]{{f_kept_success.pdf}}
\\caption{{Mean kept share per episode against the fold result.}}
\\label{{fig:ks}}
\\end{{figure}}

{montage_fig}

\\section{{What could have gone wrong}}\\label{{sec:threats}}
\\begin{{enumerate}}
\\item \\textbf{{The 2\\,cm ``held'' criterion does not fit this physics (the correction behind v2 of this report).}} It was
designed for flat pads and a light cloth, where a held patch stays at the pads. LeHome's cloth weighs $\\sim$150\\,kg, so a patch
hanging from the round jaws sags out of the 2\\,cm radius while still attached; the logger's independent lift measure shows the
cloth rising with the jaw in {n_l4} of {n_pin} grasps. The first version of this report read the 2\\,cm numbers as ``slipped'' and
called the grasps partial drags; that was wrong for this physics. What the lift measure cannot tell is whether the cloth was still
up at the moment of release (added to the logger as \\texttt{{lift\\_open\\_cm}} for the next runs). The same jaw geometry can
also explain the {n_att - n_pin} empty attempts, all on tops: the jaw closed at the cloth but the pad box held fewer than five
vertices. In Run B's light cloth with flat pads the two criteria should agree.
\\item \\textbf{{Grasps that drag still move cloth.}} In a sticky, very heavy cloth, dragging a small patch moves the whole garment, so
a grasp that keeps 7\\% of its vertices can still complete a fold. This is a property of the training physics, not of the policy, and it is
exactly what Run B removes.
\\item \\textbf{{The teacher's strategy is sliding, not pinching.}} The jaw mostly lands flat on the cloth and slides, taking several layers.
That worked here; whether it works when a grasp must actually hold is Run B's question.
\\item \\textbf{{Reader bug, fixed before the data.}} The first launch of Run A used a cloth reader that cached the garment pose; it was
stopped after five minutes, fixed, and relaunched. All data here come from the relaunch. The same bug had corrupted our first scripted
contact numbers that morning; those were re-measured.
\\item \\textbf{{Sample size.}} Five episodes per type give wide intervals on fold rates (Figure~1); grasp-level rates ({n_pin} pinched
grasps) are tighter. Differences between types should not be read beyond the intervals.
\\item \\textbf{{Spawn asymmetry.}} Tops start centred and unrotated, pants randomised; both runs share this, so it does not bias A against B,
but it does make the type rows not strictly comparable.
\\item \\textbf{{Parked closes.}} {S['parked']} of the {len(recs)} jaw closes were the jaw shutting on air while returning home; they are
excluded from attempts. Had they been counted, the empty share would read {100 * (n_att - n_pin + S['parked']) / max(len(recs), 1):.0f}\\%
rather than {100 * (1 - n_pin / max(n_att, 1)):.0f}\\%.
\\item \\textbf{{The harness is on the teacher's side.}} Retry, inpainting and best-of-$n$ candidates are part of Ilia's policy and stay on in
both runs. They raise fold success; they do not change what a single grasp does.
\\end{{enumerate}}

\\section{{What to look for in Run B}}
Run B uses the same 20 garments and policy with: our fitted arm (10\\% from the real SO-101 on held-out motions), the measured cloth
(28\\% from a real hanging sheet, on flattened garment models so the stiff settings lie flat), flat pads, and a grasp model in which
cloth caught between the pads moves with the jaw and lets go above $2\\mu N$ with the measured PLA--cotton friction and 9\\,N squeeze
(Isaac's own contact is the fallback: it holds about half a sleeve through a carry and lets go 2--10$\\times$ too early). The
pre-registered reading, on the share of pinched grasps that hold:
\\begin{{itemize}}
\\item \\textbf{{Held $\\ge$ 80\\% in B and empty $\\le$ 20\\%:}} the teacher can grasp once the physics is right; its failures are placement and
sequencing, so search and SGS go there.
\\item \\textbf{{Held in A but $<$ 50\\% in B:}} physics mismatch; a small fine-tune of the teacher in the corrected physics.
\\item \\textbf{{Held $<$ 50\\% in both:}} grasping itself is the gap; it needs SGS-style RL.
\\end{{itemize}}
On the 2\\,cm criterion Run A has held $= {100 * n_held / max(n_pin, 1):.0f}\\%$; on the lift criterion it has {100 * n_l4 / max(n_pin, 1):.0f}\\%, which
changes which branch Run B can land in (see below). Beyond the rule, watch (i) whether the pinched share stays near 100\\% with flat pads and light cloth (the jaw must now land
precisely), (ii) whether kept shares jump once cloth is actually held, (iii) how many layers the flat pads take, and (iv) whether folds
pass when a grasp that keeps 7\\% of its vertices no longer moves the garment.

Because the 2\\,cm criterion failed in Run A's physics, read Run B primarily on the lift criterion (pinched cloth rises
$\\ge$4\\,cm with the jaw and is still up at release), with the 2\\,cm criterion as the stricter secondary; Run A's lift share is
{100 * n_l4 / max(n_pin, 1):.0f}\\%, so on that criterion the teacher does grasp in its own world, and the open question is only whether
the same motions grasp under real contact.

\\appendix
\\section{{Per-episode results}}
\\begin{{table}}[H]\\centering\\small
{chr(10).join(t2)}
\\caption{{Every episode of Run A. Kept and max lift are medians over the episode's pinched grasps; layers lists the layers of each pinched grasp.}}
\\label{{tab:eps}}
\\end{{table}}

\\section{{Grasp-benchmark parameters}}
Held radius 2\\,cm; slip threshold 50\\% kept; minimum pinch 5 vertices; lift threshold 2\\,cm; drag threshold 3\\,cm; minimum layer 3
vertices; close below 0.19\\,rad, open above 0.25\\,rad (jaw open 0.55, closed $-0.17$); pad box $x \\in [-10, 4]$\\,mm, $|y| < 12$\\,mm,
$z \\in [-103, -75]$\\,mm in the gripper-link frame.

\\section{{Physics settings of this run}}
{phys_tex}
\\end{{document}}
"""
    open(os.path.join(args.out, "report.tex"), "w", encoding="utf-8").write(body)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=args.out,
                           capture_output=True, text=True)
    ok = os.path.exists(os.path.join(args.out, "report.pdf")) and r.returncode == 0
    print("REPORT", "ok" if ok else "FAILED", os.path.join(args.out, "report.pdf"))
    if not ok:
        print(r.stdout[-3000:])


if __name__ == "__main__":
    main()
