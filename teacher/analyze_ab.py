"""A/B breakdown of the teacher's grasps (grasp_log records, run_teacher_grasp.sh output dirs).

    python analyze_ab.py <run A dir> [<run B dir>] [--md report.md]

A grasp record (oracle/grasp_log.py): outcome empty / slipped / held_lifted / held_dragged / held_static, n_pinch,
layers, min_held (share of the pinched vertices still within 2 cm of the jaw before it opened), max_lift_cm,
carry_cm, slip_lift_cm, end (opened / episode_end), success (LeHome's fold check for the episode).
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

OUTCOMES = ("empty", "slipped", "held_lifted", "held_dragged", "held_static")


def load(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "grasps", "*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); r["_file"] = os.path.basename(f); recs.append(r)   # one file per episode
    return recs


def n_episode_files(d):
    """Episodes run (one jsonl per episode, written even when no jaw closed: restore-driven test episodes)."""
    return len(glob.glob(os.path.join(d, "grasps", "*.jsonl")))


def gtype(name):
    return "_".join(str(name).split("_")[:2]).lower()


def attempts(recs, cap=None):
    """Jaw closes that were grasp attempts. Full folds (cap None): an empty close at the episode's end is the arm parking,
    not an attempt. Step-capped restore episodes (cap = max steps): every empty close counts as a missed attempt, except
    one made within the last 8 steps, which had no time to pinch (truncated)."""
    if cap is None:
        return [r for r in recs if not (r["outcome"] == "empty" and r.get("end") == "episode_end")]
    return [r for r in recs if not (r["outcome"] == "empty" and r.get("end") == "episode_end"
                                    and (r.get("t_open") or cap) - r["t_close"] < 8)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def pct(k, n):
    return f"{100 * k / n:.0f}% ({k}/{n})" if n else "n/a"


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.median(xs):.1f}" if xs else "n/a"


def stats(recs, cap=None):
    att = attempts(recs, cap)
    ne = [r for r in att if r["outcome"] != "empty"]
    held = [r for r in ne if r["outcome"].startswith("held")]
    lift4 = [r for r in ne if (r.get("max_lift_cm") or 0) >= 4]       # the pinched cloth rose >= 4 cm with the jaw
    lift6 = [r for r in ne if (r.get("max_lift_cm") or 0) >= 6]
    eps = {}
    for r in recs:
        k = r.get("_file", r["episode"])
        eps[k] = bool(r.get("success")) or eps.get(k, False)
    return dict(episodes=len(eps), folds=sum(eps.values()), attempts=len(att), pinched=len(ne), held=len(held), lifted4=len(lift4), lifted6=len(lift6),
                outcomes=Counter(r["outcome"] for r in att),
                kept=[100 * r["min_held"] for r in ne if r.get("min_held") is not None],
                slip_lift=[r.get("slip_lift_cm") for r in ne], max_lift=[r.get("max_lift_cm") for r in ne],
                carry=[r.get("carry_cm") for r in ne], layers=Counter(r["layers"] for r in ne),
                n_pinch=[r["n_pinch"] for r in ne], parked=len(recs) - len(att))


def row(name, s):
    return (f"| {name} | {s['episodes']} | {pct(s['folds'], s['episodes'])} | {s['attempts']} | "
            f"{pct(s['pinched'], s['attempts'])} | {pct(s['lifted4'], s['pinched'])} | {pct(s['held'], s['pinched'])} | {med(s['kept'])} | "
            f"{med(s['slip_lift'])} | {med(s['max_lift'])} | {med(s['carry'])} | "
            f"{' '.join(f'{k}:{v}' for k, v in sorted(s['layers'].items()))} |")


HEADER = ("| run / type | episodes | folds passed | attempts | pinched cloth | lifted ≥4 cm | held (≥50% kept, 2 cm) | kept % (median) | "
          "slip at lift cm | max lift cm | carry cm | layers taken |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def fisher(k1, n1, k2, n2):
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
    except Exception:
        return None


def mwu(a, b):
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return None


def decide(a, b):
    """Pre-registered rule (summarize_grasps.py): shares among non-empty grasps. a = new physics, b = original."""
    ha = a["held"] / a["pinched"] if a["pinched"] else None
    hb = b["held"] / b["pinched"] if b["pinched"] else None
    ea = 1 - a["pinched"] / a["attempts"] if a["attempts"] else None
    if ha is None or hb is None:
        return "no data"
    if ha >= 0.8 and ea <= 0.2:
        return "the teacher can grasp in the corrected physics -> put search / SGS into placement and chaining"
    if hb >= 0.8 and ha < 0.5:
        return "physics mismatch: holds in the original physics, fails in the corrected one -> small fine-tune"
    if ha < 0.5 and hb < 0.5:
        return "grasp needs SGS / RL (fails in both)"
    return "in between: decide with Roy"


def report(dirs, labels):
    out = ["# Teacher grasp A/B", ""]
    S = {}
    for d, lab in zip(dirs, labels):
        recs = load(d)
        S[lab] = stats(recs)
        out += [f"## {lab}: `{os.path.basename(d.rstrip('/'))}`", "", HEADER]
        for t in sorted({gtype(r["episode"]) for r in recs}):
            out.append(row(t, stats([r for r in recs if gtype(r["episode"]) == t])))
        out.append(row("ALL", S[lab]))
        s = S[lab]
        nf = n_episode_files(d)
        out += ["", f"- Episodes run: {nf}; episodes in which no jaw closed at all: {nf - s['episodes']}.",
                f"- Parked closes (jaw shut on nothing while returning home): {s['parked']}, not counted as attempts.",
                f"- Held share 95% CI (Wilson): {wilson(s['held'], s['pinched'])}; fold pass 95% CI: {wilson(s['folds'], s['episodes'])}.",
                f"- Outcomes among attempts: {dict(s['outcomes'])}.", ""]
        # kept % vs fold success
        byep = defaultdict(list)
        for r in attempts(recs):
            if r["outcome"] != "empty" and r.get("min_held") is not None:
                byep[r.get("_file", r["episode"])].append((100 * r["min_held"], bool(r.get("success"))))
        ok = [np.mean([k for k, _ in v]) for v in byep.values() if v and v[0][1]]
        bad = [np.mean([k for k, _ in v]) for v in byep.values() if v and not v[0][1]]
        out.append(f"- Mean kept % per episode: passed folds {med(ok)} (n={len(ok)}), failed folds {med(bad)} (n={len(bad)}).")
        out.append("")
    if len(dirs) == 2:
        a, b = S[labels[1]], S[labels[0]]          # a = corrected physics (second dir), b = original (first dir)
        out += ["## A vs B", ""]
        out.append(f"- Lifted share (pinched cloth rose >= 4 cm with the jaw): {labels[0]} {pct(b['lifted4'], b['pinched'])} vs {labels[1]} "
                   f"{pct(a['lifted4'], a['pinched'])}, Fisher p = {fisher(b['lifted4'], b['pinched'], a['lifted4'], a['pinched'])}")
        out.append(f"- Held share: {labels[0]} {pct(b['held'], b['pinched'])} vs {labels[1]} {pct(a['held'], a['pinched'])}, "
                   f"Fisher p = {fisher(b['held'], b['pinched'], a['held'], a['pinched'])}")
        out.append(f"- Pinched share: {labels[0]} {pct(b['pinched'], b['attempts'])} vs {labels[1]} {pct(a['pinched'], a['attempts'])}, "
                   f"Fisher p = {fisher(b['pinched'], b['attempts'], a['pinched'], a['attempts'])}")
        out.append(f"- Folds passed: {labels[0]} {pct(b['folds'], b['episodes'])} vs {labels[1]} {pct(a['folds'], a['episodes'])}, "
                   f"Fisher p = {fisher(b['folds'], b['episodes'], a['folds'], a['episodes'])}")
        out.append(f"- Kept % (median): {med(b['kept'])} vs {med(a['kept'])}, Mann-Whitney p = {mwu(b['kept'], a['kept'])}")
        out.append(f"- Max lift cm (median): {med(b['max_lift'])} vs {med(a['max_lift'])}, p = {mwu(b['max_lift'], a['max_lift'])}")
        out.append(f"- Carry cm (median): {med(b['carry'])} vs {med(a['carry'])}, p = {mwu(b['carry'], a['carry'])}")
        out.append(f"- Layers taken (median): {med(list(b['layers'].elements()))} vs {med(list(a['layers'].elements()))}")
        out += ["", f"**Pre-registered decision:** {decide(a, b)}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    labels = args.labels or (["A original physics", "B corrected physics"][:len(args.dirs)])
    text = report(args.dirs, labels)
    print(text)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(text)
