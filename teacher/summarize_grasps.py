"""Grasp table for teacher runs (run_teacher_grasp.sh output dirs) and the pre-registered decision.

    python summarize_grasps.py <run dir> [<control run dir>]

One dir: per garment type and overall. Two dirs (new physics first, original physics second): also the decision:
  held >= 80% of non-empty grasps in the new physics and <= 20% empty  -> the teacher can grasp
  held >= 80% in the original physics but < 50% in the new one        -> physics mismatch (small fine-tune)
  held < 50% in both                                                   -> grasp needs SGS/RL
  anything else                                                        -> in between: decide with Roy
"""
import glob
import json
import os
import sys

OUTCOMES = ("empty", "slipped", "held_lifted", "held_dragged", "held_static")


def load(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "grasps", "*.jsonl"))):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    return recs, len(glob.glob(os.path.join(d, "grasps", "*.jsonl")))


def gtype(name):
    return "_".join(str(name).split("_")[:2]).lower()


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 2) if xs else None


def table(recs):
    # a jaw closed on nothing at the episode's end (parking) is not a grasp attempt
    att = [r for r in recs if not (r["outcome"] == "empty" and r.get("end") == "episode_end")]
    n = len(att)
    row = {"grasps": len(recs), "attempts": n, "parked_closes": len(recs) - n,
           **{k: sum(r["outcome"] == k for r in att) for k in OUTCOMES}}
    ne = [r for r in att if r["outcome"] != "empty"]
    row["empty_share"] = round(row["empty"] / n, 3) if n else None
    row["held_share"] = round(sum(r["outcome"].startswith("held") for r in ne) / len(ne), 3) if ne else None
    row["kept_pct_median"] = _median([100 * r["min_held"] for r in ne if r.get("min_held") is not None])
    row["slip_lift_cm_median"] = _median([r.get("slip_lift_cm") for r in ne])
    row["max_lift_cm_median"] = _median([r.get("max_lift_cm") for r in ne])
    row["carry_cm_median"] = _median([r.get("carry_cm") for r in ne])
    row["layers(non-empty)"] = {str(k): sum(r["layers"] == k for r in ne) for k in sorted({r["layers"] for r in ne})}
    eps = {}
    for r in recs:
        eps[r["episode"]] = bool(r.get("success")) or eps.get(r["episode"], False)
    row["episodes"] = len(eps)
    row["fold_success"] = sum(eps.values())
    return row


def report(d):
    recs, eps = load(d)
    print(f"== {d}: {eps} episodes, {len(recs)} grasps")
    for t in sorted({gtype(r["episode"]) for r in recs}):
        sub = [r for r in recs if gtype(r["episode"]) == t]
        print(f"  {t:11s}", json.dumps(table(sub)))
    over = table(recs)
    print(f"  {'ALL':11s}", json.dumps(over))
    return over


def decide(a, b):
    ha, hb, ea = a["held_share"], b["held_share"], a["empty_share"]
    if ha is None or hb is None:
        return "no data"
    if ha >= 0.8 and (ea or 0) <= 0.2:
        return "the teacher can grasp (new physics): put search/SGS into placement and chaining"
    if hb >= 0.8 and ha < 0.5:
        return "physics mismatch: holds in the original physics, fails in the new one -> small fine-tune"
    if ha < 0.5 and hb < 0.5:
        return "grasp needs SGS/RL (fails in both)"
    return "in between: decide with Roy"


if __name__ == "__main__":
    a = report(sys.argv[1])
    if len(sys.argv) > 2:
        b = report(sys.argv[2])
        print("DECISION:", decide(a, b))
