"""Summarise a session of the new game: per shirt (episode) the grab/drop sequence and scores, the best
shirt in detail, how much the cloth 'flowed back' after each drop, and which controls were used.
Usage: python analyze_session2.py demos/<session>.jsonl"""
import sys, json
from collections import defaultdict
import numpy as np

recs = [json.loads(l) for l in open(sys.argv[1])]
eps = defaultdict(list)
for r in recs:
    eps[r.get("episode", 0)].append(r)
frames = [r for r in recs if r.get("frame")]
print(f"{len(recs)} records, {len(frames) / 20 / 60:.1f} min of play, {len(eps)} shirts")

summary = []
for e, rs in eps.items():
    drops = [r for r in rs if r.get("event") == "drop"]
    grabs = [r for r in rs if r.get("event") == "grab"]
    if not drops:
        continue
    best = min(drops, key=lambda r: r["score"]["area_cm2"])
    summary.append((best["score"]["area_vs_flat"], e, len(grabs), len(drops), best["score"]))
summary.sort()
print("\nshirts by best area (area % of flat, box, rectangularity):")
for a, e, ng, nd, s in summary[:8]:
    print(f"  shirt {e:3d}: best {100 * a:4.0f}%  box {s['rect_cm']}  rect {s['rectangularity']:.2f}  ({ng} grabs)")

# flowback: how far the cloth moved in the ~1-2 s after a drop when nothing else happened
snaps = [(r["tick"], np.array(r["cloth_cm"])) for r in recs if r.get("frame") and "cloth_cm" in r]
ev_ticks = sorted(r["tick"] for r in recs if r.get("event"))
fb = []
for r in recs:
    if r.get("event") != "drop":
        continue
    t0 = r["tick"]
    nxt = [t for t in ev_ticks if t > t0]
    later = [(t, x) for t, x in snaps if t0 + 60 <= t <= t0 + 150 and (not nxt or t < nxt[0])]
    if later:
        d = np.linalg.norm(later[-1][1] - np.array(r["cloth_cm"]), axis=1)
        fb.append((r.get("episode"), r["arm"], d.mean(), np.percentile(d, 95), d.max(), (later[-1][0] - t0) / 60))
if fb:
    arr = np.array([f[2:5] for f in fb])
    print(f"\nflowback after {len(fb)} drops (cloth movement in the ~1-2.5 s after letting go):")
    print(f"  mean over the shirt: median {np.median(arr[:, 0]):.2f} cm | 95th-pct point: median {np.median(arr[:, 1]):.2f} cm, worst {arr[:, 1].max():.1f} cm | max point: median {np.median(arr[:, 2]):.1f} cm")

# best shirt in detail
a, e, *_ = summary[0]
print(f"\n=== best shirt ({e}): {100 * a:.0f}% of flat ===")
t_first = None
for r in eps[e]:
    if r.get("event") in ("grab", "drop") or r.get("undo"):
        t_first = t_first or r["tick"]
        t = (r["tick"] - t_first) / 60
        if r.get("undo"):
            print(f"  {t:6.1f}s UNDO"); continue
        s = r["score"]
        what = f"{r['mode']:5s} n={r['n']}" if r["event"] == "grab" else f"-> {100 * s.get('area_vs_flat', 1):4.0f}%  box {s['rect_cm']}"
        x, y, z = r["tip_cm"]
        print(f"  {t:6.1f}s {r['arm']} {r['event']:4s} at ({x:6.1f},{y:5.1f}, z {z:4.1f})  {what}")

# controls used
fr = [r for r in frames]
modes = defaultdict(int)
for r in recs:
    if r.get("event") == "grab":
        modes[r["mode"]] += 1
tand = defaultdict(int)
for r in fr:
    tand[r.get("tandem", "off")] += 1
carry = np.array([[r["carry_cm"]["L"], r["carry_cm"]["R"]] for r in fr])
print(f"\ngrab types: {dict(modes)} | tandem use (share of time): { {k: round(v / len(fr), 2) for k, v in tand.items()} }")
print(f"carry height used: L median {np.median(carry[:, 0]):.1f} cm (max {carry[:, 0].max():.1f}), R median {np.median(carry[:, 1]):.1f} cm (max {carry[:, 1].max():.1f})")
# where were the arms when they sat still while carrying/hovering high? (likely the reach limit)
for a_ in ("L", "R"):
    tgt = np.array([r["target_cm"][a_] for r in fr]); cur = np.array([r["cursor_cm"][a_] for r in fr])
    still = np.r_[False, np.linalg.norm(np.diff(tgt[:, :2], axis=0), axis=1) < 1e-6]
    hi = tgt[:, 2] > 4.5
    print(f"{a_}: time with aim point above 4.5 cm: {hi.mean() * 100:.0f}%; of that, frozen in place: {(still & hi).sum() / max(hi.sum(), 1) * 100:.0f}% "
          f"(vs {(still & ~hi).sum() / max((~hi).sum(), 1) * 100:.0f}% when lower)")
