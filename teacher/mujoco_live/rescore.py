"""Re-score recorded folds with the crumple-proof metric: shirt 3 step by step, then every shirt's best."""
import sys, json
from collections import defaultdict
import numpy as np
from shirt import shirt_mesh
from neat_metric import fold_score

pts, tris = shirt_mesh(0.01)
H = 0.004
flat = np.c_[pts, np.full(len(pts), H / 2)]
FLAT_AREA = fold_score(flat, tris, 1.0, H)["area_ratio"]  # flat footprint in m2 (area_ratio with flat_area=1)
S = lambda c: fold_score(np.array(c) / 100, tris, FLAT_AREA, H)

recs = [json.loads(l) for l in open(sys.argv[1])]
eps = defaultdict(list)
for r in recs:
    eps[r.get("episode", 0)].append(r)

print(f"flat shirt: {S(flat * 100)}")
print("\nshirt 3, after each drop:")
t0 = None
for r in eps[3]:
    if r.get("event") == "drop":
        t0 = t0 or r["tick"]
        s = S(r["cloth_cm"])
        print(f"  t={(r['tick'] - t0) / 60:6.1f}s  area {100 * s['area_ratio']:4.0f}%  packing {s['packing']:.2f}  rect {s['rect']:.2f}  "
              f"height {s['height_mm']:5.1f} mm  ->  SCORE {s['score']:5.1f}")

print("\nevery shirt: best by the old metric (smallest area) vs best by the new score:")
rows = []
for e, rs in eps.items():
    drops = [r for r in rs if r.get("event") == "drop"]
    if not drops:
        continue
    scored = [(S(r["cloth_cm"]), r) for r in drops]
    by_area = min(scored, key=lambda t: t[0]["area_ratio"])
    by_score = max(scored, key=lambda t: t[0]["score"])
    rows.append((by_score[0]["score"], e, by_area[0], by_score[0]))
rows.sort(reverse=True)
for sc, e, a, b in rows[:10]:
    print(f"  shirt {e:2d}: smallest area {100 * a['area_ratio']:3.0f}% (score {a['score']:5.1f}, packing {a['packing']:.2f})   "
          f"| best score {b['score']:5.1f} at area {100 * b['area_ratio']:3.0f}%, packing {b['packing']:.2f}, rect {b['rect']:.2f}, box {b['box_cm']}")
