"""Summarise where the quarter fold's first fold leaves the cloth (Q3): rows from
`isaac/recovery_replay.py --task quarter_fold --stop-at-stage 1`.

Per seed: stack spread |v0 - v10| and |v110 - v120| (the stage-0 corners, which should lie on each other), strip width at
the west and east edges |v5 - v0|, |v115 - v110| in y (15 cm when the crease is where it should be), crease shift of
v5 / v115 in y from their reset pose, and drift of the anchors v0 / v110 from reset (cm).

    python isaac/fold1_summary.py outputs/imitation/isaac_friction/q3/fold1_base.jsonl [more.jsonl ...]
"""
import json
import sys

import numpy as np


def summarise(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("stage_end")]
    keys = ("spread_W", "spread_E", "width_W", "width_E", "crease_W", "crease_E", "drift_v0", "drift_v110")
    table = []
    for r in rows:
        e, s = r["stage_end"], r["stage_end"]["start"]
        p = {k: np.array(v) for k, v in e.items() if k not in ("t", "start")}
        p0 = {k: np.array(v) for k, v in s.items()}
        table.append({"seed": r["seed"], "t": e["t"],
                      "spread_W": np.linalg.norm(p["0"] - p["10"]), "spread_E": np.linalg.norm(p["110"] - p["120"]),
                      "width_W": abs(p["5"][1] - p["0"][1]), "width_E": abs(p["115"][1] - p["110"][1]),
                      "crease_W": p["5"][1] - p0["5"][1], "crease_E": p["115"][1] - p0["115"][1],
                      "drift_v0": np.linalg.norm(p["0"] - p0["0"]), "drift_v110": np.linalg.norm(p["110"] - p0["110"])})
    print(f"== {path}  (n = {len(table)}, cm)")
    print("seed    t  " + "  ".join(f"{k:>10s}" for k in keys))
    for row in table:
        print(f"{row['seed']} {row['t']:4d}  " + "  ".join(f"{100 * row[k]:10.1f}" for k in keys))
    if table:
        print("mean        " + "  ".join(f"{100 * np.mean([r[k] for r in table]):10.1f}" for k in keys))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(p)
