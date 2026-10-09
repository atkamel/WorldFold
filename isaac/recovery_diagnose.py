"""Per-attempt analysis of recovery_replay traces (Phase F3b diagnosis): for every close after the knock, per arm --
where the corner was relative to the jaw when it closed, whether cloth was folded over it, and whether it then rose
with the jaw. Prints one line per attempt and a mechanism tally. No Isaac.

    python isaac/recovery_diagnose.py outputs/isaac/recovery_friction/diag/traces
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LIFT_CHECK = 6          # steps after the close ends at which "did the corner rise with the jaw" is judged


def attempts(arm, knock_end):
    """(close_end, lift_window) per close that ends after the knock."""
    ph = arm["phase"]
    out = []
    for t in range(1, len(ph)):
        if ph[t - 1] == "close" and ph[t] != "close" and t >= knock_end:
            out.append(t)
    return out


def classify(arm, t):
    site, corner = np.asarray(arm["site"]), np.asarray(arm["corner"])
    d_close = float(np.linalg.norm(corner[t - 1] - site[t - 1]))
    t2 = min(t + LIFT_CHECK, len(site) - 1)
    jaw_rise = site[t2, 2] - site[t - 1, 2]
    corner_rise = corner[t2, 2] - corner[t - 1, 2]
    follows = jaw_rise > 0.01 and corner_rise > 0.5 * jaw_rise
    fold = bool(arm["fold"][t - 1])
    held_later = any(arm["g"][t:t2 + 1])
    if follows:
        mech = "good grip"
    elif d_close > 0.03:
        mech = "closed off the corner (>3 cm)"
    elif fold:
        mech = "corner under a fold"
    else:
        mech = "near the corner, corner did not rise"
    return {"t": t, "d_close_cm": round(100 * d_close, 1), "fold": fold, "jaw_rise_cm": round(100 * jaw_rise, 1),
            "corner_rise_cm": round(100 * corner_rise, 1), "held": held_later, "mech": mech}


def main(trace_dir):
    tally = Counter()
    for f in sorted(Path(trace_dir).glob("*.json")):
        tr = json.loads(f.read_text(encoding="utf-8"))
        knock_end = tr["knock"][0] + tr["knock"][1]
        for p, arm in tr["arms"].items():
            # how long each post-knock descend lasted
            ph = arm["phase"]
            runs, cur, n = [], None, 0
            for x in ph[knock_end:]:
                if x == cur:
                    n += 1
                else:
                    if cur == "descend":
                        runs.append(n)
                    cur, n = x, 1
            for t in attempts(arm, knock_end):
                c = classify(arm, t)
                tally[c["mech"]] += 1
                print(tr["seed"], p, c, "descend runs", runs)
    print("\nmechanism tally:", dict(tally))


if __name__ == "__main__":
    main(sys.argv[1])
