"""Held-corner summary of scripted-expert rows (review 2026-11-10): the pass rate hid that 31 of the stiff cloth's 35
strict passes had lost the right corner mid-carry, so the grasp is judged on its own. A row counts as held when the
episode reached its window and no arm's corner left the jaws in the carry (the expert's HOLD_DIST rule, `drop_at`).

  .venv/bin/python isaac/tools/held_summary.py outputs/isaac/rl/grasp_k1/* [--control k0_lehome]

One line per config: n, held (Wilson 95%; lenient: ride-up flags count as held), carry drops per arm, the working bar (2 / 3 cm, full gate) and the strict
bar; with --control, a two-sided Fisher p on held against that config.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ARMS = ("left_", "right_")


def carry_drops(row):
    """The arms whose corner left the jaws in the carry."""
    if "drop_at" not in row:             # rows from before drops logged their phase (2026-11-09): every flag counts
        return [p for p in ARMS if p in (row.get("dropped") or [])]
    at = row.get("drop_at") or {}
    return [p for p in ARMS if (at.get(p) or {}).get("phase") == "carry"]


def rode_up(row, arm):
    """A carry flag at the path's last waypoints with the corner over 2 cm above the site: it rode up the pads past the
    5 cm rule while still pinched (batch K review: 1-2 of 32 rows per config, each followed by a good fold)."""
    d = (row.get("drop_at") or {}).get(arm) or {}
    wp = d.get("wp") or [0, 0]
    return d.get("phase") == "carry" and d.get("dz_cm", 0) > 2.0 and wp[0] >= wp[1] - 2


def held(row, lenient=False):
    """lenient: ride-up flags count as held."""
    drops = [p for p in carry_drops(row) if not (lenient and rode_up(row, p))]
    return not row.get("no_window") and row.get("reason") != "unstable" and not drops


def within(row, mean_bar, edge_bar):
    g = row.get("gate") or {}
    return bool(row.get("reason") == "settled" and g.get("valid") and row.get("mean") is not None
                and row["mean"] <= mean_bar and row["edge_max"] <= edge_bar)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact p for k1/n1 against k2/n2."""
    def pmf(k):
        return math.comb(k1 + k2, k) * math.comb(n1 + n2 - k1 - k2, n1 - k) / math.comb(n1 + n2, n1)
    p0 = pmf(k1)
    return min(1.0, sum(pmf(k) for k in range(max(0, k1 + k2 - n2), min(n1, k1 + k2) + 1) if pmf(k) <= p0 * (1 + 1e-9)))


def summarise(rows):
    n = len(rows)
    out = {"n": n, "held": sum(held(r) for r in rows), "held_lenient": sum(held(r, lenient=True) for r in rows), "no_window": sum(bool(r.get("no_window")) for r in rows),
           "working": sum(within(r, 0.02, 0.03) for r in rows), "strict": sum(within(r, 0.01, 0.015) for r in rows),
           "held_working": sum(held(r) and within(r, 0.02, 0.03) for r in rows)}
    out |= {f"drop_{p}": sum(p in carry_drops(r) for r in rows) for p in ARMS}
    return out


def load(run):
    path = Path(run) / "demo_base" / "demo.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="config directories holding demo_base/demo.jsonl")
    ap.add_argument("--control", default="", help="config name to test held against")
    args = ap.parse_args()
    stats = {Path(r).name: summarise(rows) for r in args.runs if (rows := load(r)) is not None}
    ctrl = stats.get(args.control)
    for name, s in stats.items():
        lo, hi = wilson(s["held"], s["n"])
        line = (f"{name:24s} n {s['n']:4d}  held {s['held']:4d} ({100 * s['held'] / max(s['n'], 1):5.1f}% [{lo:.1f}, {hi:.1f}])"
                f" lenient {s['held_lenient']}"
                f"  carry drops L {s['drop_left_']} R {s['drop_right_']}  no window {s['no_window']}"
                f"  2/3 {s['working']} (held {s['held_working']})  1/1.5 {s['strict']}")
        if ctrl and s is not ctrl:
            line += f"  p {fisher(s['held'], s['n'], ctrl['held'], ctrl['n']):.3f}"
        print(line)


if __name__ == "__main__":
    main()
