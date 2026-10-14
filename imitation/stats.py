"""Benchmark statistics (Phase Q0): Wilson intervals per policy, and paired tests between policies.

Every policy is evaluated on the same seeds of a set, so two policies' outcomes pair up seed by seed. A paired test
only looks at the seeds where they disagree, which is much tighter than treating the two runs as independent
binomials (the old `dagger.gain_in_se`):

- `mcnemar(b, c)`: exact two-sided McNemar p-value. b = seeds only A solved, c = seeds only B solved.
- `paired(rows_a, rows_b)`: aligns per-episode rows (see `imitation.evaluate.episode_row`) by (set, seed), and
  returns the success difference, its paired standard error, the McNemar p-value and a paired bootstrap CI.
- `paired_gain_se(new, old, sets)`: the DAgger keep rule on eval reports; paired when both reports carry rows, the
  independent-binomial fallback otherwise (old JSONs have no rows).
"""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent."""
    if n <= 0:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b vs c discordant pairs (binomial test of b ~ Bin(b + c, 1/2))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _key(row, set_name=None):
    return (row.get("set", set_name), int(row["seed"]))


def align(rows_a, rows_b):
    """Success pairs (a, b) over the (set, seed) keys both row lists share, in a fixed order."""
    a = {_key(r): bool(r["success"]) for r in rows_a}
    b = {_key(r): bool(r["success"]) for r in rows_b}
    keys = sorted(set(a) & set(b), key=lambda k: (str(k[0]), k[1]))
    return np.array([a[k] for k in keys], dtype=bool), np.array([b[k] for k in keys], dtype=bool)


def paired(rows_a, rows_b, n_boot=2000, seed=0) -> dict:
    """Paired comparison of A against B on their shared seeds: diff = rate(A) - rate(B)."""
    a, b = align(rows_a, rows_b)
    n = len(a)
    if n == 0:
        raise ValueError("no shared (set, seed) pairs")
    d = a.astype(float) - b.astype(float)
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "rate_a": float(a.mean()), "rate_b": float(b.mean()), "diff": float(d.mean()),
            "se": se, "only_a": only_a, "only_b": only_b, "p_mcnemar": mcnemar(only_a, only_b),
            "ci95": [float(lo), float(hi)]}


def rows_of(report: dict, sets) -> list[dict] | None:
    """All per-episode rows of `sets` in an eval report's results, tagged with their set; None if any set lacks rows."""
    out = []
    for s in sets:
        r = report.get(s)
        if r is None:
            continue
        if not r.get("episodes"):
            return None
        out += [{**e, "set": s} for e in r["episodes"]]
    return out


def independent_gain_se(new, old, sets) -> float:
    """Summed success gain over `sets` in SE, treating the two reports as independent binomials."""
    def score(r):
        return sum(r[k]["success_rate"] for k in sets if k in r)
    var = sum(r[k]["success_rate"] * (1 - r[k]["success_rate"]) / r[k]["n"]
              for r in (new, old) for k in sets if k in r)
    return (score(new) - score(old)) / max(var, 1e-12) ** 0.5


def paired_gain_se(new, old, sets) -> float:
    """Gain of `new` over `old` in paired SEs (mean per-seed difference / its SE) when both reports carry
    per-episode rows; the independent-binomial fallback otherwise."""
    ra, rb = rows_of(new, sets), rows_of(old, sets)
    if ra is None or rb is None:
        return independent_gain_se(new, old, sets)
    p = paired(ra, rb)
    if p["only_a"] + p["only_b"] == 0:
        return 0.0
    return p["diff"] / p["se"]


def flip_rate(rows_a, rows_b) -> dict:
    """Run-to-run noise floor: the share of seeds whose outcome differs between two runs of the same policy."""
    a, b = align(rows_a, rows_b)
    flips = int((a != b).sum())
    lo, hi = wilson(flips, len(a))
    return {"n": len(a), "flips": flips, "rate": flips / max(len(a), 1), "wilson": [lo, hi],
            "p_mcnemar": mcnemar(int((a & ~b).sum()), int((b & ~a).sum()))}


def compare_reports(a: dict, b: dict, sets) -> dict:
    """Per-set and pooled paired comparison of two eval reports (their "results" dicts) that carry rows."""
    out = {}
    for s in list(sets) + ["pooled"]:
        ra, rb = rows_of(a, sets if s == "pooled" else [s]), rows_of(b, sets if s == "pooled" else [s])
        if ra and rb:
            out[s] = paired(ra, rb)
    return out


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="paired comparison of two eval reports on their shared seeds")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--sets", nargs="+", default=None, help="default: the sets both reports have")
    ap.add_argument("--flips", action="store_true", help="report the run-to-run flip rate (same policy twice)")
    args = ap.parse_args()
    ra, rb = (json.loads(Path(p).read_text())["results"] for p in (args.a, args.b))
    sets = args.sets or [s for s in ra if s in rb]
    if args.flips:
        for s in sets:
            print(s, json.dumps(flip_rate(rows_of(ra, [s]), rows_of(rb, [s]))))
        return
    for s, p in compare_reports(ra, rb, sets).items():
        print(f"{s:12s} n {p['n']:4d}  A {p['rate_a']:.1%}  B {p['rate_b']:.1%}  diff {p['diff']:+.1%} "
              f"[{p['ci95'][0]:+.1%}, {p['ci95'][1]:+.1%}]  only A {p['only_a']} / only B {p['only_b']}  "
              f"McNemar p {p['p_mcnemar']:.3f}")


if __name__ == "__main__":
    main()
