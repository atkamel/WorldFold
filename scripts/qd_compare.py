"""QD (2026-10-16): paired comparison of quarter-fold eval JSONs on the same seeds, with a "real second fold" count:
a success whose stage 1 took more than FREE_STEPS steps (fold 1 had not already left the crease end on its corner).
    .venv/Scripts/python.exe scripts/qd_compare.py [--set sel_id_hard] base.json other.json [more.json]
--set picks one eval set from reports that hold several (QD-2 scale); default: the first set in each report.
"""
import json
import math
import sys

FREE_STEPS = 25      # stage 1 done within the 20-step settle (+ slack): no second fold was needed


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(path, set_name=None):
    r = json.load(open(path))["results"]
    eps = (r[set_name] if set_name else next(iter(r.values())))["episodes"]
    return {e["seed"]: e for e in eps}


def real(e):
    return e["success"] and e["steps"] - (e["stage_steps"][0] if e["stage_steps"] else 0) > FREE_STEPS


def main():
    argv, set_name = list(sys.argv[1:]), None
    if argv and argv[0] == "--set":
        set_name, argv = argv[1], argv[2:]
    runs = [(p, load(p, set_name)) for p in argv]
    if set_name:
        print(f"set {set_name}")
    seeds = sorted(set.intersection(*[set(r) for _, r in runs]))
    print(f"n = {len(seeds)} paired seeds")
    print("run                                         success   (Wilson)     real 2nd fold  fold 1 done  codes")
    for p, r in runs:
        k = sum(r[s]["success"] for s in seeds)
        kr = sum(real(r[s]) for s in seeds)
        f1 = sum(r[s]["stages_done"] >= 1 for s in seeds)
        codes = {}
        for s in seeds:
            if r[s]["code"]:
                key = f"{r[s]['code']}@{r[s]['fail_stage']}"
                codes[key] = codes.get(key, 0) + 1
        lo, hi = wilson(k, len(seeds))
        print(f"{p[-42:]:42s}  {k:2d}/{len(seeds)}  ({lo:4.0f}-{hi:3.0f})   {kr:2d}/{len(seeds)}         "
              f"{f1:2d}/{len(seeds)}      {codes}")
    if len(runs) >= 2:
        (pa, a) = runs[0]
        for pb, b in runs[1:]:
            ab = sum(a[s]["success"] and not b[s]["success"] for s in seeds)
            ba = sum(b[s]["success"] and not a[s]["success"] for s in seeds)
            print(f"discordant vs first: first-only {ab}, {pb[-30:]}-only {ba}")


if __name__ == "__main__":
    main()
