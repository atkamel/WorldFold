"""Small selectors over run histories, for driver scripts (prints one line).

    python -m imitation.viz.pick best --history runs/dagger_shift/history.json
    python -m imitation.viz.pick winner --a runs/dagger_diff/history.json --b runs/dagger_v2/history.json --sets id_easy recovery
    python -m imitation.viz.pick winnerdata ...      (the winner's last aggregated dataset)
    python -m imitation.viz.pick lastdata --history runs/dagger_v2/history.json
    python -m imitation.viz.pick same --a eval1.json --b eval2.json

`winner` keeps `b` (the incumbent) unless `a`'s best checkpoint beats it by more than
`--min-se` standard errors on `--sets` (the M5b.1 exit rule).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation.dagger import gain_in_se


def _hist(p):
    return json.loads(Path(p).read_text())


def best(history):
    kept = [h for h in history if h["kept"]][-1]
    return kept["checkpoint"].replace("\\", "/"), kept["eval"]


def best_by(history, sets):
    """The round (any, kept or not) with the highest summed success on `sets`."""
    from imitation.dagger import score
    h = max(history, key=lambda r: score(r["eval"], tuple(sets)))
    return h["checkpoint"].replace("\\", "/"), h["eval"]


def winner(a, b, sets, min_se=1.0, carry_sets=None):
    """Run `a` wins if its kept best beats `b`'s by > min_se SE on `sets` (the exit rule).
    The checkpoint carried forward from the winning run is its best round on `carry_sets`
    (default: its kept best) -- a round can win on id_easy + recovery while losing id_hard."""
    ha, hb = _hist(a), _hist(b)
    (ca, ea), (cb, eb) = best(ha), best(hb)
    h, c = (ha, ca) if gain_in_se(ea, eb, tuple(sets)) > min_se else (hb, cb)
    if carry_sets:
        c = best_by(h, carry_sets)[0]
    return c, h[-1]["dataset"]


def same(a, b):
    ra, rb = _hist(a), _hist(b)
    ra, rb = ra.get("results", ra), rb.get("results", rb)
    keys = ("success_rate", "n", "failure_codes", "termination")
    diff = [k for k in ra if any(ra[k].get(x) != rb.get(k, {}).get(x) for x in keys)]
    return "identical" if not diff else f"DIFFERENT on {diff}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["best", "winner", "winnerdata", "lastdata", "same"])
    ap.add_argument("--history")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--sets", nargs="+", default=["id_easy", "recovery"])
    ap.add_argument("--min-se", type=float, default=1.0)
    ap.add_argument("--carry-sets", nargs="+", default=None)
    args = ap.parse_args()
    if args.cmd == "best":
        print(best(_hist(args.history))[0])
    elif args.cmd == "lastdata":
        print(_hist(args.history)[-1]["dataset"])
    elif args.cmd in ("winner", "winnerdata"):
        ckpt, data = winner(args.a, args.b, args.sets, args.min_se, args.carry_sets)
        print(ckpt if args.cmd == "winner" else data)
    else:
        print(same(args.a, args.b))


if __name__ == "__main__":
    main()
