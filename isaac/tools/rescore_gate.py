"""Re-score stored episode rows with the part of precision_env.gate that needs only the final cloth grid (every vertex
within GATE_Z of the table, the cloth on the table). Old rows have no gripper or motion data, so the clearance, jaw and
stillness checks can't be applied after the fact (re-evaluation 2026-10-22).

    python isaac/tools/rescore_gate.py outputs/isaac/rl/*/demo*/demo.jsonl outputs/isaac/quarter/*/episodes.jsonl
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaac.rl.precision_env import GATE_Z, TABLE_HALF  # noqa: E402
from mujoco_sim.cloth_params import TABLE_TOP_Z  # noqa: E402


def static_gate(cloth):
    c = np.asarray(cloth, float)
    return bool(c[:, 2].max() - TABLE_TOP_Z <= GATE_Z and np.all(np.abs(c[:, :2]) <= TABLE_HALF))


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        rows = [r for r in rows if r.get("cloth")]
        if not rows:
            continue
        ok = [static_gate(r["cloth"]) for r in rows]
        final = lambda r: r.get("final", r)
        passed = [bool(final(r).get("passed")) for r in rows]
        means = [final(r).get("mean") for r in rows]
        bad = [round(m * 100, 2) for m, o in zip(means, ok) if not o and m is not None]
        print(f"{path}: n {len(rows)}, fails the static gate {len(rows) - sum(ok)} (their scores, cm: {sorted(bad)[:6]}), "
              f"passed {sum(passed)}, passed and gated {sum(p and o for p, o in zip(passed, ok))}")


if __name__ == "__main__":
    main()
