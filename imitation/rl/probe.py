"""Probe an IQL critic: can it rank states, and can it rank actions? (M5.3 diagnosis)

    python -m imitation.rl.probe --run runs/iql_v1 --versions harvest_v1

For each outcome group (success / G1 / F1 / S1 / M1) it reports the mean state value V
and the advantage A = min(Q1, Q2)(s, a_logged) - V(s). A critic that only ranks states
shows V separating the groups while A is ~0 +- noise in every group. Then advantage-
weighted extraction is plain BC over all the data. That is what `harvest_v1` produced.
Writes `probe.json` next to the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from imitation.data.collect import DEFAULT_ROOT
from imitation.data.schema import load_dataset
from imitation.policies.common import default_device, load_policy
from imitation.rl.iql import Critics
from imitation.rl.transitions import CODES, build_transitions


@torch.no_grad()
def probe(run, versions, root=DEFAULT_ROOT, macro=8):
    run = Path(run)
    info = json.loads((run / "run.json").read_text())
    device = default_device()
    policy = load_policy(info["init"], device)
    critics = Critics(policy.obs_horizon * policy.obs_dim, macro * policy.action_dim).to(device)
    critics.load_state_dict(torch.load(run / "critics.pt", map_location=device))
    critics.eval()
    eps = []
    for v in versions:
        eps += load_dataset(root, v)[1]
    tr = build_transitions(eps, obs_horizon=policy.obs_horizon, macro=macro)
    s = policy.normalize_obs(torch.as_tensor(tr["obs"], device=device)).flatten(1)
    a = torch.as_tensor(tr["act"] * tr["valid"][:, :, None], device=device).flatten(1)
    v = critics.v(s).squeeze(-1)
    adv = (torch.min(*critics.q(s, a)) - v).cpu().numpy()
    v = v.cpu().numpy()
    out = {}
    for i, c in enumerate(CODES):
        m = tr["code"] == i
        if m.any():
            out[c] = {"n": int(m.sum()), "V_mean": float(v[m].mean()), "A_mean": float(adv[m].mean()),
                      "A_std": float(adv[m].std()), "A_p5": float(np.percentile(adv[m], 5)),
                      "A_p95": float(np.percentile(adv[m], 95))}
    ok, bad = tr["code"] == 0, tr["code"] != 0
    out["summary"] = {"V_gap_success_minus_failed": float(v[ok].mean() - v[bad].mean()) if bad.any() else None,
                      "A_gap_success_minus_failed": float(adv[ok].mean() - adv[bad].mean()) if bad.any() else None,
                      "A_std_all": float(adv.std())}
    (run / "probe.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--versions", nargs="+", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    print(json.dumps(probe(args.run, args.versions, args.root), indent=1))


if __name__ == "__main__":
    main()
