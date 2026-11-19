"""Train the learned half fold (policy_net.py) on planned folds (policy_data.py), on the CPU. Writes the weights
(.npz) and, beside them, the JSON policy_net.Policy reads (normalisation, rows, training summary).

    .venv/bin/python hardware/so101/policy_train.py --data outputs/policy/train.npz \
        --out hardware/so101/policies/half_fold_v0.npz

Each sample is one arm at one row of one fold: the observation that row would give, the planner's next target as
the label. The fold is padded with its last row for PAD rows, so the network learns to hold still at the end. The
"measured joints" are drawn afresh every epoch: the planner's rows up to LAG_ROWS behind the row before (a servo
that lags its target), plus up to OFFSET_DEG per joint (one that settles off it, or a jaw stopped by the cloth).
The label stays the planner's row, so the network learns to follow the plan whatever the servo does.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import policy_data
import policy_net
from policy_net import DT, MIRROR_JOINTS, OBS, START

HIDDEN = (256, 256, 256)
PAD = 20                # rows of the last pose after each fold (1 s)
LAG_ROWS = 3.0          # the measured joints are the plan up to this many rows behind the row before
OFFSET_DEG = 3.0        # plus up to this much per joint
BATCH = 1024
LR = 2e-3
STD_FLOOR = 0.01        # an input that never varies (wrist_roll) is not blown up by its tiny spread


def samples(data):
    """Per sample (arrays over N): the fixed part of the observation (both arms' points [N, 8], side [N]), the time
    [N], the 4 planner rows before the labelled one [N, 4, 6] (row 0's own for rows before it), whether it is the
    first row (its measured joints are START, without noise) and the label [N, 6]. Plus the rows played."""
    rows = max(len(targets) for _, _, targets, _ in data["episodes"]) + PAD
    fixed, side, times, before, first, label = [], [], [], [], [], []
    k = np.arange(rows)
    for _, points, targets, _ in data["episodes"]:
        named = {"left": points[0], "right": points[1]}
        for s, name in enumerate(("left", "right")):
            other = "right" if name == "left" else "left"
            plan = targets[:, s] * (MIRROR_JOINTS if name == "right" else 1.0)
            plan = np.concatenate([plan, np.repeat(plan[-1:], rows - len(plan), axis=0)])
            fixed.append(np.repeat(np.concatenate([policy_net.canonical_points(name, named[name]),
                                                   policy_net.canonical_points(other, named[other])])[None], rows, 0))
            side.append(np.full(rows, policy_net.SIDES[name]))
            times.append(k * DT)
            before.append(plan[np.clip(k[:, None] - 1 - np.arange(4), 0, None)])
            first.append(k == 0)
            label.append(plan)
    return [np.concatenate(a) for a in (fixed, side, times, before, first, label)], rows


def measured(before, first, generator):
    """The measured joints of a batch: the plan a random 0 to LAG_ROWS rows behind the row before, plus a random
    offset per joint; START for a first row."""
    lag = torch.rand(len(before), generator=generator) * LAG_ROWS
    i = lag.floor().long().clamp(max=2)
    f = (lag - i)[:, None]
    rows = torch.arange(len(before))
    joints = (1 - f) * before[rows, i] + f * before[rows, i + 1]
    joints = joints + (torch.rand(joints.shape, generator=generator) * 2 - 1) * np.radians(OFFSET_DEG)
    return torch.where(first[:, None], torch.as_tensor(START, dtype=torch.float32), joints)


def observe(fixed, side, times, joints):
    """OBS vectors from the parts, in OBS's order."""
    return torch.cat([fixed, joints, times[:, None], side[:, None]], dim=1)


def build(inputs, hidden=HIDDEN, outputs=len(policy_net.ACT)):
    sizes = [inputs, *hidden]
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(sizes[-1], outputs))


def save(model, path, meta):
    """The weights as w0, b0, ... ([in, out] and [out]) in `path`, `meta` in the JSON beside it."""
    linear = [m for m in model if isinstance(m, torch.nn.Linear)]
    weights = {}
    for i, m in enumerate(linear):
        weights[f"w{i}"] = m.weight.detach().numpy().T.astype(np.float32)
        weights[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)
    Path(path).with_suffix(".json").write_text(json.dumps(meta | {"layers": len(linear)}, indent=1))


def train(data_path, out, epochs, seed):
    started = time.time()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    data = policy_data.load(data_path)
    (fixed, side, times, before, first, label), rows = samples(data)
    clean = np.concatenate([fixed, np.where(first[:, None], START, before[:, 0]), times[:, None], side[:, None]], 1)
    obs_mean, obs_std = clean.mean(0), np.maximum(clean.std(0), STD_FLOOR)
    act_mean, act_std = label.mean(0), np.ones(label.shape[1])     # the loss is in radians, every joint alike
    t = {name: torch.as_tensor(a, dtype=torch.float32) for name, a in
         (("fixed", fixed), ("side", side), ("times", times), ("before", before), ("label", label))}
    t["first"] = torch.as_tensor(first)
    norm = [torch.as_tensor(a, dtype=torch.float32) for a in (obs_mean, obs_std, act_mean, act_std)]
    model = build(len(OBS))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    steps = epochs * int(np.ceil(len(label) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LR, total_steps=steps, pct_start=0.05)
    print(f"{len(data['episodes'])} folds, {len(label)} samples, {rows} rows, {steps} steps")
    for epoch in range(epochs):
        order = torch.randperm(len(label), generator=generator)
        total = 0.0
        for start in range(0, len(label), BATCH):
            b = order[start:start + BATCH]
            obs = observe(t["fixed"][b], t["side"][b], t["times"][b], measured(t["before"][b], t["first"][b], generator))
            predicted = model((obs - norm[0]) / norm[1]) * norm[3] + norm[2]
            loss = torch.mean((predicted - t["label"][b]) ** 2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(b)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: rms {np.degrees(np.sqrt(total / len(label))):.3f} deg, "
                  f"{time.time() - started:.0f} s", flush=True)
    with torch.no_grad():                                   # the error with the measured joints on the plan
        obs = torch.as_tensor(clean, dtype=torch.float32)
        error = (model((obs - norm[0]) / norm[1]) * norm[3] + norm[2] - t["label"]).numpy().astype(np.float64)
    layouts = [layout for layout, *_ in data["episodes"]]
    summary = {"data": str(data_path), "seed": seed, "folds": len(layouts),
               "folds_by_layout": {name: layouts.count(name) for name in policy_data.LAYOUTS},
               "refused_in_data": int(len(data["refused_layout"])), "samples": int(len(label)), "epochs": epochs,
               "train_seconds": round(time.time() - started), "torch": torch.__version__,
               "rms_error_deg_on_plan": dict(zip(policy_net.JOINTS, np.round(np.degrees(np.sqrt((error ** 2).mean(0))), 3).tolist())),
               "max_error_deg_on_plan": dict(zip(policy_net.JOINTS, np.round(np.degrees(np.abs(error).max(0)), 3).tolist()))}
    meta = {"obs": OBS, "act": policy_net.ACT, "hidden": list(HIDDEN), "dt": DT, "rows": rows,
            "obs_mean": obs_mean.tolist(), "obs_std": obs_std.tolist(), "act_mean": act_mean.tolist(),
            "act_std": act_std.tolist(), "start": START.tolist(),
            "noise": {"lag_rows": LAG_ROWS, "offset_deg": OFFSET_DEG}, "training": summary}
    save(model, out, meta)
    print(json.dumps(summary, indent=1))
    print(f"Wrote {out} and {Path(out).with_suffix('.json')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="policy_data.py output")
    ap.add_argument("--out", required=True, help="weights .npz; the JSON goes beside it")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.seed)


if __name__ == "__main__":
    main()
