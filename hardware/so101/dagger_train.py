"""Train the learned camera fold (dagger_net.py), in numpy on the CPU. Two uses:

    # to start it, from planned folds alone (dagger_data.py); the last tenth of the folds is held out and reported
    .venv/bin/python hardware/so101/dagger_train.py --base outputs/hardware/dagger/base.npz \
        --out hardware/so101/policies/camera_fold_v0.npz
    # after folds on the arms: carry on from weights, with the rows dagger_run.py saved (it runs this itself)
    .venv-hw/bin/python hardware/so101/dagger_train.py --base outputs/hardware/dagger/base.npz \
        --start WEIGHTS.npz --rows hardware/so101/runs/dagger --out NEW_WEIGHTS.npz

A planned sample is one arm at one row of one fold: the label is the planner's row. Its "measured joints" are
drawn afresh every batch as in policy_train.py: the plan up to LAG_ROWS behind the row before, plus up to
OFFSET_DEG a joint. Its "row sent last" is the planner's row before, up to SENT_DEG off in every joint, so the
network learns to come back to the plan from a row that was not on it. A row from the arms (dagger_run.py) is
used as it was there: the joints the encoders gave, the row that had been sent, and as its label the scripted
fold's row for that moment, whichever of the two drove the arm. Every batch
holds BATCH planned samples and, when there are rows from the arms, RIG_BATCH of those, so a few hundred real rows
weigh as much as a fifth of the planned ones.

A training from nothing ends with MODEL_ROUNDS rounds of the same correction in the arm model, where it costs
nothing: the network plays every planned fold, each row built on the one it sent before and the plan's row taking
over outside MODEL_TRUST, and the rows it met there, with the plan's rows as labels, are trained on beside the
planned ones. Without those rounds it drove 85% of held-out rows inside 2 deg, with them 93% (results.md).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dagger_data
import dagger_net
from dagger_net import DT, MIRROR_JOINTS, OBS

LAG_ROWS = 3.0          # the measured joints are the plan up to this many rows behind the row before
OFFSET_DEG = 3.0        # plus up to this much per joint
SENT_DEG = 2.0          # the row sent last is the plan's row before, up to this far off per joint: the band
                        # dagger_run.py keeps the arms in (TRUST_DEG). 2 deg and 30,000 steps drove 85% of held-out
                        # rows, 1 deg and 10,000 steps 85%, 3 deg and 30,000 steps 78% (results.md, 2026-11-26)
BATCH = 1024
RIG_BATCH = 256
RATE = 2e-3             # from nothing
TUNE_RATE = 2e-4        # carrying on from weights
EPOCHS = 115
TUNE_STEPS = 600
MODEL_ROUNDS = 6        # rounds of DAgger in the arm model after a training from nothing
MODEL_STEPS = 4000      # training steps a round
MODEL_RATE = 5e-4
MODEL_TRUST = 5.0       # deg: the band the network plays in during those rounds (dagger_run's widest --trust-deg)
HELD_OUT = 0.1          # share of the planned folds kept out of a training from nothing
STD_FLOOR = 0.01        # an input that hardly varies is not blown up by its tiny spread
TRUST_DEG = 2.0         # the band the held-out share of rows driven is reported for (dagger_run.TRUST_DEG)


def planned_samples(folds):
    """Per sample (arrays over N): this arm's and the other's inputs [N, 10], the side [N], the time [N], the 4
    planner rows before the labelled one [N, 4, 6] (row 0's own before it) and the label [N, 6]."""
    fixed, side, times, before, label = [], [], [], [], []
    for inputs, targets in folds:
        k = np.arange(len(targets))
        for s, name in enumerate(("left", "right")):
            plan = targets[:, s] * (MIRROR_JOINTS if name == "right" else 1.0)
            fixed.append(np.repeat(np.concatenate([inputs[s], inputs[1 - s]])[None], len(k), 0))
            side.append(np.full(len(k), dagger_net.SIDES[name]))
            times.append(k * DT)
            before.append(plan[np.clip(k[:, None] - 1 - np.arange(4), 0, None)])
            label.append(plan)
    return [np.concatenate(a).astype(np.float32) for a in (fixed, side, times, before, label)]


def measured(before, rng):
    """Measured joints for a batch: the plan a random 0 to LAG_ROWS rows behind the row before, plus a random
    offset per joint."""
    lag = rng.random(len(before)) * LAG_ROWS
    i = np.minimum(np.floor(lag).astype(int), 2)
    f = (lag - i)[:, None]
    rows = np.arange(len(before))
    joints = (1 - f) * before[rows, i] + f * before[rows, i + 1]
    return joints + (rng.random(joints.shape) * 2 - 1) * np.radians(OFFSET_DEG)


def observations(fixed, side, times, joints, sent):
    """OBS vectors from the parts, in OBS's order."""
    return np.concatenate([fixed, joints, sent, times[:, None], side[:, None]], axis=1).astype(np.float32)


def on_plan(parts):
    """The planned samples' OBS vectors with every joint, and the row sent last, on the plan's row before."""
    fixed, side, times, before, _ = parts
    return observations(fixed, side, times, before[:, 0], before[:, 0])


def rig_rows(folder):
    """(observations [M, len(OBS)], the scripted fold's rows [M, 6]) of every rows_*.npz dagger_run.py saved in
    `folder`, oldest first; two empty arrays when there are none."""
    obs, label = [np.zeros((0, len(OBS)), np.float32)], [np.zeros((0, 6), np.float32)]
    for path in sorted(Path(folder).glob("rows_*.npz")) if folder else []:
        saved = np.load(path)
        obs.append(saved["obs"].astype(np.float32))
        label.append(saved["teacher"].astype(np.float32))
    return np.concatenate(obs), np.concatenate(label)


def visited(policy, folds, trust, rng):
    """(observations, the plan's rows) the network meets playing `folds` in the arm model inside a band of `trust`
    deg: every joint reaches the row it is sent and reads up to OFFSET_DEG off it."""
    obs, label = [], []
    for inputs, targets in folds:
        named = {"left": inputs[0], "right": inputs[1]}
        for s, side in enumerate(("left", "right")):
            mirror = MIRROR_JOINTS if side == "right" else 1.0
            plan = targets[:, s].astype(np.float64)
            sent = plan[0]
            for k in range(len(plan)):
                read = sent + np.radians(rng.uniform(-OFFSET_DEG, OFFSET_DEG, 6))
                row = dagger_net.observation(named, side, read, sent, k * DT)
                wanted = dagger_net.to_arm(side, policy.act(row[None])[0])
                obs.append(row)
                label.append(plan[k] * mirror)
                sent = wanted if np.degrees(np.abs(wanted - plan[k])).max() <= trust else plan[k]
    return np.array(obs, np.float32), np.array(label, np.float32)


def free_run(policy, folds, trust=None):
    """The network played through each planned fold in the arm model, every joint reaching its row and each row
    built on the one sent before: (deg from the plan [rows, 6] over all folds and arms, share of rows sent that
    were the network's). With `trust` (deg) the plan's row replaces one further than that from it, as on the
    arms; without it nothing does, and an error can grow from row to row."""
    off, drove = [], []
    for inputs, targets in folds:
        named = {"left": inputs[0], "right": inputs[1]}
        for s, side in enumerate(("left", "right")):
            plan = targets[:, s].astype(np.float64)
            sent = plan[0]
            for k in range(len(plan)):
                wanted = policy.targets(named, {side: sent}, {side: sent}, k * DT)[side]
                away = np.degrees(wanted - plan[k])
                off.append(away)
                drove.append(trust is None or np.abs(away).max() <= trust)
                sent = wanted if drove[-1] else plan[k]
    return np.array(off), float(np.mean(drove))


def errors(policy, obs, label):
    """{"rms_deg": per joint, "max_deg": per joint} of the policy against `label` on `obs`; None with no rows."""
    if len(obs) == 0:
        return None
    off = np.degrees(policy.act(obs) - label)
    return {"rms_deg": dict(zip(dagger_net.JOINTS, np.round(np.sqrt((off ** 2).mean(0)), 3).tolist())),
            "max_deg": dict(zip(dagger_net.JOINTS, np.round(np.abs(off).max(0), 3).tolist()))}


def fit(layers, norm, parts, rig, steps, rate, rng, every=0):
    """`layers` after `steps` Adam steps: BATCH planned samples a step with fresh measured joints, plus RIG_BATCH
    rows from the arms when `rig` (obs, label) has any. The rate warms up over the first steps and falls to
    nothing along a cosine."""
    fixed, side, times, before, label = parts
    obs_mean, obs_std, act_mean = norm
    adam = dagger_net.Adam(layers)
    started = time.time()
    for step in range(steps):
        b = rng.integers(0, len(label), BATCH)
        sent = before[b, 0] + (rng.random((BATCH, 6)) * 2 - 1).astype(np.float32) * np.radians(SENT_DEG)
        obs = observations(fixed[b], side[b], times[b], measured(before[b], rng), sent)
        want = label[b]
        if len(rig[0]):
            r = rng.integers(0, len(rig[0]), RIG_BATCH)
            obs, want = np.concatenate([obs, rig[0][r]]), np.concatenate([want, rig[1][r]])
        output, seen = dagger_net.forward(layers, (obs - obs_mean) / obs_std)
        loss, grads = dagger_net.gradients(layers, seen, output, want - obs[:, dagger_net.SENT] - act_mean)
        warm = min(1.0, (step + 1) / max(1, 0.03 * steps))
        layers = adam.step(layers, grads, rate * warm * 0.5 * (1 + np.cos(np.pi * step / steps)))
        if every and (step % every == 0 or step == steps - 1):
            print(f"  step {step}: rms {np.degrees(np.sqrt(loss)):.3f} deg, {time.time() - started:.0f} s", flush=True)
    return layers


def train(base, out, start=None, rows=None, steps=None, seed=0, quiet=False, model_rounds=None):
    """Write weights to `out` (and their JSON): from nothing on the planned folds of `base`, or carrying on from
    `start` with the rows from the arms in `rows` as well. Returns the summary written into the JSON."""
    started = time.time()
    rng = np.random.default_rng(seed)
    data = dagger_data.load(base)
    if float(data["closed"]) != dagger_net.CLOSED:
        raise ValueError(f"{base} was planned with the jaw shut to {float(data['closed'])}, not {dagger_net.CLOSED}")
    folds = data["folds"]
    rig = rig_rows(rows)
    if start is None:
        kept = max(1, round(len(folds) * HELD_OUT))
        parts, held = planned_samples(folds[:-kept]), folds[-kept:]
        clean = on_plan(parts)
        norm = (clean.mean(0), np.maximum(clean.std(0), STD_FLOOR), (parts[4] - parts[3][:, 0]).mean(0))
        layers = dagger_net.initial(rng, [len(OBS), *dagger_net.HIDDEN, len(dagger_net.ACT)])
        steps = steps or EPOCHS * int(np.ceil(len(parts[4]) / BATCH))
        rate, history = RATE, []
    else:
        before = dagger_net.Policy(start)
        parts, held = planned_samples(folds), None
        norm, layers = (before.obs_mean, before.obs_std, before.act_mean), before.layers
        steps, rate, history = steps or TUNE_STEPS, TUNE_RATE, before.meta.get("history", [])
    if not quiet:
        print(f"{len(folds)} planned folds, {len(parts[4])} samples, {len(rig[0])} rows from the arms, {steps} steps")
    layers = fit(layers, norm, parts, rig, steps, rate, rng, 0 if quiet else max(1, steps // 10))
    meta = {"hidden": list(dagger_net.HIDDEN), "dt": DT, "closed": dagger_net.CLOSED,
            "obs_mean": np.asarray(norm[0], float).tolist(), "obs_std": np.asarray(norm[1], float).tolist(),
            "act_mean": np.asarray(norm[2], float).tolist(),
            "noise": {"lag_rows": LAG_ROWS, "offset_deg": OFFSET_DEG, "sent_deg": SENT_DEG}, "history": history}
    rounds = MODEL_ROUNDS if start is None and model_rounds is None else (model_rounds or 0)
    met = rig
    for n in range(rounds):                 # DAgger in the arm model: see the module's docstring
        dagger_net.save(out, layers, meta)
        new = visited(dagger_net.Policy(out), folds[:-kept] if start is None else folds, MODEL_TRUST, rng)
        met = (np.concatenate([met[0], new[0]]), np.concatenate([met[1], new[1]]))
        layers = fit(layers, norm, parts, met, MODEL_STEPS, MODEL_RATE, rng)
        if not quiet:
            print(f"  arm-model round {n + 1} of {rounds}: {len(met[0])} rows met, {time.time() - started:.0f} s",
                  flush=True)
    dagger_net.save(out, layers, meta)
    after = dagger_net.Policy(out)
    summary = {"base": str(base), "seed": seed, "planned_folds": len(folds), "steps": steps,
               "arm_model_rounds": rounds,
               "started_from": None if start is None else str(start), "rows_from_arms": int(len(rig[0])),
               "seconds": round(time.time() - started),
               "on_planned_rows": errors(after, on_plan(parts), parts[4]),
               "on_rows_from_arms": errors(after, *rig)}
    if held is not None:
        free, _ = free_run(after, held)
        _, share = free_run(after, held, TRUST_DEG)
        summary["held_out_folds"] = len(held)
        summary["held_out_free_run"] = {
            "rms_deg": dict(zip(dagger_net.JOINTS, np.round(np.sqrt((free ** 2).mean(0)), 3).tolist())),
            "max_deg": dict(zip(dagger_net.JOINTS, np.round(np.abs(free).max(0), 3).tolist()))}
        summary["held_out_share_driven"] = {"trust_deg": TRUST_DEG, "share": round(share, 4)}
    else:
        summary["on_rows_from_arms_before"] = errors(before, *rig)
    meta["history"] = history + [summary]
    dagger_net.save(out, layers, meta)
    if not quiet:
        print(json.dumps(summary, indent=1))
        print(f"Wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="dagger_data.py output")
    ap.add_argument("--out", required=True, help="weights .npz; the JSON goes beside it")
    ap.add_argument("--start", help="weights to carry on from; without it the network starts from nothing")
    ap.add_argument("--rows", help="folder of rows_*.npz from dagger_run.py")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--model-rounds", type=int, help=f"rounds of DAgger in the arm model afterwards (default: "
                    f"{MODEL_ROUNDS} from nothing, none when carrying on)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.base, args.out, args.start, args.rows, args.steps, args.seed, model_rounds=args.model_rounds)


if __name__ == "__main__":
    main()
