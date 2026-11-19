"""How a trained half-fold policy does against the planner on held-out folds (policy_data.py), in the arm model only:
closed-loop rollouts with each joint (a) reaching its target every row and (b) following it with a first-order lag
(TAU) and reading a constant offset of up to OFFSET_DEG per joint. Per condition: fingertip error against the
planner where the jaw shuts and where it opens, the least self clearance, the largest joint step per row, how many
rollouts policy_run.py --dry-run would refuse (with the rig's arm2 left and arm1 right: their limits and idle
poses), and how many would trip its --track-deg stop.

    .venv/bin/python hardware/so101/policy_eval.py --policy hardware/so101/policies/half_fold_v0.npz \
        --data outputs/policy/heldout.npz --out outputs/policy/eval.json
"""

import argparse
import collections
import contextlib
import io
import json
import re
import sys
import types

import numpy as np

try:
    import scservo_sdk  # noqa: F401
except ImportError:      # replay imports it; no port is opened here. An empty stand-in must not hide the real SDK
    sys.modules["scservo_sdk"] = types.ModuleType("scservo_sdk")

import collision  # noqa: E402
import policy_data  # noqa: E402
import policy_net  # noqa: E402
import policy_run  # noqa: E402
import replay  # noqa: E402

TAU = 0.1               # s, the lagging servo's time constant
OFFSET_DEG = 1.0        # the most a joint reads off where it is
TRACK_DEG = 8.0         # policy_run.py's default --track-deg
MAX_ERROR_MM = 5.0      # policy_run.py's default --max-error-mm


def summary(values):
    values = np.asarray(values) * 1000
    return {"median": round(float(np.median(values)), 2),
            "p95": round(float(np.percentile(values, 95, method="higher")), 2),
            "max": round(float(values.max()), 2)}


def evaluate(policy, data, followers, tau, offset_deg, seed):
    rng = np.random.default_rng(seed)
    pinch, release, clearance, step, track, refused = [], [], [], [], [], collections.Counter()
    limited = 0
    for layout, points, targets, _ in data["episodes"]:
        named = {"left": tuple(map(tuple, points[0])), "right": tuple(map(tuple, points[1]))}
        plan = {"left": targets[:, 0], "right": targets[:, 1]}
        offset = None
        if offset_deg:
            offset = {side: rng.uniform(-1, 1, 6) * np.radians(offset_deg) for side in followers}
        rows, read = policy_run.rollout(policy, named, tau, offset)
        with contextlib.redirect_stdout(io.StringIO()):
            reasons, ticks = policy_run.check(rows, plan, followers, MAX_ERROR_MM)
        for reason in reasons:
            refused[f"{layout}: " + re.sub(r"\d+(\.\d+)?", "#", reason.split(" ", 2)[2])] += 1
        refused["rollouts refused"] += bool(reasons)
        worst_track = 0.0
        for side, f in followers.items():
            sent = policy_run.angles(f, replay.convert(f, rows[side].tolist())[0])
            try:
                errors = policy_run.tip_errors(sent, plan[side])
            except ValueError:
                errors = (np.inf, np.inf)
            pinch.append(errors[0])
            release.append(errors[1])
            clearance.append(collision.closest(sent.tolist())[0])
            step.append(np.degrees(np.abs(np.diff(sent, axis=0)).max()))
            worst_track = max(worst_track, np.degrees(np.abs(read[side][1:] - rows[side][:-1]).max()))
            limited += int(np.isclose(np.abs(np.diff(rows[side], axis=0)), policy_net.STEP_LIMIT).any(axis=1).sum())
        track.append(worst_track)
    return {"n_rollouts": len(data["episodes"]), "n_arms": len(pinch),
            "pinch_error_mm": summary(pinch), "release_error_mm": summary(release),
            "least_self_clearance_mm": round(float(min(clearance)) * 1000, 1),
            "largest_joint_step_deg": round(float(max(step)), 2),
            "rows_held_to_the_step_limit": f"{limited} of {len(pinch) * (policy.rows - 1)}",
            "refused_by_dry_run_checks": dict(refused),
            "largest_read_minus_last_row_deg": round(float(max(track)), 2),
            f"would_trip_track_{TRACK_DEG:g}_deg": int(sum(t > TRACK_DEG for t in track))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True, help="held-out policy_data.py output")
    ap.add_argument("--left", default="arm2")
    ap.add_argument("--right", default="arm1")
    ap.add_argument("--out")
    args = ap.parse_args()
    policy = policy_net.Policy(args.policy)
    data = policy_data.load(args.data)
    followers = {"left": replay.Follower(args.left), "right": replay.Follower(args.right)}
    results = {"policy": args.policy, "data": args.data, "planner_refused_in_data": int(len(data["refused_layout"])),
               "perfect_tracking": evaluate(policy, data, followers, 0.0, 0.0, 0),
               f"lag_{TAU:g}s_offset_{OFFSET_DEG:g}deg": evaluate(policy, data, followers, TAU, OFFSET_DEG, 0)}
    print(json.dumps(results, indent=1))
    if args.out:
        with open(args.out, "w") as out:
            json.dump(results, out, indent=1)


if __name__ == "__main__":
    main()
