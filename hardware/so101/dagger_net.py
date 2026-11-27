"""The learned camera fold that DAgger trains (dagger_run.py): for each arm, its joint targets for the next control
row from what the scripted fold's planner was given, the arm's measured joints, the row it was sent last and the
time into the fold. Plain numpy, forward and backward, so the rig laptop can run it and retrain it between folds
without torch.

Observation of one arm (OBS), in the left arm's convention (policy_net.py's mirror for the right arm):
    pinch x, y, z and place x, y of this arm     (m, its base frame: the points fold_traj.fold was sent, which is
                                                  after the calibration's inset, overshoot and pinch tilt, and the
                                                  fingertip height it was given for the pinch)
    the same five numbers of the other arm       (both plans are stretched to the same step counts)
    this arm's 6 joint angles as measured        (rad)
    the 6 joint targets it was sent last         (rad; the row the arm was glided to, before the first row)
    time since the fold started                  (s of plan time: row * DT, whatever speed it is played at)
    side                                         (+1 left, -1 right)
Action (ACT): this arm's 6 joint targets for the row at that time (rad, same convention). The network's output is
the step from the row sent last, so a row is never far from the one before it; where the fold has got to comes
from the inputs and the time, which is what pulls a drifting arm back onto the fold.

The network is trained for one jaw target at the pinch (CLOSED) and for plans without the empty jaw reading of
--grasp-check; dagger_run.py lets the scripted fold drive any other plan.
"""

import json
from pathlib import Path

import numpy as np

from policy_net import DT, JOINTS, MIRROR_JOINTS, SIDES, canonical_points, to_arm  # noqa: F401  (DT, to_arm: callers)

OBS = (["pinch_x", "pinch_y", "pinch_z", "place_x", "place_y",
        "other_pinch_x", "other_pinch_y", "other_pinch_z", "other_place_x", "other_place_y"]
       + [f"measured_{j}" for j in JOINTS] + [f"sent_{j}" for j in JOINTS] + ["time", "side"])
ACT = [f"target_{j}" for j in JOINTS]
SENT = slice(16, 22)    # where the row sent last sits in OBS
CLOSED = -0.17          # rad: the jaw target at the pinch the training folds were planned with (fold_traj.fold)
HIDDEN = (256, 256, 256)


def fold_inputs(points, pinch_z):
    """{side: [pinch x, y, z, place x, y]} in the left arm's convention, from what fold_traj.fold was given:
    `points` {side: (pinch, place)} and `pinch_z`, one height for both arms or {side: height}."""
    out = {}
    for side in ("left", "right"):
        x, y, place_x, place_y = canonical_points(side, points[side])
        z = pinch_z[side] if isinstance(pinch_z, dict) else pinch_z
        out[side] = np.array([x, y, float(z), place_x, place_y])
    return out


def observation(inputs, side, measured, sent, t):
    """The OBS vector of `side`'s arm: `inputs` from fold_inputs, `measured` that arm's 6 joint angles and `sent`
    the 6 targets it was sent last (rad, its own convention), `t` the plan time (s)."""
    other = "right" if side == "left" else "left"
    mirror = MIRROR_JOINTS if side == "right" else 1.0
    return np.concatenate([inputs[side], inputs[other], np.asarray(measured, float) * mirror,
                           np.asarray(sent, float) * mirror, [t, SIDES[side]]])


def initial(rng, sizes):
    """Layers [(w [in, out], b [out])] for the layer sizes, small random weights."""
    return [(rng.normal(0.0, 1.0 / np.sqrt(a), (a, b)).astype(np.float32), np.zeros(b, np.float32))
            for a, b in zip(sizes, sizes[1:])]


def forward(layers, x):
    """(output, the activations of every layer, input first): tanh hidden layers, a linear output."""
    seen = [x]
    for w, b in layers[:-1]:
        seen.append(np.tanh(seen[-1] @ w + b))
    w, b = layers[-1]
    return seen[-1] @ w + b, seen


def gradients(layers, seen, output, label):
    """(mean squared error, its gradient for every (w, b))."""
    error = output - label
    delta = 2.0 * error / error.size
    grads = []
    for i in range(len(layers) - 1, -1, -1):
        grads.append((seen[i].T @ delta, delta.sum(0)))
        if i:
            delta = (delta @ layers[i][0].T) * (1.0 - seen[i] ** 2)
    return float(np.mean(error ** 2)), grads[::-1]


class Adam:
    def __init__(self, layers):
        self.m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
        self.v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
        self.steps = 0

    def step(self, layers, grads, rate):
        """`layers` after one step along `grads`."""
        self.steps += 1
        out = []
        for i, (pair, grad) in enumerate(zip(layers, grads)):
            new = []
            for j in (0, 1):
                self.m[i][j][...] = 0.9 * self.m[i][j] + 0.1 * grad[j]
                self.v[i][j][...] = 0.999 * self.v[i][j] + 0.001 * grad[j] ** 2
                mean = self.m[i][j] / (1 - 0.9 ** self.steps)
                spread = self.v[i][j] / (1 - 0.999 ** self.steps)
                new.append((pair[j] - rate * mean / (np.sqrt(spread) + 1e-8)).astype(np.float32))
            out.append(tuple(new))
        return out


def save(path, layers, meta):
    """The weights as w0, b0, ... in `path` (.npz) and `meta` in the JSON beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for i, (w, b) in enumerate(layers):
        weights[f"w{i}"], weights[f"b{i}"] = w, b
    path.with_suffix(".json").write_text(json.dumps(meta | {"obs": OBS, "act": ACT, "layers": len(layers)}, indent=1))
    part = path.with_suffix(".part")
    with open(part, "wb") as file:      # the weights appear whole, after their JSON
        np.savez(file, **weights)
    part.replace(path)


class Policy:
    """Trained weights and their normalisation."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        if self.meta["obs"] != OBS or self.meta["act"] != ACT:
            raise ValueError(f"{path} was trained on another observation or action than dagger_net.py defines")
        with np.load(self.path) as weights:
            self.layers = [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(self.meta["layers"])]
        if self.layers[0][0].shape[0] != len(OBS) or self.layers[-1][0].shape[1] != len(ACT):
            raise ValueError(f"{path} holds layers of another size than its JSON says")
        self.obs_mean, self.obs_std = (np.array(self.meta[k], np.float32) for k in ("obs_mean", "obs_std"))
        self.act_mean = np.array(self.meta["act_mean"], np.float32)

    def act(self, obs):
        """ACT vectors [N, 6] for OBS vectors [N, len(OBS)]: the row sent last plus the step the network gives."""
        obs = np.asarray(obs, np.float64)
        scaled = ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
        return obs[:, SENT] + forward(self.layers, scaled)[0].astype(np.float64) + self.act_mean

    def targets(self, inputs, measured, sent, t):
        """{side: the arm's 6 joint targets (rad, its own convention)} for the row at plan time `t`, for every
        arm in `measured` ({side: 6 joint angles}); `sent` {side: the 6 targets sent last}. `inputs` from
        fold_inputs."""
        sides = list(measured)
        action = self.act(np.stack([observation(inputs, side, measured[side], sent[side], t) for side in sides]))
        return {side: to_arm(side, a) for side, a in zip(sides, action)}
