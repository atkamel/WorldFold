"""The learned half fold: a small network that gives each arm its next joint targets from where the cloth is, where
the arm is and how far into the fold it is. Plain numpy (the rig laptop has no torch). policy_train.py writes the
weights (.npz) and, beside them, a JSON with the normalisation, the fold's length in rows and the training summary.

Observation of one arm (OBS), in the LEFT arm's convention: the right arm is seen in a mirror (y -> -y in its base
frame, shoulder_pan and wrist_roll turned the other way), so both arms present the same problem to one network.
    pinch x, y and place x, y of this arm          (m, its base frame, mirrored for the right arm)
    pinch x, y and place x, y of the other arm     (m, the other arm's base frame, mirrored if that arm is the right)
    this arm's 6 joint angles as measured          (rad; START for the first row, before the arm is at the fold)
    time since the fold started                    (s of plan time: row * DT, whatever speed it is played at)
    side                                           (+1 left, -1 right)
The other arm's points are there because both arms' carries are stretched to the same length (fold_traj.fold). The
side is there because the mirror is not exact: the arm's links are offset sideways, and the planner's right arm is
its left arm mirrored only to within 0.02 deg of shoulder_pan.
Action (ACT): this arm's 6 joint targets for the next control row, DT later (rad, same convention).

next_targets then keeps each target within STEP_LIMIT of the target sent for the row before, as replay.py's play caps
a planned row. The network alone moves the jaw up to 5 deg in a row where a fold's timing is ambiguous to it (when
the carry ends depends on both arms' reaches); the planner's own rows move under 2.8 deg.
"""

import json
from pathlib import Path

import numpy as np

DT = 0.05
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OBS = (["pinch_x", "pinch_y", "place_x", "place_y", "other_pinch_x", "other_pinch_y", "other_place_x", "other_place_y"]
       + [f"measured_{j}" for j in JOINTS] + ["time", "side"])
ACT = [f"target_{j}" for j in JOINTS]
SIDES = {"left": 1.0, "right": -1.0}
MIRROR_JOINTS = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0])     # a mirrored arm turns pan and roll the other way
# The joints the first row's observation carries: the planner's IK seed (arm out, fingers down), jaw open. Its pan
# and roll are 0, so it is the same pose mirrored.
START = np.concatenate([np.radians([0.0, 40.0, -30.0, 80.0, 0.0]), [1.0]])
STEP_LIMIT = np.radians(2.9)    # rad a target may move from the row before: under replay.py's MAX_DEG_S * DT (3 deg)


def canonical_points(side, pinch_place):
    """An arm's (pinch, place) as the 4 numbers of the observation: mirrored if it is the right arm."""
    xy = np.asarray(pinch_place, float).reshape(2, 2)
    if side == "right":
        xy = xy * [1.0, -1.0]
    return xy.ravel()


def observation(points, side, measured, t):
    """The OBS vector of `side`'s arm. `points` {"left": (pinch, place), "right": ...} (each arm's base frame),
    `measured` that arm's 6 joint angles (rad, its own convention), `t` the plan time (s)."""
    other = "right" if side == "left" else "left"
    joints = np.asarray(measured, float) * (MIRROR_JOINTS if side == "right" else 1.0)
    return np.concatenate([canonical_points(side, points[side]), canonical_points(other, points[other]), joints,
                           [t, SIDES[side]]])


def to_arm(side, action):
    """An ACT vector as the 6 joint targets of `side`'s arm, in its own convention."""
    return np.asarray(action, float) * (MIRROR_JOINTS if side == "right" else 1.0)


class Policy:
    """The trained network: tanh hidden layers, a linear output, inputs and outputs normalised."""

    def __init__(self, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["obs"] != OBS or meta["act"] != ACT:
            raise ValueError(f"{path} was trained on another observation or action than policy_net.py defines")
        weights = np.load(path)
        self.layers = [(weights[f"w{i}"].astype(np.float64), weights[f"b{i}"].astype(np.float64))
                       for i in range(meta["layers"])]
        self.obs_mean, self.obs_std = np.array(meta["obs_mean"]), np.array(meta["obs_std"])
        self.act_mean, self.act_std = np.array(meta["act_mean"]), np.array(meta["act_std"])
        self.rows = meta["rows"]        # control rows the fold is played for; the last ones hold still
        self.meta = meta

    def act(self, obs):
        """ACT vectors [N, 6] for OBS vectors [N, len(OBS)] (or one of each)."""
        h = (np.asarray(obs, float) - self.obs_mean) / self.obs_std
        for w, b in self.layers[:-1]:
            h = np.tanh(h @ w + b)
        w, b = self.layers[-1]
        return (h @ w + b) * self.act_std + self.act_mean

    def next_targets(self, points, measured, t, previous=None):
        """{side: the arm's 6 joint targets for the row at plan time `t`} from `measured` {side: 6 joint angles},
        each within STEP_LIMIT of `previous` {side: the targets of the row before}; none for row 0, which the arm
        glides to."""
        sides = list(points)
        action = self.act(np.stack([observation(points, side, measured[side], t) for side in sides]))
        targets = {side: to_arm(side, a) for side, a in zip(sides, action)}
        if previous is None:
            return targets
        return {side: previous[side] + np.clip(targets[side] - previous[side], -STEP_LIMIT, STEP_LIMIT)
                for side in sides}
