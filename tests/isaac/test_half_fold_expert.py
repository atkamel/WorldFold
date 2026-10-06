"""IsaacHalfFoldExpert's decision logic on a kinematic stand-in for the Isaac env (no Isaac Sim needed).

The stand-in is the expert's own look-ahead model: joints reach their commanded targets, the jaws move JAW_RATE per
step, and a corner held at the fingertip rides along. On it the expert must finish the fold, label its own
trajectory exactly, and recover from a dropped corner. The stub IK maps the first three joint values straight to
the fingertip position, so a plan's joint targets are its tip targets.
"""

import numpy as np
import pytest

from isaac import half_fold_expert as hfe
from mujuco.cloth_params import GRIPPER_CLOSED, GRIPPER_OPEN, JOINT_DELTA_SCALE


class StubIK:
    def site_pose(self, prefix, q):
        return np.asarray(q[:3], float).copy(), np.eye(3)

    def solve(self, prefix, tip, jaw_dir, seed_q, orientation_weight=None, iters=None):
        return np.r_[np.asarray(tip, float), 0.0, 0.0], 0.0


class Move:
    def __init__(self, prefix, corner, goal):
        self.prefix, self.corners, self.goal_vertex = prefix, (corner,), goal


class Stage:
    def __init__(self, moves):
        self.moves = moves


class KinematicEnv:
    """Both the wrapper (stages, goal, _vertex) and its base (unwrapped) of a HalfFoldEnv, kinematically."""

    def __init__(self):
        # 4 cloth vertices: 0, 1 the near (goal) corners, 2, 3 the far corners the arms carry
        self.cloth = np.array([[-0.15, -0.28, 0.75], [0.15, -0.28, 0.75], [-0.15, 0.02, 0.75], [0.15, 0.02, 0.75]])
        self.start = self.cloth.copy()
        self.stages = (Stage((Move("left_", 2, 0), Move("right_", 3, 1))),)
        self.unwrapped = self
        home = {"left_": np.array([-0.15, -0.20, 0.85]), "right_": np.array([0.15, -0.20, 0.85])}
        self.q = {p: np.r_[home[p], 0.0, 0.0, GRIPPER_OPEN] for p in home}
        self.qd = {p: np.zeros(6) for p in home}
        self._gripper_closed = {p: False for p in home}
        self.held = {p: None for p in home}

    def goal(self, move):
        return self.start[move.goal_vertex]

    def _vertex(self, i):
        return self.cloth[i]

    def cloth_positions(self):
        return self.cloth

    def joint_positions(self, p):
        return self.q[p].copy()

    def joint_velocities(self, p):
        return self.qd[p].copy()

    def step(self, action):
        for p, off, move in (("left_", 0, self.stages[0].moves[0]), ("right_", 6, self.stages[0].moves[1])):
            a = action[off:off + 6]
            if a[5] < -0.3:
                self._gripper_closed[p] = True
            elif a[5] > 0.3:
                self._gripper_closed[p] = False
            q = self.q[p].copy()
            q[:5] += a[:5] * JOINT_DELTA_SCALE
            goal = GRIPPER_CLOSED if self._gripper_closed[p] else GRIPPER_OPEN
            q[5] += float(np.clip(goal - q[5], -hfe.JAW_RATE, hfe.JAW_RATE))
            self.qd[p] = (q - self.q[p]) / hfe.CONTROL_DT
            c = move.corners[0]
            if not self._gripper_closed[p]:
                self.held[p] = None
            elif self.held[p] is None and q[5] < hfe.JAW_SHUT and abs(self.qd[p][5]) < hfe.JAW_SPEED_STILL \
                    and np.linalg.norm(self.cloth[c] - self.q[p][:3]) < hfe.HOLD_RADIUS:
                self.held[p] = self.cloth[c] - self.q[p][:3]
            self.q[p] = q
            if self.held[p] is not None:
                self.cloth[c] = q[:3] + self.held[p]
            elif self.cloth[c][2] > self.start[c][2]:
                self.cloth[c] = np.r_[self.cloth[c][:2], self.start[c][2]]     # a let-go corner drops flat


@pytest.fixture
def setup(monkeypatch):
    # the stub's joints are metres; aim the plan at the stand-in's table instead of Isaac's
    monkeypatch.setattr("isaac.half_fold_demo.TABLE_TOP_Z", 0.74)
    monkeypatch.setattr("isaac.half_fold_demo.SEED_Q", {"left_": np.zeros(5), "right_": np.zeros(5)})
    env = KinematicEnv()
    expert = hfe.IsaacHalfFoldExpert(env, ik=StubIK())
    expert.reset()
    return env, expert


def _run(env, expert, steps, on_step=None):
    for t in range(steps):
        a = expert.act()
        if on_step:
            on_step(t, a)
        env.step(a)


def test_folds_and_backs_off(setup):
    env, expert = setup
    _run(env, expert, 300)
    for p, move in zip(("left_", "right_"), env.stages[0].moves):
        assert np.linalg.norm(env.cloth[move.corners[0]] - env.goal(move)) < hfe.PLACED_DIST
        assert expert.phases()[p] == "released"
        assert not env._gripper_closed[p]
    assert np.allclose(expert.act()[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]], 0.0, atol=1e-6)     # holding still


def test_labels_match_its_own_trajectory(setup):
    env, expert = setup
    labels, actions = {}, []

    def record(t, a):
        actions.append(a)
        if t % 8 == 0:
            labels[t] = expert.label_chunk(horizon=16)

    _run(env, expert, 260, record)
    actions = np.array(actions)
    for t, chunk in labels.items():
        n = min(16, len(actions) - t)
        np.testing.assert_allclose(chunk[:n], actions[t:t + n], atol=1e-4, err_msg=f"label at step {t}")


def test_label_leaves_env_untouched(setup):
    env, expert = setup
    _run(env, expert, 90)
    before = (env.cloth.copy(), {p: env.q[p].copy() for p in env.q}, dict(env._gripper_closed))
    expert.label_chunk(horizon=16)
    assert np.array_equal(env.cloth, before[0])
    assert all(np.array_equal(env.q[p], before[1][p]) for p in env.q)
    assert env._gripper_closed == before[2]


def test_regrasps_a_dropped_corner(setup):
    env, expert = setup
    dropped = []

    def drop(t, a):
        # mid-carry, force the left jaw open for 3 steps: the corner falls where it is
        if expert.phases()["left_"] == "holding" and np.linalg.norm(env.cloth[2] - env.start[2]) > 0.1 and not dropped:
            dropped.append(t)
        if dropped and t < dropped[0] + 3:
            a[5] = 1.0

    _run(env, expert, 500, drop)
    assert dropped
    assert np.linalg.norm(env.cloth[2] - env.start[0]) < hfe.PLACED_DIST
    assert expert.phases()["left_"] == "released"


def test_arms_carry_together(setup):
    env, expert = setup
    lag = []

    def hold_right_open(t, a):
        # the right arm cannot grasp for its first 40 steps: the left must wait at its pinch, holding
        if t < 40:
            a[11] = 1.0
            if expert.phases()["left_"] == "holding":
                lag.append(np.linalg.norm(expert.read("left_").tip - expert.plans["left_"]["pinch_tip"]))

    _run(env, expert, 400, hold_right_open)
    assert lag and max(lag) < hfe.WAYPOINT_DIST
    for move in env.stages[0].moves:
        assert np.linalg.norm(env.cloth[move.corners[0]] - env.goal(move)) < hfe.PLACED_DIST
