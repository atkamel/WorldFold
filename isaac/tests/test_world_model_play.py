"""The play policy and the collector's kinds on a scripted stand-in for the simulator and a fake IK. No Isaac Sim."""

import argparse
import json

import gymnasium as gym
import numpy as np
import pytest

from cloth_angles.data.state_episode import GRASP_INDEX, StateEpisodeStore, manipulated_vertices
from cloth_angles.tasks import HALF
from isaac.experts.fold_expert import IsaacArmExpert
from isaac.sim.isaac_env import CLOTH_CENTER, PROFILES, cloth_grid_mesh
from isaac.world_model import collect, play
from isaac.world_model.half_fold import ScriptedHalfFold
from mujoco_sim.cloth_params import ARM_BASE_LEFT, ARM_BASE_QUAT, ARM_BASE_RIGHT, JOINT_DELTA_SCALE, TABLE_TOP_Z

PREFIXES = ("left_", "right_")


class FakeIK:
    """Joint angles are the gripper frame's xyz (and two spares); a tip further than reach from the arm's base, in
    the horizontal, is missed by the excess."""

    def __init__(self, reach):
        self.reach = reach
        self.root = None

    def site_pose(self, prefix, q):
        return np.asarray(q[:3], dtype=float), np.eye(3)

    def solve(self, prefix, tip, jaw_dir, seed_q, orientation_weight=0.1, iters=300, jaw_weight=None):
        base = np.asarray(self.root[prefix][0], dtype=float)
        d = np.asarray(tip[:2], dtype=float) - base[:2]
        r = np.linalg.norm(d)
        xy = base[:2] + d * min(1.0, self.reach / max(r, 1e-9))
        return np.r_[xy, tip[2], 0.0, 0.0], float(max(0.0, r - self.reach))


class PlayBase(gym.Env):
    """What the adapter, play, QuarterFoldEnv and the collector read. Each arm's joints follow its deltas (FakeIK
    joints), its jaw shuts on a gripper command below -0.3; test code sets grasp flags and failure by hand."""

    profile = "friction"

    def __init__(self):
        self._prof = PROFILES["friction"]
        self.arm_bases = {"left_": (np.asarray(ARM_BASE_LEFT), np.asarray(ARM_BASE_QUAT)),
                          "right_": (np.asarray(ARM_BASE_RIGHT), np.asarray(ARM_BASE_QUAT))}
        self.grasp_knobs = dict(self._prof["friction_grasp"])
        self._closed_q = 0.05
        self._sizes = (("proprio", 50), ("cloth_state", 69), ("task", 22))
        self.observation_space = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
                                                  for k, n in self._sizes})
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.n_tasks, self.prefixes, self.max_episode_steps = 4, list(PREFIXES), 60
        self.weld_mask, self.grasped, self.failed = {}, {p: False for p in PREFIXES}, False
        self.jaw_open = {p: None for p in PREFIXES}

    def _get_obs(self):
        return {k: np.zeros(n, dtype=np.float32) for k, n in self._sizes}

    def reset(self, seed=None, options=None):
        self.positions = np.array(cloth_grid_mesh()[0]) + [*CLOTH_CENTER, TABLE_TOP_Z + 0.004]
        self.positions[:, :2] += np.ravel((options or {}).get("cloth_pose", np.zeros(2)))[:2]
        self.q = {p: np.r_[self.arm_bases[p][0] + [0.0, 0.3, 0.2], 0.0, 0.0, 1.0] for p in PREFIXES}
        self._joint_targets = {p: self.q[p].copy() for p in PREFIXES}
        self._domain_params = {"cloth_mass_scale": 1.1}
        self._step_count, self.actions = 0, []
        return self._get_obs(), {"domain_parameters": dict(self._domain_params)}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        for i, p in enumerate(PREFIXES):
            self.q[p][:5] += JOINT_DELTA_SCALE * np.asarray(action[7 * i:7 * i + 5])
            self._joint_targets[p][5] = self._closed_q if action[7 * i + 6] < -0.3 else 1.0
        self._step_count += 1
        return self._get_obs(), 0.0, False, False, {}

    def cloth_positions(self):
        return self.positions.copy()

    def joint_positions(self, p):
        return self.q[p].copy()

    def joint_velocities(self, p):
        return np.zeros(6)

    def gripper_position(self, p):
        return self.q[p][:3].copy()

    def grasp_active(self, p):
        return self.grasped[p]

    def render_rig(self):
        return {"main": np.full((3, 16, 16), len(self.actions) % 256, dtype=np.uint8),
                "left_wrist_cam": np.zeros((3, 8, 8), dtype=np.uint8)}

    def render_rig_depth(self):
        return {"main": np.where(np.eye(16, dtype=bool), np.inf, 0.4567).astype(np.float32)}

    def _failed(self):
        return self.failed


CARRY = np.array([0.5, 0, 0, 0, 0, -1.0, -0.5, 0, 0, 0, 0, -1.0], dtype=np.float32)


class FakeHalf(ScriptedHalfFold):
    """The scripted half fold (its QuarterFoldEnv stage 0 and place-servo experts) with both arms carrying, jaws shut,
    instead of the experts' actions."""

    def act(self):
        return CARRY.copy()


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(IsaacArmExpert, "_ik", FakeIK(reach=0.30))
    b = PlayBase()
    b.reset()
    return b


def started(base, seed=0):
    policy = play.PlayPolicy(base, np.random.default_rng(seed))
    assert policy.act().shape == (12,)
    return policy


def test_each_arm_plays_on_its_own_side_with_reachable_targets(base):
    policy = play.PlayPolicy(base, np.random.default_rng(1))
    mid = base.positions[:, 0].mean()
    for _ in range(400):
        assert policy.act() is not None
        for arm in policy.arms:
            assert base.weld_mask[arm.arm.prefix] == {arm.cycle["vertex"]}    # the flag follows the played vertex
    ik = IsaacArmExpert._ik
    for arm, sign in zip(policy.arms, (-1, 1)):
        mine = [c for c in policy.cycles if c["arm"] == arm.arm.prefix]
        assert len(mine) >= 2
        reach_from = base.arm_bases[arm.arm.prefix][0][:2]
        for c in mine:
            assert {"start", "vertex", "target", "arc_height", "release", "shake", "noise", "done"} <= set(c)
            assert sign * (base.positions[c["vertex"], 0] - mid) > play.SIDE_GAP           # cloth not moved here
            assert sign * (c["target"][0] - mid) >= play.SIDE_GAP
            assert np.linalg.norm(np.asarray(c["target"][:2]) - reach_from) <= ik.reach + play.PLACE_TOL
    episode = type("Ep", (), {"__len__": lambda s: 400, "metadata": {"cycles": policy.cycles},
                              "robot": np.zeros((401, 30))})()
    vertices = manipulated_vertices(episode)
    right = [c for c in policy.cycles if c["arm"] == "right_"][1]
    assert vertices.shape == (401, 2) and vertices[right["start"], 1] == right["vertex"]


def test_nothing_reachable_ends_play(monkeypatch, base):
    monkeypatch.setattr(IsaacArmExpert, "_ik", FakeIK(reach=0.0))
    assert play.PlayPolicy(base, np.random.default_rng(0)).act() is None


def test_forced_release_opens_the_jaw_mid_carry(base):
    arm = started(base).arms[1]
    e = arm.expert
    arm.open_wp, arm.shake_left, arm.noise = 3, 0, 0.0
    e.phase, e.phase_steps, e.wp = e.PHASES.index("carry"), 0, 2
    assert arm.act()[5] == e._close_command()               # before the release waypoint: still carrying
    e.wp = 3
    action = arm.act()
    assert action[5] == 1.0 and e.PHASES[e.phase] == "release"


def test_shake_swings_the_joints_back_with_the_jaw_shut(base):
    arm = started(base).arms[0]
    e = arm.expert
    arm.open_wp, arm.shake_left, arm.noise = None, play.SHAKE_STEPS, 0.0
    e.phase, e.phase_steps, e.wp = e.PHASES.index("carry"), 4, (len(e.plan["arc"]) - 1) // 2
    actions = np.array([arm.act() for _ in range(play.SHAKE_STEPS)])
    assert np.all(actions[:, 5] == e._close_command())
    assert np.all(np.abs(actions[:, :5]) >= 0.5)
    np.testing.assert_allclose(actions[:, :5].sum(axis=0), 0.0, atol=1e-6)     # back where it started
    assert e.phase_steps == 4 and arm.shake_left == 0                          # the expert was paused


def test_noise_moves_the_joints_only(monkeypatch, base):
    fixed = np.array([0.2, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    monkeypatch.setattr(play.PlayExpert, "act", lambda self: fixed.copy())
    arm = started(base).arms[0]
    arm.shake_left, arm.open_wp, arm.noise = 0, None, 0.0
    np.testing.assert_array_equal(arm.act(), fixed)
    arm.noise = play.NOISE_STD
    noisy = np.array([arm.act() for _ in range(20)])
    assert np.all(noisy[:, 5] == -1.0) and np.all(np.abs(noisy[:, :5]) <= 1.0)
    assert np.abs(noisy[:, :5] - fixed[:5]).max() > 0.1


def test_disturbance_stats_count_both_arms():
    vertices = np.zeros((6, 121, 3), dtype=np.float32)
    vertices[:, :, 2] = TABLE_TOP_Z
    vertices[3, 7] += [0.03, 0.04, 0.0]                        # one vertex 5 cm out at state 3
    robot = np.zeros((6, 30), dtype=np.float32)
    robot[:, GRASP_INDEX] = [0, 1, 1, 0, 1, 0]                  # left arm
    robot[:, 15 + GRASP_INDEX] = [0, 0, 1, 1, 1, 0]             # right arm
    actions = np.full((5, 12), -1.0, dtype=np.float32)
    actions[2, 5] = 1.0                                         # left opened at its first loss, closed at its second
    actions[4, 11] = 1.0                                        # right opened at its loss
    s = play.disturbance_stats(vertices, robot, actions, [{"done": True}, {"done": False}, {"arm": "right_"}])
    assert s["max_disp_mm"] == 50.0 and s["mean_disp_mm"] == round(50.0 / (6 * 121), 1)
    assert s["max_height_mm"] == 0.0
    assert (s["grasped_steps"], s["grasp_events"], s["release_events"], s["slip_events"]) == (6, 3, 2, 1)
    assert (s["cycles"], s["cycles_done"]) == (3, 1)


def test_play_episode_records_the_half_fold_layout_frames_and_why_it_ended(base):
    episode, frames = collect.collect_play(base, seed=3, steps=30, frames=True)
    assert episode.states().shape == (31, HALF.state_dim) and episode.actions.shape == (30, 12)
    assert set(frames) == {"main", "left_wrist_cam", "main_depth"} and frames["main"].shape == (31, 3, 16, 16)
    assert frames["main"].dtype == np.uint8 and frames["main"][5, 0, 0, 0] == 5
    depth = frames["main_depth"]                                      # whole mm, 0 where nothing was hit
    assert depth.dtype == np.uint16 and depth.shape == (31, 16, 16) and depth[0, 0, 1] == 457 and depth[0, 0, 0] == 0
    full = base.actions[0]
    np.testing.assert_array_equal(full[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13]], episode.actions[0])
    m = episode.metadata
    assert m["termination_reason"] == "steps" and not episode.terminated.any() and episode.rewards is None
    assert m["kind"] == "play" and m["seed"] == 3 and m["domain"] == {"cloth_mass_scale": 1.1}
    assert {c["arm"] for c in m["cycles"]} == set(PREFIXES) and m["stats"]["cycles"] == len(m["cycles"])

    base.failed = True
    episode, frames = collect.collect_play(base, seed=3, steps=30)
    assert frames is None and len(episode) == 1
    assert episode.metadata["termination_reason"] == "failed" and episode.terminated[-1]


def test_half_fold_kinds_noise_forced_release_and_reward_inputs(base):
    half = FakeHalf(base)
    episode, _ = collect.collect_half(half, "expert", seed=7)
    m = episode.metadata
    assert len(episode) == base.max_episode_steps and m["termination_reason"] == "truncated"
    np.testing.assert_array_equal(episode.actions, np.tile(CARRY, (len(episode), 1)))
    np.testing.assert_allclose(m["goal"], episode.vertices[0][[0, 110]].ravel(), atol=1e-6)   # FoldReward's goal
    np.testing.assert_allclose(m["anchors0"], episode.vertices[0][[0, 10, 110, 120]], atol=1e-6)
    assert episode.rewards.shape == (60,) and episode.stage.shape == (61,) and not episode.stage.any()
    assert manipulated_vertices(episode)[0].tolist() == [10, 120]

    noisy, _ = collect.collect_half(half, "expert_noisy", seed=7)
    assert np.all(noisy.actions[:, [5, 11]] == -1.0) and np.abs(noisy.actions - CARRY).max() > 0.1
    assert np.all(np.abs(noisy.actions) <= 1.0)

    base.grasped["left_"] = True                               # the left corner held from the first step on
    released, _ = collect.collect_half(half, "expert_release", seed=7)
    t = released.metadata["release_step"]
    assert 10 <= t < 40 and released.metadata["grasp_steps"] == len(released)
    opened = released.actions[:, 5] == 1.0
    assert np.all(opened == (released.actions[:, 11] == 1.0))
    assert np.flatnonzero(opened).tolist() == list(range(t, t + collect.RELEASE_HOLD))


def test_run_seeds_splits_and_writes_frames_beside_episodes(monkeypatch, base, tmp_path):
    monkeypatch.setattr(collect, "make_base", lambda steps, cameras=None: base)
    monkeypatch.setattr(collect, "ScriptedHalfFold", FakeHalf)
    args = argparse.Namespace(output=tmp_path / "wm", kinds=["play", "expert"], per_kind=3, test_per_kind=1,
                              play_steps=12, seed_base=500, no_frames=False)
    collect.run(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert [(m["kind"], m["seed"], m["split"]) for m in manifest] == [
        ("play", 500, "test"), ("play", 501, "train"), ("play", 502, "train"),
        ("expert", 1500, "test"), ("expert", 1501, "train"), ("expert", 1502, "train")]
    assert len(list(args.output.glob("frames_*.npz"))) == 6
    episodes = StateEpisodeStore(args.output).load_all()              # frames files are not episodes
    assert [ep.metadata["split"] for ep in episodes] == [m["split"] for m in manifest]
    with np.load(args.output / "frames_000001.npz") as f:
        assert f["main"].shape == (13, 3, 16, 16) and f["main"].dtype == np.uint8
