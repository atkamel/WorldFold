"""policy_net: the numpy forward pass is the torch model's, no target moves more than the step limit, the right arm
is seen in a mirror, and the planner's own mirror symmetry holds only to within hundredths of a degree (why the
observation carries the side).

    /Users/akamel/Code/WorldFold/.venv/bin/python -m pytest hardware/so101/test_policy_net.py
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

try:
    import scservo_sdk  # noqa: F401
except ImportError:      # replay imports it; only its constants are used. An empty stand-in must not hide the real SDK
    sys.modules["scservo_sdk"] = types.ModuleType("scservo_sdk")

import fold_traj  # noqa: E402
import policy_data  # noqa: E402
import policy_net  # noqa: E402
import replay  # noqa: E402
from policy_net import ACT, MIRROR_JOINTS, OBS  # noqa: E402


def random_meta(rng):
    return {"obs": OBS, "act": ACT, "rows": 10, "obs_mean": rng.normal(size=len(OBS)).tolist(),
            "obs_std": rng.uniform(0.5, 2.0, len(OBS)).tolist(), "act_mean": rng.normal(size=len(ACT)).tolist(),
            "act_std": rng.uniform(0.5, 2.0, len(ACT)).tolist()}


def test_numpy_forward_pass_equals_the_torch_model(tmp_path):
    torch = pytest.importorskip("torch")
    import policy_train
    torch.manual_seed(0)
    model = policy_train.build(len(OBS))
    rng = np.random.default_rng(0)
    meta = random_meta(rng)
    policy_train.save(model, tmp_path / "p.npz", meta)
    policy = policy_net.Policy(tmp_path / "p.npz")
    obs = rng.normal(size=(64, len(OBS))) * 3
    with torch.no_grad():
        x = torch.as_tensor((obs - meta["obs_mean"]) / np.array(meta["obs_std"]), dtype=torch.float32)
        expected = model(x).numpy() * meta["act_std"] + meta["act_mean"]
    assert np.allclose(policy.act(obs), expected, atol=1e-5)
    assert np.allclose(policy.act(obs[0]), expected[0], atol=1e-5)


def test_a_policy_trained_on_another_observation_is_refused(tmp_path):
    pytest.importorskip("torch")
    import policy_train
    meta = random_meta(np.random.default_rng(1))
    meta["obs"] = OBS[:-1] + ["something else"]
    policy_train.save(policy_train.build(len(OBS)), tmp_path / "p.npz", meta)
    with pytest.raises(ValueError):
        policy_net.Policy(tmp_path / "p.npz")
    assert json.loads((tmp_path / "p.json").read_text())["layers"] == 4


def test_no_target_moves_more_than_the_step_limit_from_the_row_before(tmp_path):
    pytest.importorskip("torch")
    import policy_train
    rng = np.random.default_rng(2)
    meta = random_meta(rng)
    meta["act_std"] = [5.0] * len(ACT)                      # a network that asks for large jumps
    policy_train.save(policy_train.build(len(OBS)), tmp_path / "p.npz", meta)
    policy = policy_net.Policy(tmp_path / "p.npz")
    points = policy_data.cloth_points("behind")
    previous = {side: rng.normal(size=6) for side in points}
    free = policy.next_targets(points, previous, 2.0)
    held = policy.next_targets(points, previous, 2.0, previous)
    assert max(np.abs(free[side] - previous[side]).max() for side in points) > 3 * policy_net.STEP_LIMIT
    for side in points:
        assert np.abs(held[side] - previous[side]).max() <= policy_net.STEP_LIMIT + 1e-12
    assert np.degrees(policy_net.STEP_LIMIT) < replay.MAX_DEG_S * policy_net.DT     # under replay.py's cap


def test_the_right_arm_is_seen_as_a_mirrored_left_arm():
    points = policy_data.cloth_points("sides")              # the right arm's points are the left's mirrored
    q = np.array([0.3, 0.5, -0.4, 1.2, 0.1, 0.8])
    left = policy_net.observation(points, "left", q, 1.5)
    right = policy_net.observation(points, "right", q * MIRROR_JOINTS, 1.5)
    assert np.allclose(left[:-1], right[:-1])               # the same problem, but for the side
    assert (left[-1], right[-1]) == (1.0, -1.0)
    assert np.allclose(policy_net.to_arm("right", q), q * MIRROR_JOINTS)
    assert np.allclose(policy_net.to_arm("left", q), q)
    assert np.allclose(policy_net.START * MIRROR_JOINTS, policy_net.START)


def test_the_planner_mirrors_the_two_arms_only_to_within_hundredths_of_a_degree():
    traj = fold_traj.sides_trajectory(0.30, 0.16)            # mirrored points in, so mirrored rows if it were exact
    left = np.array(traj["arms"]["left"]["target"])
    right = np.array(traj["arms"]["right"]["target"]) * MIRROR_JOINTS
    differ = np.degrees(np.abs(left - right)).max(axis=0)
    assert 0 < differ[0] < 0.05                              # shoulder_pan: the arm's links are offset sideways
    assert differ[1:].max() < 0.001


def test_the_unmoved_cloth_is_where_the_planner_layouts_put_it():
    for side, y in (("left", -0.0), ("right", 0.0)):         # inset 0 for a 30 cm cloth with bases 30 cm apart
        assert np.allclose(policy_data.cloth_points("behind")[side], ((0.38, y), (0.08, y)))
    sides = policy_data.cloth_points("sides")
    assert np.allclose(sides["left"], ((0.16, 0.15), (0.16, -0.15)))
    assert np.allclose(sides["right"], ((0.16, -0.15), (0.16, 0.15)))
    moved = policy_data.cloth_points("behind", shift=(0.02, 0.0))
    assert np.allclose(moved["left"][0], (0.38, -0.02)) and np.allclose(moved["right"][0], (0.38, -0.02))
