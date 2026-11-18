"""fold_traj: the planned half fold pinches the far corner, releases on the near one, stays above the table, moves
smoothly, and passes replay.py's validation."""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    import scservo_sdk  # noqa: F401
except ImportError:      # replay imports it; only validate() is used. An empty stand-in must not hide the real SDK
    sys.modules["scservo_sdk"] = types.ModuleType("scservo_sdk")

import fold_traj  # noqa: E402
import replay  # noqa: E402


def test_half_fold_trajectory():
    traj = fold_traj.trajectory(cloth=0.24, near=0.10, spacing=0.30)
    replay.validate(traj)
    for side, right in (("left", 0.03), ("right", -0.03)):
        t, tip = np.array(traj["arms"][side]["target"]), np.array(traj["arms"][side]["tip"])[1:]
        marks = traj["marks"]
        assert t[marks["empty"], 5] < replay.GRASP_RAD and t[marks["hover"], 5] == fold_traj.OPEN    # shut in the air
        assert np.allclose(tip[marks["empty"]], tip[marks["hover"]]) and tip[marks["hover"], 2] > 0.04
        assert 0.02 < tip[marks["check"], 2] < 0.04 and t[marks["check"], 5] < replay.GRASP_RAD
        shut = marks["hover"] + np.flatnonzero(t[marks["hover"]:, 5] < replay.GRASP_RAD)
        assert np.allclose(tip[shut[0]], (right, 0.34, fold_traj.PINCH_Z), atol=0.003)       # the pinch
        short = traj["source"][side]["release_short_mm"] / 1000
        assert np.allclose(tip[shut[-1]], (right, 0.10 + short, fold_traj.PLACE_Z), atol=0.003)   # the release
        assert tip[:, 2].min() > 0.004                                   # the fingertip never goes into the table
        assert tip[shut[0]:shut[-1], 2].max() < 0.12                     # the carry stays under the cloth's half length
        assert np.abs(np.diff(t[:, :5], axis=0)).max() < np.radians(60) * fold_traj.DT      # under replay's rate cap
        assert np.all(t[:, 4] == 0.0)                                    # wrist_roll never flips
        assert np.allclose(t[0, :5], fold_traj.STAND) and np.allclose(t[-1, :5], fold_traj.STAND)
        assert t[0, 5] == t[-1, 5] == fold_traj.OPEN


def test_gripper_zero_is_ten_degrees_above_the_closed_limit(tmp_path, monkeypatch):
    limits = {str(m): {"true": {"min": 900, "max": 3200}} for m in replay.MOTOR_IDS} | {"6": {"true": {"min": 1316, "max": 2779}}}
    (tmp_path / "limits.json").write_text(__import__("json").dumps(limits))
    monkeypatch.setattr(replay.arms, "folder", lambda name: tmp_path)
    f = replay.Follower("x")
    assert f.zero[1] == 2047
    assert abs(f.to_ticks(6, np.radians(-10.0)) - 1316) < 1e-6          # the URDF's closed limit is the closed tick
    assert 2000 < f.to_ticks(6, 1.0) < 2779                             # the open target stays inside the limits


def test_the_release_stops_short_of_a_near_edge_the_arm_cannot_reach_without_hitting_its_base():
    collision = __import__("pytest").importorskip("collision")
    traj = fold_traj.trajectory(cloth=0.30, near=0.08, spacing=0.30)
    for side in ("left", "right"):
        distance, step, pair = collision.closest(traj["arms"][side]["target"])
        assert distance >= collision.CLEARANCE
        short = traj["source"][side]["release_short_mm"] / 1000
        assert 0.0 < short <= fold_traj.MAX_SHORT
        t, tip = np.array(traj["arms"][side]["target"]), np.array(traj["arms"][side]["tip"])[1:]
        shut = traj["marks"]["hover"] + np.flatnonzero(t[traj["marks"]["hover"]:, 5] < replay.GRASP_RAD)
        assert abs(tip[shut[0], 1] - 0.38) < 0.003                       # the pinch stays on the far corner
        assert abs(tip[shut[-1], 1] - (0.08 + short)) < 0.003            # the release moved away from the base


def test_a_release_far_from_the_base_is_not_shortened():
    traj = fold_traj.trajectory(cloth=0.20, near=0.17, spacing=0.30)
    assert traj["source"]["left"]["release_short_mm"] == 0.0


def test_a_pinch_against_the_base_is_refused():
    with __import__("pytest").raises(ValueError):
        fold_traj.fold({"left": ((0.10, 0.0), (0.06, 0.0)), "right": ((0.10, 0.0), (0.06, 0.0))})
