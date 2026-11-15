"""fold_traj: the planned half fold pinches the far corner, releases on the near one, stays above the table, moves
smoothly, and passes replay.py's validation."""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.modules.setdefault("scservo_sdk", types.ModuleType("scservo_sdk"))   # replay imports it; only validate() is used

import fold_traj  # noqa: E402
import replay  # noqa: E402


def test_half_fold_trajectory():
    traj = fold_traj.trajectory(cloth=0.24, near=0.10, spacing=0.30)
    replay.validate(traj)
    for side, right in (("left", 0.03), ("right", -0.03)):
        t, tip = np.array(traj["arms"][side]["target"]), np.array(traj["arms"][side]["tip"])[1:]
        shut = np.flatnonzero(t[:, 5] < replay.GRASP_RAD)
        assert np.allclose(tip[shut[0]], (right, 0.34, fold_traj.PINCH_Z), atol=0.003)       # the pinch
        assert np.allclose(tip[shut[-1]], (right, 0.10, fold_traj.PLACE_Z), atol=0.003)      # the release
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
