"""The MuJoCo profile on Isaac (Phase W, W2), via isaac/profile_check.py in its own process, and the position-only IK
the MuJoCo expert needs on the Isaac kinematics. Isaac venv only."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.isaac,
              pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None, reason="needs the Isaac venv")]

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("profile")
    r = subprocess.run([sys.executable, "-u", "isaac/profile_check.py", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads((out / "profile.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("name", ["arm_reaches_90pct_in_one_step", "cloth_centred", "cloth_offset_applied",
                                  "dr_deterministic_per_seed", "dr_differs_across_seeds", "dr_in_range",
                                  "dr_read_back", "dr_off_restores", "damping_takes_effect"])
def test_profile_check(report, name):
    assert report["checks"][name], {k: v for k, v in report.items() if k != "dr"}


def test_solve_position_reaches_fk_point():
    """A point the arm provably reaches (FK of a random in-limits pose) is solved to the expert's 6 mm tolerance."""
    from isaac.pinch import PinchIK
    ik = PinchIK()
    rng = np.random.default_rng(0)
    for prefix in ("left_", "right_"):
        q_true = rng.uniform(ik.low, ik.high) * 0.5
        target = ik.site_pose(prefix, q_true)[0]
        q, err = ik.solve_position(prefix, target, np.zeros(5), rng=np.random.default_rng(1))
        assert err < 0.006
        assert np.linalg.norm(ik.site_pose(prefix, q)[0] - target) == pytest.approx(err)
        assert np.all(q >= ik.low) and np.all(q <= ik.high)
