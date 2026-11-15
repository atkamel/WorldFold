"""kinematics.Arm against MuJoCo on the so101_nexus model, and the committed trajectory regenerating from it."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from kinematics import Arm  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    mujoco = pytest.importorskip("mujoco")
    spec = importlib.util.find_spec("so101_nexus")
    if spec is None:
        pytest.skip("so101_nexus is not installed")
    xml = os.path.join(spec.submodule_search_locations[0], "assets", "SO101", "so101_new_calib.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    return mujoco, model, mujoco.MjData(model)


def test_limits_match_model(reference):
    _, model, _ = reference
    arm = Arm()
    assert np.allclose(arm.low, model.jnt_range[:5, 0], atol=1e-12)
    assert np.allclose(arm.high, model.jnt_range[:5, 1], atol=1e-12)


def test_tip_matches_mujoco(reference):
    mujoco, model, data = reference
    site = model.site("gripperframe").id
    arm, rng = Arm(), np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(arm.low, arm.high)
        data.qpos[:5] = q
        mujoco.mj_kinematics(model, data)
        p, z = arm.tip(q)
        assert np.max(np.abs(p - data.site_xpos[site])) < 1e-6
        assert np.max(np.abs(z - data.site_xmat[site].reshape(3, 3)[:, 2])) < 1e-6


def test_solve_reaches_target():
    target = np.array([0.25, 0.05, 0.02])
    q, err, tilt = Arm().solve(target, np.radians([0, 40, -30, 80, 0]))
    assert err < 1e-3 and tilt < 1.0 and q[4] == 0.0


def test_committed_trajectory_regenerates():
    import fold_traj

    committed = json.loads((Path(__file__).parent / "trajectories" / "half_fold_30cm.json").read_text())
    fresh = fold_traj.trajectory(cloth=0.30, near=0.08)
    for side in committed["arms"]:
        old, new = np.array(committed["arms"][side]["target"]), np.array(fresh["arms"][side]["target"])
        assert old.shape == new.shape
        assert np.max(np.abs(old - new)) < 1e-4
