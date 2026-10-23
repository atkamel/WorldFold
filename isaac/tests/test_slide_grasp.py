"""IsaacArmExpert's GRASP "slide" without Isaac: phases, the planned slide and the backed-off release."""

from pathlib import Path

import numpy as np
import pytest

from isaac.fold_expert import IsaacArmExpert
from mujuco.cloth_params import TABLE_TOP_Z


class _IK:
    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0, jaw_weight=None):
        return np.r_[tip, 0.0, 0.0], 0.0          # "joints" = the tip, so the plan shows where it aims


class _Base:
    def __init__(self):
        ix, iy = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
        self.cloth = np.column_stack([ix.ravel() * 0.03 - 0.15, iy.ravel() * 0.03 - 0.15, np.full(121, TABLE_TOP_Z)])

    def cloth_positions(self):
        return self.cloth


def _expert(corner=10, goal=(-0.15, -0.15, TABLE_TOP_Z)):
    e = object.__new__(IsaacArmExpert)
    e.GRASP, e._ik, e.base, e.prefix, e.corners = "slide", _IK(), _Base(), "left_", (corner,)
    e.goal = lambda: np.asarray(goal, float)
    return e


def test_phases_gain_slide_open():
    class Slide(IsaacArmExpert):
        GRASP = "slide"
    e = object.__new__(Slide)
    e.PHASES, e.OPEN_PHASES = IsaacArmExpert.PHASES, IsaacArmExpert.OPEN_PHASES
    i = e.PHASES.index("descend") + 1
    e.PHASES = e.PHASES[:i] + ("slide",) + e.PHASES[i:]     # what __init__ does for GRASP "slide"
    assert e.PHASES[:4] == ("approach", "descend", "slide", "close")


def test_slide_plan_geometry():
    e = _expert()                                            # cloth_10 = (-0.15, +0.15): the far-left corner
    plan = e._slide_plan(from_q=np.zeros(5))
    corner = np.array([-0.15, 0.15])
    grasp = corner + [0.027, -0.02]                         # in from the corner, toward the cloth
    z = TABLE_TOP_Z + e.CONTACT_HEIGHT - e.PRESS
    assert np.allclose(plan["slide_dir"], [1.0, 0.0]) and np.allclose(plan["jaw"], [1.0, 0.0, 0.0])
    assert np.allclose(plan["pinch"][:3], [grasp[0] - e.SLIDE_LEN, grasp[1], z])     # pressed, SLIDE_LEN outside
    assert np.allclose(plan["slide"][-1][:3], [grasp[0], grasp[1], z])
    assert np.allclose(plan["above"][2], TABLE_TOP_Z + e.HOVER)
    xs = [q[0] for q in plan["slide"]]
    assert all(np.diff(xs) > 0)                              # slides inward along +x


def test_release_backs_off_then_lifts():
    e = _expert()
    plan = e._slide_plan(from_q=np.zeros(5))
    r = np.array([q[:3] for q in plan["retreat_path"]])
    place = plan["arc_tips"][-1]
    assert np.allclose(r[0][:2], place[:2] - e.RELEASE_BACKOFF * plan["slide_dir"])   # back off along the slide
    assert np.isclose(r[0][2], place[2])
    assert np.allclose(np.diff(r[:, 2]), e.RETREAT_STEP) and np.allclose(r[:, :2], r[0][:2])


ASSETS = Path(__file__).resolve().parents[2] / "outputs" / "isaac" / "assets"


@pytest.mark.skipif(not (ASSETS / "robots" / "so101_new_calib.urdf").exists(), reason="needs a local LeHome URDF copy")
def test_jaw_weight_fixes_the_half_turn():
    import isaac.layout_search as L                          # stubs lehome and builds PinchIK on the local URDF
    from isaac.isaac_env import PROFILES
    L._configure(0.0, 0.15)
    L._init(str(ASSETS))
    ik = L._IK
    ik.root = PROFILES["roy"]["arm_bases"]
    jaw = np.array([1.0, 0.0, 0.0])                          # the left arm's slide: inward along +x
    tip = np.array([-0.17, 0.13, TABLE_TOP_Z - 0.002])
    reached = 0
    for pan in np.linspace(-1.8, 1.8, 7):                   # IsaacArmExpert._home_seed's sweep
        q, err = ik.solve("left_", tip, jaw, [pan, 1.07, -1.19, 1.66, -2.05], orientation_weight=0.005, jaw_weight=0.1)
        if err > 0.005:
            continue                                         # a seed pointing away; the sweep keeps the closest
        reached += 1
        opens_along = -ik.site_pose("left_", q)[1][:, 0]     # the site's -x: fixed finger -> moving jaw
        assert opens_along @ jaw > 0.9, (pan, opens_along)
    assert reached >= 1
