"""IsaacArmExpert's GRASP "slide" without Isaac: phases, the planned slide and the backed-off release."""

import numpy as np

from isaac.fold_expert import IsaacArmExpert
from mujuco.cloth_params import TABLE_TOP_Z


class _IK:
    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0):
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
