"""The quarter-fold driver's pure parts without Isaac."""

import numpy as np

from isaac.tools.quarter_batch import EAST, TRACE_KEYS, TWO, TWO_IK, WEST, QuarterBatch, park_action


def test_park_moves_toward_home_at_the_speed_limit_with_the_jaw_open():
    a = park_action([0.5, -0.2, 0.0, 0.1, 0.0], np.zeros(5), 0.1)
    assert np.isclose(np.max(np.abs(a[:5])), 1.0) and a[0] < 0 and a[1] > 0 and a[5] == 1.0
    assert np.allclose(park_action(np.zeros(5), np.zeros(5), 0.1)[:5], 0.0)


def test_edge_middles_are_stacked_pairs_of_the_half_fold():
    for bottom, top in (WEST, EAST):
        assert bottom // 11 == top // 11 and (bottom % 11) + (top % 11) == 10     # (ix, iy) under (ix, 10 - iy)
    assert WEST[0] // 11 == 0 and EAST[0] // 11 == 10


class _IK:
    def __init__(self, tilt_deg):
        self.R = np.eye(3)
        self.R[2, 2] = -np.cos(np.radians(tilt_deg))      # the fingers' axis tilt_deg from straight down

    def site_pose(self, prefix, q):
        return np.zeros(3), self.R


class _Expert:
    PHASES = ("approach", "descend", "close", "lift", "carry")

    def __init__(self, corners, tilt_deg, phase, dropped=False):
        self.corners, self._ik, self.phase, self.dropped = corners, _IK(tilt_deg), phase, dropped

    def _corner(self):
        return np.full(3, 0.1 * len(self.corners))


class _Sub:
    _gripper_closed = {"left_": True, "right_": False}

    def cloth_positions(self):
        return np.ones((121, 3))

    def joint_positions(self, p):
        return np.r_[np.zeros(5), -0.1 if p == "left_" else 1.0]

    def gripper_position(self, p):
        return np.array([0.0, 0.0, 0.5])


def test_trace_records_each_arm_per_step_with_tilt_labels_and_float32_arrays():
    qb = QuarterBatch.__new__(QuarterBatch)
    qb.subs = [_Sub()]
    qb.experts = [{"left_": _Expert((5,), 30.0, 4, dropped=True), "right_": _Expert((0, 10), 0.0, 0)}]
    qb.parked = [set()]
    qb.trace = [{k: [] for k in ("stage", "cloth", *(p + k for p in ("left_", "right_") for k in TRACE_KEYS))}]
    qb.stage = "fold2"
    qb._record([True])                     # the right arm waits (sequenced approach)
    qb.parked = [{"left_"}]
    qb._record([False])
    t = qb.trace_arrays(0)
    assert list(t["stage"]) == ["fold2", "fold2"] and t["cloth"].shape == (2, 121, 3) and t["cloth"].dtype == np.float32
    assert list(t["left_phase"]) == ["carry", "parked"] and list(t["right_phase"]) == ["wait", "approach"]
    assert list(t["left_corners"]) == ["5", "5"] and list(t["right_corners"]) == ["0+10", "0+10"]
    assert np.allclose(t["left_tilt_deg"], 30.0) and np.allclose(t["right_tilt_deg"], 0.0, atol=1e-3)
    assert np.allclose(t["left_jaw"], -0.1) and list(t["left_closed"]) == [True, True]
    assert t["left_dropped"].all() and not t["right_dropped"].any()
    assert t["right_site"].shape == (2, 3) and np.allclose(t["right_corner"], 0.2)
    assert t["left_q"].shape == (2, 5) and t["left_q"].dtype == np.float32


class _PinchExpert:
    PINCH_ORIENTATION_WEIGHT, JAW_WEIGHT = 0.1, 0.1        # IsaacArmExpert's class defaults
    dropped = False

    def reset(self):
        self.phase = 0


class _FlatSub:
    def cloth_positions(self):
        ix, iy = np.divmod(np.arange(121), 11)
        return np.c_[0.03 * ix, 0.03 * iy, np.zeros(121)]

    def _failed(self):
        return False


def test_fold2_two_softens_only_the_right_arms_pinch_ik_and_the_next_configure_restores_it():
    qb = QuarterBatch.__new__(QuarterBatch)
    qb.n, qb.subs, qb.sequenced = 1, [_FlatSub()], [False]
    qb.experts = [{"left_": _PinchExpert(), "right_": _PinchExpert()}]
    qb._run_stage = lambda cap: [True]                      # no sim: fold 2 "settles" at once
    qb._gate = lambda i: {"valid": True}
    qb._fold2_two([{"seed": 0}])
    left, right = qb.experts[0]["left_"], qb.experts[0]["right_"]
    assert right.corners == TWO["right_"][0] and left.corners == TWO["left_"][0]
    assert (right.PINCH_ORIENTATION_WEIGHT, right.JAW_WEIGHT) == (TWO_IK["right_"]["PINCH_ORIENTATION_WEIGHT"],
                                                                  TWO_IK["right_"]["JAW_WEIGHT"])
    assert (left.PINCH_ORIENTATION_WEIGHT, left.JAW_WEIGHT) == (0.1, 0.1)
    assert not vars(left).keys() & TWO_IK["right_"].keys()
    qb._configure(0, "right_", (10,), None, None, flat=False)          # the next episode's fold 1
    assert (right.PINCH_ORIENTATION_WEIGHT, right.JAW_WEIGHT) == (0.1, 0.1)
