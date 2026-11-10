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
    qb._record([{"right_"}])               # the right arm waits (sequenced approach)
    qb.parked = [{"left_"}]
    qb._record([set()])
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
    PINCH_ORIENTATION_WEIGHT, JAW_WEIGHT, PINCH_JAW_ACROSS = 0.1, 0.1, 0        # IsaacArmExpert's class defaults
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
    assert all(getattr(right, k) == v for k, v in TWO_IK["right_"].items()) and right.PINCH_JAW_ACROSS == 1
    assert (left.PINCH_ORIENTATION_WEIGHT, left.JAW_WEIGHT, left.PINCH_JAW_ACROSS) == (0.1, 0.1, 0)
    assert not vars(left).keys() & TWO_IK["right_"].keys()
    qb._configure(0, "right_", (10,), None, None, flat=False)          # the next episode's fold 1
    assert (right.PINCH_ORIENTATION_WEIGHT, right.JAW_WEIGHT, right.PINCH_JAW_ACROSS) == (0.1, 0.1, 0)


class _SeqExpert:
    PHASES = ("approach", "descend", "close", "lift", "carry", "place", "hold", "release", "retreat", "done")
    RELEASE_PHASES = ("release", "retreat", "done")

    def __init__(self, phase, phase_steps=0):
        self.phase, self.phase_steps, self.acted = phase, phase_steps, 0

    def act(self):
        self.acted += 1
        return np.full(6, 0.5, dtype=np.float32)


class _SeqSub:
    def __init__(self):
        self.sent = []

    def step_submit(self, a):
        self.sent.append(a)

    def step_collect(self):
        pass


class _Task:
    def _expand(self, a):
        return a


class _Batch:
    def advance(self):
        pass


def _seq_step(left_phase, right_phase, left_steps=0, sequenced=True):
    qb = QuarterBatch.__new__(QuarterBatch)
    qb.subs, qb.tasks, qb.batch, qb.frames, qb.trace = [_SeqSub()], [_Task()], _Batch(), None, None
    qb.parked, qb.sequenced = [set()], [sequenced]
    left, right = _SeqExpert(left_phase, left_steps), _SeqExpert(right_phase)
    qb.experts = [{"left_": left, "right_": right}]
    qb._scripted_step()
    a = qb.subs[0].sent[0]
    held = {p for p, s in (("left_", a[:6]), ("right_", a[6:])) if np.allclose(s, [0, 0, 0, 0, 0, 1.0])}
    return held, left.acted, right.acted


def test_fold2_two_the_left_arm_waits_above_the_crease_end_until_the_right_arm_has_closed():
    assert _seq_step(0, 0) == ({"right_"}, 1, 0)        # the left arm goes first
    assert _seq_step(1, 0) == ({"left_"}, 0, 1)         # above the crease end: it waits, the right arm comes in
    assert _seq_step(1, 1) == ({"left_"}, 0, 1)         # ... while the right arm descends
    assert _seq_step(1, 2) == (set(), 1, 1)             # the right arm has closed: the left one descends
    assert _seq_step(1, 1, left_steps=3) == (set(), 1, 1)       # a descend under way is never stopped
    assert _seq_step(0, 0, sequenced=False) == (set(), 1, 1)    # fold 1 and the drag: no sequencing


class _ReachIK:
    root = {"right_": (np.array([0.25, -0.275, 0.48]), None)}

    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0, jaw_weight=None, iters=300):
        return np.r_[tip, 0.0, 0.0], 0.0          # "joints" = the tip, so the plan shows where it aims


class _HalfSheet:
    _prof = {"arm_bases": True}

    def __init__(self):
        ix, iy = np.divmod(np.arange(121), 11)
        self.cloth = np.c_[-0.15 + 0.03 * ix, -0.135 - 0.15 + 0.03 * np.minimum(iy, 10 - iy), np.full(121, 0.424)]

    def cloth_positions(self):
        return self.cloth


def test_pinch_jaw_across_opens_the_jaw_across_the_arms_reach_on_the_cloth_side():
    from isaac.experts.fold_expert import IsaacArmExpert
    from mujoco_sim.cloth_params import TABLE_TOP_Z
    plans = {}
    for across in (0, 1):
        e = object.__new__(IsaacArmExpert)
        e._ik, e.base, e.prefix, e.corners, e.resynced = _ReachIK(), _HalfSheet(), "right_", (0, 10), False
        e.goal = lambda b=e.base: b.cloth[[110, 120]].mean(axis=0)
        e.PINCH_JAW_ACROSS = across
        plans[across] = e._make_plan(from_q=np.zeros(5))
    corner = e.base.cloth[[0, 10]].mean(axis=0)
    centre = e.base.cloth.mean(axis=0)
    toward_centre = (centre - corner)[:2] / np.linalg.norm((centre - corner)[:2])
    assert np.allclose(plans[0]["jaw"][:2], toward_centre)               # off: the jaw opens toward the centre
    jaw, reach = plans[1]["jaw"], corner[:2] - _ReachIK.root["right_"][0][:2]
    assert abs(jaw[:2] @ reach) < 1e-9 and np.isclose(np.linalg.norm(jaw), 1.0) and jaw[:2] @ toward_centre > 0
    pinch = np.r_[corner[:2] - e.PINCH_INSET * jaw[:2], TABLE_TOP_Z + e.PINCH_HEIGHT]
    assert np.allclose(plans[1]["pinch"][:3], pinch) and np.allclose(plans[1]["above"][:3], pinch + [0, 0, 0.04])
