"""IsaacPlaceServoExpert without Isaac: the residual and the live goal reach the IK target, the release vote and the
hold cap, and a slip ends the carry."""

import numpy as np

from isaac.fold_expert import IsaacArmExpert
from isaac.place_servo import IsaacPlaceServoExpert
from mujuco.cloth_params import TABLE_TOP_Z


class _IK:
    def __init__(self):
        self.tips = []

    def solve(self, prefix, tip, jaw, q, orientation_weight=0.0, iters=None, jaw_weight=None):
        self.tips.append(np.asarray(tip, float).copy())
        return np.r_[tip, 0.0, 0.0], 0.0


class _Base:
    grasp_knobs = {}

    def __init__(self):
        ix, iy = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
        self.cloth = np.column_stack([ix.ravel() * 0.03 - 0.15, iy.ravel() * 0.03 - 0.15, np.full(121, TABLE_TOP_Z)])
        self.site = self.cloth[10] + [0.0, 0.0, 0.03]

    def cloth_positions(self):
        return self.cloth

    def gripper_position(self, prefix):
        return self.site

    def joint_positions(self, prefix):
        return np.zeros(6)


def _expert():
    e = object.__new__(IsaacPlaceServoExpert)
    e.base, e._ik, e.prefix, e.corners = _Base(), _IK(), "left_", (10,)
    e.PHASES = IsaacArmExpert.PHASES + IsaacArmExpert.RELEASE_PHASES
    e.goal = lambda: e.base.cloth[0].copy()
    e.live_goal = lambda: e.base.cloth[0].copy()
    e.reset()
    e.release_allowed = False
    e.plan = {"jaw": np.array([0.0, -1.0, 0.0])}
    return e


def test_residual_shifts_the_carry_target():
    e = _expert()
    e._start_carry()
    e.act()
    plain = e._ik.tips[-1]
    e.residual = np.array([0.01, -0.02, 0.0])
    e.act()
    assert np.allclose(e._ik.tips[-1] - plain, e.residual, atol=1e-9)


def test_place_follows_the_partner_where_it_is_now():
    e = _expert()
    e._go("place")
    e.base.cloth[0] += [0.04, 0.0, 0.0]                    # the bottom layer slid 4 cm
    e.base.site = e.base.cloth[10] + [0.0, 0.0, 0.03]
    e.act()
    corner, goal = e.base.cloth[10], e.base.cloth[0]
    want = e.base.site + np.clip(goal + [0, 0, e.PRESS_Z] - corner, -e.STEER_CLIP, e.STEER_CLIP)
    assert np.allclose(e._ik.tips[-1], want)


def test_release_waits_for_the_gate_and_the_vote_or_the_cap():
    e = _expert()
    e._start_hold()
    e.release_vote = False
    e.act()
    assert e.PHASES[e.phase] == "hold"                     # gate closed
    e.release_allowed = True
    for _ in range(e.HOLD_CAP - 2):
        e.act()
    assert e.PHASES[e.phase] == "hold"                     # gate open, no vote, under the cap
    e.act()
    assert e.PHASES[e.phase] == "release"                  # the cap


def test_vote_releases_at_once_once_the_gate_opens():
    e = _expert()
    e._start_hold()
    e.release_allowed = True
    e.act()
    assert e.PHASES[e.phase] == "release"


def test_slip_in_the_carry_marks_a_drop_and_holds():
    e = _expert()
    e._start_carry()
    e.base.site = e.base.cloth[10] + [0.0, 0.0, 0.2]       # the corner is far from the jaw
    for _ in range(e.SLIP_STEPS):
        e.act()
    assert e.dropped and e.PHASES[e.phase] == "hold"


def test_live_goal_is_capped_around_the_start(monkeypatch):
    e = _expert()
    monkeypatch.setattr(IsaacPlaceServoExpert, "LIVE_MAX", 0.03)
    start = e.base.cloth[0].copy()
    e.goal = lambda: start
    e.base.cloth[0] += [0.08, -0.01, 0.0]
    assert np.allclose(e._goal_now(), start + [0.03, -0.01, 0.0])


def test_the_floor_keeps_a_downward_residual_off_the_table():
    e = _expert()
    e._go("place")
    e.residual = np.array([0.0, 0.0, -0.06])
    e.act()
    assert e._ik.tips[-1][2] >= TABLE_TOP_Z + e.Z_FLOOR - 1e-12


def test_latched_goal_stops_following_the_partner(monkeypatch):
    e = _expert()
    monkeypatch.setattr(IsaacPlaceServoExpert, "LATCH_GOAL", 1)
    e._start_carry()
    e.sv["k"] = len(e.sv["path"]) - 1
    e.base.site = e.sv["path"][-1] + [0.0, 0.0, 0.03]
    e.base.cloth[10] = e.sv["path"][-1]                      # the corner reached the last waypoint
    e.act()
    assert e.PHASES[e.phase] == "place"
    before = e._goal_now().copy()
    e.base.cloth[0] += [0.05, 0.0, 0.0]                      # the partner is pushed away
    assert np.allclose(e._goal_now(), before)


def test_gfold_caps_the_path_height_by_distance(monkeypatch):
    e = _expert()
    monkeypatch.setattr(IsaacPlaceServoExpert, "GFOLD_SLOPE", 0.5)
    corner, goal = e.base.cloth[10], e.base.cloth[0]
    for p in e._corner_path(corner, goal):
        assert p[2] <= goal[2] + e.GFOLD_Z0 + 0.5 * np.linalg.norm(goal[:2] - p[:2]) + 1e-12


def test_backoff_retreat_slides_out_before_lifting(monkeypatch):
    e = _expert()
    monkeypatch.setattr(IsaacPlaceServoExpert, "BACKOFF", 0.02)
    e._start_hold()
    tips = [q[:3] for q in e.plan["retreat_path"]]
    jaw = e.plan["jaw"]
    assert np.allclose(tips[0], e.base.site - 0.02 * np.r_[jaw[:2], 0.0])
    assert len(tips) == 2 and np.allclose(tips[1], tips[0] + [0.0, 0.0, 0.05])


def test_touchup_comes_down_behind_presses_slides_and_lifts():
    e = _expert()
    e.PHASES = e.PHASES + ("touch",)
    corner = e.base.cloth[10].copy()
    e.start_touchup([0.0, -0.02])
    path = e.touch["path"]
    d = np.array([0.0, -1.0, 0.0])
    start = np.r_[corner[:2], corner[2] + e.TOUCH_PRESS] - e.TOUCH_BEHIND * d
    assert np.allclose(path[1], start) and path[0][2] > path[1][2]
    assert np.allclose(path[-2], start + [0.0, -0.02, 0.0]) and path[-1][2] > path[-2][2]
    assert e.PHASES[e.phase] == "touch"
    for _ in range(13 * len(path)):         # the fake site never arrives: each waypoint waits out 13 steps
        e.act()
    assert e.PHASES[e.phase] == "done"


class _JawBase(_Base):
    """A binary gripper whose jaw sits at its target (isaac_env.set_gripper with jaw_open)."""

    def __init__(self):
        super().__init__()
        self._joint_targets = {"left_": np.r_[np.zeros(5), -0.1]}
        self.jaw_open = {"left_": None}

    def joint_positions(self, prefix):
        return self._joint_targets[prefix].copy()

    def apply(self, action):
        if action[5] > 0.3:
            self._joint_targets["left_"][5] = 1.0 if self.jaw_open["left_"] is None else self.jaw_open["left_"]


def _releasing():
    e = _expert()
    e.base = _JawBase()
    e._start_hold()
    e.release_allowed = True
    e.base.apply(e.act())
    assert e.PHASES[e.phase] == "release"
    return e


def test_gentle_release_opens_part_way_slowly_with_the_arm_still():
    e = _releasing()
    site = e.release_site.copy()
    jaws = []
    while e.PHASES[e.phase] == "release":
        e.base.apply(e.act())
        jaws.append(e.base._joint_targets["left_"][5])
        assert np.allclose(e._ik.tips[-1], site)            # holds where it let go
    assert np.allclose(np.diff(jaws[:-1]), e.RELEASE_STEP)
    assert max(jaws) == e.RELEASE_JAW and len(jaws) == int(round((e.RELEASE_JAW + 0.1) / e.RELEASE_STEP))


def test_gentle_retreat_lifts_straight_up_slowly_then_opens_fully():
    e = _releasing()
    while e.PHASES[e.phase] == "release":
        e.base.apply(e.act())
    site, tips = e.release_site.copy(), []
    for _ in range(int(0.05 / e.RETREAT_STEP)):
        e.base.apply(e.act())
        tips.append(e._ik.tips[-1].copy())
        assert e.base._joint_targets["left_"][5] == e.RELEASE_JAW
    tips = np.array(tips)
    assert np.allclose(tips[:, :2], site[:2]) and np.allclose(np.diff(tips[:, 2]), e.RETREAT_STEP)
    e.base.site = site + [0.0, 0.0, 0.05]                   # the fake site arrives at the top
    e.base.apply(e.act())
    assert e.PHASES[e.phase] == "done"
    e.base.apply(e.act())
    assert e.base.jaw_open["left_"] is None and e.base._joint_targets["left_"][5] == 1.0


def test_release_jaw_none_keeps_the_old_release(monkeypatch):
    monkeypatch.setattr(IsaacPlaceServoExpert, "RELEASE_JAW", None)
    e = _releasing()
    action = e.act()
    assert action[5] == 1.0 and e.base.jaw_open["left_"] is None


def test_gentle_retreat_slides_out_by_backoff_before_lifting(monkeypatch):
    monkeypatch.setattr(IsaacPlaceServoExpert, "BACKOFF", 0.02)
    e = _releasing()
    while e.PHASES[e.phase] == "release":
        e.base.apply(e.act())
    site, jaw, n = e.release_site.copy(), e.plan["jaw"], int(round(0.02 / e.RETREAT_STEP))
    tips = []
    for _ in range(n + 3):
        e.act()
        tips.append(e._ik.tips[-1].copy())
    assert np.allclose(tips[n - 1], site - 0.02 * np.r_[jaw[:2], 0.0])      # out at the release height
    assert np.allclose(tips[-1], tips[n - 1] + [0.0, 0.0, 3 * e.RETREAT_STEP])
