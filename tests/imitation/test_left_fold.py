"""R2 option 1 (user 2026-10-15): the square's second fold by the left arm alone, the right arm anchoring the east half."""

import types

import numpy as np

from cloth_fold_rl.quarter_fold_env import (CLOTH_0, CLOTH_10, CLOTH_110, CLOTH_115, CLOTH_5, CLOTH_ANCHOR, CLOTH_E_MID,
                                            CLOTH_W_MID, CLOTH_W_MID_TOP, GRASP_CORNERS_LEFT_FOLD, N, STAGES,
                                            STAGES_LEFT_FOLD, QuarterFoldEnv)


def test_task_selects_the_left_fold(monkeypatch):
    from imitation.tasks.half_fold import task_grasp_corners, task_stages
    monkeypatch.setenv("WORLDFOLD_QF_SECOND", "left_fold")
    assert task_stages("quarter_fold", "isaac_friction") == STAGES_LEFT_FOLD
    assert task_grasp_corners("quarter_fold", "isaac_friction") == GRASP_CORNERS_LEFT_FOLD
    assert STAGES_LEFT_FOLD[0] == STAGES[0]
    moves = {m.prefix: m for m in STAGES_LEFT_FOLD[1].moves}
    assert moves["left_"].kind == "carry" and moves["right_"].kind == "anchor"
    assert moves["right_"].corners == (moves["right_"].goal,)
    for stage in STAGES_LEFT_FOLD:      # every point a carrying arm takes is one it may grasp
        for m in stage.moves:
            if m.kind == "carry":
                assert set(m.corners) <= set(GRASP_CORNERS_LEFT_FOLD[m.prefix])
    assert task_stages("quarter_fold", "mujoco") == STAGES


def test_left_fold_points_on_the_half_folded_strip():
    from isaac.diag_clean_fold import fold_half
    h = 0.15
    xs = np.linspace(-h, h, N)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    flat = np.stack([gx.ravel(), gy.ravel(), np.full(N * N, 0.424)], axis=1)
    p = fold_half(flat, flat[CLOTH_10], flat[CLOTH_0], radius=0.003)
    # the carried pair is the west edge's midpoint through both layers, its goal the east edge's midpoint
    assert np.linalg.norm(p[CLOTH_W_MID, :2] - p[CLOTH_W_MID_TOP, :2]) < 1e-6
    assert p[CLOTH_W_MID_TOP, 2] > p[CLOTH_W_MID, 2]
    assert np.allclose(p[CLOTH_E_MID, :2], p[CLOTH_W_MID, :2] * [-1, 1])
    assert p[CLOTH_W_MID, 0] == -h and p[CLOTH_W_MID, 1] > p[CLOTH_0, 1] and p[CLOTH_W_MID, 1] < p[CLOTH_5, 1]
    # the anchor presses the east half, inside the strip and clear of its edges
    a = p[CLOTH_ANCHOR]
    assert 0 < a[0] < h - 0.03 and p[CLOTH_0, 1] + 0.03 < a[1] < p[CLOTH_5, 1] - 0.03


def test_square_checks_gate_the_stage():
    from isaac.diag_clean_fold import begin_stage
    env = QuarterFoldEnv(seed=0, stages=STAGES_LEFT_FOLD)
    env.start_hook = lambda e: begin_stage(e, 1)
    env.reset(seed=0)
    checks = STAGES_LEFT_FOLD[1].checks
    assert ((CLOTH_0, CLOTH_10), CLOTH_110) in checks and ((CLOTH_5,), CLOTH_115) in checks
    # unfolded cloth: the west corners are 30 cm from the east ones, so the stage cannot settle
    assert not env._check_met((CLOTH_0, CLOTH_10), CLOTH_110)
    assert env._check_met((CLOTH_110,), CLOTH_110)
    env._placed = lambda m: True
    env._grasped = lambda p: False
    assert not env._stage_settled()
    env.close()


def test_anchor_presses_first_and_lifts_off_when_the_flap_is_placed():
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
    carry_ph = ("approach", "descend", "close", "lift", "carry", "place", "hold", "release", "retreat", "done")
    anchor_ph = ("approach", "descend", "hold", "release", "retreat", "park", "done")

    class Arm:
        RELEASE_PHASES = ("release", "retreat", "done")

        def __init__(self, phases, phase, anchor=False):
            self.PHASES, self.anchor = phases, anchor
            self.phase, self.waiting, self.release_allowed, self.lift_allowed = phases.index(phase), False, False, True

    q = QuarterFoldExpert.__new__(QuarterFoldExpert)
    q.env = types.SimpleNamespace(stage=1, stages=STAGES_LEFT_FOLD)
    left, right = Arm(carry_ph, "approach"), Arm(anchor_ph, "descend", anchor=True)
    q.experts = {(1, "left_"): left, (1, "right_"): right}
    q.turn_done = set()
    assert q._stage_role() == "left_fold"
    q._update_sequence()
    assert left.waiting                                   # the left arm waits until the anchor presses
    right.phase = anchor_ph.index("hold")
    q._update_sequence()
    q._update_lift_gate()
    q._update_release_gate()
    assert not left.waiting and not right.release_allowed
    left.phase = carry_ph.index("close")
    q._update_lift_gate()
    assert left.lift_allowed                               # the pressing anchor counts as ready
    left.phase = carry_ph.index("lift")
    q._update_release_gate()
    assert not right.release_allowed                       # still pressing through the lift
    left.phase = carry_ph.index("carry")
    q._update_release_gate()
    assert right.release_allowed and not left.release_allowed   # clears off before the carry crosses the east half
    left.phase = carry_ph.index("hold")
    q._update_release_gate()
    assert left.release_allowed


def test_square_up_stages(monkeypatch):
    from cloth_fold_rl.quarter_fold_env import (GRASP_CORNERS_LEFT_FOLD_SQUARE, STAGE1_LEFT_FOLD,
                                                STAGES_LEFT_FOLD_SQUARE)
    from imitation.tasks.half_fold import task_grasp_corners, task_stages
    monkeypatch.setenv("WORLDFOLD_QF_SECOND", "left_fold_square")
    st = task_stages("quarter_fold", "isaac_friction")
    assert st == STAGES_LEFT_FOLD_SQUARE and len(st) == 4 and st[0] == STAGES[0]
    assert task_grasp_corners("quarter_fold", "isaac_friction") == GRASP_CORNERS_LEFT_FOLD_SQUARE
    fold, stack, crease = st[1:]
    # the rough fold settles on its own move, within ROUGH_TOL, and the anchor
    assert [(m.prefix, m.corners, m.goal, m.kind) for m in fold.moves] == [
        (m.prefix, m.corners, m.goal, m.kind) for m in STAGE1_LEFT_FOLD.moves]
    assert fold.checks == () and fold.moves[0].tol == 0.10 and STAGE1_LEFT_FOLD.moves[0].tol is None
    assert [(m.prefix, m.corners, m.goal) for m in stack.moves] == [("right_", (CLOTH_0, CLOTH_10), CLOTH_110)]
    assert [(m.prefix, m.corners, m.goal) for m in crease.moves] == [("right_", (CLOTH_5,), CLOTH_115)]
    assert crease.checks == STAGE1_LEFT_FOLD.checks                       # the square is checked at the end
    for stage in st:
        for m in stage.moves:
            if m.kind == "carry":
                assert set(m.corners) <= set(GRASP_CORNERS_LEFT_FOLD_SQUARE[m.prefix])


def test_one_arm_stage_parks_the_idle_arm():
    from cloth_fold_rl.quarter_fold_env import STAGES_LEFT_FOLD_SQUARE
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert

    class Arm:
        PHASES = ("approach", "close", "hold", "release", "done")
        RELEASE_PHASES = ("release", "done")
        phase, waiting, release_allowed, lift_allowed = 0, False, False, True

        def act(self):
            return np.full(6, 0.5, dtype=np.float32)

    class Cls:
        @staticmethod
        def park_action(base, prefix):
            return np.full(6, -0.25, dtype=np.float32)

    q = QuarterFoldExpert.__new__(QuarterFoldExpert)
    q.env = types.SimpleNamespace(stage=2, stages=STAGES_LEFT_FOLD_SQUARE, _placed=lambda m: True)
    q.base = types.SimpleNamespace(prefixes=("left_", "right_"))
    q.expert_cls = Cls
    q.experts = {(2, "right_"): Arm()}
    q.turn_done, q.done_steps, q.retries, q.correction = set(), {(2, "right_"): 0}, {(2, "right_"): 0}, {}
    act = q.act()
    assert act.shape == (12,) and np.all(act[:6] == -0.25) and np.all(act[6:] == 0.5)
    assert q.phases() == {"left_": "idle", "right_": "approach"}


def test_push_square_up_stages_and_timing(monkeypatch):
    from cloth_fold_rl.quarter_fold_env import CLOTH_FOLD_MID, STAGE1_LEFT_FOLD, STAGES_LEFT_FOLD_PUSH
    from cloth_fold_rl.quarter_fold_expert import QuarterFoldExpert
    from imitation.tasks.half_fold import task_stages
    monkeypatch.setenv("WORLDFOLD_QF_SECOND", "left_fold_push")
    st = task_stages("quarter_fold", "isaac_friction")
    assert st == STAGES_LEFT_FOLD_PUSH and len(st) == 4
    for stage, (pts, goal) in zip(st[2:], (((CLOTH_0, CLOTH_10), CLOTH_110), ((CLOTH_5,), CLOTH_115))):
        moves = {m.prefix: m for m in stage.moves}
        assert (moves["right_"].kind, moves["right_"].corners, moves["right_"].goal) == ("push", pts, goal)
        assert (moves["left_"].kind, moves["left_"].corners) == ("anchor", (CLOTH_FOLD_MID,))
    assert st[3].checks == STAGE1_LEFT_FOLD.checks

    push_ph = ("approach", "descend", "carry", "place", "hold", "release", "retreat", "done")
    anchor_ph = ("approach", "descend", "hold", "release", "retreat", "park", "done")

    class Arm:
        RELEASE_PHASES = ("release", "retreat", "done")

        def __init__(self, phases, phase, anchor=False, push=False):
            self.PHASES, self.anchor, self.push = phases, anchor, push
            self.phase, self.waiting, self.release_allowed, self.lift_allowed = phases.index(phase), False, False, True

    q = QuarterFoldExpert.__new__(QuarterFoldExpert)
    q.env = types.SimpleNamespace(stage=2, stages=STAGES_LEFT_FOLD_PUSH)
    left, right = Arm(anchor_ph, "descend", anchor=True), Arm(push_ph, "approach", push=True)
    q.experts = {(2, "left_"): left, (2, "right_"): right}
    q.turn_done = set()
    assert q._stage_role() == "square_push"
    q._update_sequence()
    assert right.waiting and not left.waiting             # the anchor presses before the push starts
    left.phase = anchor_ph.index("hold")
    q._update_sequence()
    assert not right.waiting
    right.phase = push_ph.index("carry")
    q._update_release_gate()
    assert not left.release_allowed                        # held down through the slide
    right.phase = push_ph.index("hold")
    q._update_release_gate()
    assert left.release_allowed and right.release_allowed
    q.env = types.SimpleNamespace(stage=1, stages=STAGES_LEFT_FOLD_PUSH)   # the rough fold keeps its role
    assert q._stage_role() == "left_fold"
