"""The residual-RL window's pure parts without Isaac: observation layout, shaping, terminal score, action mapping,
and the arms' gates."""

import numpy as np

from isaac import precision_env as pe
from mujuco.cloth_params import TABLE_TOP_Z

S = 0.027


def _flat():
    ix, iy = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    return np.column_stack([ix.ravel() * S - 0.135, iy.ravel() * S - 0.135, np.full(121, TABLE_TOP_Z)])


def _folded():
    c = _flat()
    top = (np.arange(121) % 11) > 5
    c[top, 1] = -c[top, 1]                 # north half mirrored onto the south half
    c[top, 2] += 0.008
    return c


def _arm(phase="carry"):
    return {"corner": np.zeros(3), "partner": np.zeros(3), "site": np.zeros(3), "site_vel": np.zeros(3), "jaw": 0.0,
            "residual": np.zeros(3), "phase": phase, "phase_steps": 3}


def test_pairs_are_the_half_folds_mirror_pairs():
    assert (0 * 11 + 10, 0) in [tuple(p) for p in pe.PAIRS]          # cloth_10 onto cloth_0
    assert (10 * 11 + 10, 110) in [tuple(p) for p in pe.PAIRS]       # cloth_120 onto cloth_110
    assert all((t % 11) + (b % 11) == 10 and t // 11 == b // 11 for t, b in pe.PAIRS)


def test_features_have_the_declared_size():
    f = pe.features(_flat(), _flat(), (_arm(), _arm("hold")), np.zeros(pe.ACT_DIM))
    assert f.shape == (pe.OBS_DIM,) and f.dtype == np.float32


def test_potential_rises_as_the_layers_stack():
    assert pe.potential(_folded()) > pe.potential(_flat())


def test_shaping_telescopes_without_action_or_drift():
    states = [_flat(), (_flat() + _folded()) / 2, _folded()]
    phis = [pe.potential(c) for c in states]
    total = sum(pe.step_reward(phis[k], phis[k + 1], np.zeros(pe.ACT_DIM), 0.0, 0.0) for k in range(2))
    assert np.isclose(total, phis[-1] - phis[0])


def test_drift_and_action_cost_lower_the_step_reward():
    a = np.zeros(pe.ACT_DIM)
    a[0] = 1.0
    assert pe.step_reward(0.0, 0.0, a, 0.0, 0.0) < 0.0
    assert pe.step_reward(0.0, 0.0, np.zeros(pe.ACT_DIM), 0.0, 0.01) < 0.0
    assert pe.step_reward(0.0, 0.0, np.zeros(pe.ACT_DIM), 0.02, 0.01) == 0.0        # drift only counts when new


def test_terminal_reward_pays_the_pass_bonus_for_a_clean_fold():
    r, s = pe.terminal_reward(_folded())
    assert s["passed"] and r > pe.PASS_BONUS
    r_bad, s_bad = pe.terminal_reward(_flat())
    assert not s_bad["passed"] and r_bad < 0.0


class _E:
    PHASES = ("approach", "descend", "close", "lift", "carry", "place", "hold", "release", "retreat", "done")
    RELEASE_PHASES = ("release", "retreat", "done")

    def __init__(self, phase):
        self.phase = self.PHASES.index(phase)
        self.residual = np.zeros(3)
        self.release_vote = True
        self.release_allowed = False
        self.lift_allowed = True


def test_actions_move_the_residual_within_bounds_and_set_the_vote():
    ex = {"left_": _E("carry"), "right_": _E("carry")}
    for _ in range(100):
        pe.apply_action(ex, [1.0, -1.0, 0.0, -0.9, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(ex["left_"].residual, [pe.RES_MAX, -pe.RES_MAX, 0.0])
    assert not ex["left_"].release_vote and ex["right_"].release_vote      # the zero action votes to release


def test_gates_hold_the_lift_and_the_release():
    ex = {"left_": _E("close"), "right_": _E("descend")}
    pe.coordinate(ex)
    assert not ex["left_"].lift_allowed
    ex = {"left_": _E("hold"), "right_": _E("place")}
    pe.coordinate(ex)
    assert not ex["left_"].release_allowed
    ex["right_"].phase = _E.PHASES.index("hold")
    pe.coordinate(ex)
    assert ex["left_"].release_allowed and ex["right_"].release_allowed
