"""The RL corrector's pure parts without Isaac: observation size, action decoding, the score, the scripted push."""

import numpy as np

from isaac.rl import corrector_env as ce
from mujoco_sim.cloth_params import TABLE_TOP_Z

S = 0.03


def _flat():
    ix, iy = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
    return np.column_stack([ix.ravel() * S - 0.15, iy.ravel() * S - 0.15, np.full(121, TABLE_TOP_Z)])


def _folded(shift=(0.0, 0.0)):
    c = _flat()
    top = (np.arange(121) % 11) > 5
    c[top, 1] = -c[top, 1]
    c[top, 2] += 0.005
    c[top, :2] += shift
    return c


def test_features_have_the_declared_size():
    assert ce.features(_folded(), 1).shape == (ce.OBS_DIM,)


def test_zero_action_pushes_nothing_and_does_not_stop():
    pushes, stop = ce.decode(np.zeros(ce.ACT_DIM))
    assert not stop and all(np.linalg.norm(d) < ce.MIN_PUSH for d, _ in pushes.values())
    lo, hi = ce.PRESS_RANGE
    assert all(np.isclose(press, (lo + hi) / 2) for _, press in pushes.values())


def test_score_prefers_the_folded_cloth():
    assert ce.score_s(_folded())[0] > ce.score_s(_folded((0.0, -0.03)))[0] > ce.score_s(_flat())[0]


def test_scripted_action_stops_on_a_clean_fold_and_pushes_a_shifted_one():
    _, stop = ce.decode(ce.scripted_action(_folded()))
    assert stop
    pushes, stop = ce.decode(ce.scripted_action(_folded((0.0, -0.02))))
    assert not stop
    for d, press in pushes.values():
        assert np.allclose(d, [0.0, 0.02]) and np.isclose(press, -0.002)       # back onto the partner, 2 mm in


def test_noise_around_zero_rarely_stops():
    rng = np.random.default_rng(0)
    stops = sum(ce.decode(0.3 * rng.standard_normal(ce.ACT_DIM))[1] for _ in range(2000))
    assert stops / 2000 < 0.1
