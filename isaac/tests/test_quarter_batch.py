"""The quarter-fold driver's pure parts without Isaac."""

import numpy as np

from isaac.quarter_batch import EAST, WEST, park_action


def test_park_moves_toward_home_at_the_speed_limit_with_the_jaw_open():
    a = park_action([0.5, -0.2, 0.0, 0.1, 0.0], np.zeros(5), 0.1)
    assert np.isclose(np.max(np.abs(a[:5])), 1.0) and a[0] < 0 and a[1] > 0 and a[5] == 1.0
    assert np.allclose(park_action(np.zeros(5), np.zeros(5), 0.1)[:5], 0.0)


def test_edge_middles_are_stacked_pairs_of_the_half_fold():
    for bottom, top in (WEST, EAST):
        assert bottom // 11 == top // 11 and (bottom % 11) + (top % 11) == 10     # (ix, iy) under (ix, 10 - iy)
    assert WEST[0] // 11 == 0 and EAST[0] // 11 == 10
