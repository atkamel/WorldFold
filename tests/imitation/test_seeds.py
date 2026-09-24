import itertools

import numpy as np

from imitation.dagger import round_seeds
from imitation.seeds import EVAL_SEED_BASE, SEED_RANGES, SHIFT_SEED_BASE, shifted_pose


def test_seed_ranges_never_overlap():
    for (a, (a0, a1)), (b, (b0, b1)) in itertools.combinations(SEED_RANGES.items(), 2):
        assert a1 <= b0 or b1 <= a0, f"{a} overlaps {b}"
    for name, base in EVAL_SEED_BASE.items():
        assert SEED_RANGES[name][0] == base


def test_shifted_pose_is_deterministic_and_outside_the_demo_jitter():
    for seed in range(50):
        p = shifted_pose(seed)["cloth_pose"]
        np.testing.assert_array_equal(p, shifted_pose(seed)["cloth_pose"])
        assert np.abs(p).max() >= 0.025 and np.abs(p).max() <= 0.04


def test_round_seeds_mix_shifted_poses_from_their_own_range():
    seeds, opts = round_seeds(2, 10, 0.5)
    shifted = [s for s in seeds if s >= SHIFT_SEED_BASE]
    assert len(seeds) == 10 and len(shifted) == 5 and len(set(seeds)) == 10
    assert opts(shifted[0]) is not None and opts(seeds[0]) is None
    assert round_seeds(2, 10, 0.0)[1] is None
