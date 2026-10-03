import itertools

import numpy as np

from imitation.dagger import round_seeds
from imitation.seeds import (EVAL_SEED_BASE, ISAAC_HARD_RMAX, SEED_RANGES, SHIFT_SEED_BASE, TUNE_SEED_BASE,
                             eval_set, shifted_pose, shifted_pose_isaac)


def test_seed_ranges_never_overlap():
    for (a, (a0, a1)), (b, (b0, b1)) in itertools.combinations(SEED_RANGES.items(), 2):
        assert a1 <= b0 or b1 <= a0, f"{a} overlaps {b}"
    for name, base in EVAL_SEED_BASE.items():
        assert SEED_RANGES[name][0] == base
    assert SEED_RANGES["tune"][0] == TUNE_SEED_BASE


def test_mujoco_eval_sets_unchanged():
    # values computed with the pre-backend code
    np.testing.assert_array_equal(shifted_pose(200000)["cloth_pose"], [-0.0032904250896706733, -0.0371428743901908])
    np.testing.assert_array_equal(shifted_pose(200007)["cloth_pose"], [0.034382680542832784, -0.017548458103994795])
    seeds, opts, perturb = eval_set("id_hard", 3)
    assert seeds == [200000, 200001, 200002] and opts is shifted_pose and perturb is None
    assert eval_set("id_easy", 2) == ([100000, 100001], None, None)
    for seed in range(40):
        p = eval_set("recovery", 1)[2](seed, np.random.default_rng(seed))
        assert 15 <= p.t < 60 and 8 <= p.k < 16


def test_isaac_id_hard_poses():
    seeds, opts, perturb = eval_set("id_hard", 5, backend="isaac")
    assert seeds[0] == 200000 and opts is shifted_pose_isaac and perturb is None
    for seed in range(200000, 200100):
        p = shifted_pose_isaac(seed)["cloth_pose"]
        np.testing.assert_array_equal(p, shifted_pose_isaac(seed)["cloth_pose"])
        assert np.abs(p).max() <= ISAAC_HARD_RMAX + 1e-12 and np.abs(p).max() >= 0.01


def test_isaac_recovery_onset_window():
    assert eval_set("id_easy", 2, backend="isaac") == ([100000, 100001], None, None)
    ts = [eval_set("recovery", 1, backend="isaac")[2](s, np.random.default_rng(s)).t for s in range(300)]
    assert min(ts) >= 35 and max(ts) < 140 and max(ts) > 100


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
