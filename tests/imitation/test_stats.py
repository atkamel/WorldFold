import math

import pytest

from imitation.stats import (compare_reports, flip_rate, independent_gain_se, mcnemar, paired, paired_gain_se,
                             wilson)


def _rows(outcomes, base=0):
    return [{"seed": base + i, "success": bool(o)} for i, o in enumerate(outcomes)]


def test_mcnemar_matches_a_hand_computed_binomial():
    # b = 1, c = 6 discordant: p = 2 * P(X <= 1 | Bin(7, 1/2)) = 2 * (1 + 7) / 128
    assert math.isclose(mcnemar(1, 6), 16 / 128)
    assert mcnemar(0, 0) == 1.0 and mcnemar(5, 5) == 1.0
    assert mcnemar(0, 10) < 0.002


def test_wilson_is_reexported_by_verify():
    from imitation import verify
    assert verify.wilson is wilson
    lo, hi = wilson(97, 100)
    assert 91 < lo < 92 and 98.9 < hi < 99


def test_paired_counts_only_discordant_seeds_and_aligns_by_seed():
    a = _rows([1, 1, 1, 0, 1, 1, 0, 1])
    b = list(reversed(_rows([1, 0, 1, 0, 0, 1, 0, 1])))     # order must not matter
    p = paired(a, b)
    assert p["n"] == 8 and p["only_a"] == 2 and p["only_b"] == 0
    assert math.isclose(p["diff"], 2 / 8) and p["ci95"][0] <= p["diff"] <= p["ci95"][1]
    with pytest.raises(ValueError):
        paired(_rows([1]), _rows([1], base=100))


def test_paired_gain_is_tighter_than_independent_when_outcomes_correlate():
    # 200 seeds, both solve the same 140, A also solves 10 more: a real but small paired gain
    old = [1] * 140 + [0] * 60
    new = [1] * 150 + [0] * 50
    rep = lambda o: {"id_easy": {"success_rate": sum(o) / len(o), "n": len(o), "episodes": _rows(o)}}
    assert paired_gain_se(rep(new), rep(old), ["id_easy"]) > 3.0
    assert independent_gain_se(rep(new), rep(old), ["id_easy"]) < 1.5
    # reports without rows fall back to the independent test (old JSONs)
    norows = lambda o: {"id_easy": {"success_rate": sum(o) / len(o), "n": len(o)}}
    assert paired_gain_se(norows(new), norows(old), ["id_easy"]) == independent_gain_se(norows(new), norows(old),
                                                                                         ["id_easy"])


def test_flip_rate_and_pooled_compare():
    a, b = _rows([1, 1, 0, 0]), _rows([1, 0, 0, 1])
    f = flip_rate(a, b)
    assert f["flips"] == 2 and f["rate"] == 0.5
    rep = lambda r: {"s1": {"episodes": r}, "s2": {"episodes": r}}
    out = compare_reports(rep(a), rep(b), ["s1", "s2"])
    assert out["pooled"]["n"] == 8 and out["s1"]["n"] == 4
