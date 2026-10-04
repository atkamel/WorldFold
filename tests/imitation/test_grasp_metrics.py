"""Friction-grasp bench metrics (track G, IG.1): sim-free, on synthetic traces."""

import numpy as np

from isaac.grasp_metrics import arm_metrics, meets_ig2, summarize, wilson

GOAL = np.array([0.0, -0.3, 0.42])
REST = 0.42


def _trace(drop_at=None, lift=0.05, stuck=False, place_off=0.0):
    """approach 5, close 5, lift 5, carry 10, place 5, hold 1, release 6, retreat 6, done 20."""
    phases = (["approach"] * 5 + ["close"] * 5 + ["lift"] * 5 + ["carry"] * 10 + ["place"] * 5 + ["hold"]
              + ["release"] * 6 + ["retreat"] * 6 + ["done"] * 20)
    T = len(phases)
    site = np.zeros((T, 3))
    corner = np.zeros((T, 3))
    start = np.array([0.0, 0.0, REST])
    end = GOAL + np.array([place_off, 0.0, 0.0])
    for t, ph in enumerate(phases):
        if ph in ("approach", "close"):
            site[t] = start + [0, 0, 0.01]
            corner[t] = start
        elif ph in ("lift", "carry", "place", "hold"):
            s = (t - 10) / 20.0
            pos = start + (end - start) * s
            pos[2] = REST + lift * np.sin(np.pi * min(s, 1.0)) if ph != "hold" else REST
            site[t] = pos + [0, 0, 0.01]
            corner[t] = pos
        else:
            site[t] = end + [0, 0, 0.01 + 0.01 * (t - 31)]
            corner[t] = end if not stuck else site[t] - [0, 0, 0.01]
    if drop_at is not None:
        corner[drop_at:] = corner[drop_at] * [1, 1, 0] + [0, 0, REST]
        corner[drop_at:, 0] += 0.1
    anchor = np.tile([0.0, -0.3, REST], (T, 1))
    return {"phase": phases, "site": site, "corner": corner, "anchor": anchor}


def test_clean_grasp_passes_all():
    m = arm_metrics(_trace(), GOAL, REST)
    assert (m["acquired"], m["held"], m["placed"], m["released"]) == (True, True, True, True)
    assert m["failed_at"] is None and m["anchor_drift"] == 0.0


def test_no_lift_is_not_acquired():
    m = arm_metrics(_trace(lift=0.0), GOAL, REST)
    assert not m["acquired"] and m["failed_at"] == "acquired"


def test_drop_mid_carry_fails_held():
    m = arm_metrics(_trace(drop_at=18), GOAL, REST)
    assert m["acquired"] and not m["held"] and m["failed_at"] == "held" and m["drop_t"] == 18


def test_misplaced_and_stuck():
    assert arm_metrics(_trace(place_off=0.08), GOAL, REST)["failed_at"] == "placed"
    m = arm_metrics(_trace(stuck=True), GOAL, REST)      # carried up by the retreating jaw: neither placed nor released
    assert m["held"] and not m["placed"] and not m["released"] and m["release_z"] > 0.025


def test_summary_and_bar():
    good = {"seed": 1, "arms": {"left_": arm_metrics(_trace(), GOAL, REST), "right_": arm_metrics(_trace(), GOAL, REST)}}
    bad = {"seed": 2, "arms": {"left_": arm_metrics(_trace(drop_at=18), GOAL, REST),
                               "right_": arm_metrics(_trace(), GOAL, REST)}}
    s = summarize([good] * 49 + [bad])
    assert s["arms"]["left_"]["held"]["k"] == 49 and s["arms"]["right_"]["held"]["k"] == 50
    assert s["arms"]["left_"]["first_failure"] == {"held": 1}
    assert meets_ig2(s) and not meets_ig2(summarize([good] * 10 + [bad]))
    lo, hi = wilson(49, 50)
    assert 88 < lo < 90 and hi > 99
