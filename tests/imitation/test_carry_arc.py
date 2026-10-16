"""QD (2026-10-16): the stage-0 carry profile knobs QF0_ARC_FLAT / QF0_CHORD_CAP (isaac.fold_expert.carry_arc_z), sim-free."""
import math

import numpy as np
import pytest

from isaac.fold_expert import carry_arc_z

START = np.array([-0.15, 0.02, 0.425])      # far corner pinch
PLACE = np.array([-0.15, -0.23, 0.431])     # near corner, placed short (overshoot)
GOAL = np.array([-0.15, -0.28, 0.425])      # the anchor corner


def test_plain_profile_is_a_sine_bump():
    zs = [carry_arc_z(s, START, PLACE, GOAL, 0.12) for s in np.linspace(0, 1, 11)]
    assert zs[5] == pytest.approx(START[2] + (PLACE[2] - START[2]) * 0.5 + 0.12)
    assert zs[0] == pytest.approx(START[2]) and zs[-1] == pytest.approx(PLACE[2])


def test_flat_top_holds_the_peak_and_drops_late():
    zs = [carry_arc_z(s, START, PLACE, GOAL, 0.15, flat=2) for s in np.linspace(0, 1, 21)]
    peak = START[2] + 0.15
    held = [i for i, z in enumerate(zs) if abs(z - peak - (PLACE[2] - START[2]) * i / 20) < 1e-6]
    assert held and held[0] <= 4 and held[-1] >= 16        # at the peak from ~s 0.17 to ~s 0.83
    assert zs[-2] < zs[-4] < zs[-6]                          # dropping over the last waypoints


def test_chord_cap_never_pulls_the_sheet_taut():
    reach = np.linalg.norm(START[:2] - GOAL[:2]) - 0.01
    for s in np.linspace(0.01, 0.99, 50):
        z = carry_arc_z(s, START, PLACE, GOAL, 0.20, flat=3, cap=0.01)
        tip_xy = START[:2] + (PLACE[:2] - START[:2]) * (1 - math.cos(math.pi * s)) / 2
        chord = math.hypot(np.linalg.norm(tip_xy - GOAL[:2]), z - START[2])
        assert chord <= max(reach, np.linalg.norm(tip_xy - GOAL[:2])) + 1e-9   # at table height until within reach
    # the cap only ever lowers the arc
    for s in np.linspace(0.01, 0.99, 50):
        assert carry_arc_z(s, START, PLACE, GOAL, 0.20, 3, 0.01) <= carry_arc_z(s, START, PLACE, GOAL, 0.20, 3) + 1e-12


def test_laydown_is_a_quarter_circle_about_the_crease():
    from isaac.fold_expert import laydown_tip
    crease = START[:2] + 0.5 * (GOAL[:2] - START[:2])
    top = laydown_tip(0.5, START, PLACE, GOAL, 0.13)
    end = laydown_tip(1.0, START, PLACE, GOAL, 0.13)
    assert np.allclose(top[:2], crease) and top[2] == pytest.approx(PLACE[2] + 0.13)
    assert end[2] == pytest.approx(PLACE[2]) and np.linalg.norm(end[:2] - crease) == pytest.approx(0.13)
    assert end[1] < crease[1]                                        # toward the arms, along the carry
    for s in np.linspace(0.5, 1.0, 11):
        tip = laydown_tip(s, START, PLACE, GOAL, 0.13)
        assert math.hypot(np.linalg.norm(tip[:2] - crease), tip[2] - PLACE[2]) == pytest.approx(0.13)
