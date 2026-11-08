"""The cloth stiffness and resolution knobs without Isaac: the spring knobs land on LeHome's garment_config, and a
coarser particle mesh still maps the 11x11 scoring grid onto the same points."""

import numpy as np
import pytest

from isaac.sim.isaac_env import CLOTH_SPRING_KNOBS, FRICTION_GRASP, cloth_grid_mesh, garment_springs, grid_particles

LEHOME = {"particle_mass": 1e-2, "stretch_stiffness": 1e8, "bend_stiffness": 5000.0, "shear_stiffness": 5000.0,
          "spring_damping": 10.0}


def test_spring_knobs_are_grasp_knobs():
    assert set(CLOTH_SPRING_KNOBS) | {"cloth_subdiv"} <= set(FRICTION_GRASP)


def test_garment_springs_sets_only_given_knobs():
    cfg = garment_springs(dict(LEHOME), {"bend_stiffness": 5e5, "shear_stiffness": None, "adhesion": 0.1})
    assert cfg == dict(LEHOME, bend_stiffness=5e5)


def test_garment_springs_none_keeps_lehome():
    assert garment_springs(dict(LEHOME), dict(FRICTION_GRASP)) == LEHOME


@pytest.mark.parametrize("subdiv", [1, 3, 5, 10])
def test_grid_points_match_at_any_resolution(subdiv):
    grid, _ = cloth_grid_mesh(1)
    fine, _ = cloth_grid_mesh(subdiv)
    idx = grid_particles(subdiv)
    assert len(fine) == (10 * subdiv + 1) ** 2
    np.testing.assert_allclose(np.asarray(fine)[idx], np.asarray(grid), atol=1e-12)


def test_drape_metrics_on_a_bent_sheet():
    from isaac.tools.drape_probe import N, drape_metrics, row
    grid = np.asarray(cloth_grid_mesh(1)[0]) + [0.0, 0.25, 0.80]    # free edge (iy 10) at y 0.40, 0.80 up
    edge_y = 0.35
    for iy, drop in ((8, 0.01), (9, 0.03), (10, 0.06)):             # the outer rows hang down
        grid[row(iy), 2] -= drop
    m = drape_metrics(grid, edge_y)
    assert m["droop_cm"] == pytest.approx(6.0) and m["reach_cm"] == pytest.approx(5.0)
    assert m["profile"][7] == pytest.approx([-4.0, 0.0]) and m["profile"][9][1] == pytest.approx(3.0)
    assert len(m["profile"]) == N - 6
