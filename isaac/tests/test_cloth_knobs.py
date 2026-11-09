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


def _flap(n, edge_y, reach, drop, z0=0.8):
    """An n x n sheet on the table (z0) whose free edge row (iy = n - 1) sits reach past the edge and drop below."""
    p = np.zeros((n, n, 3))
    p[..., 1] = np.linspace(edge_y - 0.2, edge_y, n)[None, :]
    p[..., 2] = z0
    p[:, -1] = [0.0, edge_y + reach, z0 - drop]
    return p.reshape(-1, 3)


def test_flap_at_the_peirce_angle_gives_half_its_length():
    from isaac.tools.drape_probe import flap_metrics
    reach = 0.03
    m = flap_metrics(_flap(7, 0.35, reach, reach * np.tan(np.radians(41.5))), 7, 0.35, 0.04)
    assert m["clear"] and m["angle_deg"] == pytest.approx(41.5, abs=0.1)
    assert m["bending_length_cm"] == pytest.approx(0.509 * 4.0, abs=0.01)     # ASTM D1388: c = L / 2 at 41.5 deg


def test_flap_on_the_table_face_measures_nothing():
    from isaac.tools.drape_probe import flap_metrics
    m = flap_metrics(_flap(7, 0.35, 0.004, 0.05), 7, 0.35, 0.06)            # lying on the side face, 4 mm out
    assert not m["clear"] and m["bending_length_cm"] is None and m["drop_cm"] == pytest.approx(5.0)
