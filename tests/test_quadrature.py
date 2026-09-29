import numpy as np

from b3_micromech.problem import RVEProblem
from b3_micromech.quadrature import (
    effective_stiffnesses_for_gauss_points,
    global_stiffness_at_points,
)


def _fibre_problem(vf: float = 0.5) -> RVEProblem:
    return RVEProblem.from_config(
        {
            "domain": {"shape": "square", "size": 1.0, "mesh_resolution": [4, 4]},
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3e9,
                    "poisson_ratio": 0.35,
                },
                {
                    "name": "fibre",
                    "type": "transverse_isotropic",
                    "e_l": 230e9,
                    "e_t": 15e9,
                    "g_lt": 15e9,
                    "nu_lt": 0.2,
                    "nu_tt": 0.3,
                },
            ],
            "rve": {
                "matrix_material": "matrix",
                "fibre_material": "fibre",
                "fibre_volume_fraction": vf,
            },
        }
    )


def test_global_stiffness_at_points_switches_phase():
    problem = _fibre_problem()
    centre = problem.centre_yz
    c_in = global_stiffness_at_points(problem, centre[None, :])[0]
    c_out = global_stiffness_at_points(problem, np.array([[0.0, 0.0]]))[0]
    assert np.allclose(c_in, problem.materials["fibre"].stiffness)
    assert np.allclose(c_out, problem.materials["matrix"].stiffness)


def test_local_cloud_blends_stiffness_across_fibre_boundary():
    problem = _fibre_problem()
    matrix = problem.materials["matrix"].stiffness
    fibre = problem.materials["fibre"].stiffness
    r = problem.fibre_radius
    centre = problem.centre_yz

    # Quad straddling the fibre boundary; GP on the matrix side of the cell.
    cell_vertices = np.array(
        [
            [
                [centre[0] - 1.2 * r, centre[1] - 0.2 * r],
                [centre[0] + 1.2 * r, centre[1] - 0.2 * r],
                [centre[0] + 1.2 * r, centre[1] + 0.2 * r],
                [centre[0] - 1.2 * r, centre[1] + 0.2 * r],
            ]
        ],
        dtype=float,
    )
    gp_coords = np.array([[centre[0] - 1.1 * r, centre[1]]], dtype=float)
    gp_cell_ids = np.zeros(1, dtype=np.intp)

    c_exact = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords,
        gp_cell_ids,
        cell_vertices,
        spec={"strategy": "exact", "resolution": 3, "idw_power": 2.0},
    )[0]
    c_cloud = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords,
        gp_cell_ids,
        cell_vertices,
        spec={"strategy": "local_cloud", "resolution": 6, "idw_power": 2.0},
    )[0]
    c_centroid = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords,
        gp_cell_ids,
        cell_vertices,
        spec={"strategy": "cell_constant", "resolution": 1, "idw_power": 2.0},
    )[0]

    assert np.allclose(c_exact, matrix)
    assert np.allclose(c_centroid, fibre)
    assert not np.allclose(c_cloud, matrix)
    assert not np.allclose(c_cloud, fibre)
    assert not np.allclose(c_cloud, c_centroid)
    assert np.linalg.norm(c_cloud - matrix) < np.linalg.norm(fibre - matrix)
    assert np.linalg.norm(fibre - c_cloud) < np.linalg.norm(fibre - matrix)
