import numpy as np
import pytest

pytestmark = pytest.mark.mfem

pytest.importorskip("mfem")

from b3_micromech.amr import (
    cell_refinement_metric,
    flag_cells_for_refinement,
    iteratively_refine_mfem,
)
from b3_micromech.problem import RVEProblem


def _problem(
    *, vf: float, resolution: tuple[int, int] = (6, 6), amr: dict | None = None
) -> RVEProblem:
    solver: dict = {"cell_type": "quadrilateral"}
    if amr is not None:
        solver["amr"] = amr
    return RVEProblem.from_config(
        {
            "domain": {"size": 1.0, "mesh_resolution": list(resolution)},
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
                "fibre_material": "fibre" if vf > 0 else "matrix",
                "fibre_volume_fraction": vf,
            },
            "solver": solver,
        }
    )


def test_homogeneous_mesh_has_zero_stiffness_jump():
    from b3_micromech.mesh.cartesian import build_cartesian_mesh

    problem = _problem(vf=0.0)
    mesh = build_cartesian_mesh(problem)
    metric = cell_refinement_metric(problem, mesh, n_samples_per_cell=16)
    assert float(metric.max()) < 1e-12


def test_fibre_interface_cells_are_flagged():
    from b3_micromech.mesh.cartesian import build_cartesian_mesh

    problem = _problem(vf=0.5)
    mesh = build_cartesian_mesh(problem)
    metric = cell_refinement_metric(problem, mesh, n_samples_per_cell=16)
    flagged = flag_cells_for_refinement(metric, threshold=0.05)
    assert flagged.any()
    assert int(flagged.sum()) < mesh.GetNE()


def test_triangular_barycentric_grid_sampler():
    from b3_micromech.amr import _tri_reference_barycentric_grid

    grid = _tri_reference_barycentric_grid(15)
    assert grid is not None
    assert grid.shape == (15, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-12)
    assert _tri_reference_barycentric_grid(16) is None


@pytest.mark.mfem
def test_hex_domain_amr_refines_and_solves():
    pytest.importorskip("triangle")
    from b3_micromech.homogenize import homogenize
    from b3_micromech.mesh.build import build_mesh

    problem = RVEProblem.from_config(
        {
            "domain": {"shape": "hexagon", "size": 1.0, "mesh_resolution": [10]},
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
                "fibre_volume_fraction": 0.5,
            },
            "solver": {
                "cell_type": "triangle",
                "amr": {
                    "enabled": True,
                    "max_iterations": 1,
                    "threshold": 0.05,
                    "n_samples_per_cell": 15,
                },
            },
        }
    )
    mesh0 = build_mesh(problem)
    assert mesh0.GetNE() > 10
    metric = cell_refinement_metric(problem, mesh0, n_samples_per_cell=15)
    assert float(metric.max()) > 0.05
    result = homogenize(problem)
    assert np.all(np.linalg.eigvalsh(result.effective_stiffness) > 0)


@pytest.mark.mfem
def test_amr_refines_interface_band():
    from b3_micromech.mesh.cartesian import build_cartesian_mesh

    problem = _problem(vf=0.5)
    mesh = build_cartesian_mesh(problem)
    n0 = mesh.GetNE()
    mesh, history = iteratively_refine_mfem(
        mesh,
        problem,
        threshold=0.05,
        max_iterations=2,
        dof_budget=100_000,
        n_samples_per_cell=16,
    )
    assert mesh.GetNE() > n0
    assert history[0]["n_flagged"] > 0
