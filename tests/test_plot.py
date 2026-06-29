import numpy as np
import pytest

from b3_micromech.result import LOADCASE_LABELS
from b3_micromech.tensors import macro_displacement_at_yz, von_mises_voigt


def test_loadcase_label_count():
    assert len(LOADCASE_LABELS) == 6


def test_macro_displacement_affine():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Eyy = np.zeros(6)
    Eyy[1] = 1.0
    uyy = macro_displacement_at_yz(Eyy, pts)
    np.testing.assert_allclose(uyy[1], [0, 1, 0])
    Ezz = np.zeros(6)
    Ezz[2] = 1.0
    uzz = macro_displacement_at_yz(Ezz, pts)
    np.testing.assert_allclose(uzz[2], [0, 0, 1])


def test_von_mises_uniaxial():
    s = np.array([1e9, 0, 0, 0, 0, 0])
    vm = von_mises_voigt(s)
    assert vm == pytest.approx(1e9)


@pytest.mark.mfem
def test_render_all_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from b3_micromech.plot import render_all_figures
    from b3_micromech.postprocess import solve_all_loadcases
    from b3_micromech.problem import RVEProblem

    problem = RVEProblem.from_config(
        {
            "domain": {"size": 1.0, "mesh_resolution": [12, 12]},
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
        }
    )
    paths = render_all_figures(solve_all_loadcases(problem), tmp_path)
    assert len(paths) == 5
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 1000


@pytest.mark.mfem
def test_render_all_figures_includes_amr_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from b3_micromech.plot import render_all_figures
    from b3_micromech.postprocess import solve_all_loadcases
    from b3_micromech.problem import RVEProblem

    problem = RVEProblem.from_config(
        {
            "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
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
                "cell_type": "quadrilateral",
                "amr": {
                    "enabled": True,
                    "max_iterations": 1,
                    "threshold": 0.05,
                    "n_samples_per_cell": 16,
                },
            },
        }
    )
    paths = render_all_figures(solve_all_loadcases(problem), tmp_path)
    assert len(paths) == 6
    assert "amr_refinement" in paths
    assert paths["amr_refinement"].exists()
    assert paths["amr_refinement"].stat().st_size > 1000
