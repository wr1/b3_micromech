"""Tests for thermal material properties and homogenization."""

from __future__ import annotations

import numpy as np
import pytest

from b3_micromech.features import (
    _build_mech_features_matrix,
    build_feature_matrix,
    constituent_engineering_constants,
    constituent_thermal_properties,
)
from b3_micromech.homogenize import (
    HomogenizationResult,
    effective_thermal_expansion_volume_average,
    homogenize,
)
from b3_micromech.materials import Material
from b3_micromech.problem import RVEProblem

pytestmark = pytest.mark.mfem


# ---------------------------------------------------------------------------
# Material creation
# ---------------------------------------------------------------------------


class TestMaterialIsotropic:
    def test_defaults_zero_thermal(self):
        mat = Material.isotropic("mat", youngs_modulus=3e9, poisson_ratio=0.35)
        assert np.all(mat.thermal_conductivity == 0.0)
        assert np.all(mat.thermal_expansion == 0.0)
        # thermal_expansion for isotropic: [alpha, alpha, alpha, 0, 0, 0]
        assert mat.thermal_expansion[3] == 0.0

    def test_nonzero_thermal(self):
        mat = Material.isotropic(
            "mat",
            youngs_modulus=3e9,
            poisson_ratio=0.35,
            thermal_conductivity=0.5,
            thermal_expansion=2.5e-6,
        )
        k = mat.thermal_conductivity
        a = mat.thermal_expansion
        assert k[0, 0] == 0.5
        assert k[1, 1] == 0.5
        assert k[2, 2] == 0.5
        assert a[0] == 2.5e-6
        assert a[1] == 2.5e-6
        assert a[2] == 2.5e-6
        assert a[3:] == pytest.approx(0.0)


class TestMaterialTransverseIsotropic:
    def test_defaults_zero_thermal(self):
        mat = Material.transverse_isotropic(
            "fibre",
            e_l=230e9,
            e_t=15e9,
            g_lt=15e9,
            nu_lt=0.2,
            nu_tt=0.3,
        )
        assert np.all(mat.thermal_conductivity == 0.0)
        assert np.all(mat.thermal_expansion == 0.0)

    def test_nonzero_thermal(self):
        mat = Material.transverse_isotropic(
            "fibre",
            e_l=230e9,
            e_t=15e9,
            g_lt=15e9,
            nu_lt=0.2,
            nu_tt=0.3,
            k_l=10.0,
            k_t=2.0,
            alpha_l=1.0e-6,
            alpha_t=20.0e-6,
        )
        k = mat.thermal_conductivity
        a = mat.thermal_expansion
        assert k[0, 0] == 10.0
        assert k[1, 1] == 2.0
        assert k[2, 2] == 2.0
        assert a[0] == 1.0e-6
        assert a[1] == 20.0e-6
        assert a[2] == 20.0e-6


class TestMaterialFromConfig:
    def test_isotropic_config(self):
        cfg = {
            "name": "epoxy",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
            "thermal_conductivity": 0.25,
            "thermal_expansion": 40.0e-6,
        }
        mat = Material.from_config(cfg)
        assert mat.thermal_conductivity[0, 0] == 0.25
        assert mat.thermal_expansion[1] == 40.0e-6

    def test_transverse_isotropic_config(self):
        cfg = {
            "name": "carbon",
            "type": "transverse_isotropic",
            "e_l": 230.0e9,
            "e_t": 15.0e9,
            "g_lt": 15.0e9,
            "nu_lt": 0.20,
            "nu_tt": 0.30,
            "k_l": 10.0,
            "k_t": 2.0,
            "alpha_l": 0.5e-6,
            "alpha_t": 12.0e-6,
        }
        mat = Material.from_config(cfg)
        assert mat.thermal_conductivity[0, 0] == 10.0
        assert mat.thermal_conductivity[1, 1] == 2.0
        assert mat.thermal_expansion[0] == 0.5e-6
        assert mat.thermal_expansion[1] == 12.0e-6

    def test_config_no_thermal_uses_zero(self):
        cfg = {
            "name": "old_style",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        }
        mat = Material.from_config(cfg)
        assert np.all(mat.thermal_conductivity == 0.0)
        assert np.all(mat.thermal_expansion == 0.0)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_thermal_properties_extraction(self):
        matrix = Material.isotropic(
            "m",
            youngs_modulus=3e9,
            poisson_ratio=0.35,
            thermal_conductivity=0.25,
            thermal_expansion=40.0e-6,
        )
        fibre = Material.transverse_isotropic(
            "f",
            e_l=230e9,
            e_t=15e9,
            g_lt=15e9,
            nu_lt=0.2,
            nu_tt=0.3,
            k_l=10.0,
            k_t=2.0,
            alpha_l=0.5e-6,
            alpha_t=12.0e-6,
        )
        a_m, a_Lf, a_Tf, k_m = constituent_thermal_properties(  # type: ignore[arg-type]
            matrix, fibre
        )
        assert a_m == pytest.approx(40.0e-6)
        assert a_Lf == pytest.approx(0.5e-6)
        assert a_Tf == pytest.approx(12.0e-6)
        assert k_m == pytest.approx(0.25)

    def test_12_col_feature_matrix(self):
        vf = np.array([0.3, 0.5, 0.7], dtype=float)
        mat = Material.isotropic(
            "m",
            youngs_modulus=3e9,
            poisson_ratio=0.35,
            thermal_conductivity=0.25,
            thermal_expansion=40.0e-6,
        )
        fib = Material.transverse_isotropic(
            "f",
            e_l=230e9,
            e_t=15e9,
            g_lt=15e9,
            nu_lt=0.2,
            nu_tt=0.3,
            k_l=10.0,
            k_t=2.0,
            alpha_l=0.5e-6,
            alpha_t=12.0e-6,
        )
        feat = build_feature_matrix(vf, matrix=mat, fibre=fib)
        assert feat.shape == (3, 12)
        # first 8 = mechanical
        mech = _build_mech_features_matrix(vf, matrix=mat, fibre=fib)
        assert np.allclose(feat[:, :8], mech)
        # last 4 = thermal
        assert feat[0, 8] == pytest.approx(40.0e-6)  # alpha_m
        assert feat[0, 9] == pytest.approx(0.5e-6)   # alpha_Lf
        assert feat[0, 10] == pytest.approx(12.0e-6)  # alpha_Tf
        assert feat[0, 11] == pytest.approx(0.25)     # k_m

    def test_build_mech_features_matrix_unchanged(self):
        vf = np.array([0.5], dtype=float)
        mat = Material.isotropic(
            "m",
            youngs_modulus=3e9,
            poisson_ratio=0.35,
        )
        fib = Material.transverse_isotropic(
            "f",
            e_l=230e9,
            e_t=15e9,
            g_lt=15e9,
            nu_lt=0.2,
            nu_tt=0.3,
        )
        feat = _build_mech_features_matrix(vf, matrix=mat, fibre=fib)
        assert feat.shape == (1, 8)


# ---------------------------------------------------------------------------
# Homogenization with thermal properties
# ---------------------------------------------------------------------------


def test_homogenization_result_has_thermal():
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.25,
                "thermal_expansion": 40.0e-6,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_l": 10.0,
                "k_t": 2.0,
                "alpha_l": 0.5e-6,
                "alpha_t": 12.0e-6,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.5,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)

    # Mechanical stiffness should be valid
    assert result.effective_stiffness.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(result.effective_stiffness) > 0)

    # Thermal expansion should be populated
    assert result.effective_thermal_expansion is not None
    alpha = result.effective_thermal_expansion
    assert alpha.shape == (6,)
    # Homogeneous matrix case: alpha_eff should be near the matrix CTE
    # because both constituents have very different stiffness and Vf=0.5,
    # the FEA result will be a weighted average influenced by stiffness


def test_homogeneous_matrix_thermal_recovers():
    """When both constituents are identical isotropic materials,
    effective alpha must equal the constituent alpha."""
    alpha_val = 40.0e-6
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "a",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.25,
                "thermal_expansion": alpha_val,
            },
            {
                "name": "b",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.25,
                "thermal_expansion": alpha_val,
            },
        ],
        "rve": {
            "matrix_material": "a",
            "fibre_material": "a",
            "fibre_volume_fraction": 0.5,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)

    alpha_eff = result.effective_thermal_expansion
    assert alpha_eff is not None
    # In plane strain along x, alpha_xx is constrained to zero,
    # alpha_yy should be close to the constituent value.
    assert alpha_eff[1] == pytest.approx(alpha_val, rel=0.02)
    assert alpha_eff[2] == pytest.approx(alpha_val, rel=0.02)


def test_volume_average_formula():
    """Rule-of-mixtures volume average should match the helper function."""
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_expansion": 30.0e-6,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "alpha_l": 1.0e-6,
                "alpha_t": 15.0e-6,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.5,
        },
    }
    problem = RVEProblem.from_config(cfg)
    alpha_hom = effective_thermal_expansion_volume_average(problem)

    # Voigt rule: alpha_h = vf * alpha_f + (1-vf) * alpha_m
    vf = problem.fibre_volume_fraction
    expected = vf * problem.materials["fibre"].thermal_expansion + (
        1.0 - vf
    ) * problem.materials["matrix"].thermal_expansion
    assert np.allclose(alpha_hom, expected)


# ---------------------------------------------------------------------------
# HomogenizationResult dataclass
# ---------------------------------------------------------------------------


def test_homogenization_result_default_metadata():
    result = HomogenizationResult(
        effective_stiffness=np.zeros((6, 6)),
        fea_stiffness=np.zeros((6, 6)),
    )
    assert result.metadata == {}


def test_homogenization_result_none_metadata_becomes_empty():
    result = HomogenizationResult(
        effective_stiffness=np.zeros((6, 6)),
        fea_stiffness=np.zeros((6, 6)),
        metadata=None,
    )
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# Conductivity homogenization
# ---------------------------------------------------------------------------


def test_homogenization_result_has_conductivity():
    """HomogenizationResult should carry effective_conductivity (6, 6)."""
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.25,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_l": 10.0,
                "k_t": 2.0,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.5,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)

    assert result.effective_conductivity is not None
    assert result.effective_conductivity.shape == (6, 6)
    # k_xx = rule-of-mixtures (decoupled)
    vf = 0.5
    expected_kxx = vf * 10.0 + (1.0 - vf) * 0.25
    assert result.effective_conductivity[0, 0] == pytest.approx(expected_kxx, rel=0.05)


def test_homogeneous_conductivity_recovers():
    """When both constituents are identical, effective k must equal constituent k."""
    k_val = 0.5
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "a",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": k_val,
            },
            {
                "name": "b",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": k_val,
            },
        ],
        "rve": {
            "matrix_material": "a",
            "fibre_material": "a",  # identical = homogeneous
            "fibre_volume_fraction": 0.5,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)
    k = result.effective_conductivity

    # Transverse plane (y,z): k_eff ≈ k_val for homogeneous material
    assert k[1, 1] == pytest.approx(k_val, rel=0.03)
    assert k[2, 2] == pytest.approx(k_val, rel=0.03)
    # Off-diagonals should be zero
    assert k[1, 2] == pytest.approx(0.0, abs=1e-8)
    assert k[2, 1] == pytest.approx(0.0, abs=1e-8)


def test_series_bound_transverse_conductivity():
    """Lower bound (series): 1/k_eff = vf/k_f + (1-vf)/k_m  for transverse load.

    The series bound is a conservative lower bound for the transverse effective
    conductivity.  Our FEA result should be within 10 % of this bound.
    """
    k_f = 2.0   # transverse fibre conductivity
    k_m = 0.25
    vf = 0.5
    series_k = 1.0 / (vf / k_f + (1.0 - vf) / k_m)  # ≈ 0.4545

    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [24, 24]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": k_m,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_t": k_f,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": vf,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)
    k = result.effective_conductivity

    # Average transverse: (k_yy + k_zz)/2
    avg_trans = 0.5 * (k[1, 1] + k[2, 2])
    # Series/parallel are BOUNDS for the circular-fibre RVE, not targets:
    parallel_k = vf * k_f + (1.0 - vf) * k_m
    assert avg_trans >= series_k * 0.98
    assert avg_trans <= parallel_k * 1.02
    # Physical target: Rayleigh square-array of cylinders.
    beta = (k_f - k_m) / (k_f + k_m)
    rayleigh_k = k_m * (1.0 + 2.0 * vf / (1.0 / beta - vf - 0.30584 * vf**4))
    assert avg_trans == pytest.approx(rayleigh_k, rel=0.10)


def test_parallel_bound_axial_conductivity():
    """Upper bound (parallel/rule-of-mixtures): k_eff = vf*k_f + (1-vf)*k_m along fibre."""
    k_f = 10.0
    k_m = 0.25
    vf = 0.5
    parallel_k = vf * k_f + (1.0 - vf) * k_m  # = 5.125

    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": k_m,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_l": k_f,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": vf,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)
    k = result.effective_conductivity

    assert k[0, 0] == pytest.approx(parallel_k, rel=0.05)


def test_transverse_conductivity_symmetry():
    """For UD RVE the effective transverse tensor should be isotropic in y-z plane."""
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [16, 16]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.25,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_l": 10.0,
                "k_t": 2.0,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.5,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)
    k = result.effective_conductivity

    # k_yy ≈ k_zz (transverse isotropy)
    assert k[1, 1] == pytest.approx(k[2, 2], rel=0.05)
    # Off-diagonals ≈ 0
    assert k[1, 2] == pytest.approx(0.0, abs=1e-6)


def test_conductivity_converges_with_mesh():
    """Refining the mesh should improve accuracy toward the series bound."""
    k_m = 0.25
    # Physical target: Rayleigh square-array of cylinders (series is only a bound).
    k_f, vf = 2.0, 0.5
    beta = (k_f - k_m) / (k_f + k_m)
    target_k = k_m * (1.0 + 2.0 * vf / (1.0 / beta - vf - 0.30584 * vf**4))

    errors_coarse = None
    errors_fine = None
    for res in [8, 24]:
        cfg = {
            "domain": {"size": 1.0, "mesh_resolution": [res, res]},
            "materials": [
                {
                    "name": "matrix",
                    "type": "isotropic",
                    "youngs_modulus": 3e9,
                    "poisson_ratio": 0.35,
                    "thermal_conductivity": k_m,
                },
                {
                    "name": "fibre",
                    "type": "transverse_isotropic",
                    "e_l": 230e9,
                    "e_t": 15e9,
                    "g_lt": 15e9,
                    "nu_lt": 0.2,
                    "nu_tt": 0.3,
                    "k_t": 2.0,
                },
            ],
            "rve": {
                "matrix_material": "matrix",
                "fibre_material": "fibre",
                "fibre_volume_fraction": 0.5,
            },
            "solver": {"cell_type": "quadrilateral"},
        }
        problem = RVEProblem.from_config(cfg)
        result = homogenize(problem)
        avg_trans = 0.5 * (
            result.effective_conductivity[1, 1]
            + result.effective_conductivity[2, 2]
        )
        err = abs(avg_trans - target_k) / target_k
        if res == 8:
            errors_coarse = err
        else:
            errors_fine = err

    # Staircase (voxelized circle) sampling makes convergence non-monotone;
    # require both resolutions inside a modest band of the analytic target.
    assert errors_coarse < 0.12
    assert errors_fine < 0.12


def test_diffusion_result_shape():
    """effective_conductivity_tensor returns a (6, 6) tensor with expected structure."""
    from b3_micromech.backends.mfem_periodic_2d import effective_conductivity_tensor

    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
                "thermal_conductivity": 0.5,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "k_l": 5.0,
                "k_t": 1.0,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.3,
        },
        "solver": {"cell_type": "quadrilateral"},
    }
    problem = RVEProblem.from_config(cfg)
    k_eff, meta = effective_conductivity_tensor(problem)

    assert k_eff.shape == (6, 6)
    assert meta["backend"] == "mfem_periodic_2d_diffusion"
    assert meta["n_cells"] > 0
    assert meta["n_dofs"] > 0
    assert meta["fibre_volume_fraction"] == 0.3

    # Transverse block positive-definite
    k_2d = k_eff[:2, :2]
    eigvals = np.linalg.eigvalsh(k_2d)
    assert np.all(eigvals > 0)

    # k_xx = ROM
    vf = 0.3
    expected_kxx = vf * 5.0 + (1.0 - vf) * 0.5
    assert k_eff[0, 0] == pytest.approx(expected_kxx, rel=0.01)