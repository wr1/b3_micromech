import numpy as np
import pytest

from b3_micromech.homogenize import homogenize
from b3_micromech.problem import RVEProblem
from b3_micromech.reference import mori_tanaka_cylinder
from b3_micromech.tensors import engineering_constants_transverse_iso

pytestmark = pytest.mark.mfem

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"


def test_homogeneous_matrix_recovers_stiffness():
    cfg = {
        "domain": {"size": 1.0, "mesh_resolution": [12, 12]},
        "materials": [
            {"name": "matrix", "type": "isotropic", "youngs_modulus": 3e9, "poisson_ratio": 0.35},
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
            "fibre_material": "matrix",
            "fibre_volume_fraction": 0.0,
        },
    }
    problem = RVEProblem.from_config(cfg)
    result = homogenize(problem)
    C = result.effective_stiffness
    assert np.allclose(C, problem.materials["matrix"].stiffness, rtol=0.05, atol=1e8)


def test_ud_rve_agrees_with_mori_tanaka_transverse():
    problem = RVEProblem.from_yaml(EXAMPLES / "ud_transverse.yaml")
    problem = RVEProblem.from_config(
        {
            "domain": {"size": 1.0, "mesh_resolution": [48, 48]},
            "materials": [
                {"name": "matrix", "type": "isotropic", "youngs_modulus": 3e9, "poisson_ratio": 0.35},
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
            "solver": {"cell_type": "quadrilateral"},
        }
    )
    result = homogenize(problem)
    matrix = problem.materials["matrix"]
    fibre = problem.materials["fibre"]
    Cmt = mori_tanaka_cylinder(
        matrix=matrix, fibre=fibre, fibre_volume_fraction=0.5
    )
    ec_fea = engineering_constants_transverse_iso(result.effective_stiffness)
    ec_mt = engineering_constants_transverse_iso(Cmt)
    assert abs(ec_fea["e_t"] - ec_mt["e_t"]) / ec_mt["e_t"] < 0.07
    assert ec_fea["e_l"] > 100e9
    assert np.all(np.linalg.eigvalsh(result.effective_stiffness) > 0)