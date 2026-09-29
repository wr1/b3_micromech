"""Constituent contract: features, g_tt, and named sweep materials."""

from __future__ import annotations

import numpy as np
import pytest

from b3_micromech.contract import (
    FEATURE_NAMES,
    Constituents,
    ContractError,
)
from b3_micromech.features import build_feature_matrix
from b3_micromech.materials import Material
from b3_micromech.problem import RVEProblem
from b3_micromech.sweep import problem_from_sweep_point
from b3_micromech.tensors import (
    engineering_constants_transverse_iso,
    isotropic_stiffness,
    orthotropic_stiffness,
    transverse_isotropic_stiffness,
)


class _Mat:
    def __init__(self, name: str, stiffness: np.ndarray) -> None:
        self.name = name
        self.stiffness = stiffness


def _template(**rve_overrides):
    cfg = {
        "domain": {"shape": "square", "size": 1.0, "mesh_resolution": [4, 4]},
        "materials": [
            {
                "name": "epoxy",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "carbon",
                "type": "transverse_isotropic",
                "e_l": 230.0e9,
                "e_t": 15.0e9,
                "g_lt": 15.0e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
            },
        ],
        "rve": {
            "matrix_material": "epoxy",
            "fibre_material": "carbon",
            "fibre_volume_fraction": 0.5,
        },
        "sweep": {"mesh": {"resolution": [6, 6], "cell_type": "triangle"}},
    }
    cfg["rve"].update(rve_overrides)
    return cfg


def test_g_ttf_and_named_materials_drive_the_fibre():
    problem = problem_from_sweep_point(_template(), {"vf": 0.4, "G_TTf": 6.0e9})
    assert problem.matrix_material == "epoxy"
    assert problem.fibre_material == "carbon"
    assert problem.mesh_resolution == (6, 6)
    assert problem.cell_type == "triangle"
    assert problem.fibre_volume_fraction == pytest.approx(0.4)
    g_tt = engineering_constants_transverse_iso(problem.materials["carbon"].stiffness)[
        "g_tt"
    ]
    assert g_tt == pytest.approx(6.0e9)


def test_nu_ttf_sweep_key_derives_g_tt():
    problem = problem_from_sweep_point(_template(), {"nu_TTf": 0.25})
    nu_tt = engineering_constants_transverse_iso(problem.materials["carbon"].stiffness)[
        "nu_tt"
    ]
    assert nu_tt == pytest.approx(0.25)


def test_point_cannot_carry_both_nu_ttf_and_g_ttf():
    with pytest.raises(ContractError, match="both"):
        problem_from_sweep_point(_template(), {"nu_TTf": 0.25, "G_TTf": 6.0e9})


def test_orthotropic_matrix_and_y_axis_fibre_raise():
    fibre = _Mat(
        "f",
        transverse_isotropic_stiffness(
            e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.2, nu_tt=0.3
        ),
    )
    matrix = _Mat(
        "m",
        orthotropic_stiffness(
            e1=3e9,
            e2=4e9,
            e3=5e9,
            nu12=0.3,
            nu13=0.3,
            nu23=0.3,
            g12=1.2e9,
            g13=1.2e9,
            g23=1.5e9,
        ),
    )
    with pytest.raises(ContractError, match="isotropic"):
        Constituents.from_materials(matrix, fibre)

    perm = [1, 0, 2, 4, 3, 5]
    fibre_y = _Mat("f", fibre.stiffness[np.ix_(perm, perm)])
    iso = _Mat("m", isotropic_stiffness(3e9, 0.35))
    with pytest.raises(ContractError, match="transverse-isotropic"):
        Constituents.from_materials(iso, fibre_y)


def test_material_g_tt_derives_nu_tt_and_disagreement_raises():
    agreed = Material.from_config(
        {
            "name": "f",
            "type": "transverse_isotropic",
            "e_l": 230e9,
            "e_t": 15e9,
            "g_lt": 15e9,
            "nu_lt": 0.2,
            "g_tt": 15e9 / (2.0 * 1.3),
        }
    )
    nu = engineering_constants_transverse_iso(agreed.stiffness)["nu_tt"]
    assert nu == pytest.approx(0.3)

    with pytest.raises(ContractError, match="disagree"):
        Material.from_config(
            {
                "name": "f",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
                "g_tt": 5e9,
            }
        )


@pytest.mark.filterwarnings("always::DeprecationWarning:b3_micromech")
def test_partial_thermal_feature_matrix_raises():
    with (
        pytest.warns(DeprecationWarning, match="build_feature_matrix"),
        pytest.raises(ContractError, match="thermal"),
    ):
        build_feature_matrix(
            np.array([0.5]),
            E_m=3e9,
            nu_m=0.35,
            E_Lf=230e9,
            E_Tf=15e9,
            G_LTf=15e9,
            nu_LTf=0.2,
            G_TTf=6e9,
            alpha_m=1e-5,
        )


def test_round_trip_config_uses_named_materials():
    problem = RVEProblem.from_config(_template())
    again = Constituents.from_materials(
        problem.materials["epoxy"], problem.materials["carbon"]
    )
    assert again.feature_matrix([0.5]).shape == (1, len(FEATURE_NAMES))
