"""Homogenize a transverse UD RVE to a (6, 6) stiffness tensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.backends.mfem_periodic_2d import (
    solve_periodic_plane_strain,
    solve_thermal_loadcase,
    effective_conductivity_tensor,
)
from b3_micromech.features import build_feature_matrix
from b3_micromech.problem import RVEProblem


def effective_thermal_expansion_volume_average(
    problem: RVEProblem,
) -> NDArray[np.float64]:
    """Rule-of-mixtures effective CTE from constituent volume fractions.

    Returns the ``alpha_yy`` (index 1) component of the effective
    thermal-expansion vector ``[alpha_xx, alpha_yy, alpha_zz, 0, 0, 0]``
    as a simple volume-weighted average.  This is the Voigt upper bound
    and is correct for the longitudinal direction but a rough estimate
    for the transverse direction.
    """
    vf = problem.fibre_volume_fraction
    vm = 1.0 - vf
    alpha_fibre = problem.materials[problem.fibre_material].thermal_expansion
    alpha_matrix = problem.materials[problem.matrix_material].thermal_expansion
    alpha_hom = vf * alpha_fibre + vm * alpha_matrix
    return alpha_hom


@dataclass(frozen=True)
class HomogenizationResult:
    effective_stiffness: NDArray[np.float64]
    fea_stiffness: NDArray[np.float64]
    effective_thermal_expansion: NDArray[np.float64] | None = None
    effective_conductivity: NDArray[np.float64] | None = None
    metadata: dict = field(default_factory=dict)


def homogenize(problem: RVEProblem) -> HomogenizationResult:
    """Periodic plane-strain homogenization; returns symmetrised ``(6, 6)`` stiffness, CTE, and k."""
    C_fea, meta = solve_periodic_plane_strain(problem)
    alpha_eff, thermal_meta = solve_thermal_loadcase(problem)
    meta.update(thermal_meta)

    k_eff, k_meta = effective_conductivity_tensor(problem)
    meta.update(k_meta)

    return HomogenizationResult(
        effective_stiffness=C_fea,
        fea_stiffness=C_fea,
        effective_thermal_expansion=alpha_eff,
        effective_conductivity=k_eff,
        metadata=meta,
    )


def surrogate_features(problem: RVEProblem) -> NDArray[np.float64]:
    """Feature vector aligned with ``b3_tex.micromodels.SurrogateModel``."""
    matrix = problem.materials[problem.matrix_material]
    fibre = problem.materials[problem.fibre_material]
    return build_feature_matrix(
        np.array([problem.fibre_volume_fraction], dtype=float),
        matrix=matrix,
        fibre=fibre,
    )[0]
