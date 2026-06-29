"""Homogenize a transverse UD RVE to a (6, 6) stiffness tensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_micromech.backends.mfem_periodic_2d import solve_periodic_plane_strain
from b3_micromech.features import build_feature_matrix
from b3_micromech.problem import RVEProblem


@dataclass(frozen=True)
class HomogenizationResult:
    effective_stiffness: NDArray[np.float64]
    fea_stiffness: NDArray[np.float64]
    metadata: dict


def homogenize(problem: RVEProblem) -> HomogenizationResult:
    """Periodic plane-strain homogenization; returns symmetrised ``(6, 6)`` stiffness."""
    C_fea, meta = solve_periodic_plane_strain(problem)
    return HomogenizationResult(
        effective_stiffness=C_fea,
        fea_stiffness=C_fea,
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
