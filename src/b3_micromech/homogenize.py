"""Homogenize a transverse UD RVE to a (6, 6) stiffness tensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_micromech.backends.mfem_periodic_2d import solve_periodic_plane_strain
from b3_micromech.problem import RVEProblem
from b3_micromech.reference import _engineering_constants_isotropic
from b3_micromech.tensors import engineering_constants_transverse_iso


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
    em, num = _engineering_constants_isotropic(matrix.stiffness)
    fc = engineering_constants_transverse_iso(fibre.stiffness)
    return np.array(
        [
            problem.fibre_volume_fraction,
            em,
            num,
            fc["e_l"],
            fc["e_t"],
            fc["g_lt"],
            fc["nu_lt"],
            fc["g_tt"],
        ],
        dtype=float,
    )