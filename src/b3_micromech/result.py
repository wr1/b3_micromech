"""Result types for homogenization loadcases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LOADCASE_LABELS: tuple[tuple[str, str], ...] = (
    ("ε₁₁ = 1", "axial_fibre"),
    ("ε₂₂ = 1", "transverse_y"),
    ("ε₃₃ = 1", "transverse_z"),
    ("γ₂₃ = 1", "shear_yz"),
    ("γ₁₃ = 1", "shear_xz"),
    ("γ₁₂ = 1", "shear_xy"),
)


@dataclass(frozen=True)
class LoadcaseResult:
    macro_strain: NDArray[np.float64]
    macro_stress: NDArray[np.float64]
    u_at_vertices: NDArray[np.float64]
    u_tilde_at_vertices: NDArray[np.float64]
    eps_per_gp: NDArray[np.float64]
    sigma_per_gp: NDArray[np.float64]
    von_mises_per_elem: NDArray[np.float64]
