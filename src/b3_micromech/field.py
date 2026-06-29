"""Implicit transverse fibre field — pointwise material lookup."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_micromech.geometry import classify_points

# Material IDs returned by sample_arrays (matrix = 1, fibre = 2).
MATRIX_ID = 1
FIBRE_ID = 2


def sample_material_ids(
    problem,
    points_yz: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Return per-point material IDs for the centred circular fibre RVE."""
    pts = np.asarray(points_yz, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 2:
        raise ValueError(f"points_yz must have shape (N, 2), got {pts.shape}")
    is_fibre = classify_points(
        pts, centre=problem.centre_yz, radius=problem.fibre_radius
    )
    ids = np.full(pts.shape[0], MATRIX_ID, dtype=np.intp)
    ids[is_fibre] = FIBRE_ID
    return ids
