"""Transverse RVE geometry: square domain with centred circular fibre."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def fibre_volume_fraction_from_radius(*, domain_size: float, radius: float) -> float:
    if domain_size <= 0 or radius < 0:
        raise ValueError("domain_size must be positive and radius non-negative")
    vf = math.pi * radius * radius / (domain_size * domain_size)
    if vf > 1.0 + 1e-12:
        raise ValueError(f"fibre volume fraction {vf:.4f} exceeds unity")
    return float(vf)


def radius_from_fibre_volume_fraction(*, domain_size: float, vf: float) -> float:
    if not 0.0 <= vf <= 1.0:
        raise ValueError("vf must be in [0, 1]")
    return float(math.sqrt(vf * domain_size * domain_size / math.pi))


def fibre_centre(domain_size: float) -> NDArray[np.float64]:
    half = 0.5 * domain_size
    return np.array([half, half], dtype=float)


def classify_points(
    points_yz: NDArray[np.float64],
    *,
    centre: NDArray[np.float64],
    radius: float,
) -> NDArray[np.bool_]:
    pts = np.asarray(points_yz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_yz must have shape (N, 2), got {pts.shape}")
    dist2 = np.sum((pts - centre[None, :]) ** 2, axis=1)
    return dist2 <= radius * radius + 1e-14