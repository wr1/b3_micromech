"""Transverse RVE geometry: square or hexagon domain with centred circular fibre."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DomainShape = Literal["square", "hexagon"]

_SQRT3 = math.sqrt(3.0)


def domain_area(*, shape: DomainShape, size: float) -> float:
    """Domain area in the transverse plane (y–z)."""
    if size <= 0:
        raise ValueError("size must be positive")
    if shape == "square":
        return size * size
    if shape == "hexagon":
        return 1.5 * _SQRT3 * size * size
    raise ValueError(f"unknown domain shape {shape!r}")


def fibre_volume_fraction_from_radius(
    *,
    shape: DomainShape = "square",
    domain_size: float,
    radius: float,
) -> float:
    if domain_size <= 0 or radius < 0:
        raise ValueError("domain_size must be positive and radius non-negative")
    vf = math.pi * radius * radius / domain_area(shape=shape, size=domain_size)
    if vf > 1.0 + 1e-12:
        raise ValueError(f"fibre volume fraction {vf:.4f} exceeds unity")
    return float(vf)


def max_fibre_radius(
    *,
    shape: DomainShape,
    domain_size: float,
    standoff: float = 0.01,
) -> float:
    """Largest centred fibre radius before the disc touches the domain boundary."""
    if not 0.0 <= standoff < 1.0:
        raise ValueError("standoff must be in [0, 1)")
    if shape == "square":
        return (1.0 - standoff) * 0.5 * domain_size
    if shape == "hexagon":
        inradius = domain_size * _SQRT3 / 2.0
        return (1.0 - standoff) * inradius
    raise ValueError(f"unknown domain shape {shape!r}")


def max_fibre_volume_fraction(
    *,
    shape: DomainShape,
    domain_size: float,
    standoff: float = 0.01,
) -> float:
    """Upper Vf for a centred circular fibre (hex flat-edge limit ≈ 0.89 at 1% standoff)."""
    return fibre_volume_fraction_from_radius(
        shape=shape,
        domain_size=domain_size,
        radius=max_fibre_radius(
            shape=shape, domain_size=domain_size, standoff=standoff
        ),
    )


def packing_limit_volume_fraction(*, shape: DomainShape) -> float:
    """Theoretical Vf at zero standoff (disc tangent to boundary)."""
    return max_fibre_volume_fraction(shape=shape, domain_size=1.0, standoff=0.0)


# Mid-range Vf samples plus a dense cluster at the hex flat-edge (wide-side) limit.
HEX_VF_MID_RANGE: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.68)
HEX_VF_MID_RANGE_COMPACT: tuple[float, ...] = (0.35, 0.50, 0.60, 0.68)
HexVfSweepPreset = Literal["full", "compact", "high"]


def hex_vf_packing_cluster_values(
    *,
    shape: DomainShape = "hexagon",
    domain_size: float = 1.0,
    standoff: float = 0.01,
    dense_start: float = 0.74,
    n_dense: int = 8,
) -> tuple[float, ...]:
    """Uniformly spaced Vf samples from ``dense_start`` up to the standoff-limited maximum."""
    if not 0.0 < dense_start < 1.0:
        raise ValueError("dense_start must lie in (0, 1)")
    if n_dense < 2:
        raise ValueError("n_dense must be at least 2")
    vf_max = max_fibre_volume_fraction(
        shape=shape, domain_size=domain_size, standoff=standoff
    )
    vf_end = vf_max * 0.999
    if dense_start >= vf_end:
        raise ValueError(
            f"dense_start {dense_start} must be below standoff-limited Vf {vf_end}"
        )
    cluster = np.linspace(dense_start, vf_end, n_dense, dtype=float)
    return tuple(float(v) for v in cluster)


# Default high-Vf cluster for hex surrogates (ends just inside the 1% standoff limit).
HEX_VF_PACKING_CLUSTER: tuple[float, ...] = hex_vf_packing_cluster_values()


def hex_vf_sweep_values(
    *,
    preset: HexVfSweepPreset = "full",
    vf_min: float = 0.20,
    mid_values: tuple[float, ...] | None = None,
    packing_cluster: tuple[float, ...] | None = None,
    standoff: float = 0.01,
    dense_start: float = 0.74,
    n_dense: int = 8,
) -> list[float]:
    """Vf grid for hex surrogate sweeps.

    Presets:

    - ``full``: anchor at ``vf_min``, mid-range coverage, dense packing-limit cluster.
    - ``compact``: same cluster with fewer mid-range samples (3-D response sweeps).
    - ``high``: packing-limit cluster only (refinement near the flat-edge limit).
    """
    cluster = packing_cluster or hex_vf_packing_cluster_values(
        standoff=standoff,
        dense_start=dense_start,
        n_dense=n_dense,
    )
    if cluster[-1] > 1.0:
        raise ValueError("vf sweep values must lie in (0, 1]")
    if preset == "high":
        return list(cluster)
    if mid_values is None:
        mid_values = (
            HEX_VF_MID_RANGE_COMPACT if preset == "compact" else HEX_VF_MID_RANGE
        )
    if vf_min <= 0.0:
        raise ValueError("vf_min must be positive")
    values = sorted(
        {float(vf_min), *(float(v) for v in mid_values), *(float(v) for v in cluster)}
    )
    return values


def radius_from_fibre_volume_fraction(
    *,
    shape: DomainShape = "square",
    domain_size: float,
    vf: float,
) -> float:
    if not 0.0 <= vf <= 1.0:
        raise ValueError("vf must be in [0, 1]")
    return float(math.sqrt(vf * domain_area(shape=shape, size=domain_size) / math.pi))


def fibre_centre(shape: DomainShape, domain_size: float) -> NDArray[np.float64]:
    if shape == "square":
        half = 0.5 * domain_size
        return np.array([half, half], dtype=float)
    if shape == "hexagon":
        return hexagon_centre(domain_size)
    raise ValueError(f"unknown domain shape {shape!r}")


def hexagon_corners(circumradius: float) -> NDArray[np.float64]:
    """Flat-top regular hexagon vertices (6, 2), CCW, shifted so min coordinates are 0."""
    angles = np.pi / 6.0 + np.arange(6, dtype=float) * (np.pi / 3.0)
    pts = circumradius * np.column_stack([np.cos(angles), np.sin(angles)])
    pts -= pts.min(axis=0)
    return pts.astype(float)


def hexagon_centre(circumradius: float) -> NDArray[np.float64]:
    return hexagon_corners(circumradius).mean(axis=0)


def hexagon_bounding_box(circumradius: float) -> tuple[float, float]:
    corners = hexagon_corners(circumradius)
    return float(corners[:, 0].max()), float(corners[:, 1].max())


def hexagon_side_length(circumradius: float) -> float:
    """Side length of a regular hexagon (equals circumradius)."""
    return float(circumradius)


def point_in_hexagon(
    points_yz: NDArray[np.float64],
    *,
    circumradius: float,
    tol: float = 1e-10,
) -> NDArray[np.bool_]:
    """True for points inside the flat-top hexagon (inclusive boundary)."""
    pts = np.asarray(points_yz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_yz must have shape (N, 2), got {pts.shape}")
    corners = hexagon_corners(circumradius)
    inside = np.ones(pts.shape[0], dtype=bool)
    for i in range(6):
        v0 = corners[i]
        v1 = corners[(i + 1) % 6]
        cross = (v1[0] - v0[0]) * (pts[:, 1] - v0[1]) - (v1[1] - v0[1]) * (
            pts[:, 0] - v0[0]
        )
        inside &= cross >= -tol
    return inside


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
