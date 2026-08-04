"""Structured 3-axis response grids and surface plots for stiffness surrogates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from typing import Any, Protocol

from b3_micromech.export import SURROGATE_FEATURE_NAMES
from b3_micromech.surrogate import engineering_constants_batch


class _PredictsStiffness(Protocol):
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


SURROGATE_FEATURE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(SURROGATE_FEATURE_NAMES)
}

DEFAULT_THREE_AXIS_RESPONSE: tuple[str, str, str] = ("vf", "E_m", "E_Lf")


def midpoint_feature_vector(bounds: NDArray[np.float64]) -> NDArray[np.float64]:
    """Feature vector at the hypercube centre, shape ``(8,)``."""
    b = np.asarray(bounds, dtype=float)
    if b.shape != (8, 2):
        raise ValueError(f"bounds must have shape (8, 2), got {b.shape}")
    return 0.5 * (b[:, 0] + b[:, 1])


def build_structured_response_grid(
    bounds: NDArray[np.float64],
    varying_axes: tuple[str, ...],
    grid_sizes: tuple[int, ...],
    *,
    fixed_axis_values: dict[str, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Enumerate a structured grid over three (or fewer) varying surrogate axes.

    Non-varying features are set from ``fixed_axis_values`` or the bound midpoints.
    Returns ``(features, axis_coordinates)`` where ``features`` has shape
    ``(prod(grid_sizes), 8)``.
    """
    if len(varying_axes) != len(grid_sizes):
        raise ValueError("varying_axes and grid_sizes must have the same length")
    if len(varying_axes) == 0:
        raise ValueError("at least one varying axis is required")
    if len(varying_axes) > 3:
        raise ValueError("structured response grids support at most three varying axes")

    b = np.asarray(bounds, dtype=float)
    base = midpoint_feature_vector(b)
    if fixed_axis_values:
        for name, value in fixed_axis_values.items():
            base[SURROGATE_FEATURE_INDEX[name]] = float(value)

    axis_mesh: list[NDArray[np.float64]] = []
    axis_coords: dict[str, NDArray[np.float64]] = {}
    for name, n_pts in zip(varying_axes, grid_sizes, strict=True):
        idx = SURROGATE_FEATURE_INDEX[name]
        coords = np.linspace(b[idx, 0], b[idx, 1], int(n_pts))
        axis_coords[name] = coords
        axis_mesh.append(coords)

    mesh = np.meshgrid(*axis_mesh, indexing="ij")
    features = np.tile(base, (int(np.prod(grid_sizes)), 1))
    for name, grid in zip(varying_axes, mesh, strict=True):
        idx = SURROGATE_FEATURE_INDEX[name]
        features[:, idx] = grid.ravel()
    return features, axis_coords


def predict_structured_response_grid(
    model: _PredictsStiffness | Any,
    bounds: NDArray[np.float64],
    varying_axes: tuple[str, ...],
    grid_sizes: tuple[int, ...],
    *,
    fixed_axis_values: dict[str, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """Predict stiffness on a structured response grid."""
    features, axis_coords = build_structured_response_grid(
        bounds,
        varying_axes,
        grid_sizes,
        fixed_axis_values=fixed_axis_values,
    )
    stiffness = model.predict(features)
    return features, axis_coords, stiffness


def render_three_axis_response_surfaces(
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    varying_axes: tuple[str, str, str],
    plot_dir: Path,
    *,
    grid_sizes: tuple[int, int, int],
) -> dict[str, Path]:
    """Write 3-D scatter surfaces and a median-$E_m$ slice for $E_1$ and $E_2$."""
    import matplotlib.pyplot as plt

    if len(varying_axes) != 3:
        raise ValueError(
            "render_three_axis_response_surfaces expects exactly three varying axes"
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ec = engineering_constants_batch(stiffness)
    n0, n1, n2 = grid_sizes
    expected = n0 * n1 * n2
    if features.shape[0] != expected:
        raise ValueError(f"expected {expected} grid points, got {features.shape[0]}")

    idx = [SURROGATE_FEATURE_INDEX[name] for name in varying_axes]
    x0 = features[:, idx[0]].reshape(grid_sizes)
    x1 = features[:, idx[1]].reshape(grid_sizes)
    x2 = features[:, idx[2]].reshape(grid_sizes)

    axis_labels = {
        "vf": "fibre volume fraction",
        "E_m": r"$E_m$ [GPa]",
        "E_Lf": r"$E_{Lf}$ [GPa]",
        "E_Tf": r"$E_{Tf}$ [GPa]",
    }
    axis_scale = {"E_m": 1e-9, "E_Lf": 1e-9, "E_Tf": 1e-9, "vf": 1.0}

    def _scaled(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values * axis_scale.get(name, 1.0)

    for response_key, response_label, filename in (
        ("e_l", r"$E_1$ [GPa]", "response_surface_e1_3d.png"),
        ("e_t", r"$E_2$ [GPa]", "response_surface_e2_3d.png"),
    ):
        response = (ec[response_key] / 1e9).reshape(grid_sizes)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            _scaled(varying_axes[0], x0.ravel()),
            _scaled(varying_axes[1], x1.ravel()),
            _scaled(varying_axes[2], x2.ravel()),
            c=response.ravel(),
            cmap="viridis",
            s=18,
            alpha=0.85,
        )
        ax.set_xlabel(axis_labels.get(varying_axes[0], varying_axes[0]))
        ax.set_ylabel(axis_labels.get(varying_axes[1], varying_axes[1]))
        ax.set_zlabel(axis_labels.get(varying_axes[2], varying_axes[2]))
        ax.set_title(f"Surrogate {response_label} over ({', '.join(varying_axes)})")
        fig.colorbar(sc, ax=ax, shrink=0.65, label=response_label)
        fig.tight_layout()
        path_key = f"response_{response_key}_3d"
        paths[path_key] = plot_dir / filename
        fig.savefig(paths[path_key], dpi=150)
        plt.close(fig)

    em_idx = SURROGATE_FEATURE_INDEX["E_m"]
    em_values = features[:, em_idx].reshape(grid_sizes)[:, 0, 0]
    mid_em = em_values[len(em_values) // 2]
    slice_mask = np.isclose(features[:, em_idx], mid_em)
    slice_features = features[slice_mask]
    slice_e1 = (ec["e_l"][slice_mask] / 1e9).reshape(n0, n2)
    vf_coords = slice_features[:, SURROGATE_FEATURE_INDEX["vf"]].reshape(n0, n2)[:, 0]
    elf_coords = (
        slice_features[:, SURROGATE_FEATURE_INDEX["E_Lf"]].reshape(n0, n2)[0, :] / 1e9
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        slice_e1,
        origin="lower",
        aspect="auto",
        extent=[elf_coords.min(), elf_coords.max(), vf_coords.min(), vf_coords.max()],
        cmap="magma",
    )
    ax.set_xlabel(r"$E_{Lf}$ [GPa]")
    ax.set_ylabel("fibre volume fraction")
    ax.set_title(rf"$E_1$ slice at $E_m$ = {mid_em / 1e9:.2f} GPa")
    fig.colorbar(im, ax=ax, label=r"$E_1$ [GPa]")
    fig.tight_layout()
    paths["e1_slice"] = plot_dir / "response_slice_e1_vf_elf.png"
    fig.savefig(paths["e1_slice"], dpi=150)
    plt.close(fig)

    return paths
