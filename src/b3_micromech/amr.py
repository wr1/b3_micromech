"""Adaptive refinement driven by in-cell stiffness jumps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.field import sample_material_ids
from b3_micromech.mesh.cartesian import element_cell_vertices_yz
from b3_micromech.quadrature import global_stiffness_at_points

if TYPE_CHECKING:
    from b3_micromech.problem import RVEProblem

DEFAULT_AMR_SUB_SAMPLES: int = 64


def _resolve_amr_spec(solver: dict[str, Any]) -> dict[str, Any]:
    amr = solver.get("amr", {})
    return {
        "enabled": bool(amr.get("enabled", False)),
        "max_iterations": int(amr.get("max_iterations", 3)),
        "threshold": float(amr.get("threshold", 0.15)),
        "dof_budget": int(amr.get("dof_budget", 50_000)),
        "n_samples_per_cell": int(
            amr.get("n_samples_per_cell", DEFAULT_AMR_SUB_SAMPLES)
        ),
        "marker": str(amr.get("marker", "stiffness_jump")),
        "n_uniform_refines": int(amr.get("n_uniform_refines", 0)),
    }


def _triangular_number_side(n_samples: int) -> int | None:
    side = int((8 * n_samples + 1) ** 0.5 - 1) // 2
    if side > 0 and side * (side + 1) // 2 == n_samples:
        return side
    return None


def _tri_reference_barycentric_grid(n_samples: int) -> NDArray[np.float64] | None:
    """Deterministic triangular lattice when ``n_samples = k(k+1)/2``."""
    side = _triangular_number_side(n_samples)
    if side is None:
        return None
    step = 1.0 / side
    weights: list[list[float]] = []
    for i in range(side):
        for j in range(side - i):
            u = (i + 0.5) * step
            v = (j + 0.5) * step
            weights.append([1.0 - u - v, u, v])
    return np.asarray(weights, dtype=float)


def _tri_barycentric_weights_2d(n_samples: int, rng) -> NDArray[np.float64]:
    grid = _tri_reference_barycentric_grid(n_samples)
    if grid is not None:
        return grid
    w = rng.exponential(scale=1.0, size=(n_samples, 3))
    w /= w.sum(axis=1, keepdims=True)
    return w


def _quad_reference_unit_points_2d(n_samples: int) -> NDArray[np.float64]:
    side = round(n_samples**0.5)
    if side * side != n_samples:
        raise ValueError(
            f"n_samples_per_cell must be a perfect square for 2D quad cells, got {n_samples}"
        )
    ax = (np.arange(side) + 0.5) / side
    yy, zz = np.meshgrid(ax, ax, indexing="ij")
    return np.column_stack([yy.ravel(), zz.ravel()])


def _sample_points_per_cell(
    mesh,
    *,
    n_samples_per_cell: int,
    seed: int = 0,
) -> NDArray[np.float64]:
    import mfem.ser as mfem

    rng = np.random.default_rng(seed)
    cell_verts = element_cell_vertices_yz(mesh)
    geom = mesh.GetElement(0).GetGeometryType()

    if geom == mfem.Geometry.SQUARE:
        unit = _quad_reference_unit_points_2d(n_samples_per_cell)
        lo = cell_verts.min(axis=1)
        hi = cell_verts.max(axis=1)
        return lo[:, None, :] + unit[None, :, :] * (hi - lo)[:, None, :]

    if geom == mfem.Geometry.TRIANGLE:
        bary = _tri_barycentric_weights_2d(n_samples_per_cell, rng)
        return np.einsum("tb,cbd->ctd", bary, cell_verts)

    raise NotImplementedError(
        f"AMR sub-point sampler for geometry {geom} not implemented"
    )


def stiffness_jump_metric(
    problem: RVEProblem,
    mesh,
    *,
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-cell score = max relative Frobenius stiffness jump among sub-samples."""
    n_cells = mesh.GetNE()
    sample_pts = _sample_points_per_cell(
        mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
    )
    flat = sample_pts.reshape(-1, 2)
    c_flat = global_stiffness_at_points(problem, flat)
    c_samples = c_flat.reshape(n_cells, n_samples_per_cell, 6, 6)
    mean_c = c_samples.mean(axis=1)
    jumps = np.linalg.norm(c_samples - mean_c[:, None], axis=(-2, -1))
    denom = np.maximum(np.linalg.norm(mean_c, axis=(-2, -1)), 1e-12)
    return jumps.max(axis=1) / denom


def material_heterogeneity_metric(
    problem: RVEProblem,
    mesh,
    *,
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Fraction of sub-samples whose material ID differs from the cell majority."""
    n_cells = mesh.GetNE()
    sample_pts = _sample_points_per_cell(
        mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
    )
    flat = sample_pts.reshape(-1, 2)
    ids = sample_material_ids(problem, flat).reshape(n_cells, n_samples_per_cell)
    counts = np.zeros((n_cells, 3), dtype=int)
    for mat_id in (1, 2):
        counts[:, mat_id] = (ids == mat_id).sum(axis=1)
    majority = counts.argmax(axis=1)
    return (ids != majority[:, None]).mean(axis=1)


def cell_refinement_metric(
    problem: RVEProblem,
    mesh,
    *,
    marker: str = "stiffness_jump",
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    seed: int = 0,
) -> NDArray[np.float64]:
    name = marker.lower()
    if name in ("stiffness_jump", "stiffness", "jump"):
        return stiffness_jump_metric(
            problem, mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
        )
    if name in ("material", "phase", "heterogeneity"):
        return material_heterogeneity_metric(
            problem, mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
        )
    if name in ("combined", "both"):
        jump = stiffness_jump_metric(
            problem, mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
        )
        hetero = material_heterogeneity_metric(
            problem, mesh, n_samples_per_cell=n_samples_per_cell, seed=seed
        )
        return jump + 0.5 * hetero
    raise ValueError(f"unknown AMR marker {marker!r}")


def flag_cells_for_refinement(
    metric: NDArray[np.float64], threshold: float
) -> NDArray[np.bool_]:
    return metric > threshold


def refine_flagged_cells_mfem(mesh, flagged: NDArray[np.bool_]):
    import mfem.ser as mfem

    if mesh.ncmesh is None:
        mesh.EnsureNCMesh()
    refs = mfem.intArray()
    for c in np.where(flagged)[0]:
        refs.Append(int(c))
    mesh.GeneralRefinement(refs)
    return mesh


def iteratively_refine_mfem(
    mesh,
    problem: RVEProblem,
    *,
    threshold: float = 0.15,
    max_iterations: int = 3,
    dof_budget: int = 50_000,
    n_samples_per_cell: int = DEFAULT_AMR_SUB_SAMPLES,
    marker: str = "stiffness_jump",
) -> tuple[Any, list[dict[str, Any]]]:
    """Refine flagged cells until the stiffness-jump metric drops or DOF budget is hit."""
    history: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        metric = cell_refinement_metric(
            problem,
            mesh,
            marker=marker,
            n_samples_per_cell=n_samples_per_cell,
            seed=iteration,
        )
        flagged = flag_cells_for_refinement(metric, threshold)
        history.append(
            {
                "iteration": iteration,
                "n_cells": int(mesh.GetNE()),
                "n_vertices": int(mesh.GetNV()),
                "max_metric": float(metric.max()),
                "mean_metric": float(metric.mean()),
                "n_flagged": int(flagged.sum()),
            }
        )
        if not flagged.any():
            break
        n_flag = int(flagged.sum())
        if 3 * (mesh.GetNV() + 4 * n_flag) > dof_budget:
            break
        refine_flagged_cells_mfem(mesh, flagged)
    return mesh, history


def apply_optional_refinement(
    mesh, problem: RVEProblem
) -> tuple[Any, list[dict[str, Any]]]:
    spec = _resolve_amr_spec(problem.solver)
    history: list[dict[str, Any]] = []
    for _ in range(spec["n_uniform_refines"]):
        mesh.UniformRefinement()
    if not spec["enabled"]:
        return mesh, history
    mesh, history = iteratively_refine_mfem(
        mesh,
        problem,
        threshold=spec["threshold"],
        max_iterations=spec["max_iterations"],
        dof_budget=spec["dof_budget"],
        n_samples_per_cell=spec["n_samples_per_cell"],
        marker=spec["marker"],
    )
    return mesh, history
