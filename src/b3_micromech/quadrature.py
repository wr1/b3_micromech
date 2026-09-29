"""Per-Gauss-point stiffness lookup (implicit fibre field, b3_tex-style)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.config import ConfigError, SolverConfig
from b3_micromech.field import FIBRE_ID, MATRIX_ID, sample_material_ids

if TYPE_CHECKING:
    from b3_micromech.problem import RVEProblem


def _resolve_material_sampling_spec(
    solver: SolverConfig | dict[str, Any],
) -> dict[str, Any]:
    """Deprecated. Read ``problem.solver.material_sampling`` instead."""
    warnings.warn(
        "_resolve_material_sampling_spec is deprecated; read "
        "problem.solver.material_sampling (removed in 0.3.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(solver, SolverConfig):
        solver = SolverConfig.from_mapping(solver)
    ms = solver.material_sampling
    return {
        "strategy": ms.strategy,
        "resolution": ms.resolution,
        "idw_power": ms.idw_power,
    }


def global_stiffness_at_points(
    problem: RVEProblem, points_yz: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Implicit pointwise ``(N, 6, 6)`` stiffness from the fibre disc field."""
    pts = np.asarray(points_yz, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ids = sample_material_ids(problem, pts)
    matrix = problem.materials[problem.matrix_material].stiffness
    fibre = problem.materials[problem.fibre_material].stiffness
    out = np.empty((pts.shape[0], 6, 6), dtype=float)
    out[ids == MATRIX_ID] = matrix
    out[ids == FIBRE_ID] = fibre
    return out


def _unit_material_grid_2d(resolution: int) -> NDArray[np.float64]:
    if resolution < 1:
        raise ValueError("resolution must be a positive integer")
    ax = (np.arange(resolution) + 0.5) / resolution
    yy, zz = np.meshgrid(ax, ax, indexing="ij")
    return np.column_stack([yy.ravel(), zz.ravel()])


def _idw_per_cell_2d(
    gp_coords: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    phys_material: NDArray[np.float64],
    c_per_cell_material: NDArray[np.float64],
    *,
    power: float = 2.0,
) -> NDArray[np.float64]:
    n_gps = gp_coords.shape[0]
    n_cells = phys_material.shape[0]
    if n_cells == 0 or n_gps % n_cells != 0:
        raise ValueError("gp_cell_ids must be a regular repeat partition")
    nq = n_gps // n_cells
    del gp_cell_ids
    gp_by_cell = gp_coords.reshape(n_cells, nq, 2)
    out = np.empty((n_cells, nq, 6, 6), dtype=float)
    for c in range(n_cells):
        diff = gp_by_cell[c, :, None, :] - phys_material[c, None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)
        w = 1.0 / (dist**power)
        w /= w.sum(axis=1, keepdims=True)
        out[c] = np.einsum("qm,mij->qij", w, c_per_cell_material[c])
    return out.reshape(n_gps, 6, 6)


def effective_stiffnesses_for_gauss_points(
    problem: RVEProblem,
    gp_coords_yz: NDArray[np.float64],
    gp_cell_ids: NDArray[np.intp],
    cell_vertices_yz: NDArray[np.float64],
    *,
    spec: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """``(N_gps, 6, 6)`` stiffness per GP.

    Strategies (mirrors ``b3_tex.quadrature``):

    - ``exact``: sample implicit field at each GP
    - ``cell_constant``: centroid sample, broadcast to all GPs in the cell
    - ``local_cloud``: ``resolution²`` sub-samples in the cell AABB, IDW to each GP
      (gives intermediate stiffness on elements crossing the fibre boundary)
    """
    if spec is None:
        ms = problem.solver.material_sampling
        strategy = ms.strategy
        resolution = ms.resolution
        idw_power = ms.idw_power
    else:
        strategy = str(spec["strategy"])
        resolution = int(spec["resolution"])
        idw_power = float(spec["idw_power"])
    if strategy not in ("exact", "cell_constant", "local_cloud"):
        raise ConfigError(
            f"unknown material sampling strategy {strategy!r}; "
            "allowed ['cell_constant', 'exact', 'local_cloud']"
        )
    n_cells = cell_vertices_yz.shape[0]

    if strategy == "exact":
        return global_stiffness_at_points(problem, gp_coords_yz)

    if strategy == "cell_constant":
        centroids = cell_vertices_yz.mean(axis=1)
        return global_stiffness_at_points(problem, centroids)[gp_cell_ids]

    ref_pts = _unit_material_grid_2d(resolution)
    mins = cell_vertices_yz.min(axis=1)
    maxs = cell_vertices_yz.max(axis=1)
    scales = maxs - mins
    phys_material = mins[:, None, :] + ref_pts[None, :, :] * scales[:, None, :]
    c_all = global_stiffness_at_points(problem, phys_material.reshape(-1, 2))
    c_per_cell_material = c_all.reshape(n_cells, -1, 6, 6)
    return _idw_per_cell_2d(
        gp_coords_yz,
        gp_cell_ids,
        phys_material,
        c_per_cell_material,
        power=idw_power,
    )
