"""MFEM Cartesian transverse meshes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_micromech.geometry import classify_points
from b3_micromech.problem import RVEProblem


def _resolve_mfem_cell_type(cell_type: str):
    import mfem.ser as mfem

    name = cell_type.lower()
    if name in ("quadrilateral", "quad", "hex"):
        return mfem.Element.QUADRILATERAL
    if name in ("triangle", "tri", "simplex"):
        return mfem.Element.TRIANGLE
    raise ValueError(
        f"unknown cell_type {cell_type!r}; expected quadrilateral or triangle"
    )


def build_cartesian_mesh(problem: RVEProblem):
    """Build a 2D Cartesian mesh on the y–z transverse plane."""
    import mfem.ser as mfem

    nx, ny = problem.mesh_resolution
    sy, sz = problem.size_yz
    mfem_type = _resolve_mfem_cell_type(problem.cell_type)
    mesh = mfem.Mesh.MakeCartesian2D(nx, ny, mfem_type, True, sy, sz)
    return mesh


def element_cell_vertices_yz(mesh) -> NDArray[np.float64]:
    """Per-element vertex coordinates ``(n_elem, n_verts, 2)``."""
    n_elem = mesh.GetNE()
    n_verts = len(mesh.GetElement(0).GetVerticesArray())
    out = np.empty((n_elem, n_verts, 2), dtype=float)
    for e in range(n_elem):
        vids = mesh.GetElement(e).GetVerticesArray()
        out[e] = np.array([mesh.GetVertexArray(int(v)) for v in vids], dtype=float)
    return out


def element_centroids_yz(mesh) -> NDArray[np.float64]:
    n_elem = mesh.GetNE()
    centroids = np.empty((n_elem, 2), dtype=float)
    for e in range(n_elem):
        el = mesh.GetElement(e)
        vids = el.GetVerticesArray()
        pts = np.array([mesh.GetVertexArray(int(v)) for v in vids], dtype=float)
        centroids[e] = pts.mean(axis=0)
    return centroids


def element_material_ids(problem: RVEProblem, mesh) -> NDArray[np.int32]:
    """Per-element attribute: 1 = matrix, 2 = fibre."""
    centroids = element_centroids_yz(mesh)
    centre = problem.centre_yz
    is_fibre = classify_points(centroids, centre=centre, radius=problem.fibre_radius)
    attrs = np.ones(mesh.GetNE(), dtype=np.int32)
    attrs[is_fibre] = 2
    for e in range(mesh.GetNE()):
        mesh.GetElement(e).SetAttribute(int(attrs[e]))
    return attrs


def mesh_vertices_and_cells(mesh) -> tuple[NDArray[np.float64], list[NDArray[np.intp]]]:
    """Vertex coordinates ``(nv, 2)`` and per-cell vertex index lists."""
    nv = mesh.GetNV()
    vertices = np.array([mesh.GetVertexArray(i) for i in range(nv)], dtype=float)
    cells: list[NDArray[np.intp]] = []
    for e in range(mesh.GetNE()):
        cells.append(np.asarray(mesh.GetElement(e).GetVerticesArray(), dtype=np.intp))
    return vertices, cells


def stiffness_per_element(
    problem: RVEProblem, material_ids: NDArray[np.int32]
) -> NDArray[np.float64]:
    matrix = problem.materials[problem.matrix_material].stiffness
    fibre = problem.materials[problem.fibre_material].stiffness
    out = np.empty((material_ids.shape[0], 6, 6), dtype=float)
    out[material_ids == 1] = matrix
    out[material_ids == 2] = fibre
    return out
