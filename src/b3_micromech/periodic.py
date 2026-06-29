"""Periodic vertex pairing for transverse RVE meshes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_micromech.geometry import DomainShape, hexagon_corners


def periodic_vertex_master_map(
    mesh,
    *,
    shape: DomainShape,
    domain_size: float,
    tol: float,
) -> NDArray[np.intp]:
    if shape == "square":
        return _periodic_vertex_master_map_square(mesh, domain_size, tol)
    if shape == "hexagon":
        return _periodic_vertex_master_map_hex(mesh, domain_size, tol)
    raise ValueError(f"unknown domain shape {shape!r}")


def origin_vertex_index(
    mesh, *, shape: DomainShape, domain_size: float, tol: float
) -> int:
    """Vertex at the domain origin corner (used for rigid-body pin)."""
    nv = mesh.GetNV()
    coords = np.array([mesh.GetVertexArray(i) for i in range(nv)], dtype=float)
    if shape == "square":
        target = np.array([0.0, 0.0], dtype=float)
    elif shape == "hexagon":
        target = coords.min(axis=0)
    else:
        raise ValueError(f"unknown domain shape {shape!r}")
    dists = np.linalg.norm(coords - target[None, :], axis=1)
    hits = np.where(dists <= tol)[0]
    if hits.size:
        return int(hits[0])
    return int(np.argmin(dists))


def _periodic_vertex_master_map_square(
    mesh, domain_size: float, tol: float
) -> NDArray[np.intp]:
    Ly = Lz = float(domain_size)
    nv = mesh.GetNV()
    master_of = np.empty(nv, dtype=np.intp)
    canonical_to_master: dict[tuple[int, int], int] = {}
    for v in range(nv):
        coords = np.asarray(mesh.GetVertexArray(v), dtype=float)
        canon = coords.copy()
        if abs(canon[0] - Ly) < tol:
            canon[0] = 0.0
        if abs(canon[1] - Lz) < tol:
            canon[1] = 0.0
        key = (round(canon[0] / tol), round(canon[1] / tol))
        if key not in canonical_to_master:
            canonical_to_master[key] = v
        master_of[v] = canonical_to_master[key]
    return master_of


def _distance_to_segment(
    point: NDArray[np.float64], v0: NDArray[np.float64], v1: NDArray[np.float64]
) -> tuple[float, float]:
    edge = v1 - v0
    length2 = float(np.dot(edge, edge))
    if length2 < 1e-30:
        return float(np.linalg.norm(point - v0)), 0.0
    t = float(np.dot(point - v0, edge) / length2)
    t_clamped = min(1.0, max(0.0, t))
    proj = v0 + t_clamped * edge
    return float(np.linalg.norm(point - proj)), t_clamped


def _periodic_vertex_master_map_hex(
    mesh, circumradius: float, tol: float
) -> NDArray[np.intp]:
    corners = hexagon_corners(circumradius)
    nv = mesh.GetNV()
    master_of = np.arange(nv, dtype=np.intp)

    corner_vertices: list[list[int]] = [[] for _ in range(6)]
    edge_vertices: list[list[tuple[int, float]]] = [[] for _ in range(6)]

    for v in range(nv):
        coord = np.asarray(mesh.GetVertexArray(v), dtype=float)
        corner_id = None
        for k in range(6):
            if np.linalg.norm(coord - corners[k]) <= tol:
                corner_id = k
                break
        if corner_id is not None:
            corner_vertices[corner_id].append(v)
            continue

        best_edge = -1
        best_dist = tol
        best_param = 0.0
        for edge in range(6):
            dist, param = _distance_to_segment(
                coord, corners[edge], corners[(edge + 1) % 6]
            )
            if dist <= best_dist:
                best_dist = dist
                best_edge = edge
                best_param = param
        if best_edge >= 0:
            if best_param > tol and best_param < 1.0 - tol:
                edge_vertices[best_edge].append((v, best_param))

    def _pick_master(vertices: list[int]) -> int:
        return min(vertices)

    for corner in range(3):
        slaves = corner_vertices[corner]
        masters = corner_vertices[corner + 3]
        if not slaves or not masters:
            continue
        master = _pick_master(masters)
        for v in slaves:
            master_of[v] = master

    for edge in range(3):
        slave_edge = edge
        master_edge = edge + 3
        slaves = sorted(edge_vertices[slave_edge], key=lambda item: item[1])
        masters = sorted(edge_vertices[master_edge], key=lambda item: item[1])
        if not slaves or not masters:
            continue
        master_params = np.array([p for _, p in masters], dtype=float)
        for v_slave, param in slaves:
            idx = int(np.argmin(np.abs(master_params - param)))
            master_of[v_slave] = masters[idx][0]

    return master_of
