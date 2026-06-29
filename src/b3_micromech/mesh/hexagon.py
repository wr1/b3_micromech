"""Triangle-based meshes on hexagonal transverse domains."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from b3_micromech.geometry import (
    hexagon_corners,
    hexagon_side_length,
    point_in_hexagon,
)

if TYPE_CHECKING:
    from b3_micromech.problem import RVEProblem


def _require_triangle():
    try:
        import triangle  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "hexagon domains require the triangle package — pip install triangle"
        ) from exc


def _mesh_stream_from_triangulation(vertices: np.ndarray, triangles: np.ndarray) -> str:
    nv = len(vertices)
    ne = len(triangles)
    lines = ["MFEM mesh v1.0", "", "dimension", "2", "", "elements", str(ne)]
    for tri in triangles:
        lines.append(f"1 2 {int(tri[0])} {int(tri[1])} {int(tri[2])}")
    lines += ["", "boundary", "0", "", "vertices", str(nv), "2"]
    for v in vertices:
        lines.append(f"{float(v[0]):.16e} {float(v[1]):.16e}")
    lines.append("")
    return "\n".join(lines)


def _hexagon_boundary_polyline(
    circumradius: float, edge_divisions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Closed boundary polyline with ``6 * edge_divisions`` vertices."""
    corners = hexagon_corners(circumradius)
    vertices: list[np.ndarray] = []
    for edge in range(6):
        v0 = corners[edge]
        v1 = corners[(edge + 1) % 6]
        for k in range(edge_divisions):
            t = k / edge_divisions
            vertices.append(v0 + t * (v1 - v0))
    verts = np.asarray(vertices, dtype=float)
    n = len(verts)
    segments = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.intp)
    return verts, segments


def triangulate_hexagon(
    circumradius: float,
    *,
    edge_divisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, triangles)`` for a flat-top hexagon."""
    _require_triangle()
    import triangle as tr

    if edge_divisions < 2:
        raise ValueError("edge_divisions must be at least 2")
    vertices, segments = _hexagon_boundary_polyline(circumradius, edge_divisions)
    inp: dict = {"vertices": vertices, "segments": segments}

    side = hexagon_side_length(circumradius)
    max_area = (side / float(edge_divisions)) ** 2 * 0.5
    opts = f"pq30a{max_area:.6e}"
    out = tr.triangulate(inp, opts)

    vertices = np.asarray(out["vertices"], dtype=float)
    triangles = np.asarray(out["triangles"], dtype=np.intp)

    inside = point_in_hexagon(
        vertices[triangles].mean(axis=1), circumradius=circumradius
    )
    triangles = triangles[inside]
    if triangles.size == 0:
        raise RuntimeError("hexagon triangulation produced no interior elements")

    return vertices, triangles


def build_hexagon_mesh(problem: RVEProblem):
    """Build a 2D simplex mesh on a hexagonal transverse domain."""
    import mfem.ser as mfem

    edge_divisions = max(problem.mesh_resolution)
    vertices, triangles = triangulate_hexagon(
        problem.domain_size,
        edge_divisions=edge_divisions,
    )
    text = _mesh_stream_from_triangulation(vertices, triangles)
    mesh = mfem.Mesh(io.StringIO(text), 1, 1)
    mesh.FinalizeTopology()
    return mesh
