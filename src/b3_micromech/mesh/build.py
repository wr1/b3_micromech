"""Mesh builders for transverse RVE domains."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from b3_micromech.problem import RVEProblem


def build_mesh(problem: RVEProblem):
    if problem.domain_shape == "square":
        from b3_micromech.mesh.cartesian import build_cartesian_mesh

        mesh = build_cartesian_mesh(problem)
    elif problem.domain_shape == "hexagon":
        from b3_micromech.mesh.hexagon import build_hexagon_mesh

        mesh = build_hexagon_mesh(problem)
    else:
        raise ValueError(f"unknown domain shape {problem.domain_shape!r}")

    from b3_micromech.amr import apply_optional_refinement

    mesh, _history = apply_optional_refinement(mesh, problem)
    return mesh
