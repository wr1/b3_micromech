"""Per-loadcase solve driver for homogenization and plotting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_micromech.backends.mfem_periodic_2d import make_session
from b3_micromech.problem import RVEProblem
from b3_micromech.result import LOADCASE_LABELS, LoadcaseResult

__all__ = ["LOADCASE_LABELS", "LoadcaseResult", "LoadcaseSet", "solve_all_loadcases"]


@dataclass(frozen=True)
class LoadcaseSet:
    problem: RVEProblem
    session: object
    material_ids: NDArray[np.int32]
    results: tuple[LoadcaseResult, ...]


def solve_all_loadcases(problem: RVEProblem) -> LoadcaseSet:
    """Run six unit macro-strain solves and collect fields for plotting."""
    from b3_micromech.mesh.cartesian import element_material_ids

    session = make_session(problem)
    mat_ids = element_material_ids(problem, session.mesh)
    results = tuple(session.solve_macro_strain(np.eye(6)[k]) for k in range(6))
    return LoadcaseSet(
        problem=problem,
        session=session,
        material_ids=mat_ids,
        results=results,
    )
