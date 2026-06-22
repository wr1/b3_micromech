"""RVE problem specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_micromech.geometry import (
    fibre_centre,
    fibre_volume_fraction_from_radius,
    radius_from_fibre_volume_fraction,
)
from b3_micromech.materials import Material, load_materials


@dataclass(frozen=True)
class RVEProblem:
    """Transverse UD RVE with fibre along global x."""

    domain_size: float
    mesh_resolution: tuple[int, int]
    cell_type: str
    materials: dict[str, Material]
    matrix_material: str
    fibre_material: str
    fibre_radius: float
    fibre_volume_fraction: float
    centre_yz: NDArray[np.float64]
    periodic_tolerance: float
    solver: dict[str, Any]

    @property
    def size_yz(self) -> tuple[float, float]:
        s = float(self.domain_size)
        return (s, s)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RVEProblem:
        domain = config["domain"]
        size = float(domain.get("size", domain.get("side", 1.0)))
        res = domain.get("mesh_resolution", [32, 32])
        if len(res) != 2:
            raise ValueError("mesh_resolution must have length 2 for the transverse plane")
        mesh_resolution = (int(res[0]), int(res[1]))

        materials = load_materials(config["materials"])
        rve = config.get("rve", config.get("field", {}))
        matrix_material = str(rve["matrix_material"])
        fibre_material = str(rve["fibre_material"])
        if matrix_material not in materials or fibre_material not in materials:
            raise ValueError("matrix_material and fibre_material must name configured materials")

        centre = fibre_centre(size)
        if "fibre_volume_fraction" in rve:
            vf = float(rve["fibre_volume_fraction"])
            radius = radius_from_fibre_volume_fraction(domain_size=size, vf=vf)
        elif "fibre_radius" in rve:
            radius = float(rve["fibre_radius"])
            vf = fibre_volume_fraction_from_radius(domain_size=size, radius=radius)
        else:
            raise ValueError("rve must specify fibre_volume_fraction or fibre_radius")

        if "centre" in rve:
            c = rve["centre"]
            centre = np.array([float(c[0]), float(c[1])], dtype=float)

        solver = dict(config.get("solver", {}))
        cell_type = str(solver.get("cell_type", "quadrilateral"))

        return cls(
            domain_size=size,
            mesh_resolution=mesh_resolution,
            cell_type=cell_type,
            materials=materials,
            matrix_material=matrix_material,
            fibre_material=fibre_material,
            fibre_radius=radius,
            fibre_volume_fraction=vf,
            centre_yz=centre,
            periodic_tolerance=float(config.get("periodic_tolerance", 1e-8)),
            solver=solver,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RVEProblem:
        with open(path, encoding="utf-8") as f:
            return cls.from_config(yaml.safe_load(f))