"""RVE problem specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_micromech.geometry import (
    DomainShape,
    fibre_centre,
    fibre_volume_fraction_from_radius,
    hexagon_bounding_box,
    radius_from_fibre_volume_fraction,
)
from b3_micromech.materials import Material, load_materials


def _normalize_domain_shape(raw: str) -> DomainShape:
    name = str(raw).lower().strip()
    if name in ("square", "quad", "quadrilateral"):
        return "square"
    if name in ("hexagon", "hex"):
        return "hexagon"
    raise ValueError(f"unknown domain shape {raw!r}; expected square or hexagon")


@dataclass(frozen=True)
class RVEProblem:
    """Transverse UD RVE with fibre along global x."""

    domain_shape: DomainShape
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
        if self.domain_shape == "hexagon":
            return hexagon_bounding_box(self.domain_size)
        s = float(self.domain_size)
        return (s, s)

    @property
    def plot_bounds(self) -> tuple[float, float, float, float]:
        ymax, zmax = self.size_yz
        return (0.0, ymax, 0.0, zmax)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RVEProblem:
        domain = config["domain"]
        shape = _normalize_domain_shape(domain.get("shape", "square"))
        size = float(domain.get("size", domain.get("side", 1.0)))
        res = domain.get("mesh_resolution", [32, 32])
        if len(res) == 1:
            mesh_resolution = (int(res[0]), int(res[0]))
        elif len(res) == 2:
            mesh_resolution = (int(res[0]), int(res[1]))
        else:
            raise ValueError(
                "mesh_resolution must have length 1 or 2 for the transverse plane"
            )

        materials = load_materials(config["materials"])
        rve = config.get("rve", config.get("field", {}))
        matrix_material = str(rve["matrix_material"])
        fibre_material = str(rve["fibre_material"])
        if matrix_material not in materials or fibre_material not in materials:
            raise ValueError(
                "matrix_material and fibre_material must name configured materials"
            )

        centre = fibre_centre(shape, size)
        if "fibre_volume_fraction" in rve:
            vf = float(rve["fibre_volume_fraction"])
            radius = radius_from_fibre_volume_fraction(
                shape=shape, domain_size=size, vf=vf
            )
        elif "fibre_radius" in rve:
            radius = float(rve["fibre_radius"])
            vf = fibre_volume_fraction_from_radius(
                shape=shape, domain_size=size, radius=radius
            )
        else:
            raise ValueError("rve must specify fibre_volume_fraction or fibre_radius")

        if "centre" in rve:
            c = rve["centre"]
            centre = np.array([float(c[0]), float(c[1])], dtype=float)

        solver = dict(config.get("solver", {}))
        default_cell = "triangle" if shape == "hexagon" else "quadrilateral"
        cell_type = str(solver.get("cell_type", default_cell))

        return cls(
            domain_shape=shape,
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
