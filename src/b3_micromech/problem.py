"""RVE problem specification."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_micromech.config import ConfigError, SolverConfig
from b3_micromech.geometry import (
    DomainShape,
    check_fibre_volume_fraction,
    fibre_centre,
    fibre_volume_fraction_from_radius,
    hexagon_bounding_box,
    radius_from_fibre_volume_fraction,
)
from b3_micromech.materials import Material, load_materials

_TOP_KEYS = {"domain", "materials", "rve", "solver", "periodic_tolerance", "sweep"}
_DOMAIN_KEYS = {"shape", "size", "mesh_resolution"}
_RVE_KEYS = {
    "matrix_material",
    "fibre_material",
    "fibre_volume_fraction",
    "fibre_radius",
    "centre",
}


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
    solver: SolverConfig
    material_configs: tuple[str, ...] = field(default=(), hash=False, compare=False)

    @property
    def solver_dict(self) -> dict[str, Any]:
        """Dict view of :attr:`solver`. Removed in 0.3.0."""
        return self.solver.to_dict()

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

    def to_config(self) -> dict[str, Any]:
        """Canonical config. ``sweep`` is omitted; material entries are as loaded."""
        return {
            "domain": {
                "shape": self.domain_shape,
                "size": self.domain_size,
                "mesh_resolution": list(self.mesh_resolution),
            },
            "materials": [json.loads(item) for item in self.material_configs],
            "rve": {
                "matrix_material": self.matrix_material,
                "fibre_material": self.fibre_material,
                "fibre_volume_fraction": self.fibre_volume_fraction,
                "centre": [float(self.centre_yz[0]), float(self.centre_yz[1])],
            },
            "solver": self.solver.to_dict(),
            "periodic_tolerance": self.periodic_tolerance,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RVEProblem:
        raw = dict(config)
        if "field" in raw:
            warnings.warn(
                "top-level 'field' is deprecated; use 'rve' (removed in 0.3.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            raw.setdefault("rve", raw.pop("field"))
        unknown = sorted(set(raw) - _TOP_KEYS)
        if unknown:
            raise ConfigError(
                f"config: unknown key(s) {unknown}; allowed {sorted(_TOP_KEYS)}"
            )
        raw.pop("sweep", None)

        domain = dict(raw["domain"])
        if "side" in domain:
            warnings.warn(
                "domain.side is deprecated; use domain.size (removed in 0.3.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            domain.setdefault("size", domain.pop("side"))
        unknown_domain = sorted(set(domain) - _DOMAIN_KEYS)
        if unknown_domain:
            raise ConfigError(
                f"domain: unknown key(s) {unknown_domain}; allowed {sorted(_DOMAIN_KEYS)}"
            )
        if "shape" not in domain:
            warnings.warn(
                "domain.shape not given; defaulting to 'square' (will be required in 0.3.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            shape = _normalize_domain_shape("square")
        else:
            shape = _normalize_domain_shape(domain["shape"])
        size = float(domain.get("size", 1.0))
        res = domain.get("mesh_resolution", [32, 32])
        if len(res) == 1:
            mesh_resolution = (int(res[0]), int(res[0]))
        elif len(res) == 2:
            mesh_resolution = (int(res[0]), int(res[1]))
        else:
            raise ConfigError(
                "mesh_resolution must have length 1 or 2 for the transverse plane"
            )

        material_entries = list(raw["materials"])
        materials = load_materials(material_entries)
        rve = dict(raw.get("rve", {}))
        unknown_rve = sorted(set(rve) - _RVE_KEYS)
        if unknown_rve:
            raise ConfigError(
                f"rve: unknown key(s) {unknown_rve}; allowed {sorted(_RVE_KEYS)}"
            )
        matrix_material = str(rve["matrix_material"])
        fibre_material = str(rve["fibre_material"])
        if matrix_material not in materials or fibre_material not in materials:
            raise ConfigError(
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
            check_fibre_volume_fraction(shape=shape, vf=vf)
        else:
            raise ConfigError("rve must specify fibre_volume_fraction or fibre_radius")

        if "centre" in rve:
            c = rve["centre"]
            centre = np.array([float(c[0]), float(c[1])], dtype=float)

        solver = SolverConfig.from_mapping(raw.get("solver"))
        default_cell = "triangle" if shape == "hexagon" else "quadrilateral"
        cell_type = solver.cell_type or default_cell

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
            periodic_tolerance=float(raw.get("periodic_tolerance", 1e-8)),
            solver=solver,
            material_configs=tuple(
                json.dumps(entry, sort_keys=True) for entry in material_entries
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RVEProblem:
        with open(path, encoding="utf-8") as f:
            return cls.from_config(yaml.safe_load(f))
