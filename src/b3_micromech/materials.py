"""Constituent materials for the transverse RVE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.tensors import isotropic_stiffness, transverse_isotropic_stiffness


@dataclass(frozen=True)
class Material:
    name: str
    stiffness: NDArray[np.float64]

    def __post_init__(self) -> None:
        c = np.asarray(self.stiffness, dtype=float)
        if c.shape != (6, 6):
            raise ValueError(f"stiffness must have shape (6, 6), got {c.shape}")
        object.__setattr__(self, "stiffness", c)

    @classmethod
    def isotropic(
        cls, name: str, *, youngs_modulus: float, poisson_ratio: float
    ) -> Material:
        return cls(
            name=name, stiffness=isotropic_stiffness(youngs_modulus, poisson_ratio)
        )

    @classmethod
    def transverse_isotropic(
        cls,
        name: str,
        *,
        e_l: float,
        e_t: float,
        g_lt: float,
        nu_lt: float,
        nu_tt: float,
    ) -> Material:
        return cls(
            name=name,
            stiffness=transverse_isotropic_stiffness(
                e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
            ),
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Material:
        name = str(cfg["name"])
        mtype = str(cfg["type"])
        if mtype == "isotropic":
            return cls.isotropic(
                name,
                youngs_modulus=float(cfg["youngs_modulus"]),
                poisson_ratio=float(cfg["poisson_ratio"]),
            )
        if mtype == "transverse_isotropic":
            return cls.transverse_isotropic(
                name,
                e_l=float(cfg["e_l"]),
                e_t=float(cfg["e_t"]),
                g_lt=float(cfg["g_lt"]),
                nu_lt=float(cfg["nu_lt"]),
                nu_tt=float(cfg["nu_tt"]),
            )
        raise ValueError(f"unknown material type {mtype!r}")


def load_materials(config: list[dict[str, Any]]) -> dict[str, Material]:
    return {m.name: m for m in (Material.from_config(c) for c in config)}
