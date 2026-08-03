"""Constituent materials for the transverse RVE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.tensors import (
    isotropic_stiffness,
    transverse_isotropic_stiffness,
)


@dataclass(frozen=True)
class Material:
    """Mechanical and thermal material properties.

    Parameters
    ----------
    name
        Material label used in the RVE config.
    stiffness
        Symmetric elasticity tensor in Voigt form, shape ``(6, 6)``.
    thermal_conductivity
        Tensor for steady-state heat conduction in Voigt form, shape ``(6, 6)``.
        For isotropic materials this is ``k * I_3`` (rank-2 tensor stored in
        the upper-left ``(3, 3)`` block; rows 3:6 are zero).  For transverse
        isotropy the longitudinal direction is ``0`` (global *x*, fibre axis)
        and the transverse plane is ``(1, 2)`` (transverse *y–z*).
    thermal_expansion
        Thermal-expansion tensor in Voigt form, shape ``(6,)``.
        For isotropic materials this is ``[alpha, alpha, alpha, 0, 0, 0]``.
        For transverse isotropy ``[alpha_l, alpha_t, alpha_t, 0, 0, 0]``.
    """

    name: str
    stiffness: NDArray[np.float64]
    thermal_conductivity: NDArray[np.float64]
    thermal_expansion: NDArray[np.float64]

    def __post_init__(self) -> None:
        c = np.asarray(self.stiffness, dtype=float)
        if c.shape != (6, 6):
            raise ValueError(f"stiffness must have shape (6, 6), got {c.shape}")
        k = np.asarray(self.thermal_conductivity, dtype=float)
        if k.shape != (6, 6):
            raise ValueError(
                f"thermal_conductivity must have shape (6, 6), got {k.shape}"
            )
        a = np.asarray(self.thermal_expansion, dtype=float)
        if a.shape != (6,):
            raise ValueError(f"thermal_expansion must have shape (6,), got {a.shape}")
        object.__setattr__(self, "stiffness", c)
        object.__setattr__(self, "thermal_conductivity", k)
        object.__setattr__(self, "thermal_expansion", a)

    @property
    def thermal_expansion_scalar(self) -> float:
        """Return an isotropic-equivalent CTE: ``alpha_yy`` (index 1)."""
        return float(self.thermal_expansion[1])

    @classmethod
    def isotropic(
        cls,
        name: str,
        *,
        youngs_modulus: float,
        poisson_ratio: float,
        thermal_conductivity: float = 0.0,
        thermal_expansion: float = 0.0,
    ) -> Material:
        """Create an isotropic material.

        Parameters
        ----------
        thermal_conductivity
            Isotropic conductivity *k*.  Stored in the ``(3, 3)`` block of
            the ``(6, 6)`` Voigt tensor; the remaining entries are zero.
        thermal_expansion
            Isotropic coefficient of thermal expansion ``alpha``.  Stored as
            ``[alpha, alpha, alpha, 0, 0, 0]`` (plane-strain ``ε_xx`` is
            constrained but ``ε_yy``, ``ε_zz`` are free).
        """
        k = np.zeros((6, 6), dtype=float)
        k[0, 0] = thermal_conductivity
        k[1, 1] = thermal_conductivity
        k[2, 2] = thermal_conductivity
        alpha = np.full(6, thermal_expansion, dtype=float)
        alpha[3:] = 0.0
        return cls(
            name=name,
            stiffness=isotropic_stiffness(youngs_modulus, poisson_ratio),
            thermal_conductivity=k,
            thermal_expansion=alpha,
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
        k_l: float = 0.0,
        k_t: float = 0.0,
        alpha_l: float = 0.0,
        alpha_t: float = 0.0,
    ) -> Material:
        """Create a transverse-isotropic material (fibre direction = ``x``).

        Parameters
        ----------
        k_l, k_t
            Longitudinal / transthermal conductivity.  Stored so that
            ``k[0,0] = k_l`` and ``k[1,1] = k[2,2] = k_t``.
        alpha_l, alpha_t
            Longitudinal / transverse coefficient of thermal expansion.
            Stored as ``[alpha_l, alpha_t, alpha_t, 0, 0, 0]``.
        """
        C = transverse_isotropic_stiffness(
            e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
        )
        k = np.zeros((6, 6), dtype=float)
        k[0, 0] = k_l
        k[1, 1] = k_t
        k[2, 2] = k_t
        alpha = np.zeros(6, dtype=float)
        alpha[0] = alpha_l
        alpha[1] = alpha_t
        alpha[2] = alpha_t
        return cls(
            name=name,
            stiffness=C,
            thermal_conductivity=k,
            thermal_expansion=alpha,
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
                thermal_conductivity=float(cfg.get("thermal_conductivity", 0.0)),
                thermal_expansion=float(cfg.get("thermal_expansion", 0.0)),
            )
        if mtype == "transverse_isotropic":
            return cls.transverse_isotropic(
                name,
                e_l=float(cfg["e_l"]),
                e_t=float(cfg["e_t"]),
                g_lt=float(cfg["g_lt"]),
                nu_lt=float(cfg["nu_lt"]),
                nu_tt=float(cfg["nu_tt"]),
                k_l=float(cfg.get("k_l", 0.0)),
                k_t=float(cfg.get("k_t", 0.0)),
                alpha_l=float(cfg.get("alpha_l", 0.0)),
                alpha_t=float(cfg.get("alpha_t", 0.0)),
            )
        raise ValueError(f"unknown material type {mtype!r}")


def load_materials(config: list[dict[str, Any]]) -> dict[str, Material]:
    return {m.name: m for m in (Material.from_config(c) for c in config)}
