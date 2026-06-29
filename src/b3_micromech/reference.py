"""Analytical references for validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_micromech.materials import Material
from b3_micromech.tensors import (
    engineering_constants_transverse_iso,
    transverse_isotropic_stiffness,
)


def _engineering_constants_isotropic(
    stiffness: NDArray[np.float64],
) -> tuple[float, float]:
    C = np.asarray(stiffness, dtype=float)
    lam_plus_2mu = C[0, 0]
    lam = C[0, 1]
    mu = 0.5 * (lam_plus_2mu - lam)
    e = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return float(e), float(nu)


def mori_tanaka_cylinder(
    *,
    matrix: Material,
    fibre: Material,
    fibre_volume_fraction: float,
) -> NDArray[np.float64]:
    vf = float(fibre_volume_fraction)
    if not 0.0 <= vf <= 1.0:
        raise ValueError("fibre_volume_fraction must be in [0, 1]")
    vm = 1.0 - vf

    em, num = _engineering_constants_isotropic(matrix.stiffness)
    fibre_consts = engineering_constants_transverse_iso(fibre.stiffness)
    e_l_f = fibre_consts["e_l"]
    e_t_f = fibre_consts["e_t"]
    g_lt_f = fibre_consts["g_lt"]
    nu_lt_f = fibre_consts["nu_lt"]
    nu_tt_f = fibre_consts["nu_tt"]

    e_l = vf * e_l_f + vm * em
    nu_lt = vf * nu_lt_f + vm * num

    xi_e = 2.0
    eta_e = (e_t_f / em - 1.0) / (e_t_f / em + xi_e)
    e_t = em * (1.0 + xi_e * eta_e * vf) / (1.0 - eta_e * vf)

    g_lt = em * (g_lt_f * (1.0 + vf) + em * vm) / (g_lt_f * vm + em * (1.0 + vf))
    nu_tt = vf * nu_tt_f + vm * num

    return transverse_isotropic_stiffness(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )


def rule_of_mixtures_axial(
    *,
    matrix: Material,
    fibre: Material,
    fibre_volume_fraction: float,
) -> dict[str, float]:
    vf = float(fibre_volume_fraction)
    vm = 1.0 - vf
    em, num = _engineering_constants_isotropic(matrix.stiffness)
    fc = engineering_constants_transverse_iso(fibre.stiffness)
    e_l = vf * fc["e_l"] + vm * em
    nu_lt = vf * fc["nu_lt"] + vm * num
    g_lt = vf * fc["g_lt"] + vm * (em / (2 * (1 + num)))
    return {"e_l": e_l, "nu_lt": nu_lt, "g_lt": g_lt}
