"""Analytical references for validation and physics-base surrogates.

Chamis (1989) is the closed-form low-fidelity map used by the physics residual
and multi-fidelity GP surrogates (same pattern as ``b3_invsec`` capacity
physics + residual). Mori–Tanaka remains available for validation plots.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from b3_micromech.materials import Material
from b3_micromech.tensors import (
    engineering_constants_transverse_iso,
    transverse_isotropic_stiffness,
    transverse_isotropic_stiffness_batch,
)

# Surrogate feature columns: [Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf]
_FEATURE_VF = 0
_FEATURE_EM = 1
_FEATURE_NUM = 2
_FEATURE_ELF = 3
_FEATURE_ETF = 4
_FEATURE_GLTF = 5
_FEATURE_NULTF = 6
_FEATURE_GTTF = 7

# Engineering-constant keys used as residual targets (moduli log-space; ν additive).
EC_MODULUS_KEYS: tuple[str, ...] = ("e_l", "e_t", "g_lt", "g_tt")
EC_POISSON_KEYS: tuple[str, ...] = ("nu_lt", "nu_tt")
EC_TARGET_KEYS: tuple[str, ...] = EC_MODULUS_KEYS + EC_POISSON_KEYS


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


def _chamis_transverse(
    matrix_modulus: float, fibre_modulus: float, sqrt_vf: float
) -> float:
    if fibre_modulus <= 0.0:
        raise ValueError("fibre modulus must be positive")
    return matrix_modulus / (1.0 - sqrt_vf * (1.0 - matrix_modulus / fibre_modulus))


def chamis_ud_stiffness(
    *,
    matrix: Material,
    fibre: Material,
    fibre_volume_fraction: float,
) -> NDArray[np.float64]:
    """Chamis UD lamina stiffness (matches ``b3_tex.micromechanics.chamis_ud_stiffness``)."""
    vf = float(fibre_volume_fraction)
    if not 0.0 <= vf <= 1.0:
        raise ValueError("fibre_volume_fraction must be in [0, 1]")
    vm = 1.0 - vf
    em, num = _engineering_constants_isotropic(matrix.stiffness)
    gm = em / (2.0 * (1.0 + num))
    fc = engineering_constants_transverse_iso(fibre.stiffness)
    sqrt_vf = float(np.sqrt(vf))
    e_l = vf * fc["e_l"] + vm * em
    nu_lt = vf * fc["nu_lt"] + vm * num
    e_t = _chamis_transverse(em, fc["e_t"], sqrt_vf)
    g_lt = _chamis_transverse(gm, fc["g_lt"], sqrt_vf)
    g_tt = _chamis_transverse(gm, fc["g_tt"], sqrt_vf)
    nu_tt = e_t / (2.0 * g_tt) - 1.0
    return transverse_isotropic_stiffness(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )


def chamis_engineering_constants_from_features(
    features: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Vectorized Chamis engineering constants from ``(N, 8)`` surrogate features.

    Returns arrays for ``e_l``, ``e_t``, ``g_lt``, ``g_tt``, ``nu_lt``, ``nu_tt``.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] < 8:
        raise ValueError(f"expected >=8 feature columns, got shape {x.shape}")

    vf = x[:, _FEATURE_VF]
    em = x[:, _FEATURE_EM]
    num = x[:, _FEATURE_NUM]
    elf = x[:, _FEATURE_ELF]
    etf = x[:, _FEATURE_ETF]
    gltf = x[:, _FEATURE_GLTF]
    nultf = x[:, _FEATURE_NULTF]
    gttf = x[:, _FEATURE_GTTF]

    if np.any(vf < 0.0) or np.any(vf > 1.0):
        raise ValueError("fibre volume fraction must be in [0, 1]")
    if np.any(etf <= 0.0) or np.any(gltf <= 0.0) or np.any(gttf <= 0.0):
        raise ValueError("fibre moduli must be positive")
    if np.any(em <= 0.0):
        raise ValueError("matrix modulus must be positive")

    vm = 1.0 - vf
    gm = em / (2.0 * (1.0 + num))
    sqrt_vf = np.sqrt(np.clip(vf, 0.0, 1.0))

    e_l = vf * elf + vm * em
    nu_lt = vf * nultf + vm * num
    e_t = em / (1.0 - sqrt_vf * (1.0 - em / etf))
    g_lt = gm / (1.0 - sqrt_vf * (1.0 - gm / gltf))
    g_tt = gm / (1.0 - sqrt_vf * (1.0 - gm / gttf))
    nu_tt = e_t / (2.0 * g_tt) - 1.0
    return {
        "e_l": e_l,
        "e_t": e_t,
        "g_lt": g_lt,
        "g_tt": g_tt,
        "nu_lt": nu_lt,
        "nu_tt": nu_tt,
    }


def chamis_stiffness_from_features(
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized Chamis ``(N, 6, 6)`` stiffness from surrogate feature rows."""
    ec = chamis_engineering_constants_from_features(features)
    return transverse_isotropic_stiffness_batch(
        e_l=ec["e_l"],
        e_t=ec["e_t"],
        g_lt=ec["g_lt"],
        nu_lt=ec["nu_lt"],
        nu_tt=ec["nu_tt"],
    )


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
