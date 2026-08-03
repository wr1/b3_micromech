"""Surrogate feature vectors aligned with ``b3_tex.micromodels.SurrogateModel``."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from b3_micromech.reference import _engineering_constants_isotropic
from b3_micromech.tensors import engineering_constants_transverse_iso


class _StiffnessMaterial(Protocol):
    stiffness: NDArray[np.float64]


def constituent_engineering_constants(
    matrix: _StiffnessMaterial,
    fibre: _StiffnessMaterial,
) -> tuple[float, float, float, float, float, float, float]:
    """Return ``(E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)``."""
    em, num = _engineering_constants_isotropic(matrix.stiffness)
    fc = engineering_constants_transverse_iso(fibre.stiffness)
    return (
        float(em),
        float(num),
        float(fc["e_l"]),
        float(fc["e_t"]),
        float(fc["g_lt"]),
        float(fc["nu_lt"]),
        float(fc["g_tt"]),
    )


def constituent_thermal_properties(
    matrix: object,
    fibre: object,
) -> tuple[float, float, float, float]:
    """Return ``(alpha_m, alpha_Lf, alpha_Tf, k_m)``.

    *alpha_m* – matrix CTE (transverse, index 1 of ``[α,α,α,0,0,0]``).
    *alpha_Lf / alpha_Tf* – fibre longitudinal / transverse CTE.
    *k_m* – matrix isotropic thermal conductivity (``k[0,0]``).

    Materials without thermal tensors (e.g. ``b3_tex.materials.Material``) yield
    zeros so stiffness surrogates keep the original 8-feature mechanical width.
    """
    te_m = getattr(matrix, "thermal_expansion", None)
    te_f = getattr(fibre, "thermal_expansion", None)
    k_m_tensor = getattr(matrix, "thermal_conductivity", None)
    if te_m is None or te_f is None or k_m_tensor is None:
        return 0.0, 0.0, 0.0, 0.0
    a_m = float(np.asarray(te_m, dtype=float).ravel()[1])  # alpha_yy
    a_Lf = float(np.asarray(te_f, dtype=float).ravel()[0])  # alpha_xx (fibre)
    a_Tf = float(np.asarray(te_f, dtype=float).ravel()[1])  # alpha_yy
    k_arr = np.asarray(k_m_tensor, dtype=float)
    k_m = float(k_arr[0, 0]) if k_arr.ndim == 2 else float(k_arr.ravel()[0])
    return a_m, a_Lf, a_Tf, k_m


def build_feature_matrix(
    vf: NDArray[np.float64],
    *,
    matrix: _StiffnessMaterial | None = None,
    fibre: _StiffnessMaterial | None = None,
    E_m: float | None = None,
    nu_m: float | None = None,
    E_Lf: float | None = None,
    E_Tf: float | None = None,
    G_LTf: float | None = None,
    nu_LTf: float | None = None,
    G_TTf: float | None = None,
    alpha_m: float | None = None,
    alpha_Lf: float | None = None,
    alpha_Tf: float | None = None,
    k_m: float | None = None,
) -> NDArray[np.float64]:
    """Build ``(N, 12)`` feature rows for the stiffness / thermal surrogate.

    The first 8 columns are the existing mechanical features
    ``[Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf]``.
    The last 4 columns are the thermal features
    ``[alpha_m, alpha_Lf, alpha_Tf, k_m]``.
    """
    vf_arr = np.asarray(vf, dtype=float).ravel()
    if matrix is not None and fibre is not None:
        em, num, elf, etf, gltf, nultf, gttf = constituent_engineering_constants(
            matrix, fibre
        )
        a_m, a_Lf, a_Tf, k_m_val = constituent_thermal_properties(
            matrix, fibre
        )
        _thermal = bool(abs(a_m) + abs(a_Lf) + abs(a_Tf) + abs(k_m_val) > 0.0)
    else:
        scalars = (E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)
        if any(v is None for v in scalars):
            raise ValueError(
                "provide matrix and fibre, or all seven constituent scalars"
            )
        em, num, elf, etf, gltf, nultf, gttf = (float(v) for v in scalars)  # type: ignore[misc]
        if (alpha_m is None) != (alpha_Lf is None) != (alpha_Tf is None) != (k_m is None):
            raise ValueError(
                "provide all four thermal scalars (alpha_m, alpha_Lf, alpha_Tf, k_m) or None"
            )
        a_m, a_Lf, a_Tf, k_m_val = (
            float(alpha_m) if alpha_m is not None else 0.0,
            float(alpha_Lf) if alpha_Lf is not None else 0.0,
            float(alpha_Tf) if alpha_Tf is not None else 0.0,
            float(k_m) if k_m is not None else 0.0,
        )
        _thermal = any(v is not None for v in (alpha_m, alpha_Lf, alpha_Tf, k_m))
    n = vf_arr.shape[0]
    # Backward-compatible width: thermal columns only when thermal data is
    # actually present — elastic-only callers keep the original (N, 8).
    width = 12 if _thermal else 8
    out = np.empty((n, width), dtype=float)
    out[:, 0] = vf_arr
    out[:, 1] = em
    out[:, 2] = num
    out[:, 3] = elf
    out[:, 4] = etf
    out[:, 5] = gltf
    out[:, 6] = nultf
    out[:, 7] = gttf
    if _thermal:
        out[:, 8] = a_m
        out[:, 9] = a_Lf
        out[:, 10] = a_Tf
        out[:, 11] = k_m_val
    return out


def _build_mech_features_matrix(
    vf: NDArray[np.float64],
    *,
    matrix: _StiffnessMaterial | None = None,
    fibre: _StiffnessMaterial | None = None,
    E_m: float | None = None,
    nu_m: float | None = None,
    E_Lf: float | None = None,
    E_Tf: float | None = None,
    G_LTf: float | None = None,
    nu_LTf: float | None = None,
    G_TTf: float | None = None,
) -> NDArray[np.float64]:
    """Backwards-compatible ``(N, 8)`` feature rows (mechanical only)."""
    return build_feature_matrix(
        vf,
        matrix=matrix,
        fibre=fibre,
        E_m=E_m,
        nu_m=nu_m,
        E_Lf=E_Lf,
        E_Tf=E_Tf,
        G_LTf=G_LTf,
        nu_LTf=nu_LTf,
        G_TTf=G_TTf,
    )[:, :8]


def features_out_of_bounds(
    features: NDArray[np.float64],
    bounds: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Per-row mask: True where any feature lies outside training ``[min, max]`` bounds."""
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] != x.shape[1]:
        raise ValueError(
            f"bounds must have shape ({x.shape[1]}, 2), got {b.shape}"
        )
    below = x < b[:, 0]
    above = x > b[:, 1]
    return np.any(below | above, axis=1)


def warn_if_out_of_bounds(
    features: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    context: str = "batch",
) -> int:
    """Log a warning when features extrapolate beyond training bounds; return OOB count."""
    import warnings

    mask = features_out_of_bounds(features, bounds)
    n_oob = int(mask.sum())
    if n_oob:
        warnings.warn(
            f"{n_oob} of {mask.shape[0]} {context} feature rows lie outside training bounds",
            stacklevel=3,
        )
    return n_oob
