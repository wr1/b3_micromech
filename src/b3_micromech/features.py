"""Surrogate feature vectors aligned with ``b3_tex.micromodels.SurrogateModel``."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from b3_micromech.contract import (
    FEATURE_NAMES,
    THERMAL_FEATURE_NAMES,
    Constituents,
    ContractError,
    StiffnessMaterial,
)


def _constituents_from_call(
    *,
    matrix: StiffnessMaterial | None,
    fibre: StiffnessMaterial | None,
    E_m: float | None,
    nu_m: float | None,
    E_Lf: float | None,
    E_Tf: float | None,
    G_LTf: float | None,
    nu_LTf: float | None,
    G_TTf: float | None,
    alpha_m: float | None,
    alpha_Lf: float | None,
    alpha_Tf: float | None,
    k_m: float | None,
) -> Constituents:
    if matrix is not None and fibre is not None:
        return Constituents.from_materials(matrix, fibre)
    scalars = (E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)
    if any(value is None for value in scalars):
        raise ContractError(
            "provide matrix and fibre, or all seven constituent scalars"
        )
    thermal = (alpha_m, alpha_Lf, alpha_Tf, k_m)
    if any(value is not None for value in thermal) and not all(
        value is not None for value in thermal
    ):
        raise ContractError(
            "provide all four thermal scalars (alpha_m, alpha_Lf, alpha_Tf, k_m) or none"
        )
    return Constituents(
        E_m=float(E_m),  # type: ignore[arg-type]
        nu_m=float(nu_m),  # type: ignore[arg-type]
        E_Lf=float(E_Lf),  # type: ignore[arg-type]
        E_Tf=float(E_Tf),  # type: ignore[arg-type]
        G_LTf=float(G_LTf),  # type: ignore[arg-type]
        nu_LTf=float(nu_LTf),  # type: ignore[arg-type]
        G_TTf=float(G_TTf),  # type: ignore[arg-type]
        alpha_m=None if alpha_m is None else float(alpha_m),
        alpha_Lf=None if alpha_Lf is None else float(alpha_Lf),
        alpha_Tf=None if alpha_Tf is None else float(alpha_Tf),
        k_m=None if k_m is None else float(k_m),
    )


def _names_for(constituents: Constituents) -> tuple[str, ...]:
    thermal = [getattr(constituents, name) for name in THERMAL_FEATURE_NAMES]
    if all(value is not None for value in thermal):
        return FEATURE_NAMES + THERMAL_FEATURE_NAMES
    if any(value is not None for value in thermal):
        raise ContractError(
            "provide all four thermal scalars (alpha_m, alpha_Lf, alpha_Tf, k_m) or none"
        )
    return FEATURE_NAMES


def constituent_engineering_constants(
    matrix: StiffnessMaterial,
    fibre: StiffnessMaterial,
) -> tuple[float, float, float, float, float, float, float]:
    """Return ``(E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf)``.

    .. deprecated:: 0.2.0
        Use :meth:`Constituents.from_materials`. Removed in 0.3.0.
    """
    warnings.warn(
        "constituent_engineering_constants is deprecated; use "
        "Constituents.from_materials (removed in 0.3.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    c = Constituents.from_materials(matrix, fibre)
    return (c.E_m, c.nu_m, c.E_Lf, c.E_Tf, c.G_LTf, c.nu_LTf, c.G_TTf)


def constituent_thermal_properties(
    matrix: object,
    fibre: object,
) -> tuple[float, float, float, float]:
    """Return ``(alpha_m, alpha_Lf, alpha_Tf, k_m)``, or zeros when unset.

    .. deprecated:: 0.2.0
        Use :meth:`Constituents.from_materials`. Removed in 0.3.0.
    """
    warnings.warn(
        "constituent_thermal_properties is deprecated; use "
        "Constituents.from_materials (removed in 0.3.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    c = Constituents.from_materials(matrix, fibre)  # type: ignore[arg-type]
    return tuple(
        0.0 if getattr(c, name) is None else float(getattr(c, name))
        for name in THERMAL_FEATURE_NAMES
    )  # type: ignore[return-value]


def build_feature_matrix(
    vf: NDArray[np.float64],
    *,
    matrix: StiffnessMaterial | None = None,
    fibre: StiffnessMaterial | None = None,
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
    """Feature rows. Width is 8, or 12 when every thermal scalar is set.

    .. deprecated:: 0.2.0
        Use :meth:`Constituents.feature_matrix`. Removed in 0.3.0.
    """
    warnings.warn(
        "build_feature_matrix is deprecated; use Constituents.feature_matrix "
        "(removed in 0.3.0)",
        DeprecationWarning,
        stacklevel=2,
    )
    constituents = _constituents_from_call(
        matrix=matrix,
        fibre=fibre,
        E_m=E_m,
        nu_m=nu_m,
        E_Lf=E_Lf,
        E_Tf=E_Tf,
        G_LTf=G_LTf,
        nu_LTf=nu_LTf,
        G_TTf=G_TTf,
        alpha_m=alpha_m,
        alpha_Lf=alpha_Lf,
        alpha_Tf=alpha_Tf,
        k_m=k_m,
    )
    return constituents.feature_matrix(vf, names=_names_for(constituents))


def _build_mech_features_matrix(
    vf: NDArray[np.float64],
    *,
    matrix: StiffnessMaterial | None = None,
    fibre: StiffnessMaterial | None = None,
    E_m: float | None = None,
    nu_m: float | None = None,
    E_Lf: float | None = None,
    E_Tf: float | None = None,
    G_LTf: float | None = None,
    nu_LTf: float | None = None,
    G_TTf: float | None = None,
) -> NDArray[np.float64]:
    """``(N, 8)`` mechanical feature rows."""
    constituents = _constituents_from_call(
        matrix=matrix,
        fibre=fibre,
        E_m=E_m,
        nu_m=nu_m,
        E_Lf=E_Lf,
        E_Tf=E_Tf,
        G_LTf=G_LTf,
        nu_LTf=nu_LTf,
        G_TTf=G_TTf,
        alpha_m=None,
        alpha_Lf=None,
        alpha_Tf=None,
        k_m=None,
    )
    return constituents.feature_matrix(vf, names=FEATURE_NAMES)


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
        raise ValueError(f"bounds must have shape ({x.shape[1]}, 2), got {b.shape}")
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
