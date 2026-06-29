"""Voigt utilities — same conventions as b3_tex."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VOIGT_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)


def isotropic_stiffness(
    youngs_modulus: float, poisson_ratio: float
) -> NDArray[np.float64]:
    if youngs_modulus <= 0:
        raise ValueError("youngs_modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in (-1, 0.5)")
    lam = (
        youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    c = np.zeros((6, 6), dtype=float)
    c[0:3, 0:3] = lam
    for i in range(3):
        c[i, i] = lam + 2 * mu
    for i in range(3, 6):
        c[i, i] = mu
    return c


def orthotropic_stiffness(
    *,
    e1: float,
    e2: float,
    e3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    g12: float,
    g13: float,
    g23: float,
) -> NDArray[np.float64]:
    nu21 = nu12 * e2 / e1
    nu31 = nu13 * e3 / e1
    nu32 = nu23 * e3 / e2
    compliance = np.array(
        [
            [1 / e1, -nu21 / e2, -nu31 / e3, 0, 0, 0],
            [-nu12 / e1, 1 / e2, -nu32 / e3, 0, 0, 0],
            [-nu13 / e1, -nu23 / e2, 1 / e3, 0, 0, 0],
            [0, 0, 0, 1 / g23, 0, 0],
            [0, 0, 0, 0, 1 / g13, 0],
            [0, 0, 0, 0, 0, 1 / g12],
        ],
        dtype=float,
    )
    return np.linalg.inv(compliance)


def transverse_isotropic_stiffness(
    *,
    e_l: float,
    e_t: float,
    g_lt: float,
    nu_lt: float,
    nu_tt: float,
) -> NDArray[np.float64]:
    g_tt = e_t / (2 * (1 + nu_tt))
    return orthotropic_stiffness(
        e1=e_l,
        e2=e_t,
        e3=e_t,
        nu12=nu_lt,
        nu13=nu_lt,
        nu23=nu_tt,
        g12=g_lt,
        g13=g_lt,
        g23=g_tt,
    )


def voigt_b_matrix_plane_strain_x(
    dshape: ArrayLike, *, ordering: str = "byNODES"
) -> NDArray[np.float64]:
    """B matrix for plane strain along global x on a 2D y–z mesh.

    ``dshape`` is ``(nd, 2)`` with columns ``[d/dy, d/dz]``. Fluctuating
    ``ε_xx`` is not generated from ``u`` (row 0 is zero); macro ``ε_xx``
    enters through the periodic RHS / recovery path.
    """
    d = np.asarray(dshape, dtype=float)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(f"dshape must have shape (nd, 2), got {d.shape}")
    nd = d.shape[0]
    dy = d[:, 0]
    dz = d[:, 1]

    if ordering == "byNODES":
        cx = np.arange(nd, dtype=np.intp)
        cy = cx + nd
        cz = cx + 2 * nd
    elif ordering == "byVDIM":
        cx = np.arange(nd, dtype=np.intp) * 3
        cy = cx + 1
        cz = cx + 2
    else:
        raise ValueError(f"unknown ordering {ordering!r}")

    B = np.zeros((6, nd * 3), dtype=float)
    B[1, cy] = dy
    B[2, cz] = dz
    B[3, cy] = dz
    B[3, cz] = dy
    B[4, cx] = dz
    B[5, cx] = dy
    return B


def engineering_constants_transverse_iso(
    stiffness: NDArray[np.float64],
) -> dict[str, float]:
    C = np.asarray(stiffness, dtype=float)
    n = C[0, 0]
    k = 0.5 * (C[1, 1] + C[1, 2])
    m = 0.5 * (C[1, 1] - C[1, 2])
    l = C[0, 1]
    p = C[5, 5]
    e_l = n - l * l / k
    nu_lt = l / (2.0 * k)
    g_lt = p
    e_t = 1.0 / (1.0 / (4.0 * k) + 1.0 / (4.0 * m) + nu_lt * nu_lt / e_l)
    nu_tt = (e_t / (2.0 * m)) - 1.0
    return {
        "e_l": e_l,
        "e_t": e_t,
        "g_lt": g_lt,
        "nu_lt": nu_lt,
        "nu_tt": nu_tt,
        "g_tt": m,
    }


def voigt_strain_to_tensor(strain_voigt: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(strain_voigt, dtype=float)
    if v.shape != (6,):
        raise ValueError(f"strain must have shape (6,), got {v.shape}")
    return np.array(
        [
            [v[0], v[5] / 2, v[4] / 2],
            [v[5] / 2, v[1], v[3] / 2],
            [v[4] / 2, v[3] / 2, v[2]],
        ],
        dtype=float,
    )


def macro_displacement_at_yz(
    strain_voigt: ArrayLike, points_yz: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Affine displacement ``u = eps @ [0, y, z]`` for plane-strain slice at x=0."""
    eps = voigt_strain_to_tensor(strain_voigt)
    pts = np.asarray(points_yz, dtype=float)
    xyz = np.column_stack([np.zeros(len(pts)), pts[:, 0], pts[:, 1]])
    return xyz @ eps.T


def von_mises_voigt(stress_voigt: ArrayLike) -> float:
    s = np.asarray(stress_voigt, dtype=float)
    s11, s22, s33 = s[0], s[1], s[2]
    s23, s13, s12 = s[3], s[4], s[5]
    return float(
        np.sqrt(
            0.5
            * (
                (s11 - s22) ** 2
                + (s22 - s33) ** 2
                + (s33 - s11) ** 2
                + 6 * (s23**2 + s13**2 + s12**2)
            )
        )
    )


def grad_to_voigt_strain_plane_strain_x(
    grad_u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """(N, 3, 2) grad(u) on y–z → (N, 6) Voigt strain; ε_xx row is zero."""
    if grad_u.ndim != 3 or grad_u.shape[1:] != (3, 2):
        raise ValueError(f"grad_u must have shape (N, 3, 2), got {grad_u.shape}")
    n = grad_u.shape[0]
    out = np.zeros((n, 6), dtype=float)
    out[:, 1] = grad_u[:, 1, 0]
    out[:, 2] = grad_u[:, 2, 1]
    out[:, 3] = grad_u[:, 1, 1] + grad_u[:, 2, 0]
    out[:, 4] = grad_u[:, 0, 1]
    out[:, 5] = grad_u[:, 0, 0]
    return out
