"""Constituent → feature and RVE-config contract.

One ``(matrix, fibre, Vf)`` mapping for sweeps, surrogates, and FEA.
``b3_micromech`` does not import ``b3_tex``; drift is checked by the parity test.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from b3_micromech.tensors import (
    engineering_constants_transverse_iso,
    isotropic_stiffness,
    transverse_isotropic_stiffness,
)

# fmt: off
FEATURE_NAMES: tuple[str, ...] = ("vf", "E_m", "nu_m", "E_Lf", "E_Tf", "G_LTf", "nu_LTf", "G_TTf")
# fmt: on
THERMAL_FEATURE_NAMES: tuple[str, ...] = ("alpha_m", "alpha_Lf", "alpha_Tf", "k_m")
UNITS = {
    "E_*": "Pa",
    "G_*": "Pa",
    "nu_*": "1",
    "vf": "1 (in-tow)",
    "alpha_*": "1/K",
    "k_*": "W/(m K)",
}
VOIGT_ORDER = ("11", "22", "33", "23", "13", "12")  # engineering shear; 1 = fibre = x

_MECH_NAMES = tuple(name for name in FEATURE_NAMES if name != "vf")
_POINT_KEYS = set(FEATURE_NAMES) | set(THERMAL_FEATURE_NAMES) | {"nu_TTf"}


class ContractError(ValueError):
    """Constituent tensor, feature, or sweep-point contract was violated."""


class StiffnessMaterial(Protocol):
    """Satisfied by ``b3_micromech`` and ``b3_tex`` materials."""

    name: str

    @property
    def stiffness(self) -> NDArray[np.float64]: ...


def _positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ContractError(f"{name} must be positive, got {value}")


def _close(
    reference: NDArray[np.float64], candidate: NDArray[np.float64], rtol: float
) -> bool:
    scale = float(np.max(np.abs(reference)))
    return bool(
        np.allclose(candidate, reference, rtol=0.0, atol=rtol * max(scale, 1e-30))
    )


def _thermal_from_materials(
    matrix: StiffnessMaterial, fibre: StiffnessMaterial
) -> tuple[float | None, float | None, float | None, float | None]:
    te_m = getattr(matrix, "thermal_expansion", None)
    te_f = getattr(fibre, "thermal_expansion", None)
    k_m_tensor = getattr(matrix, "thermal_conductivity", None)
    if te_m is None or te_f is None or k_m_tensor is None:
        return None, None, None, None
    alpha_m = float(np.asarray(te_m, dtype=float).ravel()[1])
    alpha_lf = float(np.asarray(te_f, dtype=float).ravel()[0])
    alpha_tf = float(np.asarray(te_f, dtype=float).ravel()[1])
    k_arr = np.asarray(k_m_tensor, dtype=float)
    k_m = float(k_arr[0, 0]) if k_arr.ndim == 2 else float(k_arr.ravel()[0])
    if abs(alpha_m) + abs(alpha_lf) + abs(alpha_tf) + abs(k_m) == 0.0:
        return None, None, None, None
    return alpha_m, alpha_lf, alpha_tf, k_m


@dataclass(frozen=True, kw_only=True)
class Constituents:
    E_m: float
    nu_m: float
    E_Lf: float
    E_Tf: float
    G_LTf: float
    nu_LTf: float
    G_TTf: float
    alpha_m: float | None = None
    alpha_Lf: float | None = None
    alpha_Tf: float | None = None
    k_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("E_m", "E_Lf", "E_Tf", "G_LTf", "G_TTf"):
            _positive(name, float(getattr(self, name)))
        if not -1.0 < self.nu_m < 0.5:
            raise ContractError(f"nu_m must be in (-1, 0.5), got {self.nu_m}")
        nu_tt = self.nu_TTf
        try:
            stiffness = transverse_isotropic_stiffness(
                e_l=self.E_Lf,
                e_t=self.E_Tf,
                g_lt=self.G_LTf,
                nu_lt=self.nu_LTf,
                nu_tt=nu_tt,
            )
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if np.any(np.linalg.eigvalsh(stiffness) <= 0.0):
            raise ContractError(
                "fibre transverse-isotropic stiffness is not positive definite"
            )

    @property
    def nu_TTf(self) -> float:
        """Transverse fibre Poisson ratio. Derived, never stored."""
        return self.E_Tf / (2.0 * self.G_TTf) - 1.0

    @classmethod
    def from_materials(
        cls,
        matrix: StiffnessMaterial,
        fibre: StiffnessMaterial,
        *,
        rtol: float = 1e-6,
    ) -> Constituents:
        """Extract constants.

        Raises :class:`ContractError` unless ``matrix`` matches
        ``isotropic_stiffness`` and ``fibre`` matches
        ``transverse_isotropic_stiffness`` about global x, each within
        ``rtol * max|C|``.
        """
        from b3_micromech.reference import _engineering_constants_isotropic

        em, num = _engineering_constants_isotropic(
            np.asarray(matrix.stiffness, dtype=float)
        )
        matrix_c = np.asarray(matrix.stiffness, dtype=float)
        rebuilt_m = isotropic_stiffness(em, num)
        if matrix_c.shape != (6, 6) or not _close(matrix_c, rebuilt_m, rtol):
            raise ContractError(
                "matrix stiffness is not isotropic about the engineering constants"
            )
        fibre_c = np.asarray(fibre.stiffness, dtype=float)
        if fibre_c.shape != (6, 6):
            raise ContractError(
                f"fibre stiffness must have shape (6, 6), got {fibre_c.shape}"
            )
        try:
            fc = engineering_constants_transverse_iso(fibre_c)
            rebuilt_f = transverse_isotropic_stiffness(
                e_l=float(fc["e_l"]),
                e_t=float(fc["e_t"]),
                g_lt=float(fc["g_lt"]),
                nu_lt=float(fc["nu_lt"]),
                nu_tt=float(fc["nu_tt"]),
            )
        except ValueError as exc:
            raise ContractError(
                "fibre stiffness is not transverse-isotropic about global x"
            ) from exc
        if not _close(fibre_c, rebuilt_f, rtol):
            raise ContractError(
                "fibre stiffness is not transverse-isotropic about global x"
            )
        alpha_m, alpha_lf, alpha_tf, k_m = _thermal_from_materials(matrix, fibre)
        return cls(
            E_m=float(em),
            nu_m=float(num),
            E_Lf=float(fc["e_l"]),
            E_Tf=float(fc["e_t"]),
            G_LTf=float(fc["g_lt"]),
            nu_LTf=float(fc["nu_lt"]),
            G_TTf=float(fc["g_tt"]),
            alpha_m=alpha_m,
            alpha_Lf=alpha_lf,
            alpha_Tf=alpha_tf,
            k_m=k_m,
        )

    @classmethod
    def from_mapping(cls, point: Mapping[str, float]) -> Constituents:
        """Build from feature keys. ``vf`` is ignored. ``nu_TTf`` derives ``G_TTf``."""
        data = {str(key): float(value) for key, value in point.items() if key != "vf"}
        if "nu_TTf" in data and "G_TTf" in data:
            raise ContractError("a sweep point may not carry both nu_TTf and G_TTf")
        if "nu_TTf" in data:
            nu_tt = data.pop("nu_TTf")
            if "E_Tf" not in data:
                raise ContractError("nu_TTf requires E_Tf")
            data["G_TTf"] = float(data["E_Tf"]) / (2.0 * (1.0 + nu_tt))
        missing = [name for name in _MECH_NAMES if name not in data]
        if missing:
            raise ContractError(f"missing constituent keys {missing}")
        unknown = sorted(set(data) - set(_MECH_NAMES) - set(THERMAL_FEATURE_NAMES))
        if unknown:
            raise ContractError(f"unknown constituent key(s) {unknown}")
        thermal = {name: data[name] for name in THERMAL_FEATURE_NAMES if name in data}
        return cls(
            E_m=data["E_m"],
            nu_m=data["nu_m"],
            E_Lf=data["E_Lf"],
            E_Tf=data["E_Tf"],
            G_LTf=data["G_LTf"],
            nu_LTf=data["nu_LTf"],
            G_TTf=data["G_TTf"],
            **thermal,
        )

    def feature_matrix(
        self,
        vf: ArrayLike,
        names: Sequence[str] = FEATURE_NAMES,
    ) -> NDArray[np.float64]:
        """``(N, len(names))``. A requested thermal name that is unset raises."""
        vf_arr = np.atleast_1d(np.asarray(vf, dtype=float)).ravel()
        columns: dict[str, NDArray[np.float64] | float | None] = {
            "vf": vf_arr,
            "E_m": self.E_m,
            "nu_m": self.nu_m,
            "E_Lf": self.E_Lf,
            "E_Tf": self.E_Tf,
            "G_LTf": self.G_LTf,
            "nu_LTf": self.nu_LTf,
            "G_TTf": self.G_TTf,
            "alpha_m": self.alpha_m,
            "alpha_Lf": self.alpha_Lf,
            "alpha_Tf": self.alpha_Tf,
            "k_m": self.k_m,
        }
        out = np.empty((vf_arr.shape[0], len(names)), dtype=float)
        for index, name in enumerate(names):
            if name not in columns:
                raise ContractError(f"unknown feature {name!r}")
            value = columns[name]
            if value is None:
                raise ContractError(
                    f"feature {name!r} is not set on these constituents"
                )
            out[:, index] = value
        return out


def defaults_from(template: Mapping[str, Any]) -> dict[str, float]:
    """Unswept constants from the template materials named by ``rve``."""
    from b3_micromech.problem import RVEProblem

    problem = RVEProblem.from_config(dict(template))
    constituents = Constituents.from_materials(
        problem.materials[problem.matrix_material],
        problem.materials[problem.fibre_material],
    )
    values: dict[str, float] = {
        "vf": float(problem.fibre_volume_fraction),
        "E_m": constituents.E_m,
        "nu_m": constituents.nu_m,
        "E_Lf": constituents.E_Lf,
        "E_Tf": constituents.E_Tf,
        "G_LTf": constituents.G_LTf,
        "nu_LTf": constituents.nu_LTf,
        "G_TTf": constituents.G_TTf,
    }
    for name in THERMAL_FEATURE_NAMES:
        thermal = getattr(constituents, name)
        if thermal is not None:
            values[name] = float(thermal)
    return values


def constituents_for_point(
    template: Mapping[str, Any], point: Mapping[str, float]
) -> tuple[Constituents, float]:
    """Merge a sweep point over the template defaults."""
    unknown = sorted(set(point) - _POINT_KEYS)
    if unknown:
        raise ContractError(f"unknown sweep key(s) {unknown}")
    if "nu_TTf" in point and "G_TTf" in point:
        raise ContractError("a sweep point may not carry both nu_TTf and G_TTf")
    merged = dict(defaults_from(template))
    merged.update({key: float(value) for key, value in point.items()})
    if "nu_TTf" in point:
        merged.pop("G_TTf", None)
    vf = float(merged.pop("vf"))
    return Constituents.from_mapping(merged), vf


def _material_entry(materials: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for entry in materials:
        if str(entry.get("name")) == name:
            return entry
    raise ContractError(f"template has no material named {name!r}")


def rve_config(
    template: Mapping[str, Any], constituents: Constituents, vf: float
) -> dict[str, Any]:
    """Deep-copy ``template`` and write ``vf`` plus the constituent constants.

    Matrix and fibre entries are the ones named by
    ``template["rve"]["matrix_material"]`` and ``fibre_material``. Thermal
    constants overwrite the template only when they are set.
    """
    cfg = deepcopy(dict(template))
    rve = dict(cfg.get("rve") or {})
    matrix_name = str(rve["matrix_material"])
    fibre_name = str(rve["fibre_material"])
    rve["fibre_volume_fraction"] = float(vf)
    rve.pop("fibre_radius", None)
    cfg["rve"] = rve
    materials = [dict(entry) for entry in cfg["materials"]]
    matrix = _material_entry(materials, matrix_name)
    fibre = _material_entry(materials, fibre_name)
    matrix["youngs_modulus"] = float(constituents.E_m)
    matrix["poisson_ratio"] = float(constituents.nu_m)
    fibre["e_l"] = float(constituents.E_Lf)
    fibre["e_t"] = float(constituents.E_Tf)
    fibre["g_lt"] = float(constituents.G_LTf)
    fibre["nu_lt"] = float(constituents.nu_LTf)
    fibre["nu_tt"] = float(constituents.nu_TTf)
    fibre.pop("g_tt", None)
    if constituents.alpha_m is not None:
        matrix["thermal_expansion"] = float(constituents.alpha_m)
    if constituents.k_m is not None:
        matrix["thermal_conductivity"] = float(constituents.k_m)
    if constituents.alpha_Lf is not None:
        fibre["alpha_l"] = float(constituents.alpha_Lf)
    if constituents.alpha_Tf is not None:
        fibre["alpha_t"] = float(constituents.alpha_Tf)
    cfg["materials"] = materials
    return cfg
