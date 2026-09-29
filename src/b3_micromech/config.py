"""Strict RVE solver configuration.

Unknown keys and unknown enum values raise :class:`ConfigError`. Omitted
material sampling is ``local_cloud`` at resolution 6.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Mapping

SamplingStrategy = Literal["exact", "cell_constant", "local_cloud"]
AmrMarker = Literal["stiffness_jump", "heterogeneity", "combined"]
CellType = Literal["triangle", "quadrilateral"]

_STRATEGIES = ("exact", "cell_constant", "local_cloud")
_MARKERS = ("stiffness_jump", "heterogeneity", "combined")
_CELL_TYPES = ("triangle", "quadrilateral")
_MARKER_ALIASES = {
    "stiffness": "stiffness_jump",
    "jump": "stiffness_jump",
    "material": "heterogeneity",
    "phase": "heterogeneity",
    "both": "combined",
}
_LEGACY_SAMPLING = {
    "quadrature": "exact",
    "exact": "exact",
    "centroid": "cell_constant",
    "cell_constant": "cell_constant",
    "local_cloud": "local_cloud",
    "cloud": "local_cloud",
}


class ConfigError(ValueError):
    """Raised for unknown keys, bad enum values, or out-of-range numbers."""


def _warn_alias(where: str, old: str, new: str) -> None:
    warnings.warn(
        f"{where}: {old!r} is deprecated; use {new!r} (removed in 0.3.0)",
        DeprecationWarning,
        stacklevel=3,
    )


def _strict(
    cls: type,
    raw: Mapping[str, Any] | None,
    where: str,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    data = dict(raw or {})
    for old, new in (aliases or {}).items():
        if old in data:
            _warn_alias(where, old, new)
            data.setdefault(new, data.pop(old))
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}; allowed {sorted(allowed)}"
        )
    return data


@dataclass(frozen=True, kw_only=True)
class MaterialSamplingConfig:
    strategy: SamplingStrategy = "local_cloud"
    resolution: int = 6
    idw_power: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MaterialSamplingConfig:
        data = _strict(cls, raw, "solver.material_sampling")
        strategy = str(data.get("strategy", "local_cloud"))
        if strategy not in _STRATEGIES:
            raise ConfigError(
                "solver.material_sampling.strategy: unknown "
                f"{strategy!r}; allowed {list(_STRATEGIES)}"
            )
        resolution = int(data.get("resolution", 6))
        if resolution < 1:
            raise ConfigError("solver.material_sampling.resolution must be >= 1")
        data["strategy"] = strategy
        data["resolution"] = resolution
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class AmrConfig:
    enabled: bool = False
    max_iterations: int = 3
    threshold: float = 0.15
    dof_budget: int = 50_000
    n_samples_per_cell: int = 64
    marker: AmrMarker = "stiffness_jump"
    n_uniform_refines: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AmrConfig:
        data = _strict(cls, raw, "solver.amr")
        marker = str(data.get("marker", "stiffness_jump"))
        marker = _MARKER_ALIASES.get(marker, marker)
        if marker not in _MARKERS:
            raise ConfigError(
                f"solver.amr.marker: unknown {marker!r}; allowed {list(_MARKERS)}"
            )
        max_iterations = int(data.get("max_iterations", 3))
        threshold = float(data.get("threshold", 0.15))
        dof_budget = int(data.get("dof_budget", 50_000))
        n_samples = int(data.get("n_samples_per_cell", 64))
        n_uniform = int(data.get("n_uniform_refines", 0))
        if max_iterations < 0:
            raise ConfigError("solver.amr.max_iterations must be >= 0")
        if threshold <= 0:
            raise ConfigError("solver.amr.threshold must be > 0")
        if dof_budget <= 0:
            raise ConfigError("solver.amr.dof_budget must be > 0")
        if n_samples < 1:
            raise ConfigError("solver.amr.n_samples_per_cell must be >= 1")
        if n_uniform < 0:
            raise ConfigError("solver.amr.n_uniform_refines must be >= 0")
        data["marker"] = marker
        data["max_iterations"] = max_iterations
        data["threshold"] = threshold
        data["dof_budget"] = dof_budget
        data["n_samples_per_cell"] = n_samples
        data["n_uniform_refines"] = n_uniform
        data["enabled"] = bool(data.get("enabled", False))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class SolverConfig:
    cell_type: CellType | None = None
    material_sampling: MaterialSamplingConfig = field(
        default_factory=MaterialSamplingConfig
    )
    amr: AmrConfig = field(default_factory=AmrConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SolverConfig:
        data = dict(raw or {})
        if "stiffness_sampling" in data:
            legacy = str(data.pop("stiffness_sampling")).lower()
            warnings.warn(
                "solver.stiffness_sampling is deprecated; use "
                "solver.material_sampling.strategy (removed in 0.3.0)",
                DeprecationWarning,
                stacklevel=2,
            )
            mapped = _LEGACY_SAMPLING.get(legacy)
            if mapped is None:
                raise ConfigError(
                    "solver.stiffness_sampling: unknown "
                    f"{legacy!r}; allowed {sorted(_LEGACY_SAMPLING)}"
                )
            sampling = dict(data.get("material_sampling") or {})
            sampling.setdefault("strategy", mapped)
            data["material_sampling"] = sampling
        parsed = _strict(cls, data, "solver")
        cell_type = parsed.get("cell_type")
        if cell_type is not None:
            cell_type = str(cell_type)
            if cell_type not in _CELL_TYPES:
                raise ConfigError(
                    f"solver.cell_type: unknown {cell_type!r}; allowed {list(_CELL_TYPES)}"
                )
        return cls(
            cell_type=cell_type,
            material_sampling=MaterialSamplingConfig.from_mapping(
                parsed.get("material_sampling")
            ),
            amr=AmrConfig.from_mapping(parsed.get("amr")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-safe form used by metadata and cache keys."""
        return {
            "cell_type": self.cell_type,
            "material_sampling": self.material_sampling.to_dict(),
            "amr": self.amr.to_dict(),
        }
