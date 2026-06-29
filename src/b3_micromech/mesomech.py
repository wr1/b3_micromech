"""Vectorized FEA surrogate adapter for ``b3_tex`` mesomechanics."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_micromech.features import (
    build_feature_matrix,
    constituent_engineering_constants,
    warn_if_out_of_bounds,
)
from b3_micromech.lut_cache import (
    default_cache_dir,
    load_lut_cache,
    lut_cache_key,
    save_lut_cache,
)
from b3_micromech.surrogate import StiffnessSurrogate

logger = logging.getLogger(__name__)

DEFAULT_RVE_YAML = Path("examples/sweep_hex_hypercube.yaml")


def _load_rve_base(rve_yaml: str | Path) -> dict[str, Any]:
    with open(rve_yaml, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sweep_point_from_materials(vf: float, matrix: Any, fibre: Any) -> dict[str, float]:
    em, num, elf, etf, gltf, nultf, gttf = constituent_engineering_constants(
        matrix, fibre
    )
    return {
        "vf": float(vf),
        "E_m": em,
        "nu_m": num,
        "E_Lf": elf,
        "E_Tf": etf,
        "G_LTf": gltf,
        "nu_LTf": nultf,
        "G_TTf": gttf,
    }


def _homogenize_vf_batch(
    rve_base: dict[str, Any],
    matrix: Any,
    fibre: Any,
    vf: NDArray[np.float64],
    *,
    n_jobs: int = 1,
) -> NDArray[np.float64]:
    """MFEM homogenize one stiffness tensor per Vf sample."""
    from b3_micromech.homogenize import homogenize
    from b3_micromech.sweep import problem_from_sweep_point

    vf_arr = np.asarray(vf, dtype=float).ravel()

    def _one(value: float) -> NDArray[np.float64]:
        point = _sweep_point_from_materials(value, matrix, fibre)
        problem = problem_from_sweep_point(rve_base, point)
        return homogenize(problem).effective_stiffness

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            tensors = Parallel(n_jobs=n_jobs)(delayed(_one)(float(v)) for v in vf_arr)
        except ImportError as exc:
            raise ImportError(
                "parallel FEA batch requires joblib; pip install b3-micromech[sweep]"
            ) from exc
    else:
        tensors = [_one(float(v)) for v in vf_arr]
    return np.stack(tensors, axis=0)


@dataclass
class FeaMicromechMicromodel:
    """Surrogate and/or FEA micromodel with memory + disk LUT caching."""

    name: str
    rve_base: dict[str, Any]
    model: StiffnessSurrogate | None = None
    n_jobs: int = 1
    cache_dir: Path | None = None
    disk_cache: bool = True
    surrogate_path: str | Path | None = None
    _mem_cache: dict[str, NDArray[np.float64]] = field(default_factory=dict, repr=False)

    @property
    def mode(self) -> str:
        return "surrogate" if self.model is not None else "fea"

    def stiffness(
        self,
        *,
        matrix: Any,
        fibre: Any,
        fibre_volume_fraction: float,
    ) -> NDArray[np.float64]:
        return self.stiffness_batch(
            matrix=matrix,
            fibre=fibre,
            vf=np.array([fibre_volume_fraction], dtype=float),
        )[0]

    def stiffness_batch(
        self,
        *,
        matrix: Any,
        fibre: Any,
        vf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        vf_arr = np.asarray(vf, dtype=float).ravel()
        cache_id = lut_cache_key(
            self.rve_base,
            matrix,
            fibre,
            vf_arr,
            mode=self.mode,
            surrogate_path=self.surrogate_path,
        )

        if cache_id in self._mem_cache:
            return self._mem_cache[cache_id]

        cache_path: Path | None = None
        if self.disk_cache and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{cache_id}.npz"
            if cache_path.is_file():
                vf_loaded, stiffness = load_lut_cache(cache_path)
                if vf_loaded.shape == vf_arr.shape and np.allclose(vf_loaded, vf_arr):
                    self._mem_cache[cache_id] = stiffness
                    logger.info("FEA LUT disk cache hit: %s", cache_path.name)
                    return stiffness

        if self.model is not None:
            features = build_feature_matrix(vf_arr, matrix=matrix, fibre=fibre)
            warn_if_out_of_bounds(
                features, self.model.feature_bounds, context="stiffness_batch"
            )
            stiffness = self.model.predict(features)
        else:
            logger.info(
                "FEA homogenization batch: N=%d solves (n_jobs=%d)",
                vf_arr.shape[0],
                self.n_jobs,
            )
            stiffness = _homogenize_vf_batch(
                self.rve_base,
                matrix,
                fibre,
                vf_arr,
                n_jobs=self.n_jobs,
            )

        self._mem_cache[cache_id] = stiffness
        if self.disk_cache and cache_path is not None:
            save_lut_cache(
                cache_path,
                vf_arr,
                stiffness,
                metadata={
                    "mode": self.mode,
                    "name": self.name,
                    "n_points": int(vf_arr.shape[0]),
                },
            )
            logger.info("FEA LUT batch cached to %s", cache_path)
        return stiffness


FeaSurrogateMicromodel = FeaMicromechMicromodel


def fea_micromech_model(
    *,
    name: str = "fea_hex",
    rve_yaml: str | Path = DEFAULT_RVE_YAML,
    surrogate_path: str | Path | None = None,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
    disk_cache: bool | None = None,
) -> FeaMicromechMicromodel:
    """Build a micromodel using a surrogate and/or FEA fallback."""
    rve_base = _load_rve_base(rve_yaml)
    model: StiffnessSurrogate | None = None
    resolved_path: Path | None = None

    if surrogate_path is not None:
        resolved_path = Path(surrogate_path)
        if resolved_path.is_file():
            model = StiffnessSurrogate.load(resolved_path)
        else:
            warnings.warn(
                f"surrogate not found at {surrogate_path}; using FEA on-the-fly",
                stacklevel=2,
            )

    if disk_cache is None:
        disk_cache = model is None

    return FeaMicromechMicromodel(
        name=name,
        rve_base=rve_base,
        model=model,
        n_jobs=n_jobs,
        cache_dir=Path(cache_dir) if cache_dir is not None else default_cache_dir(),
        disk_cache=disk_cache,
        surrogate_path=resolved_path if model is not None else surrogate_path,
    )


def fea_surrogate_from_joblib(
    path: str | Path,
    *,
    name: str = "fea_surrogate",
    rve_yaml: str | Path = DEFAULT_RVE_YAML,
) -> FeaMicromechMicromodel:
    """Load a trained surrogate and wrap it for mesomechanics."""
    return fea_micromech_model(
        name=name,
        rve_yaml=rve_yaml,
        surrogate_path=path,
        disk_cache=False,
    )


def register_fea_micromech(
    surrogate_path: str | Path | None = None,
    *,
    name: str = "fea_hex",
    rve_yaml: str | Path = DEFAULT_RVE_YAML,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
    disk_cache: bool | None = None,
) -> FeaMicromechMicromodel:
    """Register a FEA micromodel in ``b3_tex.micromodels.MICROMODELS``."""
    from b3_tex.micromodels import register_micromodel

    micromodel = fea_micromech_model(
        name=name,
        rve_yaml=rve_yaml,
        surrogate_path=surrogate_path,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        disk_cache=disk_cache,
    )
    register_micromodel(micromodel)
    logger.info("registered micromodel %r (mode=%s)", name, micromodel.mode)
    return micromodel


def register_fea_surrogate(
    path: str | Path,
    *,
    name: str = "fea_surrogate",
    rve_yaml: str | Path = DEFAULT_RVE_YAML,
) -> FeaMicromechMicromodel:
    """Register a trained surrogate (requires existing joblib file)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"surrogate model not found: {resolved}")
    return register_fea_micromech(
        surrogate_path=resolved,
        name=name,
        rve_yaml=rve_yaml,
        disk_cache=False,
    )


def predict_stiffness_batch(
    model: StiffnessSurrogate | str | Path | FeaMicromechMicromodel,
    vf: NDArray[np.float64],
    matrix: Any,
    fibre: Any,
    *,
    warn_oob: bool = True,
) -> NDArray[np.float64]:
    """Predict ``(N, 6, 6)`` stiffness for a 1-D Vf array at fixed constituents."""
    if isinstance(model, FeaMicromechMicromodel):
        return model.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    surrogate = (
        model
        if isinstance(model, StiffnessSurrogate)
        else StiffnessSurrogate.load(model)
    )
    vf_arr = np.asarray(vf, dtype=float).ravel()
    features = build_feature_matrix(vf_arr, matrix=matrix, fibre=fibre)
    if warn_oob:
        warn_if_out_of_bounds(
            features, surrogate.feature_bounds, context="predict_stiffness_batch"
        )
    return surrogate.predict(features)


def predict_from_features(
    model: StiffnessSurrogate | str | Path,
    features: NDArray[np.float64],
    *,
    warn_oob: bool = True,
) -> NDArray[np.float64]:
    """Predict stiffness from an ``(N, 8)`` feature matrix."""
    surrogate = (
        model
        if isinstance(model, StiffnessSurrogate)
        else StiffnessSurrogate.load(model)
    )
    x = np.asarray(features, dtype=float)
    if warn_oob:
        warn_if_out_of_bounds(
            x, surrogate.feature_bounds, context="predict_from_features"
        )
    return surrogate.predict(x)


def constituents_from_yaml(config_path: str | Path) -> tuple[Any, Any]:
    """Load matrix and fibre materials from an RVE or sweep YAML base config."""
    from b3_micromech.problem import RVEProblem

    problem = RVEProblem.from_yaml(config_path)
    return (
        problem.materials[problem.matrix_material],
        problem.materials[problem.fibre_material],
    )
