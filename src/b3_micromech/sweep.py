"""Hypercube parameter sweeps."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from b3_micromech.export import save_dataset
from b3_micromech.homogenize import homogenize, surrogate_features
from b3_micromech.problem import RVEProblem


def _expand_param(spec: dict[str, Any]) -> list[float]:
    if "value" in spec:
        return [float(spec["value"])]
    if "values" in spec:
        return [float(v) for v in spec["values"]]
    if "linspace" in spec:
        start, stop, n = spec["linspace"]
        return [float(x) for x in np.linspace(float(start), float(stop), int(n))]
    raise ValueError(f"unsupported sweep spec: {spec!r}")


def _param_grid(sweep_cfg: dict[str, Any]) -> list[dict[str, float]]:
    keys = [k for k in sweep_cfg if k != "mesh"]
    values = [_expand_param(sweep_cfg[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _apply_point(base: dict[str, Any], point: dict[str, float]) -> RVEProblem:
    cfg = deepcopy(base)
    rve = cfg.setdefault("rve", {})
    materials = {m["name"]: m for m in cfg["materials"]}

    if "vf" in point:
        rve["fibre_volume_fraction"] = point["vf"]
    if "E_m" in point:
        materials["matrix"]["youngs_modulus"] = point["E_m"]
    if "nu_m" in point:
        materials["matrix"]["poisson_ratio"] = point["nu_m"]
    for key, mat_key in (
        ("E_Lf", "e_l"),
        ("E_Tf", "e_t"),
        ("G_LTf", "g_lt"),
        ("nu_LTf", "nu_lt"),
        ("G_TTf", "g_tt"),
    ):
        if key in point:
            materials["fibre"][mat_key] = point[key]
    cfg["materials"] = list(materials.values())

    if "mesh" in cfg.get("sweep", {}):
        mesh = cfg["sweep"]["mesh"]
        cfg.setdefault("domain", {})
        if "resolution" in mesh:
            cfg["domain"]["mesh_resolution"] = mesh["resolution"]
        if "cell_type" in mesh:
            cfg.setdefault("solver", {})
            cfg["solver"]["cell_type"] = mesh["cell_type"]

    return RVEProblem.from_config(cfg)


def run_sweep(
    config_path: str,
    *,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[dict]]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sweep_cfg = cfg.get("sweep")
    if not sweep_cfg:
        raise ValueError("config must contain a top-level 'sweep' block")

    points = _param_grid(sweep_cfg)
    features: list[NDArray[np.float64]] = []
    stiffness: list[NDArray[np.float64]] = []
    records: list[dict] = []

    def _one(point: dict[str, float]) -> tuple[NDArray[np.float64], NDArray[np.float64], dict]:
        problem = _apply_point(cfg, point)
        result = homogenize(problem)
        feat = surrogate_features(problem)
        rec = {"point": point, **result.metadata}
        return feat, result.effective_stiffness, rec

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in points)
        except ImportError as exc:
            raise ImportError("parallel sweep requires joblib; pip install b3-micromech[sweep]") from exc
    else:
        out = [_one(p) for p in points]

    for feat, c, rec in out:
        features.append(feat)
        stiffness.append(c)
        records.append(rec)

    X = np.vstack(features)
    C = np.stack(stiffness)
    return X, C, records


def sweep_to_file(config_path: str, out_path: str, *, n_jobs: int = 1) -> None:
    X, C, records = run_sweep(config_path, n_jobs=n_jobs)
    save_dataset(
        out_path,
        features=X,
        stiffness=C,
        metadata={"source": config_path, "n_points": int(X.shape[0]), "records": records},
    )