"""Disk cache for FEA / surrogate LUT stiffness batches."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from b3_micromech.contract import Constituents

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    """LUT cache root (override with env ``B3_MICROMECH_LUT_CACHE``)."""
    raw = os.environ.get("B3_MICROMECH_LUT_CACHE")
    if raw:
        return Path(raw)
    return Path("results") / "fea_lut_cache"


def rve_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash domain / solver settings that affect homogenization."""
    subset = {k: cfg[k] for k in ("domain", "periodic_tolerance", "solver") if k in cfg}
    payload = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def constituent_fingerprint(matrix: Any, fibre: Any) -> str:
    """Hash the seven surrogate feature moduli extracted from constituents."""
    c = Constituents.from_materials(matrix, fibre)
    vals = (c.E_m, c.nu_m, c.E_Lf, c.E_Tf, c.G_LTf, c.nu_LTf, c.G_TTf)
    rounded = [float(f"{v:.8g}") for v in vals]
    payload = json.dumps(rounded, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def lut_cache_key(
    rve_base: dict[str, Any],
    matrix: Any,
    fibre: Any,
    vf: NDArray[np.float64],
    *,
    mode: str,
    surrogate_path: str | Path | None = None,
) -> str:
    """Stable filename stem for a batch LUT entry."""
    vf_arr = np.asarray(vf, dtype=float).ravel()
    rve_fp = rve_config_fingerprint(rve_base)
    const_fp = constituent_fingerprint(matrix, fibre)
    vf_hash = hashlib.sha256(vf_arr.tobytes()).hexdigest()[:16]
    mode_part = mode
    if mode == "surrogate" and surrogate_path is not None:
        path = Path(surrogate_path)
        if path.is_file():
            stat = path.stat()
            mode_part = f"surrogate_{stat.st_mtime_ns}_{stat.st_size}"
    return f"{rve_fp}_{const_fp}_{mode_part}_{vf_hash}"


def load_lut_cache(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load ``(vf, C)`` from a compressed npz cache file."""
    with np.load(path) as data:
        vf = np.asarray(data["vf"], dtype=float)
        stiffness = np.asarray(data["C"], dtype=float)
    return vf, stiffness


def save_lut_cache(
    path: Path,
    vf: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write ``vf`` and ``(N, 6, 6)`` stiffness to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, vf=np.asarray(vf, dtype=float), C=np.asarray(stiffness, dtype=float)
    )
    if metadata is not None:
        meta = dict(metadata)
        meta.setdefault("version", CACHE_VERSION)
        meta.setdefault("saved_at", datetime.now(UTC).isoformat())
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
