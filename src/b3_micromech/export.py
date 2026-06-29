"""Dataset export and load for surrogate training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SURROGATE_FEATURE_NAMES: tuple[str, ...] = (
    "vf",
    "E_m",
    "nu_m",
    "E_Lf",
    "E_Tf",
    "G_LTf",
    "nu_LTf",
    "G_TTf",
)


def save_dataset(
    path: str | Path,
    *,
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    metadata: dict | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.asarray(features, dtype=float),
        C=np.asarray(stiffness, dtype=float),
        feature_names=np.array(SURROGATE_FEATURE_NAMES),
    )
    if metadata is not None:
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_dataset(
    path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict | None]:
    """Load ``(X, C, metadata)`` from a sweep ``dataset.npz``."""
    out = Path(path)
    with np.load(out) as data:
        features = np.asarray(data["X"], dtype=float)
        stiffness = np.asarray(data["C"], dtype=float)
    meta_path = out.with_suffix(".meta.json")
    metadata = None
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return features, stiffness, metadata
