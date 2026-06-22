"""Dataset export for surrogate training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


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
    )
    if metadata is not None:
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")