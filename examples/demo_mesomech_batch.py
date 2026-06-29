#!/usr/bin/env python3
"""Vectorized FEA micromodel demo for b3_tex mesomechanics.

Registers a surrogate (if available) or FEA-on-the-fly fallback with disk LUT
cache, predicts on a large Vf vector, and builds a micromechanical LUT.

Usage::

    python examples/demo_mesomech_batch.py
    make demo-mesomech-batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

DEFAULT_MODEL = REPO / "results" / "surrogate_demo" / "surrogate_model.joblib"
DEFAULT_CONSTITUENTS = REPO / "examples" / "sweep_hex_hypercube.yaml"
DEFAULT_RVE = REPO / "examples" / "sweep_hex_hypercube.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--constituents", type=Path, default=DEFAULT_CONSTITUENTS)
    ap.add_argument("--rve-yaml", type=Path, default=DEFAULT_RVE)
    ap.add_argument("--name", type=str, default="fea_hex")
    ap.add_argument("--n-vf", type=int, default=100_000)
    ap.add_argument("--vf-lo", type=float, default=0.55)
    ap.add_argument("--vf-hi", type=float, default=0.88)
    ap.add_argument("--fea-jobs", type=int, default=1)
    ap.add_argument(
        "--cache-dir", type=Path, default=REPO / "results" / "fea_lut_cache"
    )
    ap.add_argument("--no-disk-cache", action="store_true")
    args = ap.parse_args()

    pytest = __import__("pytest")
    pytest.importorskip("sklearn")
    pytest.importorskip("b3_tex")

    from b3_micromech.mesomech import (
        constituents_from_yaml,
        predict_stiffness_batch,
        register_fea_micromech,
    )
    from b3_tex.materials import MicromechanicalMaterial
    from b3_tex.micromodels import get_micromodel

    surrogate_path = args.model if args.model.is_file() else None
    if surrogate_path is None:
        print(f"no surrogate at {args.model}; using FEA on-the-fly + disk cache")

    micromodel = register_fea_micromech(
        surrogate_path,
        name=args.name,
        rve_yaml=args.rve_yaml,
        n_jobs=args.fea_jobs,
        cache_dir=args.cache_dir,
        disk_cache=False if args.no_disk_cache else None,
    )
    assert get_micromodel(args.name) is micromodel
    print(f"mode={micromodel.mode}  disk_cache={micromodel.disk_cache}")

    matrix, fibre = constituents_from_yaml(args.constituents)
    vf = np.linspace(args.vf_lo, args.vf_hi, args.n_vf, dtype=float)
    t0 = time.perf_counter()
    stiffness = predict_stiffness_batch(micromodel, vf, matrix, fibre, warn_oob=False)
    elapsed = time.perf_counter() - t0
    print(
        f"batch predict: N={vf.shape[0]}  shape={stiffness.shape}  time={elapsed:.3f}s"
    )

    yarn = MicromechanicalMaterial.from_constituents(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel=micromodel,
        nominal_vf=0.55,
        max_vf=0.90,
    )
    t1 = time.perf_counter()
    centers, table = yarn.build_lut(args.vf_lo, args.vf_hi, n_bins=256)
    lut_elapsed = time.perf_counter() - t1
    print(f"LUT: centers={centers.shape}  table={table.shape}  time={lut_elapsed:.3f}s")
    c00 = table[:, 0, 0]
    assert np.all(np.diff(c00) > 0), "axial modulus should rise with Vf"

    print("weave YAML snippet:")
    print(f"  micromodel: {args.name}")


if __name__ == "__main__":
    main()
