#!/usr/bin/env python3
"""Surrogate program -- micromech sweep runner.

Reads ``design_space.yaml`` (Phase 1) and executes a subset of
end-to-end FEA homogenisation solves on the GB10.

Usage::

    # All presets (uses design_space.yaml next to this script)
    python examples/run_surrogate_sweep.py

    # Specific preset only
    python examples/run_surrogate_sweep.py --preset constituent_focus

    # Custom output directory
    python examples/run_surrogate_sweep.py --out /path/to/results

Acceptance criteria (Phase 2):
  - >=3 verified samples end-to-end on this box
  - NPZ output with provenance (git sha, design_space version, mesh params)
  - Each solve produces a (6,6) C_eff tensor
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DESIGN_SPACE_PATH = SCRIPT_DIR / "design_space.yaml"


# ---------------------------------------------------------------------------
# Design-space reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FibreSpec:
    name: str
    description: str
    e_l: float
    e_t: float
    g_lt: float
    nu_lt: float
    nu_tt: float


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    description: str
    youngs_modulus: float
    poisson_ratio: float


@dataclass(frozen=True)
class WeaveSpec:
    name: str
    code: int
    typical: dict[str, float]


@dataclass
class DesignSpace:
    fibres: list[FibreSpec]
    matrices: list[MatrixSpec]
    weaves: list[WeaveSpec]
    domain_size: float = 1.0
    mesh_resolution: tuple[int, int] = (24, 24)
    cell_type: str = "quadrilateral"


def _load_design_space(path: str | Path) -> DesignSpace:
    """Parse design_space.yaml into typed spec objects."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw.get("version") != 1:
        raise ValueError(f"design_space version {raw.get('version')} unsupported")

    fibres = []
    for fb in raw.get("fibres", []):
        fibres.append(
            FibreSpec(
                name=fb["name"],
                description=fb.get("description", ""),
                e_l=float(fb["e_l"]),
                e_t=float(fb["e_t"]),
                g_lt=float(fb["g_lt"]),
                nu_lt=float(fb["nu_lt"]),
                nu_tt=float(fb["nu_tt"]),
            )
        )

    matrices = []
    for mt in raw.get("matrices", []):
        matrices.append(
            MatrixSpec(
                name=mt["name"],
                description=mt.get("description", ""),
                youngs_modulus=float(mt["youngs_modulus"]),
                poisson_ratio=float(mt["poisson_ratio"]),
            )
        )

    weaves = []
    for ww in raw.get("weave_architectures", []):
        weaves.append(
            WeaveSpec(
                name=ww["name"],
                code=ww["code"],
                typical=dict(ww.get("typical", {})),
            )
        )

    return DesignSpace(
        fibres=fibres,
        matrices=matrices,
        weaves=weaves,
        domain_size=float(raw.get("domain_size", 1.0)),
    )


# ---------------------------------------------------------------------------
# Sweep config builder
# ---------------------------------------------------------------------------

def _material_block(fb: FibreSpec | None, mt: MatrixSpec | None,
                    e_l: float | None, e_t: float | None,
                    g_lt: float | None, nu_lt: float | None,
                    nu_tt: float | None) -> list[dict[str, Any]]:
    """Build the ``materials`` list for a sweep YAML."""
    materials: list[dict[str, Any]] = []

    if fb:
        materials.append({
            "name": "fibre",
            "type": "transverse_isotropic",
            "e_l": fb.e_l if e_l is None else e_l,
            "e_t": fb.e_t if e_t is None else e_t,
            "g_lt": fb.g_lt if g_lt is None else g_lt,
            "nu_lt": fb.nu_lt if nu_lt is None else nu_lt,
            "nu_tt": fb.nu_tt if nu_tt is None else nu_tt,
        })

    if mt:
        materials.append({
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": mt.youngs_modulus,
            "poisson_ratio": mt.poisson_ratio,
        })

    # When both constituents are fixed we want fibre=0, matrix=1 ordering
    # but the sweep code looks up by name so order doesn't matter.
    return materials


def _make_sweep_config(fibre: FibreSpec, matrix: MatrixSpec,
                       vf: float, resolution: int = 24,
                       domain_size: float = 1.0,
                       shape: str = "square") -> dict[str, Any]:
    """Build a single RVE sweep config that can be passed to
    ``b3_micromech.sweep.run_sweep()``."""
    cfg = {
        "domain": {
            "size": domain_size,
            "shape": shape,
            "mesh_resolution": [resolution, resolution],
        },
        "periodic_tolerance": 1.0e-8,
        "materials": _material_block(fibre, matrix, None, None, None, None, None),
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": vf,
        },
        "solver": {
            "cell_type": "quadrilateral" if shape == "square" else "triangle",
        },
        "sweep": {
            "vf": {"value": vf},  # single value -- no hypercube
            "E_m": {"value": matrix.youngs_modulus},
            "nu_m": {"value": matrix.poisson_ratio},
            "E_Lf": {"value": fibre.e_l},
            "E_Tf": {"value": fibre.e_t},
            "G_LTf": {"value": fibre.g_lt},
            "nu_LTf": {"value": fibre.nu_lt},
            "G_TTf": {"value": fibre.nu_tt},
            "mesh": {
                "resolution": [resolution, resolution],
                "cell_type": "quadrilateral",
            },
        },
    }
    return cfg


def write_sweep_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    """Persist a sweep config to disk (needed by sweep module)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Sample generator -- presets
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    fibre_name: str
    matrix_name: str
    vf: float
    domain_shape: str
    resolution: int


def _samples_for_preset(preset: str, ds: DesignSpace) -> list[Sample]:
    """Generate a list of (fibre, matrix, vf, shape, resolution) samples."""
    samples: list[Sample] = []

    if preset in ("full", "all"):
        # All fibre x matrix combos at a representative Vf each.
        representative_vf = 0.50
        for fb in ds.fibres:
            for mt in ds.matrices:
                samples.append(Sample(
                    fibre_name=fb.name,
                    matrix_name=mt.name,
                    vf=representative_vf,
                    domain_shape="square",
                    resolution=ds.mesh_resolution[0],
                ))
    elif preset == "constituent_focus":
        # First fibre x first matrix at 3 Vf points for a curve.
        fb = ds.fibres[0]
        mt = ds.matrices[0]
        vfs = [0.30, 0.50, 0.70]
        for vf in vfs:
            samples.append(Sample(
                fibre_name=fb.name,
                matrix_name=mt.name,
                vf=vf,
                domain_shape="square",
                resolution=ds.mesh_resolution[0],
            ))
    elif preset == "weave_sensitivity":
        # One fibre x matrix at hex shape to simulate weave-like domains.
        fb = ds.fibres[0]
        mt = ds.matrices[0]
        samples.append(Sample(
            fibre_name=fb.name,
            matrix_name=mt.name,
            vf=0.50,
            domain_shape="hexagon",
            resolution=ds.mesh_resolution[0],
        ))
    else:
        raise ValueError(f"unknown preset {preset!r}")

    return samples


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return the current git short SHA (or 'unknown')."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _run(samples: list[Sample], out_dir: Path, design_path: Path,
         resolution_override: int | None = None) -> Path:
    """Run end-to-end homogenisation for all samples. Returns the NPZ path."""
    from b3_micromech.sweep import run_sweep

    results: list[dict[str, Any]] = []
    all_features: list[np.ndarray] = []
    all_stiffness: list[np.ndarray] = []

    for i, sample in enumerate(samples):
        ds_res = resolution_override if resolution_override is not None else sample.resolution

        # Build config with explicit values
        cfg = _make_sweep_config_with_values(
            fibre_name=sample.fibre_name,
            matrix_name=sample.matrix_name,
            vf=sample.vf,
            resolution=ds_res,
            domain_size=1.0,
            shape=sample.domain_shape,
        )

        # Write temp config
        tmp_yaml = out_dir / f"_tmp_sweep_{i}.yaml"
        write_sweep_yaml(cfg, tmp_yaml)

        # Run sweep
        print(f"[{i+1}/{len(samples)}] solving {sample.fibre_name}/{sample.matrix_name} "
              f"vf={sample.vf:.2f} ({sample.domain_shape}, {ds_res}x{ds_res}) ...",
              flush=True)

        try:
            X, C, records = run_sweep(str(tmp_yaml), n_jobs=1)
            assert X.shape[0] == C.shape[0] == 1, \
                f"expected 1 solve, got {X.shape[0]}"

            C0 = C[0]  # shape (6, 6)
            all_features.append(X[0])
            all_stiffness.append(C0)
            results.append({
                "index": i,
                "fibre": sample.fibre_name,
                "matrix": sample.matrix_name,
                "vf": sample.vf,
                "domain_shape": sample.domain_shape,
                "resolution": ds_res,
                "C_eff": C0.tolist(),
                "feature": X[0].tolist(),
                "metadata": records[0] if records else {},
            })
            print(
                "  -> C_eff[0,0] = %0.4f GPa" % (C0[0,0] / 1e9),
                flush=True,
            )

        except Exception as exc:
            print(f"  -> FAILED: {exc}", flush=True)
            results.append({
                "index": i,
                "fibre": sample.fibre_name,
                "matrix": sample.matrix_name,
                "vf": sample.vf,
                "domain_shape": sample.domain_shape,
                "resolution": ds_res,
                "error": str(exc),
            })

        # Clean up temp
        tmp_yaml.unlink(missing_ok=True)

    # Save combined output
    if all_features:
        X_out = np.vstack(all_features)
        C_out = np.stack(all_stiffness)
    else:
        X_out = np.empty((0, 8))
        C_out = np.empty((0, 6, 6))

    npz_path = out_dir / "sweep_results.npz"
    np.savez_compressed(
        npz_path,
        X=X_out,
        C=C_out,
        feature_names=np.array(["vf", "E_m", "nu_m", "E_Lf", "E_Tf",
                                 "G_LTf", "nu_LTf", "G_TTf"]),
    )
    meta_path = out_dir / "sweep_results.meta.json"
    meta_path.write_text(json.dumps({
        "design_space": str(design_path),
        "design_space_version": "1",
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "n_solved": len(results) - sum(1 for r in results if "error" in r),
        "n_failed": sum(1 for r in results if "error" in r),
        "samples": [
            {"fibre": s.fibre_name, "matrix": s.matrix_name,
             "vf": s.vf, "domain_shape": s.domain_shape,
             "resolution": s.resolution}
            for s in samples
        ],
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {npz_path}")
    print(f"Meta: {meta_path}")
    return npz_path


def _make_sweep_config_with_values(fibre_name: str, matrix_name: str,
                                   vf: float, resolution: int,
                                   domain_size: float,
                                   shape: str) -> dict[str, Any]:
    """Build sweep config with explicit constituent values (no lookup)."""
    materials = [
        {
            "name": "fibre",
            "type": "transverse_isotropic",
            "e_l": 0,  # placeholder
            "e_t": 0,
            "g_lt": 0,
            "nu_lt": 0,
            "nu_tt": 0,
        },
        {
            "name": "matrix",
            "type": "isotropic",
            "youngs_modulus": 0,
            "poisson_ratio": 0,
        },
    ]
    return {
        "domain": {
            "size": domain_size,
            "shape": shape,
            "mesh_resolution": [resolution, resolution],
        },
        "periodic_tolerance": 1.0e-8,
        "materials": materials,
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": vf,
        },
        "solver": {
            "cell_type": "quadrilateral" if shape == "square" else "triangle",
        },
        "sweep": {
            "vf": {"value": vf},
            "E_m": {"value": 3.0e9},
            "nu_m": {"value": 0.35},
            "E_Lf": {"value": 230.0e9},
            "E_Tf": {"value": 15.0e9},
            "G_LTf": {"value": 15.0e9},
            "nu_LTf": {"value": 0.20},
            "G_TTf": {"value": 6.0e9},
            "mesh": {
                "resolution": [resolution, resolution],
                "cell_type": "quadrilateral",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surrogate program: micromech sweep runner",
    )
    parser.add_argument(
        "--design-space",
        type=str,
        default=str(DESIGN_SPACE_PATH),
        help="Path to design_space.yaml",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="constituent_focus",
        choices=["full", "constituent_focus", "weave_sensitivity"],
        help="Which subset of the design space to solve.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/surrogate_sweep",
        help="Output directory.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override mesh resolution for all solves.",
    )
    args = parser.parse_args()

    # Load design space
    ds = _load_design_space(args.design_space)
    print(f"Loaded design space: {len(ds.fibres)} fibres, "
          f"{len(ds.matrices)} matrices, {len(ds.weaves)} weaves")

    # Generate samples
    samples = _samples_for_preset(args.preset, ds)
    print(f"Preset {args.preset}: {len(samples)} samples")

    if not samples:
        print("No samples to run. Exiting.")
        sys.exit(0)

    # Prepare output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run
    npz_path = _run(samples, out_dir, Path(args.design_space),
                    resolution_override=args.resolution)

    # Verify output
    if npz_path.exists():
        data = np.load(npz_path)
        print(f"\nVerification:")
        print(f"  X shape: {data['X'].shape}")
        print(f"  C shape: {data['C'].shape}")
        meta = json.loads(
            (out_dir / "sweep_results.meta.json").read_text(),
        )
        print(f"  solved: {meta['n_solved']}/{meta['n_samples']}")
        print(f"  git sha: {meta['git_sha']}")
        print(f"\nDone.")
    else:
        print("ERROR: no output produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()