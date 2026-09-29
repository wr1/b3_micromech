#!/usr/bin/env python3
"""3-D response surrogate demo: Vf × E_m × E_Lf sweep → physics residual → surfaces.

Pipeline::

    sweep_hex_3d_response.yaml  →  dataset.npz
    dataset.npz                 →  surrogate_model.joblib  (kind=physics by default)
    surrogate_model.joblib      →  structured response grid + 3-D plots

Plots (under ``<out>/plots/``):

    - ``response_surface_e1_3d.png``      — $E_1$(Vf, $E_m$, $E_{Lf}$)
    - ``response_surface_e2_3d.png``      — $E_2$(Vf, $E_m$, $E_{Lf}$)
    - ``response_slice_e1_vf_elf.png``    — $E_1$(Vf, $E_{Lf}$) at median $E_m$
    - ``holdout_parity_e1.png``           — training-grid parity for $E_1$
    - ``holdout_parity_e2.png``           — training-grid parity for $E_2$

Usage::

    python examples/demo_surrogate_3d_response.py
    make demo-surrogate-3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from b3_micromech.export import load_dataset, save_dataset
from b3_micromech.geometry import max_fibre_volume_fraction
from b3_micromech.physics_surrogate import DEFAULT_SURROGATE_KIND, train_surrogate
from b3_micromech.surrogate import (
    engineering_constants_batch,
    evaluate_training_holdout,
    predict_random_hypercube_samples,
    save_hypercube_predictions,
)
from b3_micromech.surrogate_response import (
    DEFAULT_THREE_AXIS_RESPONSE,
    predict_structured_response_grid,
    render_three_axis_response_surfaces,
)
from b3_micromech.sweep import count_sweep_points, run_sweep, varying_sweep_parameters

REPO = Path(__file__).resolve().parents[1]

DEFAULT_SWEEP_YAML = REPO / "examples" / "sweep_hex_3d_response.yaml"
DEFAULT_RESPONSE_AXES = DEFAULT_THREE_AXIS_RESPONSE
DEFAULT_RESPONSE_GRID = (24, 12, 12)


def run_three_axis_sweep(
    sweep_yaml: Path,
    dataset_path: Path,
    *,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the Vf × E_m × E_Lf FEA sweep."""
    with open(sweep_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    varying = varying_sweep_parameters(cfg["sweep"])
    n_solves = count_sweep_points(cfg["sweep"])
    print(f"sweep axes: {varying}  ({n_solves} solves)")

    features, stiffness, records = run_sweep(str(sweep_yaml), n_jobs=n_jobs)
    save_dataset(
        dataset_path,
        features=features,
        stiffness=stiffness,
        metadata={
            "source": str(sweep_yaml),
            "n_points": int(features.shape[0]),
            "varying_axes": varying,
            "records": records,
        },
    )
    print(f"sweep complete -> {dataset_path}")
    return features, stiffness, varying


def train_three_axis_surrogate(
    features: np.ndarray,
    stiffness: np.ndarray,
    model_path: Path,
    *,
    kind: str = DEFAULT_SURROGATE_KIND,
    seed: int = 0,
):
    """Train residual (default) or requested kind on the 3-axis grid."""
    train_kw: dict = {}
    if kind == "mlp":
        train_kw["random_state"] = seed
    elif kind == "mf_gp":
        train_kw["n_restarts"] = 1
    model = train_surrogate(features, stiffness, kind=kind, **train_kw)
    model.save(model_path)
    print(f"trained surrogate kind={kind!r} -> {model_path}")
    return model


def render_holdout_parity_plots(
    train_features: np.ndarray,
    train_stiffness: np.ndarray,
    model,
    plot_dir: Path,
) -> dict[str, Path]:
    """Parity plots for $E_1$ and $E_2$ on the training hypercube."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    predicted = model.predict(train_features)
    ref_ec = engineering_constants_batch(train_stiffness)
    pred_ec = engineering_constants_batch(predicted)
    paths: dict[str, Path] = {}

    for key, label, path_key, filename in (
        ("e_l", r"$E_1$", "parity_e1", "holdout_parity_e1.png"),
        ("e_t", r"$E_2$", "parity_e2", "holdout_parity_e2.png"),
    ):
        ref = ref_ec[key] / 1e9
        pred = pred_ec[key] / 1e9
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(ref, pred, s=36, alpha=0.85)
        lo = min(ref.min(), pred.min())
        hi = max(ref.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(rf"FEA {label} [GPa]")
        ax.set_ylabel(rf"surrogate {label} [GPa]")
        ax.set_title(f"Training-grid parity ({label})")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        paths[path_key] = plot_dir / filename
        fig.savefig(paths[path_key], dpi=150)
        plt.close(fig)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-yaml", type=Path, default=DEFAULT_SWEEP_YAML)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "surrogate_3d_demo")
    ap.add_argument("--response-axes", nargs=3, default=DEFAULT_RESPONSE_AXES)
    ap.add_argument("--response-grid", nargs=3, type=int, default=DEFAULT_RESPONSE_GRID)
    ap.add_argument("--n-inference", type=int, default=1000)
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--kind",
        type=str,
        default=DEFAULT_SURROGATE_KIND,
        choices=("physics", "mf_gp", "mlp"),
        help="Surrogate kind (default: physics residual)",
    )
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    varying_axes = tuple(args.response_axes)
    grid_sizes = tuple(args.response_grid)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.npz"
    model_path = out_dir / "surrogate_model.joblib"
    predictions_path = out_dir / "predictions.npz"
    response_grid_path = out_dir / "response_grid.npz"
    plot_dir = out_dir / "plots"
    report_path = out_dir / "validation_report.json"

    max_vf = max_fibre_volume_fraction(shape="hexagon", domain_size=1.0, standoff=0.01)
    print(f"hex max Vf (1% standoff) = {max_vf:.4f}")

    if args.skip_sweep:
        features, stiffness, meta = load_dataset(dataset_path)
        varying = (
            meta.get("varying_axes", list(varying_axes)) if meta else list(varying_axes)
        )
        print(
            f"loaded dataset: {dataset_path} ({features.shape[0]} points, axes={varying})"
        )
    else:
        features, stiffness, varying = run_three_axis_sweep(
            args.sweep_yaml, dataset_path, n_jobs=args.jobs
        )

    model = train_three_axis_surrogate(
        features, stiffness, model_path, kind=args.kind, seed=args.seed
    )
    holdout = evaluate_training_holdout(model, features, stiffness)
    print("hold-out report:")
    for key, value in sorted(holdout.items()):
        print(f"  {key} = {value:.4e}")

    inf_features, inf_stiffness = predict_random_hypercube_samples(
        model, args.n_inference, seed=args.seed
    )
    save_hypercube_predictions(predictions_path, inf_features, inf_stiffness)
    print(f"inference batch -> {predictions_path}")

    grid_features, axis_coords, grid_stiffness = predict_structured_response_grid(
        model,
        model.feature_bounds,
        varying_axes,
        grid_sizes,
    )
    np.savez_compressed(
        response_grid_path,
        X=grid_features,
        C=grid_stiffness,
        varying_axes=np.array(varying_axes),
        grid_sizes=np.array(grid_sizes),
        **{f"axis_{name}": coords for name, coords in axis_coords.items()},
    )
    print(f"response grid -> {response_grid_path}")

    plot_paths = render_three_axis_response_surfaces(
        grid_features,
        grid_stiffness,
        varying_axes,
        plot_dir,
        grid_sizes=grid_sizes,
    )
    plot_paths.update(render_holdout_parity_plots(features, stiffness, model, plot_dir))
    for name, path in plot_paths.items():
        print(f"  plot {name}: {path}")

    predicted = model.predict(features)
    e1_err = np.abs(
        engineering_constants_batch(predicted)["e_l"]
        - engineering_constants_batch(stiffness)["e_l"]
    ) / np.maximum(engineering_constants_batch(stiffness)["e_l"], 1e-12)

    report = {
        "sweep_yaml": str(args.sweep_yaml),
        "varying_axes": list(varying),
        "response_axes": list(varying_axes),
        "response_grid": list(grid_sizes),
        "n_training": int(features.shape[0]),
        "n_inference": args.n_inference,
        "holdout": holdout,
        "mean_relative_e1_error": float(e1_err.mean()),
        "max_relative_e1_error": float(e1_err.max()),
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"validation report -> {report_path}")


if __name__ == "__main__":
    main()
