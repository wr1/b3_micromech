#!/usr/bin/env python3
"""End-to-end surrogate demo: hex sweep → MLP train → 1000 predictions → validation.

Pipeline::

    sweep_hex_hypercube.yaml  →  dataset.npz
    dataset.npz               →  surrogate_model.joblib
    surrogate_model.joblib    →  predictions.npz + diagnostic plots

Validation:

    - In-sample hold-out error on the training grid
    - Stratified error split at Vf ≥ 0.75
    - Three random FEA spot-checks inside the training hypercube

Plots (under ``<out>/plots/``):

    - ``e2_vs_vf.png``         — transverse modulus vs Vf (inference batch)
    - ``holdout_parity_e2.png`` — predicted vs reference E2 on training grid
    - ``holdout_error_hist.png`` — Frobenius error histogram on training grid

Optional: register the trained model with ``b3_tex.micromodels.SurrogateModel``.

Usage::

    python examples/demo_surrogate_chain.py
    python examples/demo_surrogate_chain.py --skip-sweep --out results/surrogate_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from b3_micromech.export import load_dataset, save_dataset
from b3_micromech.geometry import max_fibre_volume_fraction
from b3_micromech.homogenize import homogenize
from b3_micromech.surrogate import (
    StiffnessSurrogate,
    engineering_constants_batch,
    evaluate_training_holdout,
    predict_random_hypercube_samples,
    relative_frobenius_error,
    sample_uniform_hypercube,
    save_hypercube_predictions,
)
from b3_micromech.sweep import problem_from_sweep_point, run_sweep
from b3_micromech.tensors import engineering_constants_transverse_iso

DEFAULT_SWEEP_YAML = REPO / "examples" / "sweep_hex_hypercube.yaml"


def run_hex_hypercube_sweep(
    sweep_yaml: Path,
    dataset_path: Path,
    *,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the hex hypercube FEA sweep and write ``dataset.npz``."""
    features, stiffness, records = run_sweep(str(sweep_yaml), n_jobs=n_jobs)
    save_dataset(
        dataset_path,
        features=features,
        stiffness=stiffness,
        metadata={
            "source": str(sweep_yaml),
            "n_points": int(features.shape[0]),
            "records": records,
        },
    )
    print(f"sweep: {features.shape[0]} solves -> {dataset_path}")
    return features, stiffness


def train_stiffness_surrogate(
    features: np.ndarray,
    stiffness: np.ndarray,
    model_path: Path,
    *,
    seed: int = 0,
) -> StiffnessSurrogate:
    """Fit an MLP on the sweep dataset."""
    model = StiffnessSurrogate.train(features, stiffness, random_state=seed)
    model.save(model_path)
    print(f"trained surrogate -> {model_path}")
    return model


def predict_uniform_hypercube_batch(
    model: StiffnessSurrogate,
    n_samples: int,
    predictions_path: Path,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw uniform samples in the training bounds and predict stiffness."""
    features, stiffness = predict_random_hypercube_samples(model, n_samples, seed=seed)
    save_hypercube_predictions(predictions_path, features, stiffness)
    print(f"inference: {n_samples} samples -> {predictions_path}")
    return features, stiffness


def run_fea_spot_checks(
    sweep_yaml: Path,
    model: StiffnessSurrogate,
    *,
    n_checks: int = 3,
    seed: int = 1,
) -> list[dict[str, float]]:
    """Re-homogenize random hypercube points and compare to the surrogate."""
    import yaml

    with open(sweep_yaml, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    bounds = model.feature_bounds
    rng = np.random.default_rng(seed)
    features = sample_uniform_hypercube(
        n_checks, bounds, seed=int(rng.integers(1_000_000))
    )
    predicted = model.predict(features)

    reports: list[dict[str, float]] = []
    for i in range(n_checks):
        point = {
            "vf": float(features[i, 0]),
            "E_m": float(features[i, 1]),
            "nu_m": float(features[i, 2]),
            "E_Lf": float(features[i, 3]),
            "E_Tf": float(features[i, 4]),
            "G_LTf": float(features[i, 5]),
            "nu_LTf": float(features[i, 6]),
            "G_TTf": float(features[i, 7]),
        }
        problem = problem_from_sweep_point(base_cfg, point)
        fea = homogenize(problem).effective_stiffness
        err = float(relative_frobenius_error(predicted[i : i + 1], fea[None, ...])[0])
        ec_fea = engineering_constants_transverse_iso(fea)
        ec_pred = engineering_constants_transverse_iso(predicted[i])
        reports.append(
            {
                "vf": point["vf"],
                "frobenius_error": err,
                "e2_fea_gpa": ec_fea["e_t"] / 1e9,
                "e2_pred_gpa": ec_pred["e_t"] / 1e9,
            }
        )
        print(
            f"  spot-check {i + 1}: Vf={point['vf']:.3f}  "
            f"rel_err={err:.3e}  E2_fea={ec_fea['e_t'] / 1e9:.2f} GPa  "
            f"E2_pred={ec_pred['e_t'] / 1e9:.2f} GPa"
        )
    return reports


def render_surrogate_diagnostic_plots(
    train_features: np.ndarray,
    train_stiffness: np.ndarray,
    model: StiffnessSurrogate,
    inference_features: np.ndarray,
    inference_stiffness: np.ndarray,
    plot_dir: Path,
) -> dict[str, Path]:
    """Write E2-vs-Vf, parity, and error-histogram figures."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    inf_ec = engineering_constants_batch(inference_stiffness)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(
        inference_features[:, 0],
        inf_ec["e_t"] / 1e9,
        c=inference_features[:, 1] / 1e9,
        s=12,
        alpha=0.55,
        cmap="viridis",
    )
    ax.set_xlabel("fibre volume fraction")
    ax.set_ylabel(r"$E_2$ [GPa]")
    ax.set_title("Surrogate inference batch (colour = $E_m$)")
    fig.colorbar(sc, ax=ax, label=r"$E_m$ [GPa]")
    fig.tight_layout()
    paths["e2_vs_vf"] = plot_dir / "e2_vs_vf.png"
    fig.savefig(paths["e2_vs_vf"], dpi=150)
    plt.close(fig)

    predicted_train = model.predict(train_features)
    train_ec_ref = engineering_constants_batch(train_stiffness)
    train_ec_pred = engineering_constants_batch(predicted_train)
    e2_ref = train_ec_ref["e_t"] / 1e9
    e2_pred = train_ec_pred["e_t"] / 1e9

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(e2_ref, e2_pred, s=28, alpha=0.8)
    lo = min(e2_ref.min(), e2_pred.min())
    hi = max(e2_ref.max(), e2_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"FEA $E_2$ [GPa]")
    ax.set_ylabel(r"surrogate $E_2$ [GPa]")
    ax.set_title("Training-grid hold-out parity")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    paths["holdout_parity_e2"] = plot_dir / "holdout_parity_e2.png"
    fig.savefig(paths["holdout_parity_e2"], dpi=150)
    plt.close(fig)

    errors = relative_frobenius_error(predicted_train, train_stiffness)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=12, edgecolor="white")
    ax.set_xlabel("relative Frobenius error")
    ax.set_ylabel("count")
    ax.set_title("Training-grid hold-out errors")
    fig.tight_layout()
    paths["holdout_error_hist"] = plot_dir / "holdout_error_hist.png"
    fig.savefig(paths["holdout_error_hist"], dpi=150)
    plt.close(fig)

    e2_rel = np.abs(e2_pred - e2_ref) / np.maximum(e2_ref, 1e-12)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(e2_ref, e2_rel, s=28, alpha=0.85)
    ax.set_xlabel(r"FEA $E_2$ [GPa]")
    ax.set_ylabel("relative $E_2$ error")
    ax.set_title("Hold-out $E_2$ error vs modulus")
    fig.tight_layout()
    paths["holdout_e2_error"] = plot_dir / "holdout_e2_error.png"
    fig.savefig(paths["holdout_e2_error"], dpi=150)
    plt.close(fig)

    return paths


def register_b3_tex_surrogate(
    model: StiffnessSurrogate, name: str = "fea_hex_surrogate"
) -> None:
    """Wrap the trained model for ``b3_tex.micromodels.SurrogateModel``."""
    try:
        from b3_tex.micromodels import SurrogateModel
    except ImportError:
        print("b3_tex not installed; skipping SurrogateModel registration")
        return

    surrogate = SurrogateModel(predict=model.as_predict_callable(), name=name)
    print(f"registered b3_tex SurrogateModel(name={surrogate.name!r})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep-yaml",
        type=Path,
        default=DEFAULT_SWEEP_YAML,
        help="Hex hypercube sweep config",
    )
    ap.add_argument("--out", type=Path, default=REPO / "results" / "surrogate_demo")
    ap.add_argument("--n-inference", type=int, default=1000)
    ap.add_argument("--n-spot-checks", type=int, default=3)
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-spot-checks", action="store_true")
    ap.add_argument("--register-b3-tex", action="store_true")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.npz"
    model_path = out_dir / "surrogate_model.joblib"
    predictions_path = out_dir / "predictions.npz"
    plot_dir = out_dir / "plots"
    report_path = out_dir / "validation_report.json"

    max_vf = max_fibre_volume_fraction(shape="hexagon", domain_size=1.0, standoff=0.01)
    print(f"hex max Vf (1% standoff) = {max_vf:.4f}")

    if args.skip_sweep:
        features, stiffness, _meta = load_dataset(dataset_path)
        print(f"loaded existing dataset: {dataset_path} ({features.shape[0]} points)")
    else:
        features, stiffness = run_hex_hypercube_sweep(
            args.sweep_yaml, dataset_path, n_jobs=args.jobs
        )

    model = train_stiffness_surrogate(features, stiffness, model_path, seed=args.seed)
    holdout = evaluate_training_holdout(model, features, stiffness)
    print("hold-out report:")
    for key, value in sorted(holdout.items()):
        print(f"  {key} = {value:.4e}")

    inf_features, inf_stiffness = predict_uniform_hypercube_batch(
        model, args.n_inference, predictions_path, seed=args.seed
    )

    plot_paths = render_surrogate_diagnostic_plots(
        features,
        stiffness,
        model,
        inf_features,
        inf_stiffness,
        plot_dir,
    )
    for name, path in plot_paths.items():
        print(f"  plot {name}: {path}")

    spot_reports: list[dict[str, float]] = []
    if not args.skip_spot_checks:
        print("FEA spot-checks:")
        spot_reports = run_fea_spot_checks(
            args.sweep_yaml, model, n_checks=args.n_spot_checks, seed=args.seed + 1
        )

    report = {
        "sweep_yaml": str(args.sweep_yaml),
        "n_training": int(features.shape[0]),
        "n_inference": args.n_inference,
        "holdout": holdout,
        "spot_checks": spot_reports,
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"validation report -> {report_path}")

    if args.register_b3_tex:
        register_b3_tex_surrogate(model)


if __name__ == "__main__":
    main()
