"""treeparse CLI for b3_micromech."""

from pathlib import Path

import numpy as np
from treeparse import argument, cli, command, option

from b3_micromech.contract import Constituents
from b3_micromech.features import _names_for, features_out_of_bounds
from b3_micromech.homogenize import homogenize, surrogate_features
from b3_micromech.mesomech import (
    constituents_from_yaml,
    predict_stiffness_batch,
    register_fea_micromech,
    register_fea_surrogate,
)
from b3_micromech.physics_surrogate import (
    DEFAULT_SURROGATE_KIND,
    chamis_vs_fea_report,
    load_surrogate,
    train_surrogate_from_dataset,
)
from b3_micromech.plot import render_all_figures
from b3_micromech.postprocess import solve_all_loadcases
from b3_micromech.problem import RVEProblem
from b3_micromech.reference import chamis_ud_stiffness, mori_tanaka_cylinder
from b3_micromech.surrogate import (
    evaluate_training_holdout,
    predict_random_hypercube_samples,
    save_hypercube_predictions,
)
from b3_micromech.sweep import sweep_to_file
from b3_micromech.tensors import engineering_constants_transverse_iso


def _validate_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    print(f"OK: loaded RVE from {config}")
    print(f"  domain_size = {problem.domain_size}")
    print(f"  mesh_resolution = {problem.mesh_resolution}")
    print(f"  cell_type = {problem.cell_type}")
    print(
        f"  vf = {problem.fibre_volume_fraction:.4f}  (r = {problem.fibre_radius:.4f})"
    )
    print(f"  materials = {sorted(problem.materials)}")


def _reference_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    matrix = problem.materials[problem.matrix_material]
    fibre = problem.materials[problem.fibre_material]
    vf = problem.fibre_volume_fraction
    print(f"fibre volume fraction = {vf:.4f}")
    for label, C in (
        (
            "Mori-Tanaka",
            mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf),
        ),
        (
            "Chamis",
            chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf),
        ),
    ):
        ec = engineering_constants_transverse_iso(C)
        print(f"{label} engineering constants (axis 1 = fibre):")
        for k, v in ec.items():
            if k.startswith(("e_", "g_")):
                print(f"  {k} = {v / 1e9:.4f} GPa")
            else:
                print(f"  {k} = {v:.4f}")


def _solve_cmd(config: str, out: str, plot: bool, plot_scale: float) -> None:
    problem = RVEProblem.from_yaml(config)
    result = homogenize(problem)
    C = result.effective_stiffness
    ec = engineering_constants_transverse_iso(C)
    print("Effective stiffness engineering constants (axis 1 = fibre):")
    print(f"  E1 = {ec['e_l'] / 1e9:.4f} GPa")
    print(f"  E2 = {ec['e_t'] / 1e9:.4f} GPa")
    print(f"  G12 = {ec['g_lt'] / 1e9:.4f} GPa")
    print(f"  nu12 = {ec['nu_lt']:.4f}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "C_eff.npz",
        C_eff=C,
        C_fea=result.fea_stiffness,
        features=surrogate_features(problem),
    )
    print(f"wrote {out_dir / 'C_eff.npz'}")
    if plot:
        _plot_cmd(config, str(out_dir / "plots"), plot_scale)


def _plot_cmd(config: str, out: str, scale: float) -> None:
    problem = RVEProblem.from_yaml(config)
    loadcases = solve_all_loadcases(problem)
    plot_scale = None if scale < 0 else scale
    paths = render_all_figures(loadcases, out, scale=plot_scale)
    print(f"Plots written to {out}/")
    for name, path in paths.items():
        print(f"  {name}: {path.name}")


def _sweep_cmd(config: str, out: str, jobs: int) -> None:
    out_path = Path(out) / "dataset.npz"
    sweep_to_file(config, str(out_path), n_jobs=jobs)
    print(f"wrote {out_path}")


def _train_surrogate_cmd(dataset: str, out: str, seed: int, kind: str) -> None:
    model_path = Path(out)
    kind_norm = kind.strip().lower()
    train_kwargs: dict = {}
    if kind_norm == "mlp":
        train_kwargs["random_state"] = seed
    elif kind_norm == "mf_gp":
        train_kwargs["n_restarts"] = 1
    model = train_surrogate_from_dataset(
        dataset, model_path=model_path, kind=kind_norm, **train_kwargs
    )
    from b3_micromech.export import load_dataset

    features, stiffness, _meta = load_dataset(dataset)
    base_report = chamis_vs_fea_report(features, stiffness)
    report = evaluate_training_holdout(model, features, stiffness)
    print(f"trained surrogate kind={kind_norm!r} -> {model_path}")
    print("  bare Chamis (pre-residual) vs FEA:")
    for key, value in sorted(base_report.items()):
        print(f"    {key} = {value:.4e}")
    print("  trained holdout:")
    for key, value in sorted(report.items()):
        print(f"    {key} = {value:.4e}")


def _predict_surrogate_cmd(model: str, out: str, n_samples: int, seed: int) -> None:
    surrogate = load_surrogate(model)
    features, stiffness = predict_random_hypercube_samples(
        surrogate, n_samples, seed=seed
    )
    out_path = Path(out)
    save_hypercube_predictions(out_path, features, stiffness)
    print(f"wrote {n_samples} predictions to {out_path}")


def _parse_vf_list(vf_csv: str, vf_npy: str, vf_linspace: str) -> np.ndarray:
    sources = sum(bool(x) for x in (vf_csv, vf_npy, vf_linspace))
    if sources != 1:
        raise SystemExit("provide exactly one of --vf, --vf-npy, or --vf-linspace")
    if vf_npy:
        return np.asarray(np.load(vf_npy), dtype=float).ravel()
    if vf_csv:
        return np.array([float(x) for x in vf_csv.split(",")], dtype=float)
    start, stop, n = vf_linspace.split(":")
    return np.linspace(float(start), float(stop), int(n), dtype=float)


def _predict_batch_cmd(
    model: str,
    out: str,
    constituents: str,
    vf: str,
    vf_npy: str,
    vf_linspace: str,
) -> None:
    surrogate = load_surrogate(model)
    vf_arr = _parse_vf_list(vf, vf_npy, vf_linspace)
    matrix, fibre = constituents_from_yaml(constituents)
    stiffness = predict_stiffness_batch(surrogate, vf_arr, matrix, fibre)
    constituents = Constituents.from_materials(matrix, fibre)
    features = constituents.feature_matrix(vf_arr, names=_names_for(constituents))
    oob = features_out_of_bounds(features, surrogate.feature_bounds)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        vf=vf_arr,
        X=features,
        C=stiffness,
        out_of_bounds=oob,
    )
    print(f"wrote {vf_arr.shape[0]} batch predictions to {out_path}")
    if oob.any():
        print(f"  warning: {int(oob.sum())} points outside training bounds")


def _register_micromech_cmd(
    model: str,
    name: str,
    rve_yaml: str,
    fea_jobs: int,
    cache_dir: str,
    no_disk_cache: bool,
) -> None:
    surrogate_path = model if model else None
    micromodel = register_fea_micromech(
        surrogate_path,
        name=name,
        rve_yaml=rve_yaml,
        n_jobs=fea_jobs,
        cache_dir=cache_dir or None,
        disk_cache=False if no_disk_cache else None,
    )
    print(f"registered micromodel {micromodel.name!r} (mode={micromodel.mode})")
    print(f"  set micromodel: {micromodel.name} in weave YAML")
    if micromodel.cache_dir is not None:
        print(f"  disk cache: {micromodel.cache_dir} (enabled={micromodel.disk_cache})")


def _register_surrogate_cmd(model: str, name: str) -> None:
    micromodel = register_fea_surrogate(model, name=name)
    print(f"registered micromodel {micromodel.name!r} (mode={micromodel.mode})")
    print(f"  set micromodel: {micromodel.name} in weave YAML")


_app = cli(
    name="b3-micromech",
    help="FEA homogenization of UD composite transverse RVEs (MFEM).",
    commands=[
        command(
            name="validate",
            help="Load and validate a YAML config without solving.",
            callback=_validate_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="reference",
            help="Print Mori-Tanaka reference constants.",
            callback=_reference_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
        ),
        command(
            name="solve",
            help="Homogenize one transverse RVE.",
            callback=_solve_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results",
                    help="Output directory for C_eff.npz.",
                ),
                option(
                    flags=["--plot", "-p"],
                    flag=True,
                    help="Also write loadcase deformation plots under <out>/plots/.",
                ),
                option(
                    flags=["--plot-scale"],
                    arg_type=float,
                    default=-1.0,
                    help="In-plane deflection exaggeration (<0 = auto).",
                ),
            ],
        ),
        command(
            name="plot",
            help="Render deformation / stress plots for six unit macro-strains.",
            callback=_plot_cmd,
            arguments=[argument(name="config", arg_type=str, help="Path to RVE YAML.")],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results/plots",
                    help="Output directory for PNG figures.",
                ),
                option(
                    flags=["--scale"],
                    arg_type=float,
                    default=-1.0,
                    help="In-plane deflection exaggeration (<0 = auto).",
                ),
            ],
        ),
        command(
            name="sweep",
            help="Run a parameter hypercube sweep.",
            callback=_sweep_cmd,
            arguments=[
                argument(name="config", arg_type=str, help="Path to sweep YAML.")
            ],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results",
                    help="Output directory for dataset.npz.",
                ),
                option(
                    flags=["--jobs", "-j"],
                    arg_type=int,
                    default=1,
                    help="Parallel workers (requires joblib).",
                ),
            ],
        ),
        command(
            name="train-surrogate",
            help=(
                "Train a stiffness surrogate from a sweep dataset "
                "(default kind: physics residual; also mlp | mf_gp)."
            ),
            callback=_train_surrogate_cmd,
            arguments=[
                argument(
                    name="dataset", arg_type=str, help="Path to dataset.npz from sweep."
                ),
            ],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results/surrogate_model.joblib",
                    help="Output path for the trained model.",
                ),
                option(
                    flags=["--kind"],
                    arg_type=str,
                    default=DEFAULT_SURROGATE_KIND,
                    help=(
                        "Surrogate kind: physics (Chamis+ridge residual, default; "
                        "best high-Vf extrapolation), mf_gp (Chamis+GP residual), "
                        "mlp (black-box)."
                    ),
                ),
                option(
                    flags=["--seed"],
                    arg_type=int,
                    default=0,
                    help="Random seed for MLP training.",
                ),
            ],
        ),
        command(
            name="predict-surrogate",
            help="Sample the training hypercube and predict stiffness tensors.",
            callback=_predict_surrogate_cmd,
            arguments=[
                argument(
                    name="model",
                    arg_type=str,
                    help="Path to trained surrogate .joblib.",
                ),
            ],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results/predictions.npz",
                    help="Output path for prediction batch.",
                ),
                option(
                    flags=["--n-samples", "-n"],
                    arg_type=int,
                    default=1000,
                    help="Number of uniform hypercube samples.",
                ),
                option(
                    flags=["--seed"],
                    arg_type=int,
                    default=0,
                    help="Random seed for hypercube sampling.",
                ),
            ],
        ),
        command(
            name="predict-batch",
            help="Vectorized surrogate prediction over a Vf sample vector.",
            callback=_predict_batch_cmd,
            arguments=[
                argument(
                    name="model",
                    arg_type=str,
                    help="Path to trained surrogate .joblib.",
                ),
            ],
            options=[
                option(
                    flags=["--out", "-o"],
                    arg_type=str,
                    default="results/batch_predictions.npz",
                    help="Output path for batch predictions.",
                ),
                option(
                    flags=["--constituents", "-c"],
                    arg_type=str,
                    default="examples/sweep_hex_hypercube.yaml",
                    help="YAML with matrix and fibre materials.",
                ),
                option(
                    flags=["--vf"],
                    arg_type=str,
                    default="",
                    help="Comma-separated Vf values.",
                ),
                option(
                    flags=["--vf-npy"],
                    arg_type=str,
                    default="",
                    help="Path to a 1-D .npy Vf array.",
                ),
                option(
                    flags=["--vf-linspace"],
                    arg_type=str,
                    default="",
                    help="Vf linspace as start:stop:n (e.g. 0.55:0.88:100000).",
                ),
            ],
        ),
        command(
            name="register-fea-micromech",
            help="Register FEA micromodel (surrogate and/or on-the-fly FEA fallback).",
            callback=_register_micromech_cmd,
            arguments=[
                argument(
                    name="model",
                    arg_type=str,
                    help="Path to surrogate .joblib (optional; empty → FEA fallback).",
                    nargs="?",
                    default="",
                ),
            ],
            options=[
                option(
                    flags=["--name"],
                    arg_type=str,
                    default="fea_hex",
                    help="Registry name for weave YAML micromodel field.",
                ),
                option(
                    flags=["--rve-yaml"],
                    arg_type=str,
                    default="examples/sweep_hex_hypercube.yaml",
                    help="RVE template for FEA fallback homogenization.",
                ),
                option(
                    flags=["--fea-jobs", "-j"],
                    arg_type=int,
                    default=1,
                    help="Parallel workers for FEA fallback batch.",
                ),
                option(
                    flags=["--cache-dir"],
                    arg_type=str,
                    default="",
                    help="LUT disk cache directory (default: results/fea_lut_cache).",
                ),
                option(
                    flags=["--no-disk-cache"],
                    flag=True,
                    help="Disable disk LUT cache.",
                ),
            ],
        ),
        command(
            name="register-surrogate",
            help="Register a trained FEA surrogate in b3_tex MICROMODELS.",
            callback=_register_surrogate_cmd,
            arguments=[
                argument(
                    name="model",
                    arg_type=str,
                    help="Path to trained surrogate .joblib.",
                ),
            ],
            options=[
                option(
                    flags=["--name"],
                    arg_type=str,
                    default="fea_hex",
                    help="Registry name for weave YAML micromodel field.",
                ),
            ],
        ),
    ],
)


def main() -> None:
    _app.run()


if __name__ == "__main__":
    main()
