"""treeparse CLI for b3_micromech."""

from pathlib import Path

import numpy as np
from treeparse import argument, cli, command, option

from b3_micromech.homogenize import homogenize, surrogate_features
from b3_micromech.plot import render_all_figures
from b3_micromech.postprocess import solve_all_loadcases
from b3_micromech.problem import RVEProblem
from b3_micromech.reference import mori_tanaka_cylinder
from b3_micromech.sweep import sweep_to_file
from b3_micromech.tensors import engineering_constants_transverse_iso


def _validate_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    print(f"OK: loaded RVE from {config}")
    print(f"  domain_size = {problem.domain_size}")
    print(f"  mesh_resolution = {problem.mesh_resolution}")
    print(f"  cell_type = {problem.cell_type}")
    print(f"  vf = {problem.fibre_volume_fraction:.4f}  (r = {problem.fibre_radius:.4f})")
    print(f"  materials = {sorted(problem.materials)}")


def _reference_cmd(config: str) -> None:
    problem = RVEProblem.from_yaml(config)
    matrix = problem.materials[problem.matrix_material]
    fibre = problem.materials[problem.fibre_material]
    Cmt = mori_tanaka_cylinder(
        matrix=matrix,
        fibre=fibre,
        fibre_volume_fraction=problem.fibre_volume_fraction,
    )
    ec = engineering_constants_transverse_iso(Cmt)
    print(f"fibre volume fraction = {problem.fibre_volume_fraction:.4f}")
    print("Mori-Tanaka engineering constants (axis 1 = fibre):")
    for k, v in ec.items():
        if k.startswith("e_") or k.startswith("g_"):
            print(f"  {k} = {v/1e9:.4f} GPa")
        else:
            print(f"  {k} = {v:.4f}")


def _solve_cmd(config: str, out: str, plot: bool, plot_scale: float) -> None:
    problem = RVEProblem.from_yaml(config)
    result = homogenize(problem)
    C = result.effective_stiffness
    ec = engineering_constants_transverse_iso(C)
    print("Effective stiffness engineering constants (axis 1 = fibre):")
    print(f"  E1 = {ec['e_l']/1e9:.4f} GPa")
    print(f"  E2 = {ec['e_t']/1e9:.4f} GPa")
    print(f"  G12 = {ec['g_lt']/1e9:.4f} GPa")
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
        scale = None if plot_scale < 0 else plot_scale
        _plot_cmd(config, str(out_dir / "plots"), scale)


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
                    arg_type=bool,
                    default=False,
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
            arguments=[argument(name="config", arg_type=str, help="Path to sweep YAML.")],
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
    ],
)


def main() -> None:
    _app.run()


if __name__ == "__main__":
    main()