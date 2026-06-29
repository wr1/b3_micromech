#!/usr/bin/env python3
"""Generate deformation and stress plots for the six homogenization loadcases.

Writes a plot bundle under ``results/plots/``:

  - ``rve_overview.png``           — undeformed mesh + fibre disc
  - ``loadcase_deformations.png``  — 2×3 in-plane deflection patterns (quiver)
  - ``fibre_displacement.png``     — 2×3 fibre-direction u_x
  - ``von_mises.png``              — 2×3 element von Mises stress
  - ``engineering_constants.png``  — bar chart from recovered C_eff

Usage::

    python examples/plot_loadcases.py
    python examples/plot_loadcases.py examples/ud_transverse.yaml results/plots
    python examples/plot_loadcases.py examples/ud_transverse.yaml --scale 0.08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from b3_micromech.plot import render_all_figures
from b3_micromech.postprocess import solve_all_loadcases
from b3_micromech.problem import RVEProblem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "config",
        nargs="?",
        default=str(REPO / "examples" / "ud_transverse.yaml"),
        help="RVE YAML (default: examples/ud_transverse.yaml)",
    )
    ap.add_argument(
        "out_dir",
        nargs="?",
        default=str(REPO / "results" / "plots"),
        help="Output directory for PNGs",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="In-plane deflection exaggeration factor (default: auto)",
    )
    args = ap.parse_args()

    problem = RVEProblem.from_yaml(args.config)
    print(f"Loaded {args.config}")
    print(f"  Vf={problem.fibre_volume_fraction:.3f}  mesh={problem.mesh_resolution}")

    loadcases = solve_all_loadcases(problem)
    paths = render_all_figures(loadcases, args.out_dir, scale=args.scale)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
