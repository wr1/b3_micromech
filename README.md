# b3_micromech

[![CI](https://github.com/wr1/b3_micromech/actions/workflows/ci.yml/badge.svg)](https://github.com/wr1/b3_micromech/actions/workflows/ci.yml)

FEA homogenization of a 2-D transverse unidirectional (UD) composite RVE — a circular
fibre in a resin matrix under periodic boundary conditions, solved with
[MFEM](https://mfem.org/). It produces `(6, 6)` stiffness tensors used to train the
surrogate consumed by `b3_tex.micromodels.SurrogateModel`.

## Conventions

- Fibre axis = global **x** (Voigt index 11); the RVE mesh lives in the **y–z** plane.
- Plane strain along x: the fluctuating `ε_xx` from FE is zero, while the macro `ε_xx`
  enters via the periodic RHS / stress recovery. All six macro-strain modes are
  recovered from FEA.
- Voigt order `(11, 22, 33, 23, 13, 12)` with engineering shear strains.

## Setup

```sh
micromamba create -n b3-micromech -c conda-forge python=3.12
micromamba activate b3-micromech
pip install -e ".[viz,sweep,surrogate,test]"
```

PyMFEM (`mfem`), SciPy, and `triangle` are core dependencies installed with the
package. The optional extras add plotting (`viz`), parallel sweeps (`sweep`), the
surrogate stack (`surrogate`), and the test runner (`test`).

## Usage

```sh
# validate a config and print the Mori–Tanaka reference
b3-micromech validate  examples/ud_transverse.yaml
b3-micromech reference examples/ud_transverse.yaml

# homogenize a single RVE (square or hexagonal unit cell)
b3-micromech solve examples/ud_transverse.yaml     --out results/square
b3-micromech solve examples/ud_transverse_hex.yaml --out results/hex --plot

# adaptive mesh refinement around the fibre interface
b3-micromech solve examples/ud_transverse_hex_amr.yaml --out results/hex_amr

# render the six unit-macro-strain deformation / stress plots
b3-micromech plot examples/ud_transverse.yaml --out results/plots

# parameter-hypercube sweep → dataset, then train / use a surrogate
b3-micromech sweep             examples/sweep_hex_hypercube.yaml --out results --jobs 4
# default kind=physics (Chamis + ridge residual; best high-Vf extrapolation)
b3-micromech train-surrogate   results/dataset.npz --out results/surrogate_model.joblib
b3-micromech train-surrogate   results/dataset.npz -o results/mf_gp.joblib --kind mf_gp
b3-micromech train-surrogate   results/dataset.npz -o results/mlp.joblib --kind mlp
b3-micromech predict-surrogate results/surrogate_model.joblib
```

A `Makefile` wraps the common workflows (`make help` lists targets); it defaults to
`uv run`. Override with `make solve RUN="micromamba run -n b3-micromech"`.

### Geometry

- **Square** unit cell: `examples/ud_transverse.yaml`.
- **Hexagonal** unit cell: `examples/ud_transverse_hex.yaml` — flat-top regular hex
  (circumradius = `domain.size`), `triangle` mesher, periodic BCs across three opposite
  edge pairs (`src/b3_micromech/periodic.py`).
- **AMR**: `examples/ud_transverse_amr.yaml`, `examples/ud_transverse_hex_amr.yaml` —
  stiffness-jump refinement marker (`solver.amr.marker: stiffness_jump`).

### Plots (`pip install -e ".[viz]"`)

```sh
make plot-hex-amr   # hex + AMR bundle (includes dual-panel amr_refinement)
make plot-hex
```

| Figure | Content |
|---|---|
| `rve_overview.png` | Undeformed mesh, fibre disc |
| `loadcase_deformations.png` | 2×3 in-plane deflection quiver per unit strain |
| `fibre_displacement.png` | 2×3 fibre-direction `u_x` |
| `von_mises.png` | 2×3 element von Mises stress |
| `engineering_constants.png` | Homogenized E, G, ν bar chart |
| `amr_refinement.png` | Dual panel: cell-size grid refine + stiffness-jump marker |

#### Hex AMR showcase

Hexagonal unit cell with stiffness-jump AMR around the fibre interface
(`examples/ud_transverse_hex_amr.yaml`, `make plot-hex-amr`):

<p align="center">
  <img src="docs/images/hex_amr_rve_overview.jpg" width="360" alt="Hex AMR RVE overview mesh"/>
</p>

<p align="center">
  <img src="docs/images/hex_amr_refinement.jpg" width="720" alt="Dual-panel AMR: cell size and stiffness-jump marker"/>
</p>

*Left: relative cell size (grid refinement). Right: stiffness-jump marker with
threshold.*

<p align="center">
  <img src="docs/images/hex_amr_loadcases.jpg" width="720" alt="Six unit-strain deformation patterns"/>
</p>

*Six unit macro-strains — exaggerated in-plane fluctuation + quiver.*

<p align="center">
  <img src="docs/images/hex_amr_von_mises.jpg" width="720" alt="Element von Mises stress per loadcase"/>
</p>

*Element von Mises stress (MPa) for each loadcase.*

### Surrogate chain

Train a stiffness surrogate and feed it into `b3_tex`:

| kind | Model |
|------|--------|
| **`physics` (default)** | Chamis × ridge residual on eng. constants — best high-Vf extrapolation |
| `mf_gp` | Chamis × GP residual (optional κ; multi-fidelity residual GP) |
| `mlp` | Multi-output MLP (log-modulus features); use only on dense FEA grids |

Residual kinds keep sign-correct Vf/modulus trends; the fit only corrects magnitude
where hex FEA differs from Chamis.

**Upper-Vf edge samples are required.** Yarn compaction queries Vf near the hex packing
limit (~0.74–0.89 with 1% standoff). Put FEA budget on a dense packing cluster
(`hex_vf_sweep: full` or `high`, `dense_start` ≈ 0.74, `n_dense` ≥ 6–8) and keep the
LUT max Vf inside the training envelope. Mid-Vf-only grids under-sample the residual
that matters most.

- `examples/sweep_hex_hypercube.yaml` (Vf × E_m × E_Tf, mid + packing cluster) → `make demo-surrogate`
- `examples/sweep_hex_high_vf.yaml` (packing cluster only) → `make sweep-hex-high-vf`
- `examples/sweep_hex_3d_response.yaml` (Vf × E_m × E_Lf) → `make demo-surrogate-3d`
- `b3-micromech register-fea-micromech` registers either the trained surrogate or an
  on-the-fly FEA fallback (two-tier memory + disk LUT cache) as a `b3_tex` micromodel.
- Agent skill (train → register → yarn LUT): package root **`SKILL.md`**.

#### Surrogate response surfaces

3-axis hex sweep (Vf × E_m × E_Lf) → physics residual → structured response grids
(`make demo-surrogate-3d`):

<p align="center">
  <img src="docs/images/surrogate_response_e1.jpg" width="420" alt="E1 response surface"/>
  <img src="docs/images/surrogate_response_e2.jpg" width="420" alt="E2 response surface"/>
</p>

*Left: $E_1$ surface. Right: $E_2$ surface over the design hypercube.*

Holdout diagnostics from the standard hypercube demo (`make demo-surrogate`):

<p align="center">
  <img src="docs/images/surrogate_e2_vs_vf.jpg" width="420" alt="E2 vs fibre volume fraction"/>
  <img src="docs/images/surrogate_holdout_e2.jpg" width="360" alt="Holdout E2 parity"/>
</p>

*$E_2$ vs Vf trend and holdout parity.*

## Layout

- `src/b3_micromech/backends/` — the only place that imports `mfem`.
- `src/b3_micromech/quadrature.py` — implicit per-Gauss-point stiffness (`local_cloud`
  default; IDW blend for elements crossing the fibre boundary).
- `src/b3_micromech/amr.py` — adaptive refinement; `surrogate.py` (MLP),
  `physics_surrogate.py` (Chamis residual / mf_gp), `surrogate_response.py`.

## Development

```sh
pip install -e ".[viz,sweep,surrogate,test]"
pytest            # mfem-marked tests auto-skip if PyMFEM is unavailable
pytest -m mfem    # the full FE suite

pip install pre-commit && pre-commit install   # ruff lint + format on commit
```

CI (GitHub Actions) runs the fast suite on Python 3.10 and 3.12 (PyMFEM is omitted, so
`mfem`-marked tests are skipped) plus a `pre-commit` lint/format check.

## License

MIT
