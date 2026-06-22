# b3_micromech

FEA-based micromechanics homogenization of unidirectional composite transverse RVEs.

A square periodic cell with a centred circular fibre is solved under MFEM with
plane-strain periodic boundary conditions. The result is a `(6, 6)` transverse-isotropic
stiffness tensor suitable for training surrogates used in [`b3_tex`](../b3_tex)
mesomechanics (`SurrogateModel` feature vector).

## Setup

```sh
micromamba create -n b3-micromech -c conda-forge python=3.12 numpy pyyaml pytest
micromamba activate b3-micromech
pip install treeparse mfem
pip install -e .
```

## Usage

```sh
b3-micromech validate examples/ud_transverse.yaml
b3-micromech solve   examples/ud_transverse.yaml --out results
b3-micromech solve   examples/ud_transverse.yaml --out results --plot
b3-micromech plot    examples/ud_transverse.yaml --out results/plots
b3-micromech sweep   examples/sweep_default.yaml --out results

# or the bundled example script
python examples/plot_loadcases.py
bash examples/run_visualization.sh
```

Plot bundle (requires `pip install b3-micromech[viz]`):

| Figure | Content |
|---|---|
| `rve_overview.png` | Undeformed mesh, fibre disc |
| `loadcase_deformations.png` | 2×3 in-plane deflection quiver per unit strain |
| `fibre_displacement.png` | 2×3 fibre-direction \(u_x\) |
| `von_mises.png` | 2×3 element von Mises stress |
| `engineering_constants.png` | Homogenized E, G, ν bar chart |

## Tests

```sh
pytest
pytest -m mfem
```