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
b3-micromech sweep   examples/sweep_default.yaml --out results
```

## Tests

```sh
pytest
pytest -m mfem
```