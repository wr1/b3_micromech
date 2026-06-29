---
name: mesomech-surrogate
description: >
  Train and register b3_micromech FEA stiffness surrogates for b3_tex mesomechanics.
  Use when the user asks about mesomech surrogate, FEA micromodel registration,
  vectorized Vf batch prediction, LUT disk cache, FEA on-the-fly fallback,
  register_fea_micromech, predict-batch, or swapping chamis for a trained surrogate
  in weave YAML (micromodel: fea_hex).
---

# Mesomech FEA surrogate integration

`b3_micromech` trains MLP surrogates on hex RVE sweeps; `b3_tex` consumes them via
the pluggable `micromodel` registry on `material.type: micromechanical` yarns.

## Feature contract (8 inputs)

`[Vf, E_m, nu_m, E_Lf, E_Tf, G_LTf, nu_LTf, G_TTf]` — built by
`b3_micromech.features.build_feature_matrix` from matrix/fibre materials + local Vf.

## Training pipeline

```sh
make demo-surrogate          # sweep → train → diagnostics
# or stepwise:
b3-micromech sweep examples/sweep_hex_hypercube.yaml --out results/surrogate_demo
b3-micromech train-surrogate results/surrogate_demo/dataset.npz -o results/surrogate_demo/surrogate_model.joblib
```

High-Vf refinement sweeps: `make sweep-hex-high-vf`, `examples/sweep_hex_high_vf.yaml`.

## Register before any b3_tex solve

Registration is **Python-only** (no YAML `surrogate_model:` path). Must run in the
same Python session before loading weave YAML:

```python
from b3_micromech.mesomech import register_fea_micromech

register_fea_micromech(
    "results/surrogate_demo/surrogate_model.joblib",  # or None → FEA fallback
    name="fea_hex",
    n_jobs=4,
    cache_dir="results/fea_lut_cache",
)
```

CLI equivalent:

```sh
b3-micromech register-fea-micromech results/surrogate_demo/surrogate_model.joblib --name fea_hex
b3-micromech register-fea-micromech   # no joblib → FEA on-the-fly fallback
```

Then in weave YAML:

```yaml
materials:
  - name: yarn
    type: micromechanical
    matrix: matrix
    fibre: fibre
    micromodel: fea_hex    # registered name, not a built-in
    nominal_fibre_volume_fraction: 0.55
    max_fibre_volume_fraction: 0.90
```

## Modes (`FeaMicromechMicromodel`)

| Mode | Trigger | `stiffness_batch` |
|------|---------|-------------------|
| **surrogate** | joblib loads | one vectorized MLP call for N Vf points |
| **fea** | no joblib / missing file | MFEM homogenize per Vf (parallel via `n_jobs`) |

b3_tex `MicromechanicalMaterial.build_lut(n_bins=256)` calls `stiffness_batch` once
with 256 bin centres — vectorization matters.

## Two-tier LUT cache

Lookup order in `stiffness_batch`:

1. In-memory `_mem_cache`
2. Disk `results/fea_lut_cache/<key>.npz` (override: `B3_MICROMECH_LUT_CACHE`)
3. Compute (MLP or FEA), then write memory + disk

Cache key hashes: RVE template (domain/mesh/solver) + constituent moduli + mode + Vf vector.

- **FEA fallback:** disk cache on by default (first LUT ≈ 256 MFEM solves once per key).
- **Surrogate:** disk cache off by default (MLP is fast).

## Batch inference (outside weave solve)

```sh
b3-micromech predict-batch MODEL --vf-linspace 0.55:0.88:100000 -o results/batch.npz
```

```python
from b3_micromech.mesomech import predict_stiffness_batch, constituents_from_yaml
matrix, fibre = constituents_from_yaml("examples/sweep_hex_hypercube.yaml")
C = predict_stiffness_batch(micromodel, vf_array, matrix, fibre)  # (N, 6, 6)
```

## Key modules

| Path | Role |
|------|------|
| `src/b3_micromech/mesomech.py` | `FeaMicromechMicromodel`, registration API |
| `src/b3_micromech/lut_cache.py` | disk cache fingerprints, load/save |
| `src/b3_micromech/features.py` | 8-feature matrix builder |
| `examples/demo_mesomech_batch.py` | end-to-end demo (`make demo-mesomech-batch`) |

## Common failures

| Error | Fix |
|-------|-----|
| `unknown micromodel 'fea_hex'` | call `register_fea_micromech` before solve |
| `FileNotFoundError` on joblib | use `register_fea_micromech(None)` for FEA fallback, or train surrogate first |
| slow first LUT | expected in FEA mode; use `--fea-jobs` / disk cache for reuse |

## Dependencies

- Surrogate: `pip install -e ".[surrogate,sweep]"`
- b3_tex integration: `pip install -e ".[tex]"` or same env as `b3-tex`