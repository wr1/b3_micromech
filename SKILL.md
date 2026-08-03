---
name: b3-micromech
description: >
  FEA micromechanics for a 2-D transverse UD composite RVE (MFEM), stiffness
  surrogates (mlp / physics / mf_gp), and registration into b3_tex as a yarn
  micromodel. Use for homogenize, sweep, train-surrogate, predict-batch,
  register_fea_micromech, LUT cache, FEA fallback, or micromodel: fea_hex.
---

# b3_micromech — agent skill

Homogenize a **2-D transverse UD RVE** (circular fibre in resin, periodic BCs) with
MFEM → `(6, 6)` stiffness. Train a surrogate; register it so **b3_tex** yarn materials
(`micromodel: fea_hex`) use a **tensorized** `stiffness_batch` LUT instead of Chamis.

**Sibling package:** weave-level RVE solves live in `b3_tex` (repo root `SKILL.md`).

## Conventions

| Item | Rule |
|------|------|
| Fibre axis | global **x** (Voigt 11) |
| Mesh plane | **y–z** |
| Strain | plane strain along x; macro `ε_xx` via periodic RHS |
| Voigt | `(11, 22, 33, 23, 13, 12)`, engineering shear |
| Surrogate features | `[Vf, E_m, ν_m, E_Lf, E_Tf, G_LTf, ν_LTf, G_TTf]` (8 mechanical) |

## Setup

```sh
pip install -e ".[viz,sweep,surrogate,test]"   # + mfem (core)
# b3_tex registration: same env as b3-tex, or pip install -e ".[tex]"
```

Makefile defaults to `micromamba run -n b3-tex` — override: `make solve RUN="…"`.

## Quick CLI

```sh
b3-micromech validate  examples/ud_transverse_hex.yaml
b3-micromech reference examples/ud_transverse_hex.yaml   # Mori–Tanaka + Chamis
b3-micromech solve     examples/ud_transverse_hex.yaml --out results/hex --plot
b3-micromech solve     examples/ud_transverse_hex_amr.yaml --out results/hex_amr

b3-micromech sweep examples/sweep_hex_hypercube.yaml --out results/surrogate_demo --jobs 4
b3-micromech train-surrogate results/surrogate_demo/dataset.npz \
  -o results/surrogate_demo/surrogate_model.joblib              # mlp
b3-micromech train-surrogate results/surrogate_demo/dataset.npz \
  -o results/surrogate_demo/physics.joblib --kind physics
b3-micromech train-surrogate results/surrogate_demo/dataset.npz \
  -o results/surrogate_demo/mf_gp.joblib --kind mf_gp

b3-micromech predict-surrogate results/surrogate_demo/surrogate_model.joblib
b3-micromech predict-batch MODEL.joblib --vf-linspace 0.55:0.88:100000 -o results/batch.npz
```

| Make target | What |
|-------------|------|
| `make demo-surrogate` | hex hypercube sweep → MLP → holdout plots |
| `make demo-surrogate-3d` | Vf×E_m×E_Lf → response surfaces |
| `make demo-mesomech-batch` | register + large Vf batch predict |
| `make sweep-hex-high-vf` | packing-limit Vf cluster only |

## Geometry examples

| YAML | Domain |
|------|--------|
| `examples/ud_transverse.yaml` | square cell |
| `examples/ud_transverse_hex.yaml` | flat-top hex (preferred for packing / surrogates) |
| `examples/ud_transverse_*_amr.yaml` | stiffness-jump AMR |
| `examples/sweep_hex_hypercube.yaml` | Vf × E_m × E_Tf training grid |
| `examples/sweep_hex_high_vf.yaml` | high-Vf packing cluster |
| `examples/sweep_hex_3d_response.yaml` | Vf × E_m × E_Lf response demo |

Hex: circumradius = `domain.size`, `triangle` mesh, three opposite edge pairs periodic.

---

## Surrogate kinds

All kinds share the same joblib load path (`load_surrogate`) and mesomech API.

| kind | Train | Form | Prefer when |
|------|-------|------|-------------|
| `mlp` | `--kind mlp` (default) | multi-output MLP, log-modulus features, E₂/Vf weights | dense FEA grid |
| `physics` | `--kind physics` | Chamis × exp(φ·c) ridge residual on eng. constants | scarce data; monotone trends |
| `mf_gp` | `--kind mf_gp` | Chamis × GP residual (optional κ) | residual uncertainty / conservative moduli |

**Tensorized path:** residual kinds map features → engineering constants →
`transverse_isotropic_stiffness_batch` → `(N, 6, 6)` with **no Python sample loop**.
MLP `predict` is batched sklearn. `as_predict_callable()` is batch-aware for
`b3_tex.micromodels.SurrogateModel`.

---

## b3_tex integration (mesomech)

Yarn stiffness at local in-tow Vf comes from a registered micromodel.
`MicromechanicalMaterial.build_lut(n_bins=256)` calls `stiffness_batch` **once** —
vectorization matters.

### Register (same Python session as the weave solve)

```python
from b3_micromech.mesomech import register_fea_micromech

register_fea_micromech(
    "results/surrogate_demo/physics.joblib",  # mlp | physics | mf_gp, or None → FEA
    name="fea_hex",
    n_jobs=4,                                 # FEA fallback parallelism
    cache_dir="results/fea_lut_cache",
)
```

```sh
b3-micromech register-fea-micromech results/surrogate_demo/physics.joblib --name fea_hex
b3-micromech register-fea-micromech   # no joblib → FEA on-the-fly
```

### Weave YAML

```yaml
materials:
  - name: yarn
    type: micromechanical
    matrix: matrix
    fibre: fibre
    micromodel: fea_hex          # registered name, not a built-in
    nominal_fibre_volume_fraction: 0.55
    max_fibre_volume_fraction: 0.90
```

### Modes (`FeaMicromechMicromodel`)

| Mode | When | `stiffness_batch` |
|------|------|-------------------|
| **surrogate** | joblib loads | one `predict((N, 8)) → (N, 6, 6)` |
| **fea** | missing / omitted joblib | MFEM homogenize per Vf (`n_jobs`) |

Features: mechanical 8-cols only for stiffness models. Optional thermal columns
from `build_feature_matrix` are clipped to `feature_bounds` width.
`b3_tex.materials.Material` (no thermal tensors) is supported.

Default RVE template for FEA fallback / cache keys:
`examples/sweep_hex_hypercube.yaml` (resolved from package root, not CWD).

### LUT cache

Order in `stiffness_batch`:

1. In-memory `_mem_cache`
2. Disk `results/fea_lut_cache/<key>.npz` (`B3_MICROMECH_LUT_CACHE` overrides dir)
3. Compute (surrogate or FEA), then store

Key = RVE template + constituent moduli + mode + Vf vector.

| Mode | Disk cache default |
|------|--------------------|
| FEA | **on** (256 MFEM solves once per key) |
| Surrogate | **off** (predict is cheap) |

### Batch API (outside weave)

```python
from b3_micromech.mesomech import predict_stiffness_batch, constituents_from_yaml

matrix, fibre = constituents_from_yaml("examples/sweep_hex_hypercube.yaml")
C = predict_stiffness_batch(model, vf_array, matrix, fibre)  # (N, 6, 6)
```

---

## Layout (where to edit)

| Path | Role |
|------|------|
| `backends/mfem_periodic_2d.py` | only MFEM import site |
| `quadrature.py` | implicit per-GP stiffness (`local_cloud`) |
| `amr.py` | stiffness-jump refinement |
| `homogenize.py` / `sweep.py` | single solve / parameter sweeps |
| `surrogate.py` | MLP train / predict |
| `physics_surrogate.py` | physics + mf_gp; `load_surrogate` / `train_surrogate` |
| `reference.py` | Chamis + Mori–Tanaka closed forms |
| `tensors.py` | Voigt + batched TI stiffness / eng. constants |
| `features.py` | 8-feature (optional +4 thermal) matrix |
| `mesomech.py` | `FeaMicromechMicromodel`, register, batch predict |
| `lut_cache.py` | disk LUT fingerprints |
| `surrogate_response.py` | 3-axis response grids |
| `examples/demo_surrogate_chain.py` | end-to-end train + plots |
| `examples/demo_mesomech_batch.py` | register + Vf batch |

---

## Common failures

| Symptom | Fix |
|---------|-----|
| `unknown micromodel 'fea_hex'` | `register_fea_micromech` in the **same** process before loading YAML |
| `FileNotFoundError` on joblib | pass `None` for FEA fallback, or train first |
| Slow first LUT | FEA mode: enable disk cache / raise `n_jobs` |
| OOB feature warnings | Vf or moduli outside training hypercube; retrain or clip Vf range |
| Wrong feature width | stiffness models need 8 cols; thermal extras are auto-sliced |
| Registration from `b3_tex` cwd | package resolves default RVE YAML by package path — OK |

## Tests

```sh
pytest                              # mfem-marked tests skip without PyMFEM
pytest tests/test_physics_surrogate.py tests/test_mesomech.py
pytest -m mfem                      # full FE suite
```
