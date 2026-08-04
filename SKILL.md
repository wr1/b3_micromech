---
name: b3-micromech
description: >
  Build consistent 3D UD/yarn stiffness sets for FEA from fibre+matrix+Vf via
  MFEM homogenization or surrogates (mlp|physics|mf_gp). Mesh defaults: hex
  packing, triangle, local_cloud res 6, mesh_resolution [22]–[24] (AMR for
  high Vf). Use for C_eff cards, Vf LUTs, train-surrogate, predict-batch,
  register_fea_micromech, micromodel: fea_hex, or sweep→dataset.
---

# b3_micromech

## Top-level use

**Produce consistent 3D material property sets for FEA** from constituents
(matrix, fibre, in-tow Vf):

| Consumer | What you hand off |
|----------|-------------------|
| Structural / continuum FEA | Transverse-isotropic yarn/`UD` card: `C` (6×6) or eng. constants `E₁,E₂,G₁₂,G₂₃,ν₁₂` |
| Weave RVE (`b3_tex`) | Registered micromodel → Vf-binned `stiffness_batch` LUT (same physics as micromech) |
| Dataset / ML | NPZ features + `C` samples; joblib surrogate for batch re-eval |

**Consistency rule:** one Voigt convention, one fibre axis (local 1 = fibre), one
feature contract, one path (FEA hex RVE *or* residual/MLP surrogate trained on
that FEA). Do **not** mix Chamis yarn cards with FEA-calibrated weave solves
without documenting the switch.

```text
constituents (E_m,ν_m, E_Lf,E_Tf,…) + Vf
        │
        ▼
  MFEM hex RVE  ──sweep──►  dataset.npz  ──train──►  *.joblib
        │                         │                      │
        ▼                         ▼                      ▼
   C_eff (6×6)              NPZ / plots           stiffness_batch(N)
        │                                              │
        └──────────► FEA material card ◄───────────────┘
                     or b3_tex micromodel: fea_hex
```

**Prefer:** hex packing RVE + **`physics` residual (default)** for production cards —
best high-Vf extrapolation. Use `mf_gp` when residual uncertainty/κ is needed; `mlp`
only on dense FEA grids. Bare Chamis is the low-fidelity base, not the FEA card.

## Conventions

| | |
|--|--|
| Fibre / local 1 | global **x** (Voigt 11) |
| Mesh | **y–z** plane strain; macro `ε_xx` via periodic RHS |
| Voigt | `(11,22,33,23,13,12)`, engineering shear |
| Features | `[Vf, E_m, ν_m, E_Lf, E_Tf, G_LTf, ν_LTf, G_TTf]` |

## Setup / CLI

```sh
pip install -e ".[viz,sweep,surrogate,test]"   # mfem core; same env as b3-tex for registration
# make RUN="micromamba run -n b3-tex" …

b3-micromech solve examples/ud_transverse_hex.yaml --out results/hex --plot
b3-micromech solve examples/ud_transverse_hex_amr.yaml --out results/hex_amr
b3-micromech sweep examples/sweep_hex_hypercube.yaml --out results/surrogate_demo --jobs 4
b3-micromech train-surrogate results/surrogate_demo/dataset.npz -o results/surrogate_model.joblib
# default kind=physics; override with --kind mf_gp|mlp
b3-micromech predict-batch results/physics.joblib --vf-linspace 0.55:0.88:10000 -o results/batch.npz
```

| Make | |
|------|--|
| `demo-surrogate` | sweep → physics residual → holdout |
| `demo-surrogate-3d` | Vf×E_m×E_Lf surfaces (physics default) |
| `demo-mesomech-batch` | register + batch Vf |
| `plot-hex-amr` | thin-edge mesh + dual-panel grid refine |

YAML: `ud_transverse_hex.yaml` (preferred), `*_amr.yaml`, `sweep_hex_{hypercube,high_vf,3d_response}.yaml`.

## Mesh parameters (good behaviour)

**Default for FEA cards / surrogate training: hexagonal packing**, not square.
Hex matches wide-side fibre packing; square is validation / legacy only.

| Knob | Where | Meaning |
|------|--------|---------|
| `domain.shape` | YAML | `hexagon` (prod) \| square (omit / non-hex) |
| `domain.size` | YAML | hex **circumradius**; square side length |
| `domain.mesh_resolution` | YAML | **hex:** `[N]` → `edge_divisions=N` on each flat; **square:** `[nx, ny]` |
| `solver.cell_type` | YAML | hex → `triangle`; square → `quadrilateral` |
| `material_sampling` | YAML | phase assignment at GPs (implicit disc, not body-fitted) |
| `solver.amr.*` | YAML | optional stiffness-jump refine on the base mesh |

### Sampling (always set explicitly for production)

```yaml
solver:
  cell_type: triangle          # hex
  material_sampling:
    strategy: local_cloud      # default preferred; not cell_constant
    resolution: 6              # cloud density; 3 = coarse, 6 = standard
    idw_power: 2.0
```

| strategy | Use |
|----------|-----|
| `local_cloud` | **default** — robust at fibre/matrix interface |
| `exact` | per-GP hard phase (no blend); debugging |
| `cell_constant` | centroid only — **too crude** for cards / sweeps |

### Resolution ladder

| Goal | Hex `mesh_resolution` | AMR | Notes |
|------|----------------------|-----|--------|
| smoke / agent iterate | `[12]`–`[16]` | off or 1 iter | seconds-class |
| **standard card / sweep** | **`[22]`–`[24]`** | off | matches `sweep_hex_hypercube` / `ud_transverse_hex` |
| high Vf (≳0.75) or publish | `[22]`+ **or** base `[16]`+AMR | on | thin matrix ligaments need interface resolution |
| AMR publish | base `[16]`, see below | 2 iters | `ud_transverse_hex_amr.yaml` |

Square smoke: `[32,32]` quads. Do **not** use square packing for production yarn cards.

**Freeze mesh for a dataset.** Changing `mesh_resolution` / AMR / sampling mid-sweep
invalidates surrogate consistency — retrain or tag provenance.

### AMR (when enabled)

```yaml
solver:
  amr:
    enabled: true
    marker: stiffness_jump
    max_iterations: 2
    threshold: 0.10          # lower → more refine; 0.10–0.15 typical
    n_samples_per_cell: 15   # tri: triangular number k(k+1)/2 (10,15,21…); quad: perfect square
    dof_budget: 80000        # stop if projected DOFs exceed
    # n_uniform_refines: 0
```

| Behaviour | |
|-----------|--|
| Marker | in-cell stiffness jump on sub-samples → flag interface cells |
| Stop | no cells flagged **or** `dof_budget` |
| Plot | `amr_refinement.png` — cell size + marker (thin edges) |
| Cost | each iter remeshes; keep base modest (`[16]`) and let AMR work |

**High-Vf sweeps:** keep a packing cluster (`hex_vf_sweep: full` or `high`) for
surrogate training — see [Training data — sample the upper Vf edge](#training-data--sample-the-upper-vf-edge).
Mesh: standard res + `local_cloud` res 6 minimum; packing limit is where coarse meshes fail.

### Quick checks

- `make plot-hex-amr` — edges should resolve the fibre ring; dual-panel shows refine.
- Homogeneous matrix only → `C` ≈ isotropic matrix (periodicity sanity).
- Raise base `N` or lower AMR `threshold` if high-Vf `E₂`/`G` still drift vs finer run.

## Surrogate kinds

| kind | Form | When |
|------|------|------|
| **`physics` (default)** | Chamis × exp(φ·c) ridge residual | production default; best high-Vf extrap |
| `mf_gp` | Chamis × GP residual (±κ) | uncertainty / conservative capacity |
| `mlp` | multi-out MLP, log-moduli, E₂/Vf weights | dense FEA only |

All: `load_surrogate` → `predict((N,8))→(N,6,6)` tensorized (batched TI assembly).
`as_predict_callable()` is batch-aware for `b3_tex.SurrogateModel`.

### Training data — sample the upper Vf edge

Yarn LUTs and compaction push Vf toward the **hex packing limit**
(~0.74–0.89 with 1% standoff; hard limit π/(2√3)≈0.907). Residuals and
errors grow fastest there (thin matrix ligaments). **Do not train only on
mid-Vf** and hope residual models extrapolate cleanly to the edge.

| Rule | Practice |
|------|----------|
| Always include a dense packing cluster | Prefer `hex_vf_sweep: full` (mid + edge) or `high` (edge only for refinement) |
| Put FEA budget at the edge | Cluster from `dense_start` ≈ 0.74 up to standoff-limited max (`n_dense` ≥ 6–8) |
| Cover the Vf range you will query | `predict-batch` / `build_lut` max Vf must sit **inside** the training Vf envelope |
| Mesh at high Vf | ≥ standard res + `local_cloud` res 6; AMR if publishing near packing |
| Prefer residual kinds | `physics` default; still **needs** edge FEA labels — base alone is not enough |

Ready-made sweeps:

- `examples/sweep_hex_hypercube.yaml` — `hex_vf_sweep: full` (mid-range + packing cluster)
- `examples/sweep_hex_high_vf.yaml` — packing cluster only (`make sweep-hex-high-vf`)

Custom:

```yaml
sweep:
  vf:
    hex_vf_sweep:
      preset: full          # or high
      standoff: 0.01
      dense_start: 0.74     # start of packing cluster
      n_dense: 8            # points along the upper edge
```

MLP training already **upweights** high-Vf / high-$E_2$ rows; residual kinds
still need those rows **present** in the dataset.

## FEA material card (API)

```python
from b3_micromech.mesomech import (
    predict_stiffness_batch, constituents_from_yaml, register_fea_micromech,
)
from b3_micromech.tensors import engineering_constants_transverse_iso

matrix, fibre = constituents_from_yaml("examples/sweep_hex_hypercube.yaml")
C = predict_stiffness_batch("results/physics.joblib", vf_array, matrix, fibre)  # (N,6,6)
ec = engineering_constants_transverse_iso(C[i])  # single tensor → E1,E2,…
```

## b3_tex yarn LUT

Same session as the weave solve:

```python
register_fea_micromech("results/physics.joblib", name="fea_hex", n_jobs=4)
# YAML: micromodel: fea_hex  on material.type: micromechanical
```

| Mode | `stiffness_batch` | Disk LUT default |
|------|-------------------|------------------|
| joblib present | one vectorized predict | off |
| missing / `None` | MFEM per Vf (`n_jobs`) | on (`results/fea_lut_cache/`, `B3_MICROMECH_LUT_CACHE`) |

`build_lut(n_bins=256)` → one batch. Features clipped to model width (8).
Default FEA template: package `examples/sweep_hex_hypercube.yaml` (not CWD-relative).

## Module map

| Path | |
|------|--|
| `mesomech.py` | register, batch predict, FEA fallback |
| `physics_surrogate.py` / `surrogate.py` | train/load kinds |
| `reference.py` / `tensors.py` | Chamis/MT; batched TI `C` |
| `homogenize.py` / `sweep.py` / `export.py` | solve → NPZ |
| `amr.py` / `plot.py` | refine; thin-edge + cell-size AMR panels |
| `backends/mfem_periodic_2d.py` | only `mfem` import |

## Failures

| Symptom | Fix |
|---------|-----|
| `unknown micromodel 'fea_hex'` | register in **same** process before load |
| missing joblib | `register_fea_micromech(None)` or train |
| slow first LUT | FEA disk cache / `n_jobs` |
| OOB warnings | Vf/moduli outside train bounds |

```sh
pytest tests/test_physics_surrogate.py tests/test_mesomech.py
pytest -m mfem
pre-commit run --all-files
```

Sibling weave package: `b3_tex/SKILL.md`.
