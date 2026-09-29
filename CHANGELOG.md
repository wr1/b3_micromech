# Changelog

## [0.2.0] - unreleased

### Added
- MIT `LICENSE`.
- Committed `uv.lock`.
- Dependabot for GitHub Actions, uv, and pre-commit.
- CI runs the MFEM solver on Python 3.11 and 3.13 (`uv sync --locked`), plus an
  import smoke test, a wheel install, and a `b3_tex` integration job pinned in
  `ci/b3_tex.ref`.
- Golden `C` / CTE / conductivity arrays for the three bundled RVEs, and a
  schema-less physics joblib fixture.

### Changed
- Omitted material sampling is `local_cloud` at resolution 6. Square and hex
  example RVEs that did not pin a strategy move by about 2.4% and 0.86% of
  max|C| respectively. Pin `material_sampling: {strategy: cell_constant}` to
  reproduce 0.1.0.
- Unknown configuration keys and enum values raise `ConfigError`.
- Fibre volume fraction above the packing limit raises; above the 1% standoff
  maximum it warns.
- Requires Python >= 3.11.
- `scipy` and `triangle` are core dependencies. `mfem` is pinned to `>=4.8,<4.11`.
  `treeparse` is pinned to `>=0.3,<0.4`. `scikit-learn` is pinned to `>=1.3,<2`.
- The `tex` extra is removed (`b3-tex` is not on PyPI). The empty `hex-mesh` extra
  remains so existing `.[hex-mesh]` installs keep working.
- Makefile default prefix is `uv run`.

### Fixed
- Mori–Tanaka longitudinal shear used the matrix Young's modulus instead of its shear modulus.
- `--plot` and `--no-disk-cache` required a value; they are boolean flags.
- Duplicate material names were silently last-wins.

## [0.1.0] - 2026-08-03

- First tagged release: MFEM periodic plane-strain UD RVE homogenisation (square/hex, implicit
  material sampling, AMR), thermal CTE/conductivity, hypercube sweeps, `physics`/`mf_gp`/`mlp`
  surrogates, b3_tex micromodel adapter. (HEAD `48d53b8`, one commit later, made `physics`
  the default surrogate without a version bump.)
