"""Physics-base + residual stiffness surrogates (b3_invsec pattern).

Closed-form Chamis carries decades and monotone trends (E1 Voigt, ET/GLT rise
with √Vf). Only a small residual is fitted on FEA labels:

    modulus:   y_hat = y_chamis · exp(φ(x)·c)     # or exp(g_GP(x))
    Poisson:   y_hat = y_chamis + φ(x)·c

``PhysicsResidualSurrogate`` uses ridge least squares on φ.
``MultiFidelityGPSurrogate`` fits one GP per engineering-constant residual
(log residual for moduli) — multi-fidelity residual regression as in
``b3_invsec.multi_fidelity_residual``.

Both expose the same surface as :class:`StiffnessSurrogate` for mesomech:
``predict``, ``feature_bounds``, ``save`` / ``load``, ``as_predict_callable``.

**Training data:** always include FEA points on the upper Vf edge (hex packing
cluster, ``hex_vf_sweep: full`` or ``high``). Yarn LUTs query near packing;
mid-Vf-only grids under-sample the residual that residual kinds are meant to
learn. See package ``SKILL.md`` / README surrogate sections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from b3_micromech.reference import (
    EC_MODULUS_KEYS,
    EC_POISSON_KEYS,
    EC_TARGET_KEYS,
    chamis_engineering_constants_from_features,
)
from b3_micromech.surrogate import feature_bounds_from_training
from b3_micromech.tensors import (
    engineering_constants_transverse_iso_batch,
    transverse_isotropic_stiffness_batch,
)

# Cap log residuals that would explode the exponential (failed / wild FEA rows).
_R_CLIP = 3.0  # |r| > 3 → ~20×; treat as outlier when fitting
_EPS = 1e-30
N_RESIDUAL_BASIS = 9  # columns of residual_basis

SurrogateKind = Literal["mlp", "physics", "mf_gp"]

# Default for train-surrogate / demos. Chosen for high-Vf extrapolation on hex
# FEA (train Vf<0.7 → test Vf≥0.7): physics residual beats mf_gp and MLP on
# mean/max Frobenius and E₂ error while staying monotone by construction.
# Use ``mf_gp`` when residual uncertainty / κ is needed; ``mlp`` only on dense grids.
DEFAULT_SURROGATE_KIND: SurrogateKind = "physics"


def residual_basis(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dimensionless residual features φ(x) for Chamis correction.

    Columns (with intercept):
    ``1, vf, √vf, ln(E_Tf/E_m), ln(E_Lf/E_m), ln(G_LTf/G_m), ln(G_TTf/G_m),
    ν_m, ν_LTf``.

    High Vf and modulus contrast are exactly where Chamis drifts from hex FEA;
    the basis is built to let a low-order fit absorb that without inventing
    inverted trends.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] < 8:
        raise ValueError(f"expected >=8 feature columns, got shape {x.shape}")

    n = x.shape[0]
    vf = x[:, 0]
    em = np.maximum(x[:, 1], _EPS)
    num = x[:, 2]
    elf = np.maximum(x[:, 3], _EPS)
    etf = np.maximum(x[:, 4], _EPS)
    gltf = np.maximum(x[:, 5], _EPS)
    nultf = x[:, 6]
    gttf = np.maximum(x[:, 7], _EPS)
    gm = em / (2.0 * (1.0 + num))

    out = np.empty((n, N_RESIDUAL_BASIS), dtype=float)
    out[:, 0] = 1.0
    out[:, 1] = vf
    out[:, 2] = np.sqrt(np.clip(vf, 0.0, 1.0))
    out[:, 3] = np.log(etf / em)
    out[:, 4] = np.log(elf / em)
    out[:, 5] = np.log(gltf / np.maximum(gm, _EPS))
    out[:, 6] = np.log(gttf / np.maximum(gm, _EPS))
    out[:, 7] = num
    out[:, 8] = nultf
    return out


def _fea_engineering_constants_batch(
    stiffness: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Batch engineering constants including ``g_tt`` / ``nu_tt`` (vectorized)."""
    return engineering_constants_transverse_iso_batch(stiffness)


def _stiffness_from_ec_dict(
    ec: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Assemble ``(N, 6, 6)`` from engineering-constant arrays (no sample loop)."""
    return transverse_isotropic_stiffness_batch(
        e_l=ec["e_l"],
        e_t=ec["e_t"],
        g_lt=ec["g_lt"],
        nu_lt=ec["nu_lt"],
        nu_tt=ec["nu_tt"],
    )


def _stack_coefs(
    coefs: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Stack per-target residual coefs into ``(n_phi, n_targets)`` for one matmul."""
    return np.column_stack([np.asarray(coefs[k], dtype=float) for k in EC_TARGET_KEYS])


def _ec_from_residual_matrix(
    base: dict[str, NDArray[np.float64]],
    residual: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Apply stacked residual columns (moduli log, Poisson additive)."""
    out: dict[str, NDArray[np.float64]] = {}
    for j, key in enumerate(EC_MODULUS_KEYS):
        out[key] = _apply_modulus_residual(base[key], residual[:, j])
    n_mod = len(EC_MODULUS_KEYS)
    for j, key in enumerate(EC_POISSON_KEYS):
        out[key] = _apply_poisson_residual(base[key], residual[:, n_mod + j])
    return out


def _fit_ridge(
    phi: NDArray[np.float64],
    residual: NDArray[np.float64],
    *,
    alpha: float,
) -> NDArray[np.float64]:
    from sklearn.linear_model import Ridge

    ok = np.isfinite(residual)
    if ok.sum() < phi.shape[1]:
        return np.zeros(phi.shape[1], dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(phi[ok], residual[ok])
    return np.asarray(model.coef_, dtype=float)


def _apply_modulus_residual(
    base: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> NDArray[np.float64]:
    r = np.clip(residual, -_R_CLIP, _R_CLIP)
    return np.maximum(base, _EPS) * np.exp(r)


def _apply_poisson_residual(
    base: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> NDArray[np.float64]:
    return base + residual


@dataclass
class PhysicsResidualSurrogate:
    """Chamis base × exp(φ·c) residual on engineering constants.

    Drop-in for :class:`~b3_micromech.surrogate.StiffnessSurrogate` prediction API.

    Batch path is fully vectorized: residual φ is ``(N, 9)``, one matmul
    ``φ @ coef_matrix`` → residual, then batched TI stiffness assembly.
    """

    coefs: dict[str, NDArray[np.float64]]
    feature_bounds: NDArray[np.float64]
    ridge_alpha: float = 1e-3
    base_model: str = "chamis"
    kind: str = "physics"
    coef_matrix: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.coef_matrix is None:
            self.coef_matrix = _stack_coefs(self.coefs)

    @classmethod
    def train(
        cls,
        features: NDArray[np.float64],
        stiffness: NDArray[np.float64],
        *,
        ridge_alpha: float = 1e-3,
        **_ignored: Any,
    ) -> PhysicsResidualSurrogate:
        x = np.asarray(features, dtype=float)
        c = np.asarray(stiffness, dtype=float)
        base = chamis_engineering_constants_from_features(x)
        fea = _fea_engineering_constants_batch(c)
        phi = residual_basis(x)
        coefs: dict[str, NDArray[np.float64]] = {}
        for key in EC_MODULUS_KEYS:
            y_b = np.maximum(base[key], _EPS)
            y_f = np.maximum(fea[key], _EPS)
            r = np.log(y_f) - np.log(y_b)
            r = np.where(np.abs(r) > _R_CLIP, np.nan, r)
            coefs[key] = _fit_ridge(phi, r, alpha=ridge_alpha)
        for key in EC_POISSON_KEYS:
            r = fea[key] - base[key]
            coefs[key] = _fit_ridge(phi, r, alpha=ridge_alpha)
        return cls(
            coefs=coefs,
            feature_bounds=feature_bounds_from_training(x),
            ridge_alpha=ridge_alpha,
            coef_matrix=_stack_coefs(coefs),
        )

    def _predict_ec(
        self, features: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        base = chamis_engineering_constants_from_features(x)
        assert self.coef_matrix is not None
        residual = residual_basis(x) @ self.coef_matrix
        return _ec_from_residual_matrix(base, residual)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return _stiffness_from_ec_dict(self._predict_ec(features))

    def predict_single(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict(features)[0]

    def save(self, path: str | Path) -> None:
        import joblib

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "physics",
                "coefs": self.coefs,
                "feature_bounds": self.feature_bounds,
                "ridge_alpha": self.ridge_alpha,
                "base_model": self.base_model,
            },
            out,
        )

    @classmethod
    def load(cls, path: str | Path) -> PhysicsResidualSurrogate:
        import joblib

        payload = joblib.load(path)
        if payload.get("kind", "physics") != "physics":
            raise ValueError(
                f"expected kind='physics' joblib, got {payload.get('kind')!r}"
            )
        return cls(
            coefs={k: np.asarray(v, dtype=float) for k, v in payload["coefs"].items()},
            feature_bounds=np.asarray(payload["feature_bounds"], dtype=float),
            ridge_alpha=float(payload.get("ridge_alpha", 1e-3)),
            base_model=str(payload.get("base_model", "chamis")),
        )

    def as_predict_callable(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Callable for ``b3_tex.micromodels.SurrogateModel`` (batch-aware)."""

        def predict(features: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.asarray(features, dtype=float)
            out = self.predict(x)
            return out[0] if x.ndim == 1 else out

        return predict


@dataclass
class MultiFidelityGPSurrogate:
    """Chamis low-fidelity × exp(GP residual) — multi-fidelity residual GP.

    Same idea as ``b3_invsec.multi_fidelity_residual.MultiFidelityGPSurrogate``:
    physics carries trends; the GP learns only ``log y_FEA − log y_physics``.
    ``kappa > 0`` uses a conservative residual ``μ − κ·σ`` on moduli (lower
    stiffness for safety-style consumers).
    """

    gps: dict[str, Any]
    residual_const: dict[str, float]
    feature_scaler: Any
    feature_bounds: NDArray[np.float64]
    kappa: float = 0.0
    base_model: str = "chamis"
    kind: str = "mf_gp"
    n_restarts: int = 2

    @classmethod
    def train(
        cls,
        features: NDArray[np.float64],
        stiffness: NDArray[np.float64],
        *,
        kappa: float = 0.0,
        n_restarts: int = 2,
        min_rows: int = 12,
        **_ignored: Any,
    ) -> MultiFidelityGPSurrogate:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(features, dtype=float)
        c = np.asarray(stiffness, dtype=float)
        base = chamis_engineering_constants_from_features(x)
        fea = _fea_engineering_constants_batch(c)
        phi = residual_basis(x)
        scaler = StandardScaler()
        phi_s = scaler.fit_transform(phi)

        gps: dict[str, Any] = {}
        residual_const: dict[str, float] = {}

        def _make_gp(n_features: int) -> Any:
            ls0 = np.ones(n_features)
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e2)) * RBF(
                length_scale=ls0, length_scale_bounds=(1e-2, 1e2)
            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
            return GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=n_restarts,
                alpha=1e-8,
                random_state=0,
            )

        for key in EC_MODULUS_KEYS:
            y_b = np.maximum(base[key], _EPS)
            y_f = np.maximum(fea[key], _EPS)
            r = np.log(y_f) - np.log(y_b)
            ok = np.isfinite(r) & (np.abs(r) <= _R_CLIP)
            if int(ok.sum()) < min_rows:
                residual_const[key] = float(np.nanmedian(r[ok])) if ok.any() else 0.0
                continue
            gp = _make_gp(phi_s.shape[1])
            gp.fit(phi_s[ok], r[ok])
            gps[key] = gp

        for key in EC_POISSON_KEYS:
            r = fea[key] - base[key]
            ok = np.isfinite(r)
            if int(ok.sum()) < min_rows:
                residual_const[key] = float(np.nanmedian(r[ok])) if ok.any() else 0.0
                continue
            gp = _make_gp(phi_s.shape[1])
            gp.fit(phi_s[ok], r[ok])
            gps[key] = gp

        return cls(
            gps=gps,
            residual_const=residual_const,
            feature_scaler=scaler,
            feature_bounds=feature_bounds_from_training(x),
            kappa=kappa,
            n_restarts=n_restarts,
        )

    def _residual(
        self,
        key: str,
        phi_s: NDArray[np.float64],
        *,
        return_std: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if key not in self.gps:
            n = phi_s.shape[0]
            r = np.full(n, float(self.residual_const.get(key, 0.0)))
            return r, np.zeros(n)
        gp = self.gps[key]
        if return_std or self.kappa != 0.0:
            mu, std = gp.predict(phi_s, return_std=True)
            mu = np.asarray(mu, dtype=float)
            std = np.asarray(std, dtype=float)
        else:
            mu = np.asarray(gp.predict(phi_s), dtype=float)
            std = np.zeros_like(mu)
        # Conservative on moduli: lower residual → lower stiffness.
        r = mu - float(self.kappa) * std
        return r, std

    def _residual_matrix(self, phi_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Stacked residual columns ``(N, n_targets)`` in :data:`EC_TARGET_KEYS` order."""
        n = phi_s.shape[0]
        residual = np.empty((n, len(EC_TARGET_KEYS)), dtype=float)
        for j, key in enumerate(EC_TARGET_KEYS):
            if key in EC_POISSON_KEYS and self.kappa != 0.0 and key in self.gps:
                # Poisson: mean residual only (kappa is a capacity lever on moduli).
                residual[:, j] = np.asarray(self.gps[key].predict(phi_s), dtype=float)
            else:
                r, _ = self._residual(key, phi_s)
                residual[:, j] = r
        return residual

    def _predict_ec(
        self, features: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        base = chamis_engineering_constants_from_features(x)
        phi_s = self.feature_scaler.transform(residual_basis(x))
        return _ec_from_residual_matrix(base, self._residual_matrix(phi_s))

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return _stiffness_from_ec_dict(self._predict_ec(features))

    def predict_single(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict(features)[0]

    def save(self, path: str | Path) -> None:
        import joblib

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "mf_gp",
                "gps": self.gps,
                "residual_const": self.residual_const,
                "feature_scaler": self.feature_scaler,
                "feature_bounds": self.feature_bounds,
                "kappa": self.kappa,
                "base_model": self.base_model,
                "n_restarts": self.n_restarts,
            },
            out,
        )

    @classmethod
    def load(cls, path: str | Path) -> MultiFidelityGPSurrogate:
        import joblib

        payload = joblib.load(path)
        if payload.get("kind") != "mf_gp":
            raise ValueError(
                f"expected kind='mf_gp' joblib, got {payload.get('kind')!r}"
            )
        return cls(
            gps=payload["gps"],
            residual_const=dict(payload.get("residual_const", {})),
            feature_scaler=payload["feature_scaler"],
            feature_bounds=np.asarray(payload["feature_bounds"], dtype=float),
            kappa=float(payload.get("kappa", 0.0)),
            base_model=str(payload.get("base_model", "chamis")),
            n_restarts=int(payload.get("n_restarts", 2)),
        )

    def as_predict_callable(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Callable for ``b3_tex.micromodels.SurrogateModel`` (batch-aware)."""

        def predict(features: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.asarray(features, dtype=float)
            out = self.predict(x)
            return out[0] if x.ndim == 1 else out

        return predict


AnyStiffnessSurrogate = (
    "StiffnessSurrogate | PhysicsResidualSurrogate | MultiFidelityGPSurrogate"
)


def train_surrogate(
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    *,
    kind: SurrogateKind = DEFAULT_SURROGATE_KIND,
    **train_kwargs: Any,
) -> Any:
    """Train a stiffness surrogate of the requested kind.

    Default is :data:`DEFAULT_SURROGATE_KIND` (``physics`` residual).
    """
    if kind == "mlp":
        from b3_micromech.surrogate import StiffnessSurrogate

        return StiffnessSurrogate.train(features, stiffness, **train_kwargs)
    if kind == "physics":
        return PhysicsResidualSurrogate.train(features, stiffness, **train_kwargs)
    if kind == "mf_gp":
        return MultiFidelityGPSurrogate.train(features, stiffness, **train_kwargs)
    raise ValueError(f"unknown surrogate kind {kind!r}; use mlp|physics|mf_gp")


def load_surrogate(path: str | Path) -> Any:
    """Load any stiffness surrogate joblib (dispatches on ``kind``).

    Legacy joblibs without a ``kind`` field are treated as ``mlp``.
    """
    import joblib

    from b3_micromech.surrogate import StiffnessSurrogate

    payload = joblib.load(path)
    kind = payload.get("kind", "mlp")
    if kind == "physics":
        return PhysicsResidualSurrogate.load(path)
    if kind == "mf_gp":
        return MultiFidelityGPSurrogate.load(path)
    # Legacy MLP payloads have no kind field.
    return StiffnessSurrogate.load(path)


def train_surrogate_from_dataset(
    dataset_path: str | Path,
    *,
    model_path: str | Path,
    kind: SurrogateKind = DEFAULT_SURROGATE_KIND,
    **train_kwargs: Any,
) -> Any:
    from b3_micromech.export import load_dataset

    features, stiffness, _meta = load_dataset(dataset_path)
    model = train_surrogate(features, stiffness, kind=kind, **train_kwargs)
    model.save(model_path)
    return model


def chamis_vs_fea_report(
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
) -> dict[str, float]:
    """Mean relative error of bare Chamis vs FEA (baseline before residual)."""
    from b3_micromech.surrogate import relative_frobenius_error

    pred = chamis_engineering_constants_from_features(features)
    # reconstruct for Frobenius
    pred_c = _stiffness_from_ec_dict(pred)
    err = relative_frobenius_error(pred_c, stiffness)
    fea_ec = _fea_engineering_constants_batch(stiffness)
    report: dict[str, float] = {
        "chamis_mean_frobenius_error": float(err.mean()),
        "chamis_max_frobenius_error": float(err.max()),
    }
    for key in EC_MODULUS_KEYS:
        rel = np.abs(pred[key] - fea_ec[key]) / np.maximum(fea_ec[key], _EPS)
        report[f"chamis_mean_{key}_error"] = float(rel.mean())
    return report
