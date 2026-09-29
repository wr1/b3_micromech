"""Train and run MLP surrogates for homogenized stiffness tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# Upper-triangle Voigt indices (row, col) for symmetric 6×6 stiffness.
_STIFFNESS_UPPER_TRIANGLE: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(6) for j in range(i, 6)
)
N_STIFFNESS_TARGETS = len(_STIFFNESS_UPPER_TRIANGLE)

# Pa-scale moduli and shear moduli: log-transform improves MLP fidelity across decades.
LOG_MODULUS_FEATURE_INDICES: tuple[int, ...] = (1, 3, 4, 5, 7)


def transform_features_for_regression(
    features: NDArray[np.float64],
    *,
    log_modulus: bool,
) -> NDArray[np.float64]:
    """Optionally map modulus features to ``log(Pa)`` before scaling."""
    x = np.asarray(features, dtype=float)
    if not log_modulus:
        return x
    out = x.copy()
    if out.ndim == 1:
        for idx in LOG_MODULUS_FEATURE_INDICES:
            out[idx] = np.log(out[idx])
        return out
    for idx in LOG_MODULUS_FEATURE_INDICES:
        out[:, idx] = np.log(out[:, idx])
    return out


def e2_emphasis_sample_weights(
    stiffness: NDArray[np.float64],
    *,
    emphasis_power: float = 2.0,
    floor_weight: float = 0.25,
) -> NDArray[np.float64]:
    """Upweight high-$E_2$ training points where the surrogate tends to deviate."""
    ec = engineering_constants_batch(stiffness)
    e2 = ec["e_t"]
    span = float(np.max(e2) - np.min(e2))
    if span <= 0.0:
        return np.ones(e2.shape[0], dtype=float)
    e2_norm = (e2 - np.min(e2)) / span
    return floor_weight + (1.0 - floor_weight) * np.power(e2_norm, emphasis_power)


def vf_emphasis_sample_weights(
    features: NDArray[np.float64],
    *,
    emphasis_power: float = 2.0,
    floor_weight: float = 0.25,
) -> NDArray[np.float64]:
    """Upweight high-Vf training points near the hex packing limit."""
    vf = np.asarray(features, dtype=float)[:, 0]
    span = float(np.max(vf) - np.min(vf))
    if span <= 0.0:
        return np.ones(vf.shape[0], dtype=float)
    vf_norm = (vf - np.min(vf)) / span
    return floor_weight + (1.0 - floor_weight) * np.power(vf_norm, emphasis_power)


def combined_emphasis_sample_weights(
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    *,
    emphasize_high_e2: bool = True,
    emphasize_high_vf: bool = True,
    e2_emphasis_power: float = 2.0,
    vf_emphasis_power: float = 2.0,
    floor_weight: float = 0.25,
) -> NDArray[np.float64] | None:
    """Product of optional $E_2$ and Vf emphasis weights, normalised to unit mean."""
    weights: NDArray[np.float64] | None = None
    if emphasize_high_e2:
        weights = e2_emphasis_sample_weights(
            stiffness,
            emphasis_power=e2_emphasis_power,
            floor_weight=floor_weight,
        )
    if emphasize_high_vf:
        vf_weights = vf_emphasis_sample_weights(
            features,
            emphasis_power=vf_emphasis_power,
            floor_weight=floor_weight,
        )
        weights = vf_weights if weights is None else weights * vf_weights
    if weights is None:
        return None
    mean = float(weights.mean())
    if mean <= 0.0:
        return None
    return weights / mean


def stiffness_to_targets(stiffness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map ``(N, 6, 6)`` symmetric tensors to ``(N, 21)`` regression targets."""
    c = np.asarray(stiffness, dtype=float)
    if c.ndim == 2:
        c = 0.5 * (c + c.T)
        return np.array([c[i, j] for i, j in _STIFFNESS_UPPER_TRIANGLE], dtype=float)[
            None, :
        ]
    if c.ndim != 3 or c.shape[1:] != (6, 6):
        raise ValueError(f"stiffness must have shape (N, 6, 6), got {c.shape}")
    c_sym = 0.5 * (c + np.transpose(c, (0, 2, 1)))
    return np.array(
        [[m[i, j] for i, j in _STIFFNESS_UPPER_TRIANGLE] for m in c_sym],
        dtype=float,
    )


def targets_to_stiffness(targets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reconstruct symmetric ``(N, 6, 6)`` tensors from ``(N, 21)`` targets."""
    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[1] != N_STIFFNESS_TARGETS:
        raise ValueError(f"expected {N_STIFFNESS_TARGETS} targets, got shape {y.shape}")
    n = y.shape[0]
    out = np.zeros((n, 6, 6), dtype=float)
    for k, (i, j) in enumerate(_STIFFNESS_UPPER_TRIANGLE):
        out[:, i, j] = y[:, k]
        if i != j:
            out[:, j, i] = y[:, k]
    return 0.5 * (out + np.transpose(out, (0, 2, 1)))


def feature_bounds_from_training(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-feature ``[min, max]`` from training data, shape ``(8, 2)``."""
    x = np.asarray(features, dtype=float)
    return np.column_stack([x.min(axis=0), x.max(axis=0)])


def sample_uniform_hypercube(
    n_samples: int,
    bounds: NDArray[np.float64],
    *,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Draw ``n_samples`` feature vectors uniformly inside ``bounds``."""
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (n, 2), got {bounds.shape}")
    n_features = bounds.shape[0]
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n_samples, n_features))


def relative_frobenius_error(
    predicted: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-sample ``||C_pred - C_ref||_F / ||C_ref||_F``."""
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = np.linalg.norm((pred - ref).reshape(pred.shape[0], -1), axis=1)
    denom = np.maximum(np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1), 1e-12)
    return diff / denom


def stratified_holdout_report(
    features: NDArray[np.float64],
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    vf_threshold: float = 0.75,
) -> dict[str, float]:
    """Mean/max Frobenius error below and above a Vf split."""
    err = relative_frobenius_error(predicted, reference)
    vf = features[:, 0]
    low = vf < vf_threshold
    high = ~low
    report: dict[str, float] = {
        "mean_error_all": float(err.mean()),
        "max_error_all": float(err.max()),
    }
    if low.any():
        report["mean_error_vf_low"] = float(err[low].mean())
        report["max_error_vf_low"] = float(err[low].max())
    if high.any():
        report["mean_error_vf_high"] = float(err[high].mean())
        report["max_error_vf_high"] = float(err[high].max())
    return report


def stratified_e2_holdout_report(
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    e2_threshold_gpa: float = 8.0,
) -> dict[str, float]:
    """Relative $E_2$ error below and above a transverse-modulus split."""
    ref_ec = engineering_constants_batch(reference)
    pred_ec = engineering_constants_batch(predicted)
    e2_ref = ref_ec["e_t"]
    rel = np.abs(pred_ec["e_t"] - e2_ref) / np.maximum(e2_ref, 1e-12)
    threshold = e2_threshold_gpa * 1e9
    low = e2_ref < threshold
    high = ~low
    report: dict[str, float] = {
        "mean_e2_error_all": float(rel.mean()),
        "max_e2_error_all": float(rel.max()),
    }
    if low.any():
        report["mean_e2_error_low"] = float(rel[low].mean())
        report["max_e2_error_low"] = float(rel[low].max())
    if high.any():
        report["mean_e2_error_high"] = float(rel[high].mean())
        report["max_e2_error_high"] = float(rel[high].max())
    return report


def engineering_constants_batch(
    stiffness: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Extract transverse-isotropic engineering constants for each stiffness tensor."""
    from b3_micromech.tensors import engineering_constants_transverse_iso_batch

    batch = engineering_constants_transverse_iso_batch(stiffness)
    return {k: batch[k] for k in ("e_l", "e_t", "g_lt", "nu_lt")}


@dataclass
class StiffnessSurrogate:
    """MLP mapping surrogate feature vectors to ``(6, 6)`` stiffness."""

    feature_scaler: Any
    target_scaler: Any
    regressor: Any
    feature_bounds: NDArray[np.float64]
    log_modulus_features: bool = True

    @property
    def scaler(self) -> Any:
        """Backward-compatible alias for :attr:`feature_scaler`."""
        return self.feature_scaler

    @classmethod
    def train(
        cls,
        features: NDArray[np.float64],
        stiffness: NDArray[np.float64],
        *,
        hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
        max_iter: int = 5000,
        random_state: int = 0,
        log_modulus_features: bool = True,
        emphasize_high_e2: bool = True,
        emphasize_high_vf: bool = True,
        e2_emphasis_power: float = 2.0,
        vf_emphasis_power: float = 2.0,
        **train_kwargs: Any,
    ) -> StiffnessSurrogate:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(features, dtype=float)
        c = np.asarray(stiffness, dtype=float)
        y = stiffness_to_targets(c)
        n_samples = x.shape[0]
        x_model = transform_features_for_regression(x, log_modulus=log_modulus_features)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        x_scaled = feature_scaler.fit_transform(x_model)
        y_scaled = target_scaler.fit_transform(y)
        mlp_kwargs = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-5,
            "learning_rate_init": 1e-3,
            "max_iter": max_iter,
            "random_state": random_state,
            "early_stopping": n_samples >= 80,
            "validation_fraction": 0.15,
        }
        mlp_kwargs.update(train_kwargs)
        if n_samples < 80 and "early_stopping" not in train_kwargs:
            mlp_kwargs["early_stopping"] = False
        regressor = MLPRegressor(**mlp_kwargs)
        fit_kwargs: dict[str, Any] = {}
        sample_weight = combined_emphasis_sample_weights(
            x,
            c,
            emphasize_high_e2=emphasize_high_e2,
            emphasize_high_vf=emphasize_high_vf,
            e2_emphasis_power=e2_emphasis_power,
            vf_emphasis_power=vf_emphasis_power,
        )
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        regressor.fit(x_scaled, y_scaled, **fit_kwargs)
        return cls(
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            regressor=regressor,
            feature_bounds=feature_bounds_from_training(x),
            log_modulus_features=log_modulus_features,
        )

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        x_model = transform_features_for_regression(
            x, log_modulus=self.log_modulus_features
        )
        x_scaled = self.feature_scaler.transform(x_model)
        y_scaled = self.regressor.predict(x_scaled)
        return targets_to_stiffness(self.target_scaler.inverse_transform(y_scaled))

    def predict_single(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict(features)[0]

    def save(self, path: str | Path) -> None:
        import joblib

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": "mlp",
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "regressor": self.regressor,
                "feature_bounds": self.feature_bounds,
                "log_modulus_features": self.log_modulus_features,
            },
            out,
        )

    @classmethod
    def load(cls, path: str | Path) -> StiffnessSurrogate:
        import joblib
        from sklearn.preprocessing import StandardScaler

        payload = joblib.load(path)
        if "feature_scaler" in payload:
            feature_scaler = payload["feature_scaler"]
            target_scaler = payload["target_scaler"]
        else:
            feature_scaler = payload["scaler"]
            target_scaler = StandardScaler()
            target_scaler.scale_ = np.ones(N_STIFFNESS_TARGETS)
            target_scaler.mean_ = np.zeros(N_STIFFNESS_TARGETS)
        return cls(
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            regressor=payload["regressor"],
            feature_bounds=np.asarray(payload["feature_bounds"], dtype=float),
            log_modulus_features=bool(payload.get("log_modulus_features", False)),
        )

    def as_predict_callable(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Callable for ``b3_tex.micromodels.SurrogateModel`` (batch-aware).

        Accepts a single feature row → ``(6, 6)`` or a batch ``(N, F)`` →
        ``(N, 6, 6)`` so ``SurrogateModel.stiffness_batch`` stays vectorized.
        """

        def predict(features: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.asarray(features, dtype=float)
            out = self.predict(x)
            return out[0] if x.ndim == 1 else out

        return predict


def train_stiffness_surrogate_from_dataset(
    dataset_path: str | Path,
    *,
    model_path: str | Path,
    **train_kwargs: Any,
) -> StiffnessSurrogate:
    from b3_micromech.export import load_dataset

    features, stiffness, _meta = load_dataset(dataset_path)
    model = StiffnessSurrogate.train(features, stiffness, **train_kwargs)
    model.save(model_path)
    return model


def predict_random_hypercube_samples(
    model: StiffnessSurrogate,
    n_samples: int,
    *,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features = sample_uniform_hypercube(n_samples, model.feature_bounds, seed=seed)
    stiffness = model.predict(features)
    return features, stiffness


def save_hypercube_predictions(
    path: str | Path,
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
) -> None:
    """Write uniform-hypercube inference samples to ``predictions.npz``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=np.asarray(features, dtype=float), C=np.asarray(stiffness, dtype=float)
    )


def evaluate_training_holdout(
    model: StiffnessSurrogate,
    features: NDArray[np.float64],
    stiffness: NDArray[np.float64],
    *,
    vf_threshold: float = 0.75,
    e2_threshold_gpa: float = 8.0,
) -> dict[str, float]:
    """Hold-out error on the training grid (in-sample diagnostic)."""
    predicted = model.predict(features)
    report = stratified_holdout_report(
        features,
        predicted,
        stiffness,
        vf_threshold=vf_threshold,
    )
    report.update(
        stratified_e2_holdout_report(
            predicted,
            stiffness,
            e2_threshold_gpa=e2_threshold_gpa,
        )
    )
    return report
