"""Physics residual and multi-fidelity GP surrogates (Chamis base)."""

from __future__ import annotations

import numpy as np
import pytest

from b3_micromech.reference import (
    chamis_engineering_constants_from_features,
    chamis_stiffness_from_features,
)
from b3_micromech.surrogate import evaluate_training_holdout, relative_frobenius_error
from b3_micromech.tensors import transverse_isotropic_stiffness


def _synthetic_fea_like_dataset(n: int = 40, seed: int = 0):
    """FEA stand-in: Chamis with a smooth high-Vf residual bias."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(
        low=[0.25, 2.8e9, 0.32, 220e9, 14e9, 12e9, 0.18, 5.5e9],
        high=[0.85, 3.4e9, 0.38, 240e9, 16e9, 16e9, 0.22, 6.5e9],
        size=(n, 8),
    )
    base = chamis_engineering_constants_from_features(features)
    # Mild residual that grows with Vf and E_Tf/E_m — what physics+φ should recover.
    vf = features[:, 0]
    contrast = np.log(features[:, 4] / features[:, 1])
    r_et = 0.05 + 0.12 * vf + 0.03 * contrast
    r_glt = 0.04 + 0.10 * vf
    stiffness = np.empty((n, 6, 6), dtype=float)
    for i in range(n):
        stiffness[i] = transverse_isotropic_stiffness(
            e_l=float(base["e_l"][i] * np.exp(0.01 * vf[i])),
            e_t=float(base["e_t"][i] * np.exp(r_et[i])),
            g_lt=float(base["g_lt"][i] * np.exp(r_glt[i])),
            nu_lt=float(base["nu_lt"][i]),
            nu_tt=float(base["nu_tt"][i] + 0.02 * vf[i]),
        )
    return features, stiffness


def test_chamis_from_features_matches_hand_formula():
    vf, em, num = 0.6, 3e9, 0.35
    elf, etf, gltf, nultf = 230e9, 15e9, 15e9, 0.2
    gttf = 15e9 / (2.0 * (1.0 + 0.25))
    features = np.array([[vf, em, num, elf, etf, gltf, nultf, gttf]])
    ec = chamis_engineering_constants_from_features(features)
    vm = 1.0 - vf
    gm = em / (2.0 * (1.0 + num))
    sqrt_vf = np.sqrt(vf)
    e_l = vf * elf + vm * em
    e_t = em / (1.0 - sqrt_vf * (1.0 - em / etf))
    g_lt = gm / (1.0 - sqrt_vf * (1.0 - gm / gltf))
    assert np.isclose(ec["e_l"][0], e_l)
    assert np.isclose(ec["e_t"][0], e_t)
    assert np.isclose(ec["g_lt"][0], g_lt)
    c = chamis_stiffness_from_features(features)
    assert c.shape == (1, 6, 6)
    assert np.allclose(c[0], c[0].T)


def test_physics_residual_beats_bare_chamis():
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import (
        PhysicsResidualSurrogate,
        chamis_vs_fea_report,
    )

    features, stiffness = _synthetic_fea_like_dataset(n=48, seed=1)
    bare = chamis_vs_fea_report(features, stiffness)
    model = PhysicsResidualSurrogate.train(features, stiffness, ridge_alpha=1e-4)
    pred = model.predict(features)
    report = evaluate_training_holdout(model, features, stiffness)
    assert report["mean_error_all"] < bare["chamis_mean_frobenius_error"]
    assert report["mean_error_all"] < 0.03
    assert pred.shape == stiffness.shape


def test_physics_residual_save_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import PhysicsResidualSurrogate, load_surrogate

    features, stiffness = _synthetic_fea_like_dataset(n=30, seed=2)
    model = PhysicsResidualSurrogate.train(features, stiffness)
    path = tmp_path / "physics.joblib"
    model.save(path)
    loaded = load_surrogate(path)
    assert getattr(loaded, "kind", None) == "physics"
    assert np.allclose(model.predict(features), loaded.predict(features), rtol=1e-12)


def test_mf_gp_recovers_smooth_residual():
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import MultiFidelityGPSurrogate

    features, stiffness = _synthetic_fea_like_dataset(n=36, seed=3)
    model = MultiFidelityGPSurrogate.train(
        features, stiffness, n_restarts=0, min_rows=8
    )
    report = evaluate_training_holdout(model, features, stiffness)
    assert report["mean_error_all"] < 0.05
    assert model.kind == "mf_gp"


def test_mf_gp_save_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import MultiFidelityGPSurrogate, load_surrogate

    features, stiffness = _synthetic_fea_like_dataset(n=28, seed=4)
    model = MultiFidelityGPSurrogate.train(
        features, stiffness, n_restarts=0, min_rows=8
    )
    path = tmp_path / "mf_gp.joblib"
    model.save(path)
    loaded = load_surrogate(path)
    assert np.allclose(model.predict(features), loaded.predict(features), rtol=1e-10)


def test_train_surrogate_kind_dispatch():
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import train_surrogate

    features, stiffness = _synthetic_fea_like_dataset(n=24, seed=5)
    phys = train_surrogate(features, stiffness, kind="physics")
    assert phys.kind == "physics"
    mlp = train_surrogate(
        features,
        stiffness,
        kind="mlp",
        hidden_layer_sizes=(32,),
        max_iter=200,
        early_stopping=False,
    )
    assert hasattr(mlp, "regressor")
    bare_chamis = chamis_stiffness_from_features(features)
    # residual model should beat bare Chamis on this synthetic residual
    err_phys = relative_frobenius_error(phys.predict(features), stiffness).mean()
    err_bare = relative_frobenius_error(bare_chamis, stiffness).mean()
    assert err_phys < err_bare


def test_ti_stiffness_batch_matches_scalar():
    from b3_micromech.tensors import (
        transverse_isotropic_stiffness,
        transverse_isotropic_stiffness_batch,
    )

    e_l = np.array([100e9, 120e9])
    e_t = np.array([8e9, 10e9])
    g_lt = np.array([4e9, 5e9])
    nu_lt = np.array([0.25, 0.28])
    nu_tt = np.array([0.35, 0.40])
    batch = transverse_isotropic_stiffness_batch(
        e_l=e_l, e_t=e_t, g_lt=g_lt, nu_lt=nu_lt, nu_tt=nu_tt
    )
    for i in range(2):
        ref = transverse_isotropic_stiffness(
            e_l=float(e_l[i]),
            e_t=float(e_t[i]),
            g_lt=float(g_lt[i]),
            nu_lt=float(nu_lt[i]),
            nu_tt=float(nu_tt[i]),
        )
        assert np.allclose(batch[i], ref, rtol=1e-12, atol=1e-6)


def test_physics_predict_large_batch_is_vectorized():
    """100k Vf samples must stay pure ndarray ops (no Python sample loop)."""
    pytest.importorskip("sklearn")
    from b3_micromech.physics_surrogate import PhysicsResidualSurrogate

    features, stiffness = _synthetic_fea_like_dataset(n=40, seed=6)
    model = PhysicsResidualSurrogate.train(features, stiffness, ridge_alpha=1e-4)

    n = 100_000
    rng = np.random.default_rng(0)
    # Fixed constituents, Vf sweep — the mesomech LUT pattern.
    big = np.broadcast_to(features[0], (n, 8)).copy()
    big[:, 0] = rng.uniform(0.3, 0.85, size=n)

    import time

    t0 = time.perf_counter()
    pred = model.predict(big)
    elapsed = time.perf_counter() - t0
    assert pred.shape == (n, 6, 6)
    assert np.isfinite(pred).all()
    # Conservative wall budget: vectorized path is O(ms–100ms), looped would be multi-s.
    assert elapsed < 2.0, f"physics batch predict too slow: {elapsed:.3f}s for N={n}"
