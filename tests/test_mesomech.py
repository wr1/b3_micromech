import numpy as np
import pytest

from b3_micromech.features import (
    build_feature_matrix,
    constituent_engineering_constants,
    features_out_of_bounds,
)
from b3_micromech.homogenize import surrogate_features
from b3_micromech.lut_cache import (
    load_lut_cache,
    lut_cache_key,
    save_lut_cache,
)
from b3_micromech.materials import Material
from b3_micromech.mesomech import (
    FeaMicromechMicromodel,
    _load_rve_base,
    fea_micromech_model,
    fea_surrogate_from_joblib,
    predict_stiffness_batch,
    register_fea_micromech,
)
from b3_micromech.problem import RVEProblem
from b3_micromech.surrogate import StiffnessSurrogate


def _constituents():
    matrix = Material.isotropic("m", youngs_modulus=3e9, poisson_ratio=0.35)
    fibre = Material.transverse_isotropic(
        "f", e_l=230e9, e_t=15e9, g_lt=15e9, nu_lt=0.2, nu_tt=0.3
    )
    return matrix, fibre


def _synthetic_surrogate() -> StiffnessSurrogate:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n_train = 48
    features = rng.uniform(
        low=[0.2, 2.5e9, 0.30, 200e9, 12e9, 10e9, 0.15, 5e9],
        high=[0.8, 3.5e9, 0.40, 250e9, 18e9, 20e9, 0.25, 7e9],
        size=(n_train, 8),
    )
    stiffness = np.zeros((n_train, 6, 6), dtype=float)
    for i, row in enumerate(features):
        vf, em, _num, _elf, etf, _glt, _nult, _gtt = row
        base = 1e9 * (1.0 + 2.0 * vf + 0.4 * em / 3e9 + 0.2 * etf / 15e9)
        stiffness[i] = np.diag(
            [base * 10, base, base, base * 0.4, base * 0.4, base * 0.3]
        )
        stiffness[i] = 0.5 * (stiffness[i] + stiffness[i].T)
    return StiffnessSurrogate.train(
        features,
        stiffness,
        hidden_layer_sizes=(64, 32),
        max_iter=4000,
        random_state=0,
        early_stopping=False,
    )


def _surrogate_micromodel(
    model: StiffnessSurrogate, *, name: str = "test_fea"
) -> FeaMicromechMicromodel:
    return FeaMicromechMicromodel(
        name=name,
        rve_base=_load_rve_base("examples/sweep_hex_hypercube.yaml"),
        model=model,
        disk_cache=False,
        cache_dir=None,
    )


def test_build_feature_matrix_matches_surrogate_features():
    problem = RVEProblem.from_yaml("examples/ud_transverse_hex.yaml")
    matrix = problem.materials[problem.matrix_material]
    fibre = problem.materials[problem.fibre_material]
    vf = np.array([0.2, problem.fibre_volume_fraction, 0.8])
    built = build_feature_matrix(vf, matrix=matrix, fibre=fibre)
    assert built.shape == (3, 8)
    np.testing.assert_allclose(built[1], surrogate_features(problem))


def test_constituent_engineering_constants():
    matrix, fibre = _constituents()
    scalars = constituent_engineering_constants(matrix, fibre)
    assert len(scalars) == 7
    assert scalars[0] == pytest.approx(3e9)
    assert scalars[2] == pytest.approx(230e9)


def test_features_out_of_bounds():
    bounds = np.column_stack(
        [
            [0.2, 2.5e9, 0.30, 180e9, 12e9, 10e9, 0.15, 5e9],
            [0.8, 3.5e9, 0.40, 280e9, 18e9, 20e9, 0.25, 7e9],
        ]
    )
    inside = np.array([[0.5, 3e9, 0.35, 230e9, 15e9, 15e9, 0.2, 6e9]])
    outside = np.array([[0.9, 3e9, 0.35, 230e9, 15e9, 15e9, 0.2, 6e9]])
    assert not features_out_of_bounds(inside, bounds)[0]
    assert features_out_of_bounds(outside, bounds)[0]


def test_lut_cache_key_changes_with_rve_or_constituents():
    matrix, fibre = _constituents()
    rve_a = _load_rve_base("examples/sweep_hex_hypercube.yaml")
    rve_b = dict(rve_a)
    rve_b["domain"] = dict(rve_a["domain"])
    rve_b["domain"]["mesh_resolution"] = [12]
    vf = np.linspace(0.4, 0.8, 8)
    key_a = lut_cache_key(rve_a, matrix, fibre, vf, mode="fea")
    key_b = lut_cache_key(rve_b, matrix, fibre, vf, mode="fea")
    assert key_a != key_b

    matrix2 = Material.isotropic("m2", youngs_modulus=3.5e9, poisson_ratio=0.35)
    key_c = lut_cache_key(rve_a, matrix2, fibre, vf, mode="fea")
    assert key_a != key_c


def test_disk_cache_roundtrip(tmp_path):
    matrix, fibre = _constituents()
    vf = np.linspace(0.4, 0.7, 5)
    stiffness = np.stack([np.eye(6) * (i + 1) for i in range(5)])
    cache_path = tmp_path / "entry.npz"
    save_lut_cache(cache_path, vf, stiffness, metadata={"mode": "fea"})
    vf_loaded, c_loaded = load_lut_cache(cache_path)
    np.testing.assert_allclose(vf_loaded, vf)
    np.testing.assert_allclose(c_loaded, stiffness)


def _fake_homogenize_batch(rve_base, matrix, fibre, vf, *, n_jobs=1):
    vf_arr = np.asarray(vf, dtype=float).ravel()
    return np.stack([np.eye(6) * v for v in vf_arr], axis=0)


def test_fea_fallback_when_no_surrogate(monkeypatch):
    matrix, fibre = _constituents()
    calls = {"n": 0}

    def counting_batch(rve_base, matrix, fibre, vf, *, n_jobs=1):
        calls["n"] += int(np.asarray(vf).size)
        return _fake_homogenize_batch(rve_base, matrix, fibre, vf, n_jobs=n_jobs)

    monkeypatch.setattr("b3_micromech.mesomech._homogenize_vf_batch", counting_batch)
    mm = fea_micromech_model(
        name="fea_only",
        surrogate_path=None,
        cache_dir=None,
        disk_cache=False,
    )
    vf = np.array([0.4, 0.5, 0.6])
    out = mm.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert out.shape == (3, 6, 6)
    assert calls["n"] == 3


def test_fea_fallback_memory_cache_hit(monkeypatch):
    matrix, fibre = _constituents()
    calls = {"n": 0}

    def counting_batch(rve_base, matrix, fibre, vf, *, n_jobs=1):
        calls["n"] += 1
        return _fake_homogenize_batch(rve_base, matrix, fibre, vf, n_jobs=n_jobs)

    monkeypatch.setattr("b3_micromech.mesomech._homogenize_vf_batch", counting_batch)
    mm = fea_micromech_model(surrogate_path=None, cache_dir=None, disk_cache=False)
    vf = np.array([0.45, 0.55])
    mm.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    mm.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert calls["n"] == 1


def test_fea_fallback_disk_cache_hit(monkeypatch, tmp_path):
    matrix, fibre = _constituents()
    calls = {"n": 0}

    def counting_batch(rve_base, matrix, fibre, vf, *, n_jobs=1):
        calls["n"] += 1
        return _fake_homogenize_batch(rve_base, matrix, fibre, vf, n_jobs=n_jobs)

    monkeypatch.setattr("b3_micromech.mesomech._homogenize_vf_batch", counting_batch)
    vf = np.linspace(0.4, 0.6, 3)
    mm1 = fea_micromech_model(surrogate_path=None, cache_dir=tmp_path, disk_cache=True)
    out1 = mm1.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert calls["n"] == 1

    mm2 = fea_micromech_model(surrogate_path=None, cache_dir=tmp_path, disk_cache=True)
    out2 = mm2.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert calls["n"] == 1
    np.testing.assert_allclose(out2, out1)


def test_fea_surrogate_stiffness_batch_matches_scalar_loop():
    matrix, fibre = _constituents()
    model = _synthetic_surrogate()
    micromodel = _surrogate_micromodel(model)
    vf = np.linspace(0.3, 0.7, 16)
    batch = micromodel.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf)
    assert batch.shape == (16, 6, 6)
    for i, v in enumerate(vf):
        scalar = micromodel.stiffness(
            matrix=matrix, fibre=fibre, fibre_volume_fraction=float(v)
        )
        np.testing.assert_allclose(batch[i], scalar, rtol=0.05, atol=1e8)


def test_predict_stiffness_batch_large_vector():
    matrix, fibre = _constituents()
    model = _synthetic_surrogate()
    vf = np.linspace(0.25, 0.75, 10_000)
    out = predict_stiffness_batch(model, vf, matrix, fibre, warn_oob=False)
    assert out.shape == (10_000, 6, 6)


def test_fea_surrogate_from_joblib_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    model = _synthetic_surrogate()
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = fea_surrogate_from_joblib(path, name="tmp")
    assert loaded.name == "tmp"
    matrix, fibre = _constituents()
    vf = np.array([0.4, 0.6])
    np.testing.assert_allclose(
        loaded.stiffness_batch(matrix=matrix, fibre=fibre, vf=vf),
        model.predict(build_feature_matrix(vf, matrix=matrix, fibre=fibre)),
    )


def test_register_fea_micromech_with_b3_tex(tmp_path):
    pytest.importorskip("b3_tex")
    from b3_tex.materials import MicromechanicalMaterial
    from b3_tex.micromodels import get_micromodel

    model = _synthetic_surrogate()
    path = tmp_path / "model.joblib"
    model.save(path)

    micromodel = register_fea_micromech(
        path, name="test_register", disk_cache=False, cache_dir=tmp_path
    )
    assert get_micromodel("test_register") is micromodel

    matrix, fibre = _constituents()
    yarn = MicromechanicalMaterial.from_constituents(
        "yarn",
        matrix=matrix,
        fibre=fibre,
        micromodel=micromodel,
        nominal_vf=0.5,
        max_vf=0.9,
    )
    centers, table = yarn.build_lut(0.4, 0.8, n_bins=32)
    assert centers.shape == (32,)
    assert table.shape == (32, 6, 6)
