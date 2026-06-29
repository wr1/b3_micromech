import math

import numpy as np
import pytest

from b3_micromech.export import SURROGATE_FEATURE_NAMES, load_dataset, save_dataset
from b3_micromech.geometry import (
    max_fibre_volume_fraction,
    packing_limit_volume_fraction,
)
from b3_micromech.surrogate import (
    StiffnessSurrogate,
    evaluate_training_holdout,
    combined_emphasis_sample_weights,
    e2_emphasis_sample_weights,
    stiffness_to_targets,
    targets_to_stiffness,
    transform_features_for_regression,
    vf_emphasis_sample_weights,
)


def test_hex_packing_limit_volume_fraction():
    vf = packing_limit_volume_fraction(shape="hexagon")
    assert abs(vf - math.pi / (2.0 * math.sqrt(3.0))) < 1e-12


def test_hex_max_fibre_volume_fraction_with_standoff():
    vf = max_fibre_volume_fraction(shape="hexagon", domain_size=1.0, standoff=0.01)
    assert 0.88 < vf < 0.90


def test_stiffness_target_roundtrip():
    c = np.eye(6) + np.arange(36, dtype=float).reshape(6, 6) * 0.01
    c = 0.5 * (c + c.T)
    y = stiffness_to_targets(c)
    c_back = targets_to_stiffness(y)[0]
    assert np.allclose(c, c_back, atol=1e-12)


def test_save_load_dataset_roundtrip(tmp_path):
    features = np.random.default_rng(0).uniform(size=(4, 8))
    stiffness = np.random.default_rng(1).uniform(size=(4, 6, 6))
    path = tmp_path / "dataset.npz"
    save_dataset(path, features=features, stiffness=stiffness, metadata={"n": 4})
    loaded_x, loaded_c, meta = load_dataset(path)
    assert loaded_x.shape == (4, 8)
    assert loaded_c.shape == (4, 6, 6)
    assert meta is not None
    assert meta["n"] == 4


def test_surrogate_feature_names_length():
    assert len(SURROGATE_FEATURE_NAMES) == 8


def test_log_modulus_feature_transform():
    x = np.array([0.5, 3e9, 0.35, 230e9, 15e9, 15e9, 0.2, 6e9])
    out = transform_features_for_regression(x, log_modulus=True)
    assert out[0] == 0.5
    assert np.isclose(out[1], np.log(3e9))


def test_vf_emphasis_weights_increase_with_volume_fraction():
    features = np.array([[0.2], [0.5], [0.88]])
    weights = vf_emphasis_sample_weights(features)
    assert weights[0] < weights[1] < weights[2]


def test_combined_emphasis_upweights_high_vf_and_high_e2():
    features = np.array([[0.2], [0.88]])
    stiffness = np.zeros((2, 6, 6), dtype=float)
    stiffness[0] = np.diag([50e9, 5e9, 5e9, 2e9, 2e9, 1e9])
    stiffness[1] = np.diag([100e9, 20e9, 20e9, 8e9, 8e9, 4e9])
    weights = combined_emphasis_sample_weights(features, stiffness)
    assert weights is not None
    assert weights[1] > weights[0]


def test_e2_emphasis_weights_increase_with_modulus():
    stiffness = np.zeros((3, 6, 6), dtype=float)
    for i, scale in enumerate([1.0, 2.0, 4.0]):
        stiffness[i] = np.diag(
            [
                scale * 50e9,
                scale * 5e9,
                scale * 5e9,
                scale * 2e9,
                scale * 2e9,
                scale * 1e9,
            ]
        )
    weights = e2_emphasis_sample_weights(stiffness)
    assert weights[0] < weights[1] < weights[2]


def test_stiffness_surrogate_synthetic_roundtrip():
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

    model = StiffnessSurrogate.train(
        features,
        stiffness,
        hidden_layer_sizes=(64, 32),
        max_iter=4000,
        random_state=0,
        early_stopping=False,
    )
    predicted = model.predict(features)
    report = evaluate_training_holdout(model, features, stiffness)
    assert report["mean_error_all"] < 0.08
    assert np.allclose(predicted, stiffness, rtol=0.10, atol=3e8)
