"""Frozen solver output. Later refactors must match these arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from b3_micromech.homogenize import homogenize
from b3_micromech.physics_surrogate import load_surrogate
from b3_micromech.problem import RVEProblem

pytestmark = pytest.mark.mfem

DATA = Path(__file__).resolve().parent / "data"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

CASES = [
    ("ud_transverse.yaml", "golden_ud_transverse.npz"),
    ("ud_transverse_hex.yaml", "golden_ud_transverse_hex.npz"),
    ("ud_transverse_hex_amr.yaml", "golden_ud_transverse_hex_amr.npz"),
]


@pytest.mark.parametrize(("yaml_name", "golden_name"), CASES)
def test_homogenize_matches_golden(yaml_name: str, golden_name: str):
    result = homogenize(RVEProblem.from_yaml(EXAMPLES / yaml_name))
    golden = np.load(DATA / golden_name)
    np.testing.assert_allclose(
        result.effective_stiffness, golden["C_eff"], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        result.effective_thermal_expansion, golden["alpha_eff"], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        result.effective_conductivity, golden["k_eff"], rtol=1e-12, atol=0.0
    )


def test_legacy_physics_joblib_loads():
    model = load_surrogate(DATA / "legacy_physics_v010.joblib")
    assert model.kind == "physics"
    row = model.feature_bounds.mean(axis=1)
    predicted = model.predict(row)
    assert predicted.shape == (1, 6, 6)
    assert np.isfinite(predicted).all()
