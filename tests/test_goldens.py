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

# PyMFEM 4.8 (where these arrays were frozen) and the locked 4.10 wheels
# disagree by up to ~1e3 Pa on couplings that are numerical zeros. CPython
# 3.11 and 3.13 on the same 4.10 wheel disagree by a similar amount. That is
# ~1e-8 of max|C| (~1e11 Pa). An absolute band of 1e-6 of the tensor scale
# still rejects the ~1% move from the sampling-default change. Those
# near-zero couplings have relative gaps of order 1, so the band is absolute.
_GOLDEN_ATOL_OVER_SCALE = 1e-6

CASES = [
    ("ud_transverse.yaml", "golden_ud_transverse.npz"),
    ("ud_transverse_hex.yaml", "golden_ud_transverse_hex.npz"),
    ("ud_transverse_hex_amr.yaml", "golden_ud_transverse_hex_amr.npz"),
]


def _assert_matches_golden(actual: np.ndarray, desired: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=float)
    desired = np.asarray(desired, dtype=float)
    scale = float(np.max(np.abs(desired)))
    if scale == 0.0:
        np.testing.assert_array_equal(actual, desired)
        return
    np.testing.assert_allclose(
        actual, desired, rtol=0.0, atol=_GOLDEN_ATOL_OVER_SCALE * scale
    )


@pytest.mark.parametrize(("yaml_name", "golden_name"), CASES)
def test_homogenize_matches_golden(yaml_name: str, golden_name: str):
    result = homogenize(RVEProblem.from_yaml(EXAMPLES / yaml_name))
    golden = np.load(DATA / golden_name)
    _assert_matches_golden(result.effective_stiffness, golden["C_eff"])
    _assert_matches_golden(result.effective_thermal_expansion, golden["alpha_eff"])
    _assert_matches_golden(result.effective_conductivity, golden["k_eff"])


def test_legacy_physics_joblib_loads():
    model = load_surrogate(DATA / "legacy_physics_v010.joblib")
    assert model.kind == "physics"
    row = model.feature_bounds.mean(axis=1)
    predicted = model.predict(row)
    assert predicted.shape == (1, 6, 6)
    assert np.isfinite(predicted).all()
