import numpy as np
import pytest

from b3_micromech.surrogate_response import (
    DEFAULT_THREE_AXIS_RESPONSE,
    SURROGATE_FEATURE_INDEX,
    build_structured_response_grid,
    midpoint_feature_vector,
)
from b3_micromech.geometry import hex_vf_sweep_values
from b3_micromech.sweep import count_sweep_points, varying_sweep_parameters


def _bounds() -> np.ndarray:
    lo = np.array([0.2, 2.5e9, 0.35, 180e9, 12e9, 10e9, 0.2, 5e9])
    hi = np.array([0.8, 3.5e9, 0.35, 280e9, 18e9, 20e9, 0.2, 7e9])
    return np.column_stack([lo, hi])


def test_midpoint_feature_vector_shape():
    mid = midpoint_feature_vector(_bounds())
    assert mid.shape == (8,)


def test_build_structured_response_grid_three_axes():
    bounds = _bounds()
    axes = DEFAULT_THREE_AXIS_RESPONSE
    sizes = (4, 3, 2)
    features, axis_coords = build_structured_response_grid(bounds, axes, sizes)
    assert features.shape == (24, 8)
    assert axis_coords["vf"].shape == (4,)
    assert axis_coords["E_m"].shape == (3,)
    assert axis_coords["E_Lf"].shape == (2,)
    assert features[:, SURROGATE_FEATURE_INDEX["nu_m"]].tolist() == [0.35] * 24


def test_build_structured_response_grid_fixed_axes():
    bounds = _bounds()
    features, _ = build_structured_response_grid(
        bounds,
        ("vf", "E_Lf"),
        (5, 4),
        fixed_axis_values={"E_m": 3.0e9},
    )
    assert features.shape == (20, 8)
    assert np.allclose(features[:, SURROGATE_FEATURE_INDEX["E_m"]], 3.0e9)


def test_varying_sweep_parameters_3d_response_yaml():
    sweep_cfg = {
        "vf": {"hex_vf_sweep": "compact"},
        "E_m": {"linspace": [2.5e9, 3.5e9, 3]},
        "E_Lf": {"linspace": [180e9, 280e9, 3]},
        "E_Tf": {"value": 15e9},
    }
    varying = varying_sweep_parameters(sweep_cfg)
    assert varying == ["vf", "E_m", "E_Lf"]
    n_vf = len(hex_vf_sweep_values(preset="compact"))
    assert count_sweep_points(sweep_cfg) == n_vf * 3 * 3


def test_build_structured_response_grid_rejects_four_axes():
    with pytest.raises(ValueError, match="at most three"):
        build_structured_response_grid(
            _bounds(), ("vf", "E_m", "E_Lf", "E_Tf"), (2, 2, 2, 2)
        )
