from b3_micromech.geometry import hex_vf_sweep_values
from b3_micromech.sweep import (
    _expand_param,
    count_sweep_points,
    varying_sweep_parameters,
)


def test_expand_hex_vf_sweep_preset_string():
    values = _expand_param({"hex_vf_sweep": "full"})
    assert values == hex_vf_sweep_values(preset="full")


def test_expand_hex_vf_sweep_option_dict():
    values = _expand_param(
        {
            "hex_vf_sweep": {
                "preset": "high",
                "dense_start": 0.80,
                "n_dense": 5,
            }
        }
    )
    assert values[0] == 0.80
    assert len(values) == 5


def test_hex_high_vf_sweep_point_count():
    sweep_cfg = {
        "vf": {
            "hex_vf_sweep": {
                "preset": "high",
                "standoff": 0.01,
                "dense_start": 0.74,
                "n_dense": 8,
            }
        },
        "E_m": {"linspace": [2.5e9, 3.5e9, 3]},
        "E_Tf": {"linspace": [12.0e9, 18.0e9, 2]},
        "E_Lf": {"value": 230e9},
    }
    varying = varying_sweep_parameters(sweep_cfg)
    assert varying == ["vf", "E_m", "E_Tf"]
    assert count_sweep_points(sweep_cfg) == 48
