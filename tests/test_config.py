"""Strict RVE configuration."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from b3_micromech.config import ConfigError, MaterialSamplingConfig
from b3_micromech.problem import RVEProblem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _base(**overrides):
    cfg = {
        "domain": {"shape": "square", "size": 1.0, "mesh_resolution": [4, 4]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3.0e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230.0e9,
                "e_t": 15.0e9,
                "g_lt": 15.0e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": 0.5,
        },
    }
    cfg.update(overrides)
    return cfg


def test_unknown_top_level_key_raises():
    with pytest.raises(ConfigError, match="bogus_top_key"):
        RVEProblem.from_config(_base(bogus_top_key=1))


def test_unknown_amr_key_names_the_typo():
    cfg = _base()
    cfg["solver"] = {"amr": {"enabeld": True}}
    with pytest.raises(ConfigError, match="enabeld") as exc:
        RVEProblem.from_config(cfg)
    assert "enabled" in str(exc.value)


def test_unknown_sampling_strategy_raises():
    cfg = _base()
    cfg["solver"] = {"material_sampling": {"strategy": "local-cloud"}}
    with pytest.raises(ConfigError, match="local-cloud"):
        RVEProblem.from_config(cfg)


def test_hex_example_uses_local_cloud_resolution_6():
    problem = RVEProblem.from_yaml(EXAMPLES / "ud_transverse_hex.yaml")
    assert problem.solver.material_sampling == MaterialSamplingConfig(
        strategy="local_cloud", resolution=6
    )


def test_vf_above_packing_raises_and_thin_ligament_warns():
    cfg = _base()
    cfg["domain"]["shape"] = "hexagon"
    cfg["domain"]["mesh_resolution"] = [8]
    cfg["rve"]["fibre_volume_fraction"] = 0.95
    with pytest.raises(ConfigError, match="packing"):
        RVEProblem.from_config(cfg)
    cfg["rve"]["fibre_volume_fraction"] = 0.89
    with pytest.warns(UserWarning, match="thin matrix ligament"):
        RVEProblem.from_config(cfg)

    square = _base()
    square["rve"]["fibre_volume_fraction"] = 0.80
    with pytest.raises(ConfigError, match="packing"):
        RVEProblem.from_config(square)


def test_examples_load_without_deprecation_warnings():
    for path in sorted(EXAMPLES.glob("*.yaml")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            RVEProblem.from_yaml(path)


@pytest.mark.filterwarnings("always::DeprecationWarning:b3_micromech")
def test_missing_domain_shape_warns_and_defaults_to_square():
    cfg = _base()
    del cfg["domain"]["shape"]
    with pytest.warns(
        DeprecationWarning,
        match=(
            "domain.shape not given; defaulting to 'square' "
            r"\(will be required in 0.3.0\)"
        ),
    ):
        problem = RVEProblem.from_config(cfg)
    assert problem.domain_shape == "square"


def test_to_config_round_trips():
    problem = RVEProblem.from_config(_base())
    again = RVEProblem.from_config(problem.to_config())
    assert again.domain_shape == problem.domain_shape
    assert again.mesh_resolution == problem.mesh_resolution
    assert again.fibre_volume_fraction == problem.fibre_volume_fraction
    assert again.solver == problem.solver
    assert again.material_configs == problem.material_configs
    assert again.solver_dict == problem.solver.to_dict()


@pytest.mark.filterwarnings("always::DeprecationWarning:b3_micromech")
def test_legacy_stiffness_sampling_maps_with_warning():
    cfg = _base()
    cfg["solver"] = {"stiffness_sampling": "centroid"}
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        problem = RVEProblem.from_config(cfg)
    assert problem.solver.material_sampling.strategy == "cell_constant"
    assert problem.solver.material_sampling.resolution == 6


@pytest.mark.filterwarnings("always::DeprecationWarning:b3_micromech")
def test_unknown_legacy_stiffness_sampling_raises():
    cfg = _base()
    cfg["solver"] = {"stiffness_sampling": "nope"}
    with pytest.warns(DeprecationWarning, match="stiffness_sampling"):
        with pytest.raises(ConfigError, match="nope"):
            RVEProblem.from_config(cfg)
