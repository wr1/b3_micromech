"""Material loading."""

import pytest

from b3_micromech.config import ConfigError
from b3_micromech.materials import load_materials


def test_duplicate_material_name_raises():
    config = [
        {
            "name": "epoxy",
            "type": "isotropic",
            "youngs_modulus": 3.0e9,
            "poisson_ratio": 0.35,
        },
        {
            "name": "epoxy",
            "type": "isotropic",
            "youngs_modulus": 3.2e9,
            "poisson_ratio": 0.30,
        },
    ]
    with pytest.raises(ConfigError, match="epoxy"):
        load_materials(config)
