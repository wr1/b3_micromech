from b3_micromech.homogenize import surrogate_features
from b3_micromech.problem import RVEProblem


def _problem(vf: float = 0.5) -> RVEProblem:
    cfg = {
        "domain": {"shape": "square", "size": 1.0, "mesh_resolution": [8, 8]},
        "materials": [
            {
                "name": "matrix",
                "type": "isotropic",
                "youngs_modulus": 3e9,
                "poisson_ratio": 0.35,
            },
            {
                "name": "fibre",
                "type": "transverse_isotropic",
                "e_l": 230e9,
                "e_t": 15e9,
                "g_lt": 15e9,
                "nu_lt": 0.2,
                "nu_tt": 0.3,
            },
        ],
        "rve": {
            "matrix_material": "matrix",
            "fibre_material": "fibre",
            "fibre_volume_fraction": vf,
        },
    }
    return RVEProblem.from_config(cfg)


def test_surrogate_features_shape():
    feat = surrogate_features(_problem())
    assert feat.shape == (8,)
    assert abs(feat[0] - 0.5) < 1e-12
    assert abs(feat[1] - 3e9) < 1.0
