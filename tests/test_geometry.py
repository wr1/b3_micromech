import math

import numpy as np
import pytest

from b3_micromech.geometry import (
    classify_points,
    fibre_volume_fraction_from_radius,
    radius_from_fibre_volume_fraction,
)


def test_vf_radius_roundtrip():
    vf = 0.5
    r = radius_from_fibre_volume_fraction(domain_size=1.0, vf=vf)
    assert abs(fibre_volume_fraction_from_radius(domain_size=1.0, radius=r) - vf) < 1e-12


def test_classify_points():
    centre = np.array([0.5, 0.5])
    pts = np.array([[0.5, 0.5], [0.0, 0.0]])
    mask = classify_points(pts, centre=centre, radius=0.2)
    assert mask.tolist() == [True, False]


def test_radius_invalid_vf():
    with pytest.raises(ValueError):
        radius_from_fibre_volume_fraction(domain_size=1.0, vf=1.2)