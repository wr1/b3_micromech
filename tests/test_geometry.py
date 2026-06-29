import numpy as np
import pytest

from b3_micromech.geometry import (
    HEX_VF_PACKING_CLUSTER,
    classify_points,
    fibre_volume_fraction_from_radius,
    hex_vf_packing_cluster_values,
    hex_vf_sweep_values,
    max_fibre_volume_fraction,
    radius_from_fibre_volume_fraction,
)


def test_vf_radius_roundtrip():
    vf = 0.5
    r = radius_from_fibre_volume_fraction(domain_size=1.0, vf=vf)
    assert (
        abs(fibre_volume_fraction_from_radius(domain_size=1.0, radius=r) - vf) < 1e-12
    )


def test_classify_points():
    centre = np.array([0.5, 0.5])
    pts = np.array([[0.5, 0.5], [0.0, 0.0]])
    mask = classify_points(pts, centre=centre, radius=0.2)
    assert mask.tolist() == [True, False]


def test_radius_invalid_vf():
    with pytest.raises(ValueError):
        radius_from_fibre_volume_fraction(domain_size=1.0, vf=1.2)


def test_hex_vf_packing_cluster_reaches_standoff_limit():
    cluster = hex_vf_packing_cluster_values()
    vf_max = max_fibre_volume_fraction(shape="hexagon", domain_size=1.0, standoff=0.01)
    assert cluster[0] == 0.74
    assert cluster[-1] > 0.88
    assert cluster[-1] < vf_max
    assert len(cluster) == 8


def test_hex_vf_sweep_values_mid_range_and_packing_cluster():
    values = hex_vf_sweep_values()
    assert values[0] == 0.20
    assert 0.50 in values
    assert 0.60 in values
    assert values[-len(HEX_VF_PACKING_CLUSTER) :] == list(HEX_VF_PACKING_CLUSTER)
    assert len(values) == 14
    # Wide-side cluster is denser than mid-range spacing.
    cluster_spacings = np.diff(values[-len(HEX_VF_PACKING_CLUSTER) :])
    mid_spacings = np.diff([v for v in values if 0.30 <= v <= 0.68])
    assert cluster_spacings.mean() < mid_spacings.mean()


def test_hex_vf_sweep_presets():
    compact = hex_vf_sweep_values(preset="compact")
    high = hex_vf_sweep_values(preset="high")
    assert len(compact) == 13
    assert compact[:5] == [0.20, 0.35, 0.50, 0.60, 0.68]
    assert high == list(HEX_VF_PACKING_CLUSTER)
