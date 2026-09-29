import math

import numpy as np
import pytest

from b3_micromech.geometry import (
    domain_area,
    fibre_volume_fraction_from_radius,
    hexagon_bounding_box,
    hexagon_corners,
    point_in_hexagon,
    radius_from_fibre_volume_fraction,
)
from b3_micromech.homogenize import homogenize
from b3_micromech.periodic import periodic_vertex_master_map
from b3_micromech.problem import RVEProblem

pytestmark = pytest.mark.mfem


def _hex_homogeneous_config(*, edge_divisions: int = 12) -> dict:
    return {
        "domain": {
            "shape": "hexagon",
            "size": 1.0,
            "mesh_resolution": [edge_divisions],
        },
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
            "fibre_material": "matrix",
            "fibre_volume_fraction": 0.0,
        },
        "solver": {"cell_type": "triangle"},
    }


def test_hexagon_area_and_vf_roundtrip():
    r = 1.0
    assert math.isclose(
        domain_area(shape="hexagon", size=r), 1.5 * math.sqrt(3) * r * r
    )
    vf = 0.4
    radius = radius_from_fibre_volume_fraction(shape="hexagon", domain_size=r, vf=vf)
    assert (
        abs(
            fibre_volume_fraction_from_radius(
                shape="hexagon", domain_size=r, radius=radius
            )
            - vf
        )
        < 1e-12
    )


def test_point_in_hexagon_corners_and_centre():
    r = 1.0
    corners = hexagon_corners(r)
    centre = corners.mean(axis=0)
    mask = point_in_hexagon(
        np.vstack([corners, centre, centre + [10.0, 0.0]]), circumradius=r
    )
    assert mask.tolist() == [True] * 7 + [False]


def test_hexagon_periodic_pairing_covers_opposite_edges():
    pytest.importorskip("triangle")
    from b3_micromech.mesh.hexagon import build_hexagon_mesh

    problem = RVEProblem.from_config(_hex_homogeneous_config(edge_divisions=10))
    mesh = build_hexagon_mesh(problem)
    master_of = periodic_vertex_master_map(
        mesh,
        shape="hexagon",
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    corners = hexagon_corners(problem.domain_size)
    tol = problem.periodic_tolerance
    nv = mesh.GetNV()
    for v in range(nv):
        coord = np.asarray(mesh.GetVertexArray(v), dtype=float)
        on_boundary = False
        for i in range(6):
            v0, v1 = corners[i], corners[(i + 1) % 6]
            edge = v1 - v0
            normal = np.array([edge[1], -edge[0]])
            if np.dot(normal, corners.mean(axis=0) - v0) < 0:
                normal = -normal
            if float((coord - v0) @ normal) <= tol:
                dist_edge = np.linalg.norm(
                    np.cross(np.append(edge, 0.0), np.append(coord - v0, 0.0))
                ) / (np.linalg.norm(edge) + 1e-30)
                if (
                    dist_edge <= 10 * tol
                    or np.linalg.norm(coord - corners).min() <= 10 * tol
                ):
                    on_boundary = True
                    break
        if on_boundary:
            assert master_of[v] == v or master_of[master_of[v]] == master_of[v]


def test_homogeneous_matrix_recovers_stiffness_on_hex():
    pytest.importorskip("triangle")
    problem = RVEProblem.from_config(_hex_homogeneous_config(edge_divisions=14))
    result = homogenize(problem)
    C = result.effective_stiffness
    assert np.allclose(C, problem.materials["matrix"].stiffness, rtol=0.06, atol=1e8)


@pytest.mark.mfem
def test_hex_amr_example_yaml_loads():
    examples = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"
    problem = RVEProblem.from_yaml(examples / "ud_transverse_hex_amr.yaml")
    assert problem.domain_shape == "hexagon"
    assert problem.solver.amr.enabled is True


def test_hex_domain_loads_from_yaml():
    examples = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples"
    problem = RVEProblem.from_yaml(examples / "ud_transverse_hex.yaml")
    assert problem.domain_shape == "hexagon"
    ymax, zmax = hexagon_bounding_box(problem.domain_size)
    assert ymax > 0 and zmax > 0
