"""Parity of the forked UD tensors against the pinned b3_tex."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from b3_micromech.contract import FEATURE_NAMES, Constituents
from b3_micromech.reference import chamis_ud_stiffness, mori_tanaka_cylinder
from b3_micromech.tensors import (
    engineering_constants_transverse_iso,
    isotropic_stiffness,
    orthotropic_stiffness,
    transverse_isotropic_stiffness,
)

pytestmark = pytest.mark.b3tex


def _require_b3_tex():
    try:
        import b3_tex
    except ImportError as exc:
        if os.environ.get("B3_MICROMECH_ALLOW_NO_B3TEX") == "1":
            pytest.skip("b3_tex not installed (B3_MICROMECH_ALLOW_NO_B3TEX=1)")
        raise RuntimeError(
            "b3_tex is not installed. Set B3_MICROMECH_ALLOW_NO_B3TEX=1 to skip."
        ) from exc
    return b3_tex


def _span(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return min(values), max(values)


def test_tensors_chamis_mori_tanaka_and_features_match_b3_tex():
    _require_b3_tex()
    from b3_tex.materials import Material as TexMaterial
    from b3_tex.micromechanics import chamis_ud_stiffness as tex_chamis
    from b3_tex.micromodels import SurrogateModel
    from b3_tex.reference import (
        engineering_constants_transverse_iso as tex_ti_constants,
        mori_tanaka_cylinder as tex_mt,
    )
    from b3_tex.tensors import (
        isotropic_stiffness as tex_isotropic,
        orthotropic_stiffness as tex_orthotropic,
        transverse_isotropic_stiffness as tex_ti,
    )

    from b3_micromech.materials import Material

    design = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "design_space.yaml").read_text()
    )
    fibres = design["fibres"]
    matrices = design["matrices"]
    bounds = {
        "E_m": _span(matrices, "youngs_modulus"),
        "nu_m": _span(matrices, "poisson_ratio"),
        "E_Lf": _span(fibres, "e_l"),
        "E_Tf": _span(fibres, "e_t"),
        "G_LTf": _span(fibres, "g_lt"),
        "nu_LTf": _span(fibres, "nu_lt"),
        "nu_TTf": _span(fibres, "nu_tt"),
    }
    rng = np.random.default_rng(0)
    n = 200

    def draw(key: str) -> float:
        lo, hi = bounds[key]
        if lo == hi:
            return lo
        return float(rng.uniform(lo, hi))

    for _ in range(n):
        e_m = draw("E_m")
        nu_m = draw("nu_m")
        e_l = draw("E_Lf")
        e_t = draw("E_Tf")
        g_lt = draw("G_LTf")
        nu_lt = draw("nu_LTf")
        nu_tt = draw("nu_TTf")
        vf = float(rng.uniform(0.05, 0.8))

        np.testing.assert_allclose(
            isotropic_stiffness(e_m, nu_m),
            tex_isotropic(e_m, nu_m),
            rtol=1e-12,
            atol=0.0,
        )
        ortho = {
            "e1": e_l,
            "e2": e_t,
            "e3": e_t,
            "nu12": nu_lt,
            "nu13": nu_lt,
            "nu23": nu_tt,
            "g12": g_lt,
            "g13": g_lt,
            "g23": e_t / (2.0 * (1.0 + nu_tt)),
        }
        np.testing.assert_allclose(
            orthotropic_stiffness(**ortho),
            tex_orthotropic(**ortho),
            rtol=1e-12,
            atol=0.0,
        )
        ti_kwargs = {
            "e_l": e_l,
            "e_t": e_t,
            "g_lt": g_lt,
            "nu_lt": nu_lt,
            "nu_tt": nu_tt,
        }
        ours_ti = transverse_isotropic_stiffness(**ti_kwargs)
        theirs_ti = tex_ti(**ti_kwargs)
        np.testing.assert_allclose(ours_ti, theirs_ti, rtol=1e-12, atol=0.0)
        ours_ec = engineering_constants_transverse_iso(ours_ti)
        theirs_ec = tex_ti_constants(theirs_ti)
        for key in ours_ec:
            assert ours_ec[key] == pytest.approx(theirs_ec[key], rel=1e-12, abs=0.0)

        matrix = Material.isotropic("m", youngs_modulus=e_m, poisson_ratio=nu_m)
        fibre = Material.transverse_isotropic("f", **ti_kwargs)
        tex_matrix = TexMaterial.isotropic("m", youngs_modulus=e_m, poisson_ratio=nu_m)
        tex_fibre = TexMaterial.transverse_isotropic("f", **ti_kwargs)
        np.testing.assert_allclose(
            chamis_ud_stiffness(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf),
            tex_chamis(matrix=tex_matrix, fibre=tex_fibre, fibre_volume_fraction=vf),
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            mori_tanaka_cylinder(matrix=matrix, fibre=fibre, fibre_volume_fraction=vf),
            tex_mt(matrix=tex_matrix, fibre=tex_fibre, fibre_volume_fraction=vf),
            rtol=1e-12,
            atol=0.0,
        )
        model = SurrogateModel(predict=lambda rows: np.zeros((6, 6)))
        vf_arr = np.array([vf])
        ours_x = Constituents.from_materials(matrix, fibre).feature_matrix(
            vf_arr, names=FEATURE_NAMES
        )
        theirs_x = model._feature_matrix(tex_matrix, tex_fibre, vf_arr)
        np.testing.assert_allclose(ours_x, theirs_x, rtol=1e-12, atol=0.0)
