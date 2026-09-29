"""MFEM 2D periodic homogenization under plane strain along fibre axis x."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_micromech.mesh.build import build_mesh
from b3_micromech.mesh.cartesian import element_cell_vertices_yz, element_material_ids
from b3_micromech.quadrature import effective_stiffnesses_for_gauss_points
from b3_micromech.periodic import origin_vertex_index, periodic_vertex_master_map
from b3_micromech.result import LoadcaseResult
from b3_micromech.problem import RVEProblem
from b3_micromech.tensors import (
    grad_to_voigt_strain_plane_strain_x,
    macro_displacement_at_yz,
    von_mises_voigt,
    voigt_b_matrix_plane_strain_x,
)


def _mfem_spmat_to_scipy(spmat):
    import scipy.sparse as sp

    return sp.csr_matrix(
        (
            np.asarray(spmat.GetDataArray()).copy(),
            np.asarray(spmat.GetJArray()).copy(),
            np.asarray(spmat.GetIArray()).copy(),
        ),
        shape=(spmat.Height(), spmat.Width()),
    )


@dataclass(frozen=True)
class _ElementGPData:
    gp_coords_yz: NDArray[np.float64]
    gp_dshapes: NDArray[np.float64]
    gp_weights: NDArray[np.float64]
    elem_vdofs: NDArray[np.intp]
    c_per_gp: NDArray[np.float64]
    n_elem: int
    nq: int
    nd: int


def _collect_element_gp_data(mesh, fespace, problem: RVEProblem) -> _ElementGPData:
    import mfem.ser as mfem

    n_elem = mesh.GetNE()
    if fespace.GetOrdering() != mfem.Ordering.byNODES:
        raise NotImplementedError("mfem_periodic_2d assumes byNODES dof ordering")

    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    total = n_elem * nq
    gp_coords_yz = np.empty((total, 2), dtype=float)
    gp_dshapes = np.empty((total, nd, 2), dtype=float)
    gp_weights = np.empty(total, dtype=float)
    elem_vdofs = np.empty((n_elem, 3, nd), dtype=np.intp)
    c_per_gp = np.empty((total, 6, 6), dtype=float)
    gp_cell_ids = np.repeat(np.arange(n_elem, dtype=np.intp), nq)

    dshape_ref = mfem.DenseMatrix(nd, 2)
    J_inv = mfem.DenseMatrix(2, 2)
    dshape_phys = mfem.DenseMatrix(nd, 2)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError("mixed 2D meshes are not supported")
        elem_vdofs[e] = np.asarray(fespace.GetElementVDofs(e), dtype=np.intp).reshape(
            3, nd
        )
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            idx = e * nq + q
            gp_coords_yz[idx] = np.asarray(T.Transform(ip))
            fe.CalcDShape(ip, dshape_ref)
            mfem.CalcInverse(T.Jacobian(), J_inv)
            mfem.Mult(dshape_ref, J_inv, dshape_phys)
            gp_dshapes[idx] = np.asarray(dshape_phys.GetDataArray())
            gp_weights[idx] = ip.weight * T.Weight()

    cell_vertices = element_cell_vertices_yz(mesh)
    c_per_gp[:] = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords_yz,
        gp_cell_ids,
        cell_vertices,
    )

    return _ElementGPData(
        gp_coords_yz=gp_coords_yz,
        gp_dshapes=gp_dshapes,
        gp_weights=gp_weights,
        elem_vdofs=elem_vdofs,
        c_per_gp=c_per_gp,
        n_elem=n_elem,
        nq=nq,
        nd=nd,
    )


def _collect_u_gradient_at_gps(
    u_array: NDArray[np.float64], data: _ElementGPData
) -> NDArray[np.float64]:
    u_elem = u_array[data.elem_vdofs]
    dsh = data.gp_dshapes.reshape(data.n_elem, data.nq, data.nd, 2)
    grad = np.einsum("ein,eqnj->eqij", u_elem, dsh)
    return grad.reshape(data.n_elem * data.nq, 3, 2)


def _make_bilinear_integrator(c_per_gp: NDArray[np.float64], data: _ElementGPData):
    import mfem.ser as mfem

    c_view = c_per_gp.reshape(data.n_elem, data.nq, 6, 6)
    dsh_view = data.gp_dshapes.reshape(data.n_elem, data.nq, data.nd, 2)
    w_view = data.gp_weights.reshape(data.n_elem, data.nq)
    nq, nd = data.nq, data.nd

    class _Integrator(mfem.PyBilinearFormIntegrator):
        def AssembleElementMatrix(self, fe, T, elmat):
            e = T.ElementNo
            elmat.SetSize(nd * 3)
            local = np.zeros((nd * 3, nd * 3), dtype=float)
            for q in range(nq):
                B = voigt_b_matrix_plane_strain_x(dsh_view[e, q])
                local += B.T @ c_view[e, q] @ B * w_view[e, q]
            elmat.GetDataArray()[:] = local

    return _Integrator()


def _make_thermal_integrator(alpha_per_gp: NDArray[np.float64], data: _ElementGPData):
    """Bilinear integrator for thermal eigenstrain RHS (delta_T = 1).

    Returns a ``PyLinearFormIntegrator`` that assembles the thermal
    load vector ``f = -∫ B^T : C : alpha dV`` per Gauss point.
    """
    import mfem.ser as mfem

    c_view = data.c_per_gp.reshape(data.n_elem, data.nq, 6, 6)
    dsh_view = data.gp_dshapes.reshape(data.n_elem, data.nq, data.nd, 2)
    w_view = data.gp_weights.reshape(data.n_elem, data.nq)
    alpha_view = alpha_per_gp.reshape(data.n_elem, data.nq, 6)
    nq, nd = data.nq, data.nd

    class _ThermalRHS(mfem.PyLinearFormIntegrator):
        def AssembleRHSElementVect(self, el, Tr, elvect):
            e = Tr.ElementNo
            elvect.SetSize(nd * 3)
            local = np.zeros(nd * 3, dtype=float)
            for q in range(nq):
                B = voigt_b_matrix_plane_strain_x(dsh_view[e, q])
                # Thermal stress at this GP: sigma_th = C : alpha * delta_T
                sigma_th = c_view[e, q] @ alpha_view[e, q]
                local -= B.T @ sigma_th * w_view[e, q]
            elvect.GetDataArray()[:] = local

    return _ThermalRHS()


def _collect_alpha_at_gps(mesh, fespace, problem: RVEProblem) -> NDArray[np.float64]:
    """Per-GP thermal-expansion vector (N_gps, 6) from the RVEProblem."""
    n_elem = mesh.GetNE()
    import mfem.ser as mfem

    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()
    total = n_elem * nq

    # Same cell-ID mapping as _collect_element_gp_data uses.
    gp_cell_ids = np.repeat(np.arange(n_elem, dtype=np.intp), nq)
    gp_coords_yz = np.empty((total, 2), dtype=float)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError("mixed 2D meshes are not supported")
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            idx = e * nq + q
            gp_coords_yz[idx] = np.asarray(T.Transform(ip))

    # Re-use the quadrature module to look up C per GP; extract alpha at
    # the same sample locations.
    from b3_micromech.quadrature import effective_stiffnesses_for_gauss_points

    cell_vertices = element_cell_vertices_yz(mesh)
    c_all = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords_yz,
        gp_cell_ids,
        cell_vertices,
    )

    # For each GP, pick the material that the stiffness field assigned
    # by inspecting which constituent stiffness it equals.
    matrix = problem.materials[problem.matrix_material].thermal_expansion
    fibre = problem.materials[problem.fibre_material].thermal_expansion

    alpha_per_gp = np.empty((total, 6), dtype=float)
    for idx in range(total):
        # Compare the stiffness at this GP to the two constituent stiffnesses.
        diff_m = np.abs(
            c_all[idx] - problem.materials[problem.matrix_material].stiffness
        ).max()
        diff_f = np.abs(
            c_all[idx] - problem.materials[problem.fibre_material].stiffness
        ).max()
        if diff_f < diff_m:
            alpha_per_gp[idx] = fibre
        else:
            alpha_per_gp[idx] = matrix
    return alpha_per_gp


def _make_rhs_integrator(sigma_macro_per_gp: NDArray[np.float64], data: _ElementGPData):
    import mfem.ser as mfem

    sm_view = sigma_macro_per_gp.reshape(data.n_elem, data.nq, 6)
    dsh_view = data.gp_dshapes.reshape(data.n_elem, data.nq, data.nd, 2)
    w_view = data.gp_weights.reshape(data.n_elem, data.nq)
    nq, nd = data.nq, data.nd

    class _RHS(mfem.PyLinearFormIntegrator):
        def AssembleRHSElementVect(self, el, Tr, elvect):
            e = Tr.ElementNo
            elvect.SetSize(nd * 3)
            local = np.zeros(nd * 3, dtype=float)
            for q in range(nq):
                B = voigt_b_matrix_plane_strain_x(dsh_view[e, q])
                local -= B.T @ sm_view[e, q] * w_view[e, q]
            elvect.GetDataArray()[:] = local

    return _RHS()


@dataclass
class PeriodicPlaneStrainSession:
    problem: RVEProblem
    mesh: object
    fespace: object
    _data: _ElementGPData
    _P_NC: object
    _n_T: int
    _n_constraints: int
    _n_scalar_L: int
    _nv: int
    _lu: object

    @property
    def gp_weights(self) -> NDArray[np.float64]:
        return self._data.gp_weights

    @property
    def c_per_gp(self) -> NDArray[np.float64]:
        return self._data.c_per_gp

    def solve_macro_strain(self, E_voigt: NDArray[np.float64]) -> LoadcaseResult:
        import mfem.ser as mfem

        E_voigt = np.asarray(E_voigt, dtype=float)
        data = self._data
        P_NC = self._P_NC
        nv = self._nv
        n_scalar_L = self._n_scalar_L

        sigma_macro = np.einsum("nij,j->ni", data.c_per_gp, E_voigt)
        b_lf = mfem.LinearForm(self.fespace)
        b_lf.AddDomainIntegrator(_make_rhs_integrator(sigma_macro, data))
        b_lf.Assemble()
        b_L = np.asarray(b_lf.GetDataArray()).copy()
        b_T = P_NC.T @ b_L
        b_aug = np.concatenate([b_T, np.zeros(self._n_constraints)])
        sol = self._lu.solve(b_aug)
        u_L = P_NC @ sol[: self._n_T]

        grad_u = _collect_u_gradient_at_gps(u_L, data)
        eps_fluct = grad_to_voigt_strain_plane_strain_x(grad_u)
        eps_total = eps_fluct + E_voigt[None, :]
        sigma_per_gp = np.einsum("nij,nj->ni", data.c_per_gp, eps_total)
        macro_stress = (data.gp_weights[:, None] * sigma_per_gp).sum(
            axis=0
        ) / data.gp_weights.sum()

        u_tilde = np.column_stack(
            [u_L[d * n_scalar_L : d * n_scalar_L + nv] for d in range(3)]
        )
        vertices_yz = np.array(
            [self.mesh.GetVertexArray(i) for i in range(nv)], dtype=float
        )
        u_macro = macro_displacement_at_yz(E_voigt, vertices_yz)
        u_total = u_tilde + u_macro

        nq = data.nq
        von_mises = np.empty(data.n_elem, dtype=float)
        for e in range(data.n_elem):
            vals = [von_mises_voigt(sigma_per_gp[e * nq + q]) for q in range(nq)]
            von_mises[e] = float(np.mean(vals))

        return LoadcaseResult(
            macro_strain=E_voigt.copy(),
            macro_stress=macro_stress,
            u_at_vertices=u_total,
            u_tilde_at_vertices=u_tilde,
            eps_per_gp=eps_total,
            sigma_per_gp=sigma_per_gp,
            von_mises_per_elem=von_mises,
        )


def make_session(problem: RVEProblem) -> PeriodicPlaneStrainSession:
    import mfem.ser as mfem
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mesh = build_mesh(problem)
    element_material_ids(problem, mesh)

    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)
    data = _collect_element_gp_data(mesh, fespace, problem)

    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_bilinear_integrator(data.c_per_gp, data))
    a.Assemble()
    a.Finalize()

    p_nc = fespace.GetConformingProlongation()
    n_L = a.SpMat().Height()
    if p_nc is None:
        P_NC = sp.eye(n_L, format="csr")
        n_T = n_L
    else:
        P_NC = _mfem_spmat_to_scipy(p_nc)
        n_T = P_NC.shape[1]

    K_L = _mfem_spmat_to_scipy(a.SpMat())
    K_T = (P_NC.T @ K_L @ P_NC).tocsr()

    n_scalar_L = fespace.GetNDofs()
    master_of = periodic_vertex_master_map(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    pin_vertex = origin_vertex_index(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    nv = mesh.GetNV()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_constraints = 0

    def add_row(row_sparse) -> None:
        nonlocal n_constraints
        coo = row_sparse.tocoo()
        for c, v in zip(coo.col, coo.data, strict=True):
            rows.append(n_constraints)
            cols.append(int(c))
            vals.append(float(v))
        n_constraints += 1

    is_hanging = np.zeros(nv, dtype=bool)
    for v in range(nv):
        row = P_NC.getrow(v)
        if row.nnz != 1 or abs(row.data[0] - 1.0) > 1e-12:
            is_hanging[v] = True

    for v in range(nv):
        m = int(master_of[v])
        if m == v or is_hanging[v] or is_hanging[m]:
            continue
        for d in range(3):
            l_slave = v + d * n_scalar_L
            l_master = m + d * n_scalar_L
            diff_row = P_NC.getrow(l_slave) - P_NC.getrow(l_master)
            if diff_row.nnz > 0:
                add_row(diff_row)

    for d in range(3):
        add_row(P_NC.getrow(d * n_scalar_L + pin_vertex))

    C = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_T)).tocsr()
    Z = sp.csr_matrix((n_constraints, n_constraints))
    A_aug = sp.bmat([[K_T, C.T], [C, Z]], format="csr").tocsc()
    lu = spla.splu(A_aug)

    return PeriodicPlaneStrainSession(
        problem=problem,
        mesh=mesh,
        fespace=fespace,
        _data=data,
        _P_NC=P_NC,
        _n_T=n_T,
        _n_constraints=n_constraints,
        _n_scalar_L=n_scalar_L,
        _nv=nv,
        _lu=lu,
    )


def solve_periodic_plane_strain(
    problem: RVEProblem,
) -> tuple[NDArray[np.float64], dict]:
    """Six unit macro-strain solves → volume-averaged stress columns of C_eff."""
    session = make_session(problem)
    eye6 = np.eye(6)
    cols = np.zeros((6, 6))
    for k in range(6):
        cols[:, k] = session.solve_macro_strain(eye6[k]).macro_stress
    C_eff = 0.5 * (cols + cols.T)
    meta = {
        "backend": "mfem_periodic_2d_plane_strain",
        "domain_shape": problem.domain_shape,
        "mesh_resolution": list(problem.mesh_resolution),
        "cell_type": problem.cell_type,
        "n_cells": int(session.mesh.GetNE()),
        "n_dofs": int(session.fespace.GetTrueVSize()),
        "fibre_volume_fraction": problem.fibre_volume_fraction,
        "material_sampling": problem.solver.material_sampling.to_dict(),
        "amr": problem.solver.amr.to_dict(),
    }
    return C_eff, meta


def solve_thermal_loadcase(
    problem: RVEProblem,
    *,
    delta_t: float = 1.0,
    c_eff: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """One-temperature-rise solve → effective thermal-expansion vector.

    Runs a thermal eigenstrain solve (``ΔT = delta_T``) on the RVE,
    computes the volume-averaged thermal stress at zero macroscopic strain
    ``<sigma> = -C_eff : alpha_eff * ΔT`` and returns
    ``alpha_eff = -C_eff^{-1} <sigma> / ΔT`` (shape ``(6,)``), the effective
    coefficient-of-thermal-expansion vector in Voigt form. ``c_eff`` may be
    passed in (e.g. from the elastic solve) to avoid recomputing it.

    Returns the full ``(6,)`` vector ``[alpha_xx, alpha_yy, alpha_zz,
    gamma_yz, gamma_xz, gamma_xy]``.  Only indices 0, 1, 2 (normal
    components) carry physical meaning for plane strain along *x*.
    """
    import mfem.ser as mfem
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mesh = build_mesh(problem)
    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 3)
    data = _collect_element_gp_data(mesh, fespace, problem)
    alpha_per_gp = _collect_alpha_at_gps(mesh, fespace, problem)

    # --- build stiffness matrix (same as mechanical) ---
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(_make_bilinear_integrator(data.c_per_gp, data))
    a.Assemble()
    a.Finalize()

    p_nc = fespace.GetConformingProlongation()
    n_L = a.SpMat().Height()
    P_NC = _mfem_spmat_to_scipy(p_nc) if p_nc is not None else sp.eye(n_L, format="csr")
    n_T = P_NC.shape[1]
    K_L = _mfem_spmat_to_scipy(a.SpMat())
    K_T = (P_NC.T @ K_L @ P_NC).tocsr()

    n_scalar_L = fespace.GetNDofs()
    master_of = periodic_vertex_master_map(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    pin_vertex = origin_vertex_index(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    nv = mesh.GetNV()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_constraints = 0

    def add_row(row_sparse) -> None:
        nonlocal n_constraints
        coo = row_sparse.tocoo()
        for c, v in zip(coo.col, coo.data, strict=True):
            rows.append(n_constraints)
            cols.append(int(c))
            vals.append(float(v))
        n_constraints += 1

    is_hanging = np.zeros(nv, dtype=bool)
    for v in range(nv):
        row = P_NC.getrow(v)
        if row.nnz != 1 or abs(row.data[0] - 1.0) > 1e-12:
            is_hanging[v] = True

    for v in range(nv):
        m = int(master_of[v])
        if m == v or is_hanging[v] or is_hanging[m]:
            continue
        for d in range(3):
            l_slave = v + d * n_scalar_L
            l_master = m + d * n_scalar_L
            diff_row = P_NC.getrow(l_slave) - P_NC.getrow(l_master)
            if diff_row.nnz > 0:
                add_row(diff_row)

    for d in range(3):
        add_row(P_NC.getrow(d * n_scalar_L + pin_vertex))

    C_mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_T)).tocsr()
    Z = sp.csr_matrix((n_constraints, n_constraints))
    A_aug = sp.bmat([[K_T, C_mat.T], [C_mat, Z]], format="csr").tocsc()
    lu = spla.splu(A_aug)

    # --- assemble thermal RHS (delta_T = 1 at each GP) ---
    b_lf = mfem.LinearForm(fespace)
    b_lf.AddDomainIntegrator(_make_thermal_integrator(alpha_per_gp, data))
    b_lf.Assemble()
    b_L = np.asarray(b_lf.GetDataArray()).copy()
    b_T = P_NC.T @ b_L
    b_aug = np.concatenate([b_T, np.zeros(n_constraints)])
    sol = lu.solve(b_aug)
    u_L = P_NC @ sol[:n_T]

    # --- compute volume-averaged thermal stress ---
    grad_u = _collect_u_gradient_at_gps(u_L, data)
    eps_fluct = grad_to_voigt_strain_plane_strain_x(grad_u)

    # Constitutive law with thermal eigenstrain: sigma = C : (eps - alpha*dT).
    # Periodic BC without applied macro strain -> <eps_fluct> = 0, so
    # <sigma> = -C_eff : alpha_eff * dT.
    eps_mech = eps_fluct - alpha_per_gp * delta_t
    sigma_per_gp = np.einsum("nij,nj->ni", data.c_per_gp, eps_mech)
    vol_avg_sigma = (data.gp_weights[:, None] * sigma_per_gp).sum(
        axis=0
    ) / data.gp_weights.sum()

    if c_eff is None:
        c_eff, _ = solve_periodic_plane_strain(problem)
    alpha_eff = -np.linalg.solve(c_eff, vol_avg_sigma) / delta_t

    meta = {
        "backend": "mfem_periodic_2d_thermal",
        "domain_shape": problem.domain_shape,
        "mesh_resolution": list(problem.mesh_resolution),
        "cell_type": problem.cell_type,
        "n_cells": int(mesh.GetNE()),
        "n_dofs": int(fespace.GetTrueVSize()),
        "fibre_volume_fraction": problem.fibre_volume_fraction,
        "delta_t": delta_t,
    }

    return alpha_eff, meta


# ---------------------------------------------------------------------------
# Steady-state thermal diffusion (conductivity homogenization)
# ---------------------------------------------------------------------------


def _collect_k_at_gps(mesh, fespace, problem: RVEProblem) -> NDArray[np.float64]:
    """Per-GP conductivity tensor (N_gps, 6, 6) from the RVEProblem.

    Reuses the same element/grid topology as _collect_element_gp_data
    but substitutes the conductivity tensor (k) instead of stiffness (C).
    """
    import mfem.ser as mfem

    n_elem = mesh.GetNE()
    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()
    total = n_elem * nq

    gp_coords_yz = np.empty((total, 2), dtype=float)
    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError("mixed 2D meshes are not supported")
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            idx = e * nq + q
            gp_coords_yz[idx] = np.asarray(T.Transform(ip))

    from b3_micromech.field import FIBRE_ID, MATRIX_ID, sample_material_ids

    k_per_gp_ids = sample_material_ids(problem, gp_coords_yz)
    k_matrix = problem.materials[problem.matrix_material].thermal_conductivity
    k_fibre = problem.materials[problem.fibre_material].thermal_conductivity

    k_per_gp = np.empty((total, 6, 6), dtype=float)
    k_per_gp[k_per_gp_ids == MATRIX_ID] = k_matrix
    k_per_gp[k_per_gp_ids == FIBRE_ID] = k_fibre

    return k_per_gp


def _make_diffusion_integrator(k_per_gp: NDArray[np.float64], data: _ElementGPData):
    """PyBilinearFormIntegrator for scalar diffusion: ∫ k ∇φ·∇ψ dV."""
    import mfem.ser as mfem

    nq, nd = data.nq, data.nd

    class _DiffusionInt(mfem.PyBilinearFormIntegrator):
        def AssembleElementMatrix(self, fe, T, elmat):
            e = T.ElementNo
            elmat.SetSize(nd)
            local = np.zeros((nd, nd), dtype=float)
            for q in range(nq):
                # gp_dshapes are stored ALREADY in physical coords (see
                # _precompute: Mult(dshape_ref, J_inv, dshape_phys)).
                dsh_phys = data.gp_dshapes[e * nq + q]  # (nd, 2) physical
                w = data.gp_weights[e * nq + q]
                k_val = k_per_gp[e * nq + q]  # (6, 6)

                # Transverse-plane conductivity: k[0:2, 0:2] in (y, z)
                k_2d = k_val[1:3, 1:3]
                for i in range(nd):
                    for j in range(nd):
                        grad_i = dsh_phys[i]  # (dy, dz)
                        grad_j = dsh_phys[j]  # (dy, dz)
                        local[i, j] += grad_i @ k_2d @ grad_j * w
            elmat.GetDataArray()[:] = local

    return _DiffusionInt()


@dataclass(frozen=True)
class _DiffGPData:
    """Minimal data view for the diffusion integrator (scalar field)."""

    nq: int
    nd: int
    n_elem: int
    gp_dshapes: NDArray[np.float64]
    gp_weights: NDArray[np.float64]
    elem_vdofs: NDArray[np.intp]


def effective_conductivity_tensor(
    problem: RVEProblem,
) -> tuple[NDArray[np.float64], dict]:
    """Compute effective conductivity tensor via steady-state diffusion.

    Solves for 2 unit temperature-gradient loadcases in the transverse plane
    (y and z), assembles the effective ``(2, 2)`` tensor, and embeds it in
    ``(6, 6)`` Voigt form.  The fibre-direction component ``k_xx`` is
    computed by rule-of-mixtures (decoupled in 2D).

    Returns ``(k_eff, metadata)`` where ``k_eff`` has shape ``(6, 6)``.
    """
    import mfem.ser as mfem
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # ---- Step 1: build scalar mesh & collect GP data ----
    mesh = build_mesh(problem)

    fec = mfem.H1_FECollection(1, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec, 1)  # scalar (temperature)
    n_elem = mesh.GetNE()

    fe0 = fespace.GetFE(0)
    nd = fe0.GetDof()
    ir0 = mfem.IntRules.Get(fe0.GetGeomType(), 2 * fe0.GetOrder())
    nq = ir0.GetNPoints()

    total = n_elem * nq
    gp_coords_yz = np.empty((total, 2), dtype=float)
    gp_dshapes = np.empty((total, nd, 2), dtype=float)
    gp_weights = np.empty(total, dtype=float)
    elem_vdofs = np.empty((n_elem, nd), dtype=np.intp)

    dshape_ref = mfem.DenseMatrix(nd, 2)
    J_inv = mfem.DenseMatrix(2, 2)
    dshape_phys_tmp = mfem.DenseMatrix(nd, 2)

    for e in range(n_elem):
        T = mesh.GetElementTransformation(e)
        fe = fespace.GetFE(e)
        ir = mfem.IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder())
        if fe.GetDof() != nd or ir.GetNPoints() != nq:
            raise NotImplementedError("mixed 2D meshes are not supported")
        elem_vdofs[e] = np.asarray(fespace.GetElementVDofs(e), dtype=np.intp)
        for q in range(nq):
            ip = ir.IntPoint(q)
            T.SetIntPoint(ip)
            idx = e * nq + q
            gp_coords_yz[idx] = np.asarray(T.Transform(ip))
            fe.CalcDShape(ip, dshape_ref)
            mfem.CalcInverse(T.Jacobian(), J_inv)
            mfem.Mult(dshape_ref, J_inv, dshape_phys_tmp)
            gp_dshapes[idx] = np.asarray(dshape_phys_tmp.GetDataArray())
            gp_weights[idx] = ip.weight * T.Weight()

    # Collect k per GP
    k_per_gp = _collect_k_at_gps(mesh, fespace, problem)

    # Guard: zero in-plane conductivity in any constituent (e.g. axial-only
    # material data) leaves empty operator rows -> singular factorization.
    # Floor the in-plane diagonal with a tiny fraction of the global maximum
    # BEFORE assembling K; skip the transverse solve entirely if there is no
    # in-plane conduction at all.
    k_inplane_max = float(np.max(np.abs(k_per_gp[:, 1:3, 1:3])))
    if k_inplane_max <= 0.0:
        # No in-plane conduction at all (or thermal data absent): skip the
        # 2D solve entirely; only the axial rule-of-mixtures entry is set.
        k_eff = np.zeros((6, 6), dtype=float)
        vf = problem.fibre_volume_fraction
        k_m = problem.materials[problem.matrix_material].thermal_conductivity
        k_f = problem.materials[problem.fibre_material].thermal_conductivity
        k_eff[0, 0] = vf * k_f[0, 0] + (1.0 - vf) * k_m[0, 0]
        return k_eff, {
            "backend": "mfem_periodic_2d_diffusion",
            "skipped_transverse": True,
            "fibre_volume_fraction": vf,
        }
    _floor = 1e-9 * k_inplane_max
    for _i in (1, 2):
        np.maximum(k_per_gp[:, _i, _i], _floor, out=k_per_gp[:, _i, _i])

    # ---- Step 2: build diffusion stiffness matrix ----
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(
        _make_diffusion_integrator(
            k_per_gp,
            type(
                "DummyGPData",
                (),
                {
                    "nq": nq,
                    "nd": nd,
                    "n_elem": n_elem,
                    "gp_dshapes": gp_dshapes,
                    "gp_weights": gp_weights,
                    "elem_vdofs": elem_vdofs,
                },
            )(),
        )
    )
    a.Assemble()
    a.Finalize()

    p_nc = fespace.GetConformingProlongation()
    n_L = a.SpMat().Height()
    P_NC = _mfem_spmat_to_scipy(p_nc) if p_nc is not None else sp.eye(n_L, format="csr")
    n_T = P_NC.shape[1]
    K_L = _mfem_spmat_to_scipy(a.SpMat())
    K_T = (P_NC.T @ K_L @ P_NC).tocsr()

    # ---- Step 3: periodic constraints ----
    master_of = periodic_vertex_master_map(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    pin_vertex = origin_vertex_index(
        mesh,
        shape=problem.domain_shape,
        domain_size=problem.domain_size,
        tol=problem.periodic_tolerance,
    )
    nv = mesh.GetNV()

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    n_constraints = 0

    def add_row(row_sparse) -> None:
        nonlocal n_constraints
        coo = row_sparse.tocoo()
        for c, v in zip(coo.col, coo.data, strict=True):
            rows.append(n_constraints)
            cols.append(int(c))
            vals.append(float(v))
        n_constraints += 1

    is_hanging = np.zeros(nv, dtype=bool)
    for v in range(nv):
        row = P_NC.getrow(v)
        if row.nnz != 1 or abs(row.data[0] - 1.0) > 1e-12:
            is_hanging[v] = True

    for v in range(nv):
        m = int(master_of[v])
        if m == v or is_hanging[v] or is_hanging[m]:
            continue
        l_slave = v
        l_master = m
        diff_row = P_NC.getrow(l_slave) - P_NC.getrow(l_master)
        if diff_row.nnz > 0:
            add_row(diff_row)

    # pin origin to remove rigid-body (constant-T) mode
    add_row(P_NC.getrow(pin_vertex))

    C_mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_T)).tocsr()
    Z = sp.csr_matrix((n_constraints, n_constraints))
    A_aug = sp.bmat([[K_T, C_mat.T], [C_mat, Z]], format="csr").tocsc()
    lu = spla.splu(A_aug)

    # ---- Step 4: solve for each transverse direction ----
    k_eff_2d = np.zeros((2, 2), dtype=float)

    for dir_idx in range(2):  # y=0, z=1
        applied_grad = np.zeros(2, dtype=float)
        applied_grad[dir_idx] = 1.0

        dsh_phys = np.empty((nd, 2), dtype=float)

        # Assemble RHS: f_i = -∫ k · g · ∇ψ dV
        b_L = np.zeros(n_T, dtype=float)
        for e in range(n_elem):
            for q in range(nq):
                idx = e * nq + q

                k_val = k_per_gp[idx]
                k_2d = k_val[1:3, 1:3]
                rhs_vec = k_2d @ applied_grad

                # gp_dshapes are stored ALREADY in physical coords.
                dsh_phys = gp_dshapes[idx]

                w = gp_weights[idx]
                for i in range(nd):
                    local_rhs = -(dsh_phys[i] @ rhs_vec) * w
                    global_idx = elem_vdofs[e, i]
                    P_NC_row = P_NC.getrow(global_idx)
                    if P_NC_row.nnz > 0:
                        for c_idx, c_val in zip(P_NC_row.indices, P_NC_row.data):
                            b_L[int(c_idx)] += local_rhs * c_val

        b_aug = np.concatenate([b_L, np.zeros(n_constraints)])
        sol = lu.solve(b_aug)
        phi_L = P_NC @ sol[:n_T]

        # Compute volume-averaged flux
        grad_phi = np.zeros((total, 2), dtype=float)
        for idx in range(total):
            phi_elem = phi_L[elem_vdofs[idx // nq]]
            e = idx // nq
            T = mesh.GetElementTransformation(e)
            ip = ir0.IntPoint(idx % nq)
            T.SetIntPoint(ip)
            J = T.Jacobian()
            mfem.CalcInverse(J, J_inv)
            fe = fespace.GetFE(e)
            fe.CalcDShape(ip, dshape_ref)
            mfem.Mult(dshape_ref, J_inv, dshape_phys_tmp)
            dsh_phys[:] = np.asarray(dshape_phys_tmp.GetDataArray())
            for i in range(nd):
                grad_phi[idx, 0] += phi_elem[i] * dsh_phys[i, 0]
                grad_phi[idx, 1] += phi_elem[i] * dsh_phys[i, 1]

        total_grad = grad_phi + applied_grad  # (total, 2)
        q_per_gp = np.empty((total, 2), dtype=float)
        for idx in range(total):
            k_val = k_per_gp[idx][1:3, 1:3]
            q_per_gp[idx] = -(k_val @ total_grad[idx])

        q_vol = (gp_weights[:, None] * q_per_gp).sum(axis=0) / gp_weights.sum()

        # q = -k_eff · grad  →  k_eff[:, dir] = -q
        k_eff_2d[:, dir_idx] = -q_vol

    # Symmetrize
    k_eff_2d = 0.5 * (k_eff_2d + k_eff_2d.T)

    # ---- Step 5: embed in (6,6) Voigt form ----
    k_eff = np.zeros((6, 6), dtype=float)
    # 2D solve lives in the transverse (y, z) plane -> tensor indices 1, 2
    k_eff[1:3, 1:3] = k_eff_2d

    # Fibre-direction (xx): rule-of-mixtures (decoupled in 2D)
    vf = problem.fibre_volume_fraction
    k_m = problem.materials[problem.matrix_material].thermal_conductivity
    k_f = problem.materials[problem.fibre_material].thermal_conductivity
    k_eff[0, 0] = vf * k_f[0, 0] + (1.0 - vf) * k_m[0, 0]

    meta = {
        "backend": "mfem_periodic_2d_diffusion",
        "domain_shape": problem.domain_shape,
        "mesh_resolution": list(problem.mesh_resolution),
        "cell_type": problem.cell_type,
        "n_cells": n_elem,
        "n_dofs": int(fespace.GetTrueVSize()),
        "fibre_volume_fraction": problem.fibre_volume_fraction,
    }

    return k_eff, meta
