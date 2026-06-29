"""MFEM 2D periodic homogenization under plane strain along fibre axis x."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from b3_micromech.mesh.build import build_mesh
from b3_micromech.mesh.cartesian import element_cell_vertices_yz, element_material_ids
from b3_micromech.quadrature import (
    _resolve_material_sampling_spec,
    effective_stiffnesses_for_gauss_points,
)
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
    sampling_spec = _resolve_material_sampling_spec(problem.solver)
    c_per_gp[:] = effective_stiffnesses_for_gauss_points(
        problem,
        gp_coords_yz,
        gp_cell_ids,
        cell_vertices,
        spec=sampling_spec,
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
        "material_sampling": _resolve_material_sampling_spec(problem.solver),
        "amr": problem.solver.get("amr", {}),
    }
    return C_eff, meta
