"""Matplotlib plots for RVE geometry and per-loadcase deformation patterns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from b3_micromech.amr import _resolve_amr_spec, cell_refinement_metric
from b3_micromech.geometry import hexagon_corners
from b3_micromech.mesh.cartesian import mesh_vertices_and_cells
from b3_micromech.postprocess import LoadcaseSet
from b3_micromech.result import LOADCASE_LABELS
from b3_micromech.tensors import engineering_constants_transverse_iso


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib — pip install b3-micromech[viz]"
        ) from exc
    return plt


def _material_colors(material_ids: NDArray[np.int32]) -> list:
    palette = {1: (0.88, 0.90, 0.93, 0.85), 2: (0.82, 0.35, 0.32, 0.90)}
    return [palette.get(int(m), (0.7, 0.7, 0.7, 0.8)) for m in material_ids]


def _fibre_circle(problem) -> tuple[float, float, float]:
    centre = problem.centre_yz
    return float(centre[0]), float(centre[1]), float(problem.fibre_radius)


def _set_plot_limits(ax, problem) -> None:
    ymin, ymax, zmin, zmax = problem.plot_bounds
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)


def _draw_domain_outline(ax, problem) -> None:
    if problem.domain_shape == "hexagon":
        from matplotlib.patches import Polygon

        corners = hexagon_corners(problem.domain_size)
        ax.add_patch(
            Polygon(
                corners,
                fill=False,
                edgecolor="#1a1a1a",
                linewidth=1.2,
                linestyle="-",
                zorder=2,
            )
        )
    else:
        ymax, zmax = problem.size_yz
        ax.plot([0, ymax, ymax, 0, 0], [0, 0, zmax, zmax, 0], color="#1a1a1a", lw=1.2)


def _auto_scale(loadcases: LoadcaseSet, fraction: float = 0.12) -> float:
    max_mag = 0.0
    for res in loadcases.results:
        uyz = res.u_at_vertices[:, 1:3]
        max_mag = max(max_mag, float(np.max(np.linalg.norm(uyz, axis=1))))
    if max_mag < 1e-15:
        return 1.0
    return fraction * loadcases.problem.domain_size / max_mag


def _draw_mesh_cells(ax, vertices, cells, facecolors, edgecolor="#333333", lw=0.4):
    from matplotlib.collections import PolyCollection

    polys = [vertices[cell] for cell in cells]
    coll = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidths=lw,
    )
    ax.add_collection(coll)
    return coll


def _amr_enabled(problem) -> bool:
    return bool(_resolve_amr_spec(problem.solver).get("enabled", False))


def plot_amr_refinement(loadcases: LoadcaseSet, out_path: str | Path) -> Path:
    """Mesh coloured by the AMR stiffness-jump marker (post-refinement)."""
    plt = _require_matplotlib()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle

    problem = loadcases.problem
    amr = _resolve_amr_spec(problem.solver)
    mesh = loadcases.session.mesh
    metric = cell_refinement_metric(
        problem,
        mesh,
        marker=amr["marker"],
        n_samples_per_cell=amr["n_samples_per_cell"],
    )

    vertices, cells = mesh_vertices_and_cells(mesh)
    polys = [vertices[cell] for cell in cells]
    threshold = amr["threshold"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    coll = PolyCollection(
        polys,
        array=metric,
        cmap="magma",
        edgecolors="#333333",
        linewidths=0.35,
    )
    coll.set_clim(0.0, max(float(metric.max()), threshold))
    ax.add_collection(coll)
    _draw_domain_outline(ax, problem)
    cy, cz, r = _fibre_circle(problem)
    ax.add_patch(
        Circle(
            (cy, cz), r, fill=False, edgecolor="#ffffff", linewidth=1.2, linestyle="--"
        )
    )
    ax.set_aspect("equal")
    _set_plot_limits(ax, problem)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(
        f"AMR marker ({amr['marker']})  {mesh.GetNE()} cells, threshold={threshold:g}"
    )
    cbar = fig.colorbar(coll, ax=ax, shrink=0.85)
    cbar.ax.axhline(threshold, color="#4fc3f7", linewidth=1.5, linestyle="--")
    cbar.set_label("stiffness-jump score")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_rve_overview(loadcases: LoadcaseSet, out_path: str | Path) -> Path:
    """Undeformed transverse mesh with fibre disc."""
    plt = _require_matplotlib()
    from matplotlib.patches import Circle

    problem = loadcases.problem
    vertices, cells = mesh_vertices_and_cells(loadcases.session.mesh)
    facecolors = _material_colors(loadcases.material_ids)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    _draw_mesh_cells(ax, vertices, cells, facecolors)
    cy, cz, r = _fibre_circle(problem)
    ax.add_patch(
        Circle(
            (cy, cz), r, fill=False, edgecolor="#1a1a1a", linewidth=1.5, linestyle="--"
        )
    )
    _draw_domain_outline(ax, problem)
    ax.set_aspect("equal")
    _set_plot_limits(ax, problem)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    shape = problem.domain_shape
    mesh = loadcases.session.mesh
    res_tag = (
        f"AMR {mesh.GetNE()} cells"
        if _amr_enabled(problem)
        else f"{problem.mesh_resolution[0]}×{problem.mesh_resolution[1]}"
    )
    ax.set_title(
        f"Transverse RVE ({shape})  (Vf={problem.fibre_volume_fraction:.2f}, "
        f"{res_tag} {problem.cell_type})"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _plot_loadcase_panel(
    ax,
    *,
    vertices: NDArray[np.float64],
    cells: list,
    material_ids: NDArray[np.int32],
    result,
    scale: float,
    problem,
    title: str,
    mode: str = "in_plane",
):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle

    u = result.u_at_vertices
    if mode == "in_plane":
        disp = u[:, 1:3]
        values = np.linalg.norm(disp, axis=1)
        cmap = "viridis"
        cbar_label = r"$|u_{yz}|$"
    elif mode == "fibre":
        disp = np.column_stack([np.zeros(len(u)), u[:, 0]])
        values = np.abs(u[:, 0])
        cmap = "coolwarm"
        cbar_label = r"$u_x$ (fibre axis)"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    def_y = vertices[:, 0] + scale * disp[:, 0]
    def_z = vertices[:, 1] + scale * disp[:, 1]

    polys = [np.column_stack([def_y[cell], def_z[cell]]) for cell in cells]
    facecolors = _material_colors(material_ids)
    ax.add_collection(
        PolyCollection(
            polys,
            facecolors=facecolors,
            edgecolors="#222222",
            linewidths=0.35,
            alpha=0.55,
        )
    )

    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-12
    sc = ax.scatter(
        def_y,
        def_z,
        c=values,
        s=14,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        zorder=3,
    )

    step = max(1, len(vertices) // 12)
    ax.quiver(
        vertices[::step, 0],
        vertices[::step, 1],
        scale * disp[::step, 0],
        scale * disp[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#111111",
        width=0.003,
        zorder=4,
    )

    cy, cz, r = _fibre_circle(problem)
    ax.add_patch(
        Circle(
            (cy, cz), r, fill=False, edgecolor="#000000", linewidth=1.0, linestyle=":"
        )
    )
    ax.set_aspect("equal")
    _set_plot_limits(ax, problem)
    voigt_idx = int(np.argmax(np.abs(result.macro_strain)))
    sig = result.macro_stress[voigt_idx] / 1e9
    ax.set_title(f"{title}\nσ̄[{voigt_idx}] = {sig:.2f} GPa", fontsize=9)
    return sc, cbar_label


def plot_loadcase_deformations(
    loadcases: LoadcaseSet,
    out_path: str | Path,
    *,
    scale: float | None = None,
) -> Path:
    """2×3 grid of exaggerated in-plane deformation patterns (six unit strains)."""
    plt = _require_matplotlib()

    if scale is None:
        scale = _auto_scale(loadcases)

    vertices, cells = mesh_vertices_and_cells(loadcases.session.mesh)
    problem = loadcases.problem

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    axes_flat = axes.ravel()
    mappable = None
    cbar_label = ""

    for k, (ax, (label, _), result) in enumerate(
        zip(axes_flat, LOADCASE_LABELS, loadcases.results, strict=True)
    ):
        mappable, cbar_label = _plot_loadcase_panel(
            ax,
            vertices=vertices,
            cells=cells,
            material_ids=loadcases.material_ids,
            result=result,
            scale=scale,
            problem=problem,
            title=label,
            mode="in_plane",
        )
        ax.set_xlabel("y")
        ax.set_ylabel("z")

    fig.suptitle(
        f"Periodic fluctuation + macro strain  (deflection scale ×{scale:.3g})",
        fontsize=11,
    )
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, shrink=0.7, label=cbar_label)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_fibre_displacement(loadcases: LoadcaseSet, out_path: str | Path) -> Path:
    """2×3 grid of fibre-direction displacement ``u_x`` per loadcase."""
    plt = _require_matplotlib()

    vertices, cells = mesh_vertices_and_cells(loadcases.session.mesh)
    problem = loadcases.problem

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    mappable = None
    cbar_label = ""

    for ax, (label, _), result in zip(
        axes.ravel(), LOADCASE_LABELS, loadcases.results, strict=True
    ):
        mappable, cbar_label = _plot_loadcase_panel(
            ax,
            vertices=vertices,
            cells=cells,
            material_ids=loadcases.material_ids,
            result=result,
            scale=0.0,
            problem=problem,
            title=label,
            mode="fibre",
        )
        ax.set_xlabel("y")
        ax.set_ylabel("z")

    fig.suptitle("Fibre-direction displacement $u_x$ (plane-strain slice)", fontsize=11)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, shrink=0.7, label=cbar_label)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_von_mises(loadcases: LoadcaseSet, out_path: str | Path) -> Path:
    """2×3 element von Mises stress for each macro-strain loadcase."""
    plt = _require_matplotlib()
    from matplotlib.collections import PolyCollection

    vertices, cells = mesh_vertices_and_cells(loadcases.session.mesh)
    problem = loadcases.problem

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    mappable = None

    for ax, (label, _), result in zip(
        axes.ravel(), LOADCASE_LABELS, loadcases.results, strict=True
    ):
        polys = [vertices[cell] for cell in cells]
        vm = result.von_mises_per_elem / 1e6
        coll = PolyCollection(
            polys,
            array=vm,
            cmap="magma",
            edgecolors="#333333",
            linewidths=0.3,
        )
        ax.add_collection(coll)
        ax.set_aspect("equal")
        _set_plot_limits(ax, problem)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        mappable = coll

    fig.suptitle("Element von Mises stress (MPa)", fontsize=11)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, shrink=0.7, label="σ_vm")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_engineering_constants(loadcases: LoadcaseSet, out_path: str | Path) -> Path:
    """Bar chart of homogenized engineering constants from the six loadcases."""
    plt = _require_matplotlib()

    cols = np.column_stack([r.macro_stress for r in loadcases.results])
    C = 0.5 * (cols + cols.T)
    ec = engineering_constants_transverse_iso(C)

    labels = ["E₁", "E₂", "G₁₂", "G₂₃", "ν₁₂"]
    values = [ec["e_l"], ec["e_t"], ec["g_lt"], ec["g_tt"], ec["nu_lt"]]
    units = ["GPa", "GPa", "GPa", "GPa", "–"]

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    bars = ax.bar(labels, [v / 1e9 if u != "–" else v for v, u in zip(values, units)])
    ax.set_ylabel("Value")
    ax.set_title(
        f"Homogenized constants  (Vf={loadcases.problem.fibre_volume_fraction:.2f})"
    )
    for bar, val, unit in zip(bars, values, units):
        text = f"{val / 1e9:.2f} {unit}" if unit != "–" else f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def render_all_figures(
    loadcases: LoadcaseSet,
    out_dir: str | Path,
    *,
    scale: float | None = None,
) -> dict[str, Path]:
    """Write the standard plot bundle for an RVE solve."""
    out_dir = Path(out_dir)
    paths = {
        "rve_overview": plot_rve_overview(loadcases, out_dir / "rve_overview.png"),
        "loadcase_deformations": plot_loadcase_deformations(
            loadcases, out_dir / "loadcase_deformations.png", scale=scale
        ),
        "fibre_displacement": plot_fibre_displacement(
            loadcases, out_dir / "fibre_displacement.png"
        ),
        "von_mises": plot_von_mises(loadcases, out_dir / "von_mises.png"),
        "engineering_constants": plot_engineering_constants(
            loadcases, out_dir / "engineering_constants.png"
        ),
    }
    if _amr_enabled(loadcases.problem):
        paths["amr_refinement"] = plot_amr_refinement(
            loadcases, out_dir / "amr_refinement.png"
        )
    return paths
