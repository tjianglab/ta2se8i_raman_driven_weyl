#!/usr/bin/env python3
"""
Figure 2 plotting script.

This script generates Fig. 2 (2x4 panels) including:
(a) Direct gap Eg(Γ–Z) vs phonon coordinate Q for mode A(18), with highlighted points.
(b–d) Band structures at selected Q values (Q=0, 2.0, 2.5); panel (d) includes an inset zoom.
(e–h) Brillouin-zone (BZ) projections with Weyl nodes overlaid for Q=2.0 and Q=2.5:
      - (kx, ky) projection
      - (k_parallel, kz) projection where k_parallel = (kx + ky)/sqrt(2)

Inputs
------
- GAP_VS_DISP_MODE18/combine_gap_list_vs_distortion_TSI.h5
- BANDS_MODE18_EQP/save_bands_and_klist.h5
- BANDS_MODE18_EQUAL_BOND/save_bands_and_klist.h5
- BANDS_MODE18_GAP_CLOSE/save_bands_and_klist.h5
- Q_2.0_WEYL_NODES_WITH_CHIRALITY/POSCAR, Q_2.0_WEYL_NODES_WITH_CHIRALITY/filtered_Nodes_with_Chiralities.dat
- Q_2.5_WEYL_NODES_WITH_CHIRALITY/POSCAR, Q_2.5_WEYL_NODES_WITH_CHIRALITY/filtered_Nodes_with_Chiralities.dat

Output
------
- FIG_2.png (dpi=600)

Notes
-----
- Weyl node file format is assumed:
  columns 1–3: fractional coordinates in reciprocal basis (b1, b2, b3),
  column 7: chirality (+1 or -1).
- For the (kx, ky) plot, only nodes with kz > 0 are shown (to avoid duplication),
  consistent with your original behavior.
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

from ase.io import read
from scipy.spatial import Voronoi

from matplotlib.ticker import MultipleLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# =============================================================================
# Global style
# =============================================================================
COLORS = np.asarray(
    [
        [0, 0, 0],
        [230, 159, 0],
        [86, 180, 233],
        [0, 158, 115],
        [240, 228, 66],
        [0, 114, 178],
        [213, 94, 0],
        [204, 121, 167],
    ],
    dtype=float
) / 255.0
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)


# =============================================================================
# User-editable parameters (paths, annotations, plot windows)
# =============================================================================
# --- Gap vs Q (panel a) ---
GAP_H5 = "GAP_VS_DISP_MODE18/combine_gap_list_vs_distortion_TSI.h5"
GAP_XLIM = (-4.0, 4.0)
GAP_YLIM = (0.0, 1.0)

# "Special points" on the gap curve:
# - b: point closest to Q=0 (min |Q|)
# - c: Q=2.0 (exact match in your file; could replace by np.isclose)
# - d: minimum gap point
Q_TARGET = 2.0

# --- Bands (panels b–d) ---
BANDS_Q0_H5   = "BANDS_MODE18_EQP/save_bands_and_klist.h5"         # Q=0
BANDS_Q2_H5   = "BANDS_MODE18_EQUAL_BOND/save_bands_and_klist.h5"  # Q=2.0
BANDS_Q25_H5  = "BANDS_MODE18_GAP_CLOSE/save_bands_and_klist.h5"   # Q=2.5

BAND_YLIM = (-1.0, 1.0)
BAND_YTICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Inset zoom window (panel d)
INSET_XMIN, INSET_XMAX = 217, 243
INSET_YMIN, INSET_YMAX = -0.06, 0.00

# --- BZ/Weyl (panels e–h) ---
# Edges are manually specified based on the vertex indexing returned by Voronoi.
EDGES = [
    (23,13),(13,18),(15,18),(18,7),(7,9),(9,20),(20,5),(5,15),(15,23),
    (7,11),(11,21),(21,1),(1,13),
    (1,19),(19,12),(12,4),(4,23),
    (20,2),(2,14),(14,24),(24,3),(3,9),
    (11,3),(24,16),(16,6),(6,21),
    (6,19),(16,17),(17,8),(8,12),
    (22,10),(10,4),(10,8),
    (22,5),(22,2),(14,17)
]

# Q=2.0
POSCAR_Q2 = "Q_2.0_WEYL_NODES_WITH_CHIRALITY/POSCAR"
NODES_Q2  = "Q_2.0_WEYL_NODES_WITH_CHIRALITY/filtered_Nodes_with_Chiralities.dat"

# Q=2.5
POSCAR_Q25 = "Q_2.5_WEYL_NODES_WITH_CHIRALITY/POSCAR"
NODES_Q25  = "Q_2.5_WEYL_NODES_WITH_CHIRALITY/filtered_Nodes_with_Chiralities.dat"

# BZ plot range (fixed to match your original)
BZ_LIM = (-0.55, 0.55)
BZ_TICKS = [-0.5, 0.0, 0.5]


# =============================================================================
# Utilities: reciprocal lattice, Voronoi BZ vertices, Weyl loading
# =============================================================================
def get_reciprocal_lattice_from_poscar(poscar_file, use_2pi=True):
    """
    Read real-space lattice vectors from a VASP POSCAR and compute reciprocal vectors b1,b2,b3.

    Convention:
      b1 = 2π * (a2 × a3) / V, etc. if use_2pi=True
    """
    atoms = read(poscar_file, format="vasp")
    cell = atoms.cell  # rows: a1,a2,a3
    a1 = np.array(cell[0], dtype=float)
    a2 = np.array(cell[1], dtype=float)
    a3 = np.array(cell[2], dtype=float)

    volume = np.dot(a1, np.cross(a2, a3))
    factor = 2.0 * np.pi if use_2pi else 1.0

    b1 = factor * np.cross(a2, a3) / volume
    b2 = factor * np.cross(a3, a1) / volume
    b3 = factor * np.cross(a1, a2) / volume
    return b1, b2, b3


def compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8):
    """
    Compute vertices of the 1st Brillouin zone using Voronoi (Wigner–Seitz) construction.

    Returns
    -------
    unique_vertices : (N,3) array of BZ vertices (cartesian reciprocal units)
    """
    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)

    pts = []
    origin_index = None
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            for n3 in range(-nmax, nmax + 1):
                g = n1 * b1 + n2 * b2 + n3 * b3
                pts.append(g)
                if n1 == 0 and n2 == 0 and n3 == 0:
                    origin_index = len(pts) - 1

    pts = np.array(pts)
    if origin_index is None:
        raise RuntimeError("Failed to locate Γ at (0,0,0) in the reciprocal lattice grid.")

    vor = Voronoi(pts)
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]
    if -1 in region:
        raise RuntimeError(
            "Voronoi region contains a point at infinity (-1). "
            "Increase nmax or check the lattice geometry."
        )

    bz_vertices = vor.vertices[region]
    bz_rounded = np.round(bz_vertices, round_decimals)
    unique_vertices = np.unique(bz_rounded, axis=0)

    # Sort by distance to Γ for reproducible indexing (purely cosmetic)
    radii = np.linalg.norm(unique_vertices, axis=1)
    unique_vertices = unique_vertices[np.argsort(radii)]
    return unique_vertices


def load_weyl_points(nodes_file, b1, b2, b3, kz_positive_only=True):
    """
    Load Weyl nodes and chiralities from a text file.

    Assumed format:
      col 1–3: fractional coordinates in (b1,b2,b3) basis
      col 7  : chirality (+1 or -1)

    Returns
    -------
    weyl_cart : (M,3) Weyl positions in cartesian reciprocal coordinates
    weyl_sign : (M,) array of '+'/'-'
    k_frac    : (M,3) fractional coords (filtered consistently with weyl_cart)
    """
    data = np.loadtxt(nodes_file, comments="#")
    k_frac = data[:, :3]
    chirality = data[:, 6]

    B = np.vstack([np.array(b1), np.array(b2), np.array(b3)])  # 3x3
    weyl_cart = k_frac @ B

    if kz_positive_only:
        mask = weyl_cart[:, 2] > 0.0
        weyl_cart = weyl_cart[mask]
        chirality = chirality[mask]
        k_frac = k_frac[mask]

    weyl_sign = np.where(chirality > 0, "+", "-")
    return weyl_cart, weyl_sign, k_frac


# =============================================================================
# Utilities: drawing edges in different projections
# =============================================================================
def draw_edges_kxky(ax, verts, edge_list, **kwargs):
    """Draw BZ edges in the (kx, ky) projection."""
    for i, j in edge_list:
        p1 = verts[i - 1]
        p2 = verts[j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs, zorder=1)


def draw_edges_kpar_kz(ax, verts, edge_list, **kwargs):
    """Draw BZ edges in the (k_parallel, kz) projection with k_parallel=(kx+ky)/sqrt(2)."""
    for i, j in edge_list:
        p1 = verts[i - 1]
        p2 = verts[j - 1]
        x1 = (p1[0] + p1[1]) / np.sqrt(2)
        x2 = (p2[0] + p2[1]) / np.sqrt(2)
        ax.plot([x1, x2], [p1[2], p2[2]], **kwargs, zorder=1)


# =============================================================================
# Plotting helpers: BZ + Weyl overlays
# =============================================================================
def format_bz_axes(ax, xlabel, ylabel):
    """Common formatting for BZ projection panels."""
    ax.grid(False)
    ax.set_xlim(*BZ_LIM)
    ax.set_ylim(*BZ_LIM)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)

    ax.set_xticks(BZ_TICKS)
    ax.set_yticks(BZ_TICKS)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove minor ticks for a clean schematic look
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bz_kxky(ax, verts, weyl_cart=None, weyl_sign=None):
    """
    Plot (kx, ky) projection of BZ and optional Weyl nodes.
    """
    # Vertices (not shown as points; keep as s=0 to preserve your behavior)
    for v in verts:
        ax.scatter(v[0], v[1], c="k", s=0, edgecolor="none", zorder=2)

    if weyl_cart is not None and len(weyl_cart) > 0:
        xy = weyl_cart[:, :2]
        colors = np.array(["r" if s == "+" else "b" for s in weyl_sign])
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=colors,
            s=12,
            edgecolors="white",
            linewidths=0.3,
            zorder=4,
        )

    format_bz_axes(ax, xlabel=r"$k_x \,(2\pi/a)$", ylabel=r"$k_y \,(2\pi/b)$")


def plot_bz_kpar_kz(ax, verts, weyl_cart=None, weyl_sign=None):
    """
    Plot (k_parallel, kz) projection where k_parallel=(kx+ky)/sqrt(2).
    Weyl nodes are depth-sorted by k_perp = (-kx+ky)/sqrt(2) to reduce overplotting.
    """
    for v in verts:
        kpar = (v[0] + v[1]) / np.sqrt(2)
        ax.scatter(kpar, v[2], c="k", s=0, edgecolor="none", zorder=2)

    if weyl_cart is not None and len(weyl_cart) > 0:
        kpar = (weyl_cart[:, 0] + weyl_cart[:, 1]) / np.sqrt(2)
        kperp = (-weyl_cart[:, 0] + weyl_cart[:, 1]) / np.sqrt(2)
        kz = weyl_cart[:, 2]
        colors = np.array(["r" if s == "+" else "b" for s in weyl_sign])

        # Sort so points with smaller |kperp| are plotted first (appear behind)
        idx = np.argsort(kperp)
        ax.scatter(
            kpar[idx], kz[idx],
            c=colors[idx],
            s=12,
            edgecolors="white",
            linewidths=0.3,
            zorder=4,
        )

    format_bz_axes(ax, xlabel=r"$k_{\parallel}\,(2\pi/a_{\parallel})$", ylabel=r"$k_z \,(2\pi/c)$")


# =============================================================================
# Plotting helpers: gap curve + band structures
# =============================================================================
def plot_gap_vs_q(ax, gap_h5):
    """Panel (a): Eg(Γ–Z) vs Q with highlighted markers."""
    with h5py.File(gap_h5, "r") as f:
        gap_list = f["/gap_list"][()][::-1]                 # reverse as in your original
        distortion = f["/distortion"][()]
        q = (-1.0 * distortion)[::-1]                       # reversed sign + reversed order

    # Identify special indices
    min_gap_index = int(np.argmin(gap_list))
    min_abs_q_index = int(np.argmin(np.abs(q)))
    indices_q_exact = np.where(q == Q_TARGET)[0]            # exact match (float risk)

    # Main curve
    ax.plot(q, gap_list, lw=1.0, linestyle="-", color="b")
    ax.set_xlim(*GAP_XLIM)
    ax.set_ylim(*GAP_YLIM)
    ax.set_ylabel(r"$E_{G}(\Gamma-\mathrm{Z})$ (eV)")
    ax.set_xlabel(r"$Q_{\mathit{A}(18)}$ ($\mathrm{\AA}\sqrt{amu})$")

    # Ticks: major=1.0, minor=0.5; show labels only at -4,-2,0,2,4
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    def xtick_fmt(x, pos):
        return f"{int(x)}" if x in [-4, -2, 0, 2, 4] else ""
    ax.xaxis.set_major_formatter(FuncFormatter(xtick_fmt))
    ax.xaxis.set_minor_formatter(FuncFormatter(xtick_fmt))

    ax.tick_params(axis="x", which="major", length=6, width=1.0)
    ax.tick_params(axis="x", which="minor", length=2, width=1.0)

    # Align tick labels consistently
    for lab in ax.get_xticklabels():
        lab.set_verticalalignment("baseline")
        lab.set_y(-0.05)

    # Highlight points b, c, d
    # b: closest to Q=0
    ax.scatter([q[min_abs_q_index]], [gap_list[min_abs_q_index]], s=12, c="red", zorder=40)
    ax.annotate("b", (q[min_abs_q_index] - 0.25, gap_list[min_abs_q_index] + 0.05), zorder=30)

    # c: Q=2.0 (exact match)
    if len(indices_q_exact) > 0:
        ax.scatter(q[indices_q_exact], gap_list[indices_q_exact], s=12, c="red", zorder=40)
        # Label first match only (avoid repeated 'c' labels)
        i0 = int(indices_q_exact[0])
        ax.annotate("c", (q[i0] - 0.25, gap_list[i0] + 0.05), zorder=30)

    # d: minimum gap
    ax.scatter([q[min_gap_index]], [gap_list[min_gap_index]], s=12, c="red", zorder=40)
    ax.annotate("d", (q[min_gap_index] - 0.25, gap_list[min_gap_index] + 0.05), zorder=30)


def plot_bands(ax, bands_h5, show_ylabel=False):
    """Plot band structure from an HDF5 file containing /bands_list and /klist."""
    with h5py.File(bands_h5, "r") as f:
        bands = f["/bands_list"][()]
        k_list = f["/klist"][()]

    for i in range(bands.shape[0]):
        ax.plot(k_list, bands[i, :], lw=1.0, linestyle="-", color="k", zorder=20)

    ax.set_xlim(k_list[0], k_list[-1])
    ax.set_ylim(*BAND_YLIM)
    ax.set_yticks(BAND_YTICKS)
    ax.set_xticks([k_list[0], k_list[-1]], [r"$\Gamma$", r"$\mathrm{Z}$"])
    ax.axhline(0, lw=0.5, linestyle="--", color="k", zorder=25)

    if show_ylabel:
        ax.set_ylabel(r"$E-E_f$ (eV)", labelpad=-3.5)
    else:
        ax.set_yticklabels([])

    return bands, k_list


# =============================================================================
# Main
# =============================================================================
def main():
    fig = plt.figure(constrained_layout=True, figsize=(8, 4))
    axs = fig.subplots(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])

    # -------------------------------------------------------------------------
    # (a) Gap vs Q
    # -------------------------------------------------------------------------
    plot_gap_vs_q(axs[0, 0], GAP_H5)
    axs[0, 0].annotate("(a)", xy=(-0.36, 0.94), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (b) Bands at Q=0
    # -------------------------------------------------------------------------
    plot_bands(axs[1, 0], BANDS_Q0_H5, show_ylabel=True)
    axs[1, 0].annotate("(b)", xy=(-0.36, 0.94), xycoords="axes fraction")
    axs[1, 0].annotate(r"$Q=0$", xy=(0.72, 0.8), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (c) Bands at Q=2.0
    # -------------------------------------------------------------------------
    plot_bands(axs[0, 1], BANDS_Q2_H5, show_ylabel=False)
    axs[0, 1].annotate("(c)", xy=(-0.18, 0.94), xycoords="axes fraction")
    axs[0, 1].annotate(r"$Q=2.0$", xy=(0.65, 0.8), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (d) Bands at Q=2.5 + inset zoom
    # -------------------------------------------------------------------------
    bands25, k_list25 = plot_bands(axs[1, 1], BANDS_Q25_H5, show_ylabel=False)
    axs[1, 1].annotate("(d)", xy=(-0.18, 0.94), xycoords="axes fraction")
    axs[1, 1].annotate(r"$Q=2.5$", xy=(0.65, 0.8), xycoords="axes fraction")

    axins = inset_axes(axs[1, 1], width="40%", height="40%", loc="center left")
    for i in range(bands25.shape[0]):
        axins.plot(k_list25, bands25[i, :], lw=1.0, linestyle="-", color="k", zorder=20)
    axins.set_xlim(INSET_XMIN, INSET_XMAX)
    axins.set_ylim(INSET_YMIN, INSET_YMAX)
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(
        axs[1, 1], axins,
        loc1=1, loc2=4,
        fc="none", ec="k",
        linewidth=0.8, linestyle="--",
    )

    # -------------------------------------------------------------------------
    # (e,g) BZ projections at Q=2.0
    # -------------------------------------------------------------------------
    b1, b2, b3 = get_reciprocal_lattice_from_poscar(POSCAR_Q2, use_2pi=True)
    verts = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

    weyl_cart = None
    weyl_sign = None
    if os.path.exists(NODES_Q2):
        weyl_cart, weyl_sign, _ = load_weyl_points(NODES_Q2, b1, b2, b3, kz_positive_only=True)

    # (e): (kx,ky)
    plot_bz_kxky(axs[0, 2], verts, weyl_cart=weyl_cart, weyl_sign=weyl_sign)
    draw_edges_kxky(axs[0, 2], verts, EDGES, color="k", linewidth=1.0)
    axs[0, 2].annotate("(e)", xy=(-0.2, 0.94), xycoords="axes fraction")

    # (g): (k_parallel,kz)
    # For (k_parallel,kz), load all nodes (no kz>0 filter) to match your original intent
    weyl_cart_all = None
    weyl_sign_all = None
    if os.path.exists(NODES_Q2):
        weyl_cart_all, weyl_sign_all, _ = load_weyl_points(NODES_Q2, b1, b2, b3, kz_positive_only=False)

    plot_bz_kpar_kz(axs[0, 3], verts, weyl_cart=weyl_cart_all, weyl_sign=weyl_sign_all)
    draw_edges_kpar_kz(axs[0, 3], verts, EDGES, color="k", linewidth=1.0)
    # (g) label omitted in your original; add if needed:
    # axs[0, 3].annotate("(g)", xy=(-0.2, 0.94), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (f,h) BZ projections at Q=2.5
    # -------------------------------------------------------------------------
    b1, b2, b3 = get_reciprocal_lattice_from_poscar(POSCAR_Q25, use_2pi=True)
    verts25 = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

    weyl_cart = None
    weyl_sign = None
    if os.path.exists(NODES_Q25):
        weyl_cart, weyl_sign, _ = load_weyl_points(NODES_Q25, b1, b2, b3, kz_positive_only=True)

    plot_bz_kxky(axs[1, 2], verts25, weyl_cart=weyl_cart, weyl_sign=weyl_sign)
    draw_edges_kxky(axs[1, 2], verts25, EDGES, color="k", linewidth=1.0)
    axs[1, 2].annotate("(f)", xy=(-0.2, 0.94), xycoords="axes fraction")

    weyl_cart_all = None
    weyl_sign_all = None
    if os.path.exists(NODES_Q25):
        weyl_cart_all, weyl_sign_all, _ = load_weyl_points(NODES_Q25, b1, b2, b3, kz_positive_only=False)

    plot_bz_kpar_kz(axs[1, 3], verts25, weyl_cart=weyl_cart_all, weyl_sign=weyl_sign_all)
    draw_edges_kpar_kz(axs[1, 3], verts25, EDGES, color="k", linewidth=1.0)
    # axs[1, 3].annotate("(h)", xy=(-0.2, 0.94), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.savefig("FIG_2.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

