#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIG. A3 — Element-selective decomposition of the CDW amplitude mode A(18)
and its distinct impacts on the Γ–Z gap and Weyl topology.

Panels:
(a) Ta-only distortion (image)
(b) Se-only distortion (image)
(c),(d) E_G(Γ–Z) vs Q_A(18) for Ta-only / Se-only distortions
(e) Γ–Z band structure at Ta-only Q=2.0 (with inset)
(f) BZ map (kx–ky) with Weyl-node chiralities at Ta-only Q=2.0

You renamed folders/files; paths below are updated accordingly.
Also per your request: (c) label position keeps xy=(-0.39, 0.95).
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from ase.io import read
from scipy.spatial import Voronoi


# ============================
# Global plotting style
# ============================
colors = np.asarray([
    [0, 0, 0],
    [230, 159, 0],
    [86, 180, 233],
    [0, 158, 115],
    [240, 228, 66],
    [0, 114, 178],
    [213, 94, 0],
    [204, 121, 167],
]) / 255.0
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)


# ============================
# Helper: read (distortion, gap_list) and reverse as in the original script
# ============================
def read_gap_curve(h5file):
    with h5py.File(h5file, "r") as f:
        gap_list = f["/gap_list"][()]
        gap_list = gap_list[::-1]
        distortion = f["/distortion"][()]
        reversed_distortion = (-1.0 * distortion)[::-1]
    return reversed_distortion, gap_list


# ============================
# Helper: format x tick labels (show only -4,-2,0,2,4)
# ============================
def set_custom_x_ticks(ax):
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))

    def _fmt(x, pos):
        return f"{int(x)}" if x in [-4, -2, 0, 2, 4] else ""

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_minor_formatter(FuncFormatter(_fmt))

    ax.tick_params(axis="x", which="major", length=6, width=1.0)
    ax.tick_params(axis="x", which="minor", length=2, width=1.0)

    # Force consistent baseline alignment for tick labels
    for lab in ax.get_xticklabels():
        lab.set_verticalalignment("baseline")
        lab.set_y(-0.10)


# ============================
# Brillouin zone + Weyl utilities
# ============================
def get_reciprocal_lattice_from_poscar(poscar_file="POSCAR", use_2pi=True):
    """
    Read real-space lattice vectors a1,a2,a3 from POSCAR and compute
    reciprocal lattice vectors b1,b2,b3.
    """
    mol = read(poscar_file, format="vasp")
    cell = mol.cell
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
    Compute first Brillouin-zone (Wigner–Seitz) vertices using Voronoi construction.

    Returns:
      verts: (N_vertices, 3) array, each row is a BZ vertex (kx, ky, kz)
    """
    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)

    pts = []
    origin_index = None

    # Generate a block of reciprocal lattice points: G = n1*b1 + n2*b2 + n3*b3
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            for n3 in range(-nmax, nmax + 1):
                g = n1 * b1 + n2 * b2 + n3 * b3
                pts.append(g)
                if n1 == 0 and n2 == 0 and n3 == 0:
                    origin_index = len(pts) - 1

    pts = np.array(pts)
    if origin_index is None:
        raise RuntimeError("Failed to locate the Γ point (0,0,0) in the reciprocal-lattice grid.")

    vor = Voronoi(pts)
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]

    if -1 in region:
        raise RuntimeError("Voronoi region contains -1 (point at infinity). Try increasing nmax.")

    bz_vertices = vor.vertices[region]

    # De-duplicate (numerical rounding) and sort by distance to Γ for stable indexing
    bz_rounded = np.round(bz_vertices, round_decimals)
    unique_vertices = np.unique(bz_rounded, axis=0)
    radii = np.linalg.norm(unique_vertices, axis=1)
    unique_vertices = unique_vertices[np.argsort(radii)]
    return unique_vertices


def draw_edge(ax, verts, i, j, **kwargs):
    """
    Draw a single BZ edge between vertex Vi and Vj (1-based indexing),
    using (kx, ky) projection.
    """
    p1 = verts[i - 1]
    p2 = verts[j - 1]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs, zorder=1)


def draw_edges(ax, verts, edge_list, **kwargs):
    """Draw multiple edges given a list of (i, j) pairs (1-based indices)."""
    for (i, j) in edge_list:
        draw_edge(ax, verts, i, j, **kwargs)


def load_weyl_points_with_chiralities(nodes_file, b1, b2, b3):
    """
    Load Weyl nodes from a file with columns:
      k1 k2 k3 ... ... chirality

    The first 3 columns are fractional coordinates in the (b1,b2,b3) basis,
    and the 7th column is chirality (+1 / -1).

    Returns only nodes with kz > 0 in Cartesian coordinates.
    """
    data = np.loadtxt(nodes_file, comments="#")
    k_frac = data[:, :3]
    weyl_chirality = data[:, 6]

    B = np.vstack([np.array(b1), np.array(b2), np.array(b3)])  # 3x3
    weyl_cart = k_frac @ B  # cart = k1*b1 + k2*b2 + k3*b3

    # Keep only kz > 0 nodes (as in the original workflow)
    mask = weyl_cart[:, 2] > 0.0
    return weyl_cart[mask], weyl_chirality[mask], k_frac[mask]


def plot_bz_vertices_and_weyl(ax, verts, weyl_cart=None, weyl_labels=None):
    """
    Plot the BZ outline (projected to kx-ky) and overlay Weyl nodes.

    verts      : (N,3) BZ vertices (only kx,ky are used)
    weyl_cart  : (M,3) Weyl nodes in Cartesian reciprocal coordinates
    weyl_labels: list of '+'/'-' for chirality coloring (optional)
    """
    ax.grid(False)

    # BZ vertices (hidden markers; edges define the shape)
    for v in verts:
        ax.scatter(v[0], v[1], c="k", s=0, edgecolor="none", zorder=2)

    # Weyl nodes colored by chirality
    if weyl_cart is not None and len(weyl_cart) > 0:
        weyl_xy = weyl_cart[:, :2]
        c = ["r" if lab == "+" else "b" for lab in weyl_labels] if weyl_labels is not None else "r"
        ax.scatter(
            weyl_xy[:, 0],
            weyl_xy[:, 1],
            c=c,
            s=12,
            edgecolors="white",
            linewidths=0.3,
            zorder=4,
        )

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$k_x \,(2\pi/a)$", labelpad=2)
    ax.set_ylabel(r"$k_y \,(2\pi/b)$", labelpad=2)

    # Keep tick marks but hide numeric labels
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(axis="both", which="major", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ============================
# Main
# ============================
def main():
    fig = plt.figure(constrained_layout=True, figsize=(4, 5.5))
    subfigs = fig.subfigures(2, 1, wspace=0.0, height_ratios=[4, 6])

    axstop = subfigs[0].subplots(1, 2)
    axsbottom = subfigs[1].subplots(2, 2, height_ratios=[5, 6])

    # ----------------------------
    # Updated paths after renaming
    # ----------------------------
    IMG_TA_ONLY = "TOP_VIEW_MODE18_Ta_Only.png"
    IMG_SE_ONLY = "TOP_VIEW_MODE18_Se_Only.png"

    GAP_TA_ONLY_H5 = "GAP_VS_DISP_MODE18_Ta_Only/combine_gap_list_vs_distortion_TSI.h5"
    GAP_SE_ONLY_H5 = "GAP_VS_DISP_MODE18_Se_Only/gap_list_vs_distortion_TSI.h5"

    BANDS_TA_ONLY_H5 = "BANDS_MODE18_Ta_Only/save_bands_and_klist.h5"

    WEYL_DIR = "WEYL_WITH_CHIRALITY_FOR_Q_2.0_of_MODE18_Ta_Only"
    WEYL_POSCAR = os.path.join(WEYL_DIR, "POSCAR")
    WEYL_NODES = os.path.join(WEYL_DIR, "filtered_Nodes_with_Chiralities.dat")

    # ----------------------------
    # (a) Ta-only image
    # ----------------------------
    im1 = plt.imread(IMG_TA_ONLY)
    axstop[0].imshow(im1)
    axstop[0].set_xticks([])
    axstop[0].set_yticks([])
    for sp in axstop[0].spines.values():
        sp.set_visible(False)
    axstop[0].annotate("(a)", xy=(-0.26, 1.02), xycoords="axes fraction")
    axstop[0].annotate("Ta-only", xy=(0.35, 1.02), xycoords="axes fraction")

    # ----------------------------
    # (b) Se-only image
    # ----------------------------
    im2 = plt.imread(IMG_SE_ONLY)
    axstop[1].imshow(im2)
    axstop[1].set_xticks([])
    axstop[1].set_yticks([])
    for sp in axstop[1].spines.values():
        sp.set_visible(False)
    axstop[1].annotate("(b)", xy=(-0.07, 1.02), xycoords="axes fraction")
    axstop[1].annotate("Se-only", xy=(0.35, 1.02), xycoords="axes fraction")

    # ----------------------------
    # (c) Gap vs Q : Ta-only
    # ----------------------------
    revQ1, gap1 = read_gap_curve(GAP_TA_ONLY_H5)
    min_gap_index = np.argmin(gap1)

    axsbottom[0, 0].plot(revQ1, gap1, lw=1.0, linestyle="-", color="b")
    axsbottom[0, 0].annotate("(c)", xy=(-0.39, 0.95), xycoords="axes fraction")  # keep original
    axsbottom[0, 0].set_ylabel(r"$E_{G}({\Gamma}-Z)$ (eV)")

    # Move ylabel slightly downward to match your original layout
    axsbottom[0, 0].yaxis.set_label_coords(-0.27, 0.40)

    axsbottom[0, 0].set_ylim(0, 1.0)
    axsbottom[0, 0].set_xlim(-4.0, 4.0)
    axsbottom[0, 0].set_xlabel(r"$Q_{\mathit{A}(18)}$ ($\mathrm{\AA}\sqrt{amu})$")

    set_custom_x_ticks(axsbottom[0, 0])

    # Highlight the minimum gap point (label "e")
    axsbottom[0, 0].scatter(revQ1[min_gap_index], gap1[min_gap_index], s=5, c="red", zorder=40)
    axsbottom[0, 0].annotate("e", (revQ1[min_gap_index] - 0.25, gap1[min_gap_index] + 0.05), zorder=30)

    # ----------------------------
    # (d) Gap vs Q : Se-only
    # ----------------------------
    revQ0, gap0 = read_gap_curve(GAP_SE_ONLY_H5)

    axsbottom[0, 1].plot(revQ0, gap0, lw=1.0, linestyle="-", color="b")
    axsbottom[0, 1].annotate("(d)", xy=(-0.18, 0.95), xycoords="axes fraction")

    axsbottom[0, 1].set_ylim(0, 1.0)
    axsbottom[0, 1].set_xlim(-4.0, 4.0)
    axsbottom[0, 1].set_xlabel(r"$Q_{\mathit{A}(18)}$ ($\mathrm{\AA}\sqrt{amu})$)")

    # Remove ylabel and y tick labels for the right panel (as in your original)
    axsbottom[0, 1].set_ylabel("")
    axsbottom[0, 1].tick_params(axis="y", which="both", labelleft=False)

    set_custom_x_ticks(axsbottom[0, 1])

    # ----------------------------
    # (e) Bands along Γ–Z at Ta-only Q=2.0 (with inset)
    # ----------------------------
    with h5py.File(BANDS_TA_ONLY_H5, "r") as f:
        bands = f["/bands_list"][()]
        k_list = f["/klist"][()]

    for i in range(bands.shape[0]):
        axsbottom[1, 0].plot(k_list, bands[i, :], lw=1.0, linestyle="-", color="k", zorder=20)

    axsbottom[1, 0].annotate("(e)", xy=(-0.40, 0.94), xycoords="axes fraction")
    axsbottom[1, 0].set_ylabel(r"$E-E_f$ (eV)", labelpad=-2)
    axsbottom[1, 0].set_xlim(k_list[0], k_list[-1])
    axsbottom[1, 0].set_ylim(-1.00, 1.00)
    axsbottom[1, 0].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axsbottom[1, 0].set_xticks([k_list[0], k_list[-1]], [r"$\Gamma$", r"$\mathrm{Z}$"])
    axsbottom[1, 0].annotate(r"$Q=2.0$", xy=(0.60, 0.8), xycoords="axes fraction")
    axsbottom[1, 0].axhline(0, lw=0.5, linestyle="--", color="k", zorder=25)

    # Inset (zoomed-in band window)
    #xmin, xmax = 224, 230
    xmin, xmax = 220, 240
    ymin, ymax = -0.05, -0.01
    axins = inset_axes(axsbottom[1, 0], width="40%", height="40%", loc="center left")

    for i in range(bands.shape[0]):
        axins.plot(k_list, bands[i, :], lw=1.0, linestyle="-", color="k", zorder=20)

    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(
        axsbottom[1, 0],
        axins,
        loc1=1,
        loc2=4,
        fc="none",
        ec="k",
        linewidth=0.8,
        linestyle="--",
    )

    # ----------------------------
    # (f) Weyl chirality map (kx–ky)
    # ----------------------------
    # Edge list (same as your original workflow)
    edges = [
        (20,12),(12,17),(17,13),(13,20),
        (20,1),(12,5),(17,8),(13,3),
        (1,15),(1,9),(5,23),(5,22),(8,10),(8,16),(3,24),(3,21),
        (10,22),(23,15),(9,21),(24,16),
        (10,2),(16,2),(24,6),(21,6),(9,7),(15,7),(23,4),(22,4),
        (2,19),(4,11),(7,18),(6,14),
        (19,14),(14,18),(18,11),(11,19),
        #(23, 13), (13, 18), (15, 18), (18, 7), (7, 9), (9, 20), (20, 5), (5, 15), (15, 23),
        #(7, 11), (11, 21), (21, 1), (1, 13),
        #(1, 19), (19, 12), (12, 4), (4, 23),
        #(20, 2), (2, 14), (14, 24), (24, 3), (3, 9),
        #(11, 3), (24, 16), (16, 6), (6, 21),
        #(6, 19), (16, 17), (17, 8), (8, 12),
        #(22, 10), (10, 4), (10, 8),
        #(22, 5), (22, 2), (14, 17),
    ]

    b1, b2, b3 = get_reciprocal_lattice_from_poscar(WEYL_POSCAR, use_2pi=True)
    verts = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

    weyl_cart = None
    weyl_sign = None
    if os.path.exists(WEYL_NODES):
        weyl_cart, weyl_chirality, _ = load_weyl_points_with_chiralities(WEYL_NODES, b1, b2, b3)
        weyl_sign = np.where(weyl_chirality > 0, "+", "-")
        print(f"Loaded {len(weyl_cart)} Weyl points from {WEYL_NODES}")

    plot_bz_vertices_and_weyl(axsbottom[1, 1], verts, weyl_cart=weyl_cart, weyl_labels=weyl_sign)
    axsbottom[1, 1].annotate("(f)", xy=(-0.2, 0.94), xycoords="axes fraction")
    draw_edges(axsbottom[1, 1], verts, edges, color="k", linewidth=1.0)

    plt.savefig("FIG_A3_v1.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

