#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIG_A1 plotting script
======================

This script reproduces FIG_A1 consisting of:
  (a) 3D BZ sketch (Voronoi Wigner–Seitz) + (optional) Weyl points chirality
  (b) band structure of MODE18 (colored segments)
  (c)-(k) 3x3 grid: gap vs displacement for modes [18,30,33,40,41,44,47,50,62]

IMPORTANT USER REQUIREMENT:
- Keep panel (a) basic logic unchanged (layout/ratio/view/limits/annotation logic).
- Other parts can be cleaned/organized.

Author: (your name)
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from ase.io import read
from scipy.spatial import Voronoi
from matplotlib.ticker import NullLocator


# ============================================================
# Global plot style (your original color cycle)
# ============================================================
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
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)


# ============================================================
# 1. From POSCAR -> reciprocal lattice vectors
# ============================================================
def get_reciprocal_lattice_from_poscar(poscar_file="POSCAR", use_2pi=True):
    """
    From POSCAR read real-space lattice vectors a1,a2,a3 and compute reciprocal b1,b2,b3.
    """
    mol = read(poscar_file, format="vasp")
    cell = mol.cell  # 3x3 matrix, rows are a1,a2,a3
    a1 = np.array(cell[0], dtype=float)
    a2 = np.array(cell[1], dtype=float)
    a3 = np.array(cell[2], dtype=float)

    volume = np.dot(a1, np.cross(a2, a3))
    factor = 2.0 * np.pi if use_2pi else 1.0

    b1 = factor * np.cross(a2, a3) / volume
    b2 = factor * np.cross(a3, a1) / volume
    b3 = factor * np.cross(a1, a2) / volume

    return b1, b2, b3


# ============================================================
# 2. Voronoi + Wigner–Seitz -> BZ vertices
# ============================================================
def compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8):
    """
    Use Voronoi decomposition on reciprocal lattice points to obtain the 1st BZ vertices.

    Returns:
      unique_vertices: (N,3) array, vertices (kx,ky,kz)
    """
    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)

    pts = []
    origin_index = None

    # generate reciprocal lattice points: G = n1*b1 + n2*b2 + n3*b3
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            for n3 in range(-nmax, nmax + 1):
                g = n1 * b1 + n2 * b2 + n3 * b3
                pts.append(g)
                if n1 == 0 and n2 == 0 and n3 == 0:
                    origin_index = len(pts) - 1

    pts = np.array(pts)
    if origin_index is None:
        raise RuntimeError("Cannot find Γ point in reciprocal lattice points.")

    vor = Voronoi(pts)

    # Voronoi region associated with Γ is the 1st BZ
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]

    if -1 in region:
        raise RuntimeError(
            "Voronoi region contains -1 (infinite vertex). Increase nmax."
        )

    bz_vertices = vor.vertices[region]

    # deduplicate + sort by distance to Γ
    bz_rounded = np.round(bz_vertices, round_decimals)
    unique_vertices = np.unique(bz_rounded, axis=0)
    radii = np.linalg.norm(unique_vertices, axis=1)
    order = np.argsort(radii)
    unique_vertices = unique_vertices[order]

    return unique_vertices


# ============================================================
# 3. Draw edges on 3D BZ sketch
# ============================================================
def draw_edge(ax, verts, i, j, **kwargs):
    """
    Draw an edge between vertices Vi and Vj (1-based index).
    """
    p1 = verts[i - 1]
    p2 = verts[j - 1]
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    zs = [p1[2], p2[2]]
    ax.plot(xs, ys, zs, **kwargs, zorder=1)

def draw_edges(ax, verts, edge_list, **kwargs):
    """
    Draw a batch of edges. edge_list = [(i1,j1), (i2,j2), ...], 1-based index.
    """
    for (i, j) in edge_list:
        draw_edge(ax, verts, i, j, **kwargs)


# ============================================================
# 4. Load Weyl points with chiralities (k_frac + chirality column)
# ============================================================
def load_weyl_points_with_chiralities(nodes_file, b1, b2, b3):
    data = np.loadtxt(nodes_file, comments="#")

    # fractional coordinates in reciprocal basis (b1,b2,b3)
    k_frac = data[:, :3]
    weyl_chirality = data[:, 6]

    b1 = np.array(b1, dtype=float)
    b2 = np.array(b2, dtype=float)
    b3 = np.array(b3, dtype=float)
    B = np.vstack([b1, b2, b3])  # 3x3

    # cart = k1*b1 + k2*b2 + k3*b3
    weyl_cart = k_frac @ B

    # only keep kz > 0
    mask = weyl_cart[:, 2] > 0.0
    weyl_cart = weyl_cart[mask]
    weyl_chirality = weyl_chirality[mask]
    k_frac = k_frac[mask]

    return weyl_cart, weyl_chirality, k_frac


# ============================================================
# 5. (a) panel: BZ vertices + special k-point labels + (optional) Weyl
#     NOTE: keep the "basic logic" as your original version
# ============================================================
def get_mid_pos(V1, V2):
    x_mid = (V1[0] + V2[0]) / 2
    y_mid = (V1[1] + V2[1]) / 2
    z_mid = (V1[2] + V2[2]) / 2
    return [x_mid, y_mid, z_mid]


def plot_bz_vertices_and_weyl(ax, verts, weyl_cart=None, weyl_labels=None):
    """
    Keep your original core logic:
      - ortho projection
      - fixed x/y/z limits (your chosen numbers)
      - turn off axes + remove ticks
      - view_init(elev=12, azim=-72)
      - special labeled k points (D0, X, G0, H0, Y, C0, Z, B0, A0, L, Γ)
      - (optional) Weyl labels not shown (kept as empty strings)
    """
    # 关键：关闭透视，改用正交投影
    ax.set_proj_type('ortho')
    # 关掉网格
    ax.grid(False)

    # ---- BZ 顶点：黑色实心点 + 标签（此处你用 s=0 并且 label 为空）----
    for i, v in enumerate(verts, start=1):
        ax.scatter(
            v[0], v[1], v[2],
            c="k",
            s=0,
            edgecolor='none',
            zorder=2,
        )
        ax.text(
            v[0]*1.05,
            v[1]*1.05,
            v[2]*1.05,
            f"",
            fontsize=8,
            ha="center", va="center",
            zorder=3,
        )

    # Γ
    ax.text(
        0.05, 0, 0.05,
        f"$\Gamma$",
        fontsize=8,
        ha="center", va="center",
        zorder=3,
    )
    ax.scatter(0, 0, 0, s=10, color='gray', edgecolor='none', zorder=2)

    # ---- special points (kept as your original) ----
    d0_pos = get_mid_pos(verts[10], verts[10])
    ax.scatter(d0_pos[0], d0_pos[1], d0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(0.2*d0_pos[0], 1.4*d0_pos[1], 1.0*d0_pos[2]-0.08,
            r"$\mathrm{D}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    x_pos = get_mid_pos(verts[8], verts[10])
    ax.scatter(x_pos[0], x_pos[1], x_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.0*x_pos[0], 1.0*x_pos[1], 1.0*x_pos[2]-0.10,
            r"$\mathrm{X}$", fontsize=8, ha="center", va="center", zorder=3)

    g0_pos = get_mid_pos(verts[6], verts[6])
    ax.scatter(g0_pos[0], g0_pos[1], g0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.3*g0_pos[0], 1.1*g0_pos[1], 1.1*g0_pos[2],
            r"$\mathrm{G}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    h0_pos = get_mid_pos(verts[0], verts[0])
    ax.scatter(h0_pos[0], h0_pos[1], h0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.2*h0_pos[0], 1.0*h0_pos[1], 1.3*h0_pos[2],
            r"$\mathrm{H}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    y_pos = get_mid_pos(verts[0], verts[5])
    ax.scatter(y_pos[0], y_pos[1], y_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.2*y_pos[0], 1.0*y_pos[1], 1.0*y_pos[2],
            r"$\mathrm{Y}$", fontsize=8, ha="center", va="center", zorder=3)

    c0_pos = get_mid_pos(verts[20], verts[20])
    ax.scatter(c0_pos[0], c0_pos[1], c0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.7*c0_pos[0], 1.0*c0_pos[1], 1.0*c0_pos[2],
            r"$\mathrm{C}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    z_pos = get_mid_pos(verts[14], verts[12])
    ax.scatter(z_pos[0], z_pos[1], z_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.0*z_pos[0], 1.0*z_pos[1], 1.2*z_pos[2],
            r"$\mathrm{Z}$", fontsize=8, ha="center", va="center", zorder=3)

    b0_pos = get_mid_pos(verts[12], verts[12])
    ax.scatter(b0_pos[0], b0_pos[1], b0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.8*b0_pos[0], 1.0*b0_pos[1], 0.95*b0_pos[2],
            r"$\mathrm{B}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    a0_pos = get_mid_pos(verts[17], verts[17])
    ax.scatter(a0_pos[0], a0_pos[1], a0_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.5*a0_pos[0], 1.1*a0_pos[1], 1.1*a0_pos[2],
            r"$\mathrm{A}_{0}$", fontsize=8, ha="center", va="center", zorder=3)

    l_pos = get_mid_pos(verts[20], verts[17])
    ax.scatter(l_pos[0], l_pos[1], l_pos[2], c="k", s=10, edgecolor='none', zorder=2)
    ax.text(1.2*l_pos[0], 1.4*l_pos[1], 1.0*l_pos[2],
            r"$\mathrm{L}$", fontsize=8, ha="center", va="center", zorder=3)

    # ---- Weyl labels (your original effectively hides them) ----
    if weyl_cart is not None and len(weyl_cart) > 0:
        # colors = ['r' if lab == '+' else 'b' for lab in weyl_labels]
        # You had scatter commented out; keep it “no extra points shown” to preserve look.
        if weyl_labels is None:
            weyl_labels = ["" for _ in range(len(weyl_cart))]
        for (x, y, z), lab in zip(weyl_cart, weyl_labels):
            ax.text(
                x, y, z,
                '',
                fontsize=20,
                ha="center", va="center",
                zorder=5,
            )

    # ---- fixed axis range (keep) ----
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.4, 0.4)

    # ---- remove ticks/panes (keep) ----
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())

    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=8)
    ax.tick_params(axis='z', which='major', pad=8)

    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor((1, 1, 1, 0))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    # ---- view (keep) ----
    ax.view_init(elev=12, azim=-72)


# ============================================================
# 6. (b) panel: plot MODE18 bands with colored segments
# ============================================================
def plot_mode18_bands_panel(ax, h5_file="MODE18/save_bands_and_klist.h5"):
    """
    Panel (b): MODE18 band structure with colored k-segments.
    (This is your original logic, wrapped into a function.)
    """
    with h5py.File(h5_file, "r") as f:
        bands = f["/bands_list"][()]
        k_list = f["/klist"][()]

    # plot bands with segment colors (same slicing as your original)
    for i in range(bands.shape[0]):
        ax.plot(k_list[0:40],    bands[i, 0:40],    lw=1.0, linestyle='-', color=colors[1], zorder=20)
        ax.plot(k_list[40:120],  bands[i, 40:120],  lw=1.0, linestyle='-', color='k',       zorder=20)
        ax.plot(k_list[120:200], bands[i, 120:200], lw=1.0, linestyle='-', color=colors[2], zorder=20)
        ax.plot(k_list[200:360], bands[i, 200:360], lw=1.0, linestyle='-', color='k',       zorder=20)
        ax.plot(k_list[360:400], bands[i, 360:400], lw=1.0, linestyle='-', color=colors[3], zorder=20)

    ax.annotate("(b)", xy=(-0.2, 0.94), size=16, xycoords='axes fraction')
    ax.annotate(r"$Q=0$", xy=(0.73, 0.7), xycoords='axes fraction')
    ax.set_ylabel(r"$E-E_f$ (eV)", labelpad=-3.5)

    ax.set_xlim(k_list[0], k_list[-1])
    ax.set_ylim(-1.00, 1.00)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # ticks & labels
    tick_positions = list(range(0, 401, 40))
    k_name = [
        r'$\Gamma$',
        r'$\mathrm{Z}$',
        r'$\mathrm{B}_{0}\,|\,\mathrm{D}_{0}$',
        r'$\mathrm{X}$',
        r'$\mathrm{G}_{0}\,|\,\mathrm{H}_{0}$',
        r'$\mathrm{Y}$',
        r'$\mathrm{C}_{0}\,|\,\mathrm{A}_{0}$',
        r'$\mathrm{Z}\,|\,\mathrm{Y}$',
        r'$\Gamma$',
        r'$\mathrm{X}/\Gamma$',
        r'$\mathrm{L}$'
    ]
    ax.set_xticks(tick_positions, labels=k_name)

    for t in ax.get_xticklabels():
        t.set_va('bottom')
    ax.tick_params(axis='x', pad=15)

    # vertical separators
    npoints_per_group = 40
    for i in range(1, len(k_name) - 1):
        ax.axvline(k_list[int(i * npoints_per_group)] + 0.5, lw=0.5, linestyle='-', color='k')

    ax.axhline(0, lw=0.5, linestyle='--', color='k', zorder=25)


# ============================================================
# 7. (c)-(k) panel: 3x3 grid gap vs distortion for modes
# ============================================================
def plot_gap_grid(axs, mode_list, panel_labels, set_xlim_list, reversed_mode_list):
    """
    Plot the 3x3 gap panels (c)-(k).
    This wraps your original nested itertools loops and keeps plotting logic unchanged.
    """
    labels = ['Gap 1 (0-100)', 'Gap 2 (100-200)', 'Gap 3 (200-300)', 'Gap 4 (300-400)']
    markers = ['o', 's', '^', 'd']  # you only plot first 3 gaps -> first 3 markers

    for idx, (i1, i2) in enumerate(itertools.product(range(3), repeat=2)):
        mode = mode_list[idx]

        with h5py.File(f"MODE{mode}/combine_results_displace_vs_gap.h5", "r") as f:
            gap = f["/gap_list"][()]
            distortion = f["/distortion"][()]
            reversed_gap = gap[::-1]
            reversed_distortion = (-1 * distortion)[::-1]

        if mode in reversed_mode_list:
            x = reversed_distortion
            y = reversed_gap
        else:
            x = distortion
            y = gap

        # (you plotted i in range(3))
        for i in range(3):
            axs[i1, i2].plot(
                x, y[:, i],
                marker=markers[i],
                label=labels[i],
                linewidth=1.5,
                color=colors[i + 1]
            )

        axs[i1, i2].set_ylim(0.0, 0.15)
        axs[i1, i2].set_yticks([0.00, 0.05, 0.10, 0.15])
        axs[i1, i2].set_xlim(set_xlim_list[idx])

        axs[i1, i2].annotate(panel_labels[idx], xy=(0.02, 0.85), size=16, xycoords='axes fraction')
        axs[i1, i2].annotate(rf"$\mathit{{A}}({mode})$", xy=(0.30, 0.85), size=16, xycoords='axes fraction')

        if i1 in [2]:
            axs[i1, i2].set_xlabel(rf"$Q$ ($\mathrm{{\AA}}$ $\sqrt{{amu}}$)")

        if i2 in [0]:
            axs[i1, i2].set_ylabel('Energy Gap (eV)')
        else:
            axs[i1, i2].set_yticklabels([])

        axs[i1, i2].grid(True, linestyle='--', alpha=0.6)


# ============================================================
# 8. Main: assemble FIG_A1 layout (keep figsize and ratios)
# ============================================================
def main():
    # ---------- inputs ----------
    poscar_file = "MODE18/POSCAR"
    nodes_file = "MODE18/filtered_Nodes_with_Chiralities.dat"

    # ---------- compute BZ ----------
    b1, b2, b3 = get_reciprocal_lattice_from_poscar(poscar_file, use_2pi=True)
    verts = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

    # ---------- optional Weyl points ----------
    weyl_cart = None
    weyl_sign = None
    if os.path.exists(nodes_file):
        weyl_cart, weyl_chirality, _ = load_weyl_points_with_chiralities(nodes_file, b1, b2, b3)
        weyl_sign = np.where(weyl_chirality > 0, '+', '-')

    # ---------- figure layout (keep your original ratios) ----------
    fig = plt.figure(constrained_layout=True, figsize=(6.5, 8.5))
    main_subfigs = fig.subfigures(2, 1, height_ratios=[1, 3])

    # Top row: (a) + (b)
    top_subfigs = main_subfigs[0].subfigures(1, 2, width_ratios=[1, 2.2])

    # (a) 3D BZ
    ax3d = top_subfigs[0].add_subplot(111, projection="3d")
    plot_bz_vertices_and_weyl(
        ax3d,
        verts,
        weyl_cart=weyl_cart,
        weyl_labels=weyl_sign,
    )

    # edges exactly as your original usage (solid + dashed)
    solid_edges = [
        (23,13),(13,18),(15,18),(18,7),(7,9),(15,23),
        (7,11),(11,21),(21,1),(1,13),
        (1,19),(19,12),(12,4),(4,23),
        (24,3),(3,9),
        (11,3),(24,16),(16,6),(6,21),
        (6,19),(16,17),(17,8),(8,12),
    ]
    dash_edges = [
        (9,20),(20,5),(5,15),
        (20,2),(2,14),(14,24),
        (22,10),(10,4),(10,8),
        (22,5),(22,2),(14,17),
    ]

    draw_edges(ax3d, verts, solid_edges, linestyle='-', color="k", linewidth=1.0)
    draw_edges(ax3d, verts, dash_edges, linestyle=':', color="gray", linewidth=1.0)

    ax3d.annotate("(a)", xy=(0.10, 0.95), size=16, xycoords='axes fraction')

    # (b) band structure
    ax_band = top_subfigs[1].subplots(1, 1)
    plot_mode18_bands_panel(ax_band, "MODE18/save_bands_and_klist.h5")

    # Bottom: 3x3 grid (c)-(k)
    axs = main_subfigs[1].subplots(
        3, 3,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1]
    )

    panel_labels = ['(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']
    mode_list = [18, 30, 33, 40, 41, 44, 47, 50, 62]
    reversed_mode_list = [18, 30, 41, 47]
    set_xlim_list = [
        [2.0, 3.0],
        [5.5, 6.5],
        [3.0, 4.0],
        [6.0, 7.0],
        [3.5, 4.5],
        [3.5, 4.5],
        [4.5, 5.5],
        [3.0, 4.0],
        [6.0, 7.0],
    ]

    plot_gap_grid(
        axs,
        mode_list=mode_list,
        panel_labels=panel_labels,
        set_xlim_list=set_xlim_list,
        reversed_mode_list=reversed_mode_list
    )

    plt.savefig("FIG_A1.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

