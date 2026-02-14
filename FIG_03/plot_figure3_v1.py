#!/usr/bin/env python3
"""
Figure 3 plotting script.

This script generates a multi-panel figure (Fig. 3) for eight Raman-active A modes:
A(30), A(33), A(40), A(41), A(44), A(47), A(50), A(62).

Layout
------
Top subfigure (2 rows × 8 cols):
  - Row 1: top-view eigenvector visualizations (pre-rendered PNGs)
  - Row 2: side-view eigenvector visualizations (pre-rendered PNGs)

Bottom subfigure (3 rows × 8 cols):
  - Row 1: direct gap Eg(Γ–Z) vs phonon coordinate Q for each mode
  - Row 2: band structure at a selected configuration (gap-close dataset), with an inset zoom
  - Row 3: (kx, ky) projection of the BZ with Weyl nodes (colored by chirality)

Inputs
------
Images:
  TOP_VIEWS/35x35x72ppi_MODE{mode}_TOP.png  (except mode 62 uses SHORT_ prefix)
  SIDE_VIEWS/35x45x72ppi_MODE{mode}_SIDE.png (except mode 62 uses SHORT_ prefix)

Gap curves:
  GAP_VS_DISP_Se_MODES/MODE{mode}/combine_gap_list_vs_distortion_TSI.h5

Bands (gap-close configurations):
  BANDS_WHEN_GAP_CLOSE_Se_MODES/MODE{mode}/save_bands_and_klist.h5
  (optionally cached in BANDS_WHEN_GAP_CLOSE_Se_MODES/save_all_k_and_bands_list_for_pc.h5)

Weyl nodes:
  WEYL_NODES_WITH_CHIRALITY_Se_MODES/MODE{mode}/POSCAR
  WEYL_NODES_WITH_CHIRALITY_Se_MODES/MODE{mode}/filtered_Nodes_with_Chiralities.dat

Output
------
FIG_3.png (dpi=600)

Notes
-----
- Weyl node file is assumed to contain:
  columns 1–3: fractional coordinates in the reciprocal basis (b1,b2,b3),
  column 7: chirality (+1 or -1).
- In the BZ plot (kx,ky), only kz>0 Weyl nodes are kept to avoid duplicates.
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

from ase.io import read
from scipy.spatial import Voronoi
from matplotlib.ticker import NullLocator
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
# User-editable configuration
# =============================================================================
MODE_LIST = [30, 33, 40, 41, 44, 47, 50, 62]
TOP_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

# Image naming rule: mode 62 uses "SHORT_" prefix in your original script
TOP_IMG_TEMPLATE = "TOP_VIEWS/{prefix}35x35x72ppi_MODE{mode}_TOP.png"
SIDE_IMG_TEMPLATE = "SIDE_VIEWS/{prefix}35x45x72ppi_MODE{mode}_SIDE.png"
SHORT_PREFIX_MODES = {62}  # modes that use "SHORT_" in file names

# Gap data
GAP_H5_TEMPLATE = "GAP_VS_DISP_Se_MODES/MODE{mode}/combine_gap_list_vs_distortion_TSI.h5"
GAP_CACHE_NPZ = "GAP_VS_DISP_Se_MODES/save_all_q_and_gap_list.npz"
SPECIAL_REVERSE_MODES = {30, 41, 47}  # modes that are reversed in your original script

# Bands data
BANDS_H5_TEMPLATE = "BANDS_WHEN_GAP_CLOSE_Se_MODES/MODE{mode}/save_bands_and_klist.h5"
BANDS_CACHE_H5 = "BANDS_WHEN_GAP_CLOSE_Se_MODES/save_all_k_and_bands_list_for_pc.h5"

# Inset windows (per mode) for bands
XMIN_XMAX_LIST = [
    [233, 247], [222, 238], [216, 233], [228, 244],
    [230, 245], [222, 238], [222, 238], [242, 258]
]
YMIN_YMAX_LIST = [
    [-0.050, -0.030], [-0.025, -0.005], [-0.050, -0.030], [-0.055, -0.035],
    [-0.040, -0.020], [-0.030, -0.010], [-0.040, -0.020], [-0.050, -0.030]
]

# Weyl data
POSCAR_TEMPLATE = "WEYL_NODES_WITH_CHIRALITY_Se_MODES/MODE{mode}/POSCAR"
NODES_TEMPLATE = "WEYL_NODES_WITH_CHIRALITY_Se_MODES/MODE{mode}/filtered_Nodes_with_Chiralities.dat"

# Manual BZ edges (based on your Voronoi vertex indexing)
EDGES = [
    (23, 13), (13, 18), (15, 18), (18, 7), (7, 9), (9, 20), (20, 5), (5, 15), (15, 23),
    (7, 11), (11, 21), (21, 1), (1, 13),
    (1, 19), (19, 12), (12, 4), (4, 23),
    (20, 2), (2, 14), (14, 24), (24, 3), (3, 9),
    (11, 3), (24, 16), (16, 6), (6, 21),
    (6, 19), (16, 17), (17, 8), (8, 12),
    (22, 10), (10, 4), (10, 8),
    (22, 5), (22, 2), (14, 17),
]

# BZ plot cosmetics (fixed range as in your original script)
BZ_XLIM = (-0.55, 0.55)
BZ_YLIM = (-0.55, 0.55)
BZ_TICKS = [-0.5, 0.0, 0.5]


# =============================================================================
# Helper utilities
# =============================================================================
def hide_spines_ticks(ax):
    """Hide ticks and spines for image panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)


def get_img_prefix(mode):
    """Return filename prefix for a given mode (matches your original naming convention)."""
    return "SHORT_" if mode in SHORT_PREFIX_MODES else ""


def get_reciprocal_lattice_from_poscar(poscar_file, use_2pi=True):
    """
    Compute reciprocal lattice vectors b1,b2,b3 from a VASP POSCAR.

    If use_2pi=True, vectors include the 2π factor:
      b1 = 2π * (a2 × a3) / V, etc.
    """
    atoms = read(poscar_file, format="vasp")
    cell = atoms.cell  # rows are a1,a2,a3
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
    verts : (N,3) array of BZ vertices in cartesian reciprocal coordinates.
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
        raise RuntimeError("Failed to locate Γ point (0,0,0) in the reciprocal lattice grid.")

    vor = Voronoi(pts)
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]
    if -1 in region:
        raise RuntimeError(
            "Voronoi region contains a point at infinity (-1). "
            "Increase nmax for a more complete reciprocal lattice grid."
        )

    bz_vertices = vor.vertices[region]
    bz_rounded = np.round(bz_vertices, round_decimals)
    unique_vertices = np.unique(bz_rounded, axis=0)

    # Sort by distance to Γ for reproducible vertex ordering (cosmetic)
    radii = np.linalg.norm(unique_vertices, axis=1)
    unique_vertices = unique_vertices[np.argsort(radii)]
    return unique_vertices


def draw_edges_kxky(ax, verts, edge_list, **kwargs):
    """Draw BZ edges in the (kx, ky) projection."""
    for i, j in edge_list:
        p1 = verts[i - 1]
        p2 = verts[j - 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs, zorder=1)


def load_weyl_points_with_chiralities(nodes_file, b1, b2, b3, kz_positive_only=True):
    """
    Load Weyl nodes and their chiralities from a text file.

    Expected columns:
      - first 3 columns: fractional coordinates in reciprocal basis (b1,b2,b3)
      - 7th column: chirality (+1 or -1)

    Returns
    -------
    weyl_cart : (M,3) cartesian reciprocal coordinates
    weyl_sign : (M,) array of '+'/'-'
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

    weyl_sign = np.where(chirality > 0, "+", "-")
    return weyl_cart, weyl_sign


def plot_bz_kxky(ax, verts, weyl_cart=None, weyl_sign=None):
    """
    Plot (kx, ky) projection of the BZ and optional Weyl nodes.
    """
    ax.grid(False)

    # BZ vertices (hidden markers; kept for consistent autoscaling logic if needed)
    for v in verts:
        ax.scatter(v[0], v[1], c="k", s=0, edgecolor="none", zorder=2)

    # Weyl nodes
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

    # Fixed range and minimalist ticks
    ax.set_xlim(*BZ_XLIM)
    ax.set_ylim(*BZ_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(BZ_TICKS)
    ax.set_yticks(BZ_TICKS)
    ax.set_xticklabels([])
    # (keep y tick positions; labels may be selectively removed by caller)

    ax.tick_params(axis="both", which="major", pad=2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Data loading: gap curves and bands (with caching)
# =============================================================================
def load_all_gap_curves(mode_list):
    """
    Load gap(Q) curves for all modes.
    Since each mode may have a different number of Q points, we store arrays as dtype=object.

    Returns
    -------
    total_q_list   : list of 1D arrays
    total_gap_list : list of 1D arrays
    """
    if os.path.isfile(GAP_CACHE_NPZ):
        data = np.load(GAP_CACHE_NPZ, allow_pickle=True)
        return data["total_q_list"].tolist(), data["total_gap_list"].tolist()

    total_q_list = []
    total_gap_list = []
    for mode in mode_list:
        h5_path = GAP_H5_TEMPLATE.format(mode=mode)
        with h5py.File(h5_path, "r") as f:
            gap_list = f["/gap_list"][()]
            distortion = f["/distortion"][()]

        # Your original script uses a different sign/order for a subset of modes
        if mode in SPECIAL_REVERSE_MODES:
            gap_list = gap_list[::-1]
            q = (-1.0 * distortion)[::-1]
        else:
            q = distortion

        total_q_list.append(q)
        total_gap_list.append(gap_list)

    np.savez(
        GAP_CACHE_NPZ,
        total_q_list=np.array(total_q_list, dtype=object),
        total_gap_list=np.array(total_gap_list, dtype=object),
    )
    return total_q_list, total_gap_list


def load_all_bands(mode_list):
    """
    Load band structures for all modes at the selected (gap-close) configuration.

    Returns
    -------
    all_bands  : ndarray, shape (n_modes, n_k, n_bands) after swapaxes
    all_k_list : ndarray, shape (n_modes, n_k)
    """
    if os.path.isfile(BANDS_CACHE_H5):
        with h5py.File(BANDS_CACHE_H5, "r") as f:
            all_bands = f["/all_bands"][()]
            all_k_list = f["/all_k_list"][()]
    else:
        all_bands = []
        all_k_list = []
        for mode in mode_list:
            h5_path = BANDS_H5_TEMPLATE.format(mode=mode)
            with h5py.File(h5_path, "r") as f:
                bands = f["/bands_list"][()]
                k_list = f["/klist"][()]
            all_bands.append(bands)
            all_k_list.append(k_list)

        all_bands = np.array(all_bands)
        all_k_list = np.array(all_k_list)

        with h5py.File(BANDS_CACHE_H5, "w") as f:
            f["/all_bands"] = all_bands
            f["/all_k_list"] = all_k_list

    # Make shape consistent with your plotting call:
    # original: all_bands = np.swapaxes(all_bands, 1, 2)
    all_bands = np.swapaxes(all_bands, 1, 2)  # (n_modes, n_k, n_bands)
    return all_bands, all_k_list


# =============================================================================
# Main plotting
# =============================================================================
def main():
    # Figure canvas and subfigure layout
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    subfigs = fig.subfigures(2, 1, height_ratios=[8, 12])

    axs1 = subfigs[0].subplots(
        2, 8,
        height_ratios=[3.5, 4.5],
        width_ratios=[3.5] * 8
    )
    axs2 = subfigs[1].subplots(
        3, 8,
        height_ratios=[4, 4, 4],
        width_ratios=[3.5] * 8
    )

    # -------------------------------------------------------------------------
    # Top subfigure: eigenvector images (top/side views)
    # -------------------------------------------------------------------------
    for col, mode in enumerate(MODE_LIST):
        prefix = get_img_prefix(mode)

        # Top views
        top_path = TOP_IMG_TEMPLATE.format(prefix=prefix, mode=mode)
        im_top = plt.imread(top_path)
        axs1[0, col].imshow(im_top)
        hide_spines_ticks(axs1[0, col])

        # Panel label + mode label
        axs1[0, col].annotate(TOP_LABELS[col], xy=(0.00, 1.02), size=16, xycoords="axes fraction")
        axs1[0, col].annotate(rf"$\mathit{{A}}({mode})$", xy=(0.30, 1.02), size=16, xycoords="axes fraction")

        # Side views
        side_path = SIDE_IMG_TEMPLATE.format(prefix=prefix, mode=mode)
        im_side = plt.imread(side_path)
        axs1[1, col].imshow(im_side)
        hide_spines_ticks(axs1[1, col])

    # Dummy ylabel (you used white "hello" to enforce spacing)
    axs1[1, 0].set_ylabel(" ", color="white", labelpad=24)

    # -------------------------------------------------------------------------
    # Bottom subfigure row 1: Eg(Γ–Z) vs Q for each mode
    # -------------------------------------------------------------------------
    total_q_list, total_gap_list = load_all_gap_curves(MODE_LIST)

    for col, mode in enumerate(MODE_LIST):
        axs2[0, col].plot(total_q_list[col], total_gap_list[col], lw=1.0, linestyle="-", color="b")
        axs2[0, col].set_xlim(total_q_list[col][0], total_q_list[col][-1])
        axs2[0, col].set_ylim(0.0, 1.0)
        axs2[0, col].set_xticks([-10, -5, 0, 5, 10])
        axs2[0, col].set_xlabel(rf"Q$_{{A({mode})}}$ ($\mathrm{{\AA}}\sqrt{{\mathrm{{amu}}}}$)")

        if col != 0:
            axs2[0, col].set_yticklabels([])

    axs2[0, 0].set_ylabel(r"$E_{G}(\Gamma-\mathrm{Z})$ (eV)")

    # -------------------------------------------------------------------------
    # Bottom subfigure row 2: band structures + inset zoom
    # -------------------------------------------------------------------------
    all_bands, all_k_list = load_all_bands(MODE_LIST)  # bands shape: (n_modes, n_k, n_bands)

    for col, mode in enumerate(MODE_LIST):
        # Plot all bands at once: y is (n_k, n_bands)
        axs2[1, col].plot(all_k_list[col], all_bands[col], lw=1.0, linestyle="-", color="k", zorder=20)
        axs2[1, col].set_xlim(all_k_list[col][0], all_k_list[col][-1])
        axs2[1, col].set_ylim(-1.0, 1.0)
        axs2[1, col].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axs2[1, col].set_xticks([all_k_list[col][0], all_k_list[col][-1]])
        axs2[1, col].set_xticklabels([r"$\Gamma$", "Z"])
        axs2[1, col].axhline(0, lw=0.5, linestyle="--", color="k", zorder=25)

        if col != 0:
            axs2[1, col].set_yticklabels([])

        # Inset zoom window (mode-dependent)
        xmin, xmax = XMIN_XMAX_LIST[col]
        ymin, ymax = YMIN_YMAX_LIST[col]

        axins = inset_axes(axs2[1, col], width="40%", height="40%", loc="center left")
        axins.plot(all_k_list[col], all_bands[col], lw=1.0, linestyle="-", color="k", zorder=20)
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(ymin, ymax)
        axins.set_xticks([])
        axins.set_yticks([])

        mark_inset(
            axs2[1, col], axins,
            loc1=1, loc2=4,
            fc="none", ec="k",
            linewidth=0.8, linestyle="--",
        )

    axs2[1, 0].set_ylabel(r"$E-E_f$ (eV)", labelpad=-1.5)

    # -------------------------------------------------------------------------
    # Bottom subfigure row 3: BZ (kx,ky) + Weyl nodes for each mode
    # -------------------------------------------------------------------------
    for col, mode in enumerate(MODE_LIST):
        poscar_file = POSCAR_TEMPLATE.format(mode=mode)
        nodes_file = NODES_TEMPLATE.format(mode=mode)

        b1, b2, b3 = get_reciprocal_lattice_from_poscar(poscar_file, use_2pi=True)
        verts = compute_bz_vertices(b1, b2, b3, nmax=2, round_decimals=8)

        weyl_cart = None
        weyl_sign = None
        if os.path.exists(nodes_file):
            weyl_cart, weyl_sign = load_weyl_points_with_chiralities(
                nodes_file, b1, b2, b3, kz_positive_only=True
            )

        plot_bz_kxky(axs2[2, col], verts, weyl_cart=weyl_cart, weyl_sign=weyl_sign)
        draw_edges_kxky(axs2[2, col], verts, EDGES, color="k", linewidth=1.0)

        # Axis labels (match your original: label x for every panel; y only for the first)
        axs2[2, col].set_xlabel(r"$k_x \,(2\pi/a)$", labelpad=-0.5)

        if col != 0:
            axs2[2, col].set_yticklabels([])

    axs2[2, 0].set_ylabel(r"$k_y \,(2\pi/b)$", labelpad=-0.5)
    axs2[2, 0].set_yticks(BZ_TICKS)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.savefig("FIG_3.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

