#!/usr/bin/env python3
"""
Figure 4 plotting script.

This figure summarizes the IR-active mode B3(7) and its coupling to the Raman CDW amplitude mode A(18).

Layout
------
A 2×2 subfigure grid:

  Top-left  (a): B3(7) eigenvector (top view)  - PNG image
  Top-right (b): B3(7) eigenvector (side view) - PNG image
  Bottom-left (c): Coupled potential-energy curves vs Q_A(18) for several fixed Q_B3(7)
  Bottom-right (d): Schematic of Ta-chain tetramerization patterns under B3(7) and A(18),
                    with arrows indicating distortions and stars marking short/long bonds.

Inputs
------
Images:
  35x35x72ppi_MODE07_TOP.png
  35x40x72ppi_MODE07_SIDE.png

Coupled energy surface (HDF5):
  q_vs_total_energy_list_couple_07_18.h5
    /q_list               : 1D array, Q_A(18)
    /total_energy_list    : 2D array, E(Q_B3(7), Q_A(18))

Output
------
FIG_4.png (dpi=600)

Notes
-----
- The star markers are intentionally drawn with a black outline (edgecolor='black').
  If you want no outline, set edgecolor='none' in `create_five_pointed_star()`.
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches


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
FIGSIZE = (4, 4)
OUTPUT_FIG = "FIG_4.png"
DPI = 600

IMG_TOP = "35x35x72ppi_MODE07_TOP.png"
IMG_SIDE = "35x40x72ppi_MODE07_SIDE.png"

H5_COUPLING = "q_vs_total_energy_list_couple_07_18.h5"
H5_Q_PATH = "/q_list"
H5_E_PATH = "/total_energy_list"

# Energy-curve styling (use only indices 2..4 as in your original script)
CURVE_IDXS = [2, 3, 4]
CURVE_LABELS = [
    r"$Q_{\mathit{B}_{3}(7)}=0.0$",
    r"$Q_{\mathit{B}_{3}(7)}=\pm 1.0$",
    r"$Q_{\mathit{B}_{3}(7)}=\pm 2.0$",
]
CURVE_COLORS = ["C0", "C1", "C2"]
CURVE_STYLES = ["--", "--", "--"]
CURVE_LW = 1.5

# Panel (c) axis settings
C_XLABEL = r"$Q_{\mathit{A}(18)}$ ($\mathrm{\AA}\sqrt{\mathrm{amu}}$)"
C_YLABEL = "Potential Energy\n(meV/atom)"
C_YLIM = (-0.50, 15.5)
C_YTICKS = [0.0, 5.0, 10.0, 15.0]

# Schematic geometry for panel (d)
CIRCLE_R = 0.3
ARROW_FC = "blue"
ARROW_EC = "blue"

STAR_SIZE = 0.2
STAR_SHORT_COLOR = "red"
STAR_LONG_COLOR = "green"
STAR_EDGE = "black"  # outline color for stars (set to 'none' for no outline)


# =============================================================================
# Helper functions
# =============================================================================
def hide_axes_for_image(ax):
    """Remove ticks and spines for image panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)


def draw_circles(ax, positions, color, radius, zorder=2, alpha=1.0):
    """Draw filled circles at (x,y) positions."""
    for (x, y) in positions:
        circ = plt.Circle(
            (x, y),
            radius,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_artist(circ)


def create_five_pointed_star(center, size):
    """
    Return a 5-point star polygon (as Nx2 points).
    A 10-vertex star is built by alternating outer/inner radii.
    """
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]  # 10 vertices
    angles += np.pi / 2.0  # rotate so one point is upward
    pts = np.vstack((np.cos(angles), np.sin(angles))).T
    pts[1::2] *= 0.5  # inner points
    return center + size * pts


def draw_star(ax, center, size, facecolor, edgecolor, zorder=4):
    """Draw a star marker at a given center."""
    pts = create_five_pointed_star(np.array(center, dtype=float), size)
    poly = patches.Polygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor='none',
        linewidth=0.8 if edgecolor != "none" else 0.0,
        zorder=zorder,
    )
    ax.add_patch(poly)


def draw_stars(ax, positions, size, facecolor, edgecolor, zorder=4):
    """Draw stars at a list of (x,y) positions."""
    for (x, y) in positions:
        draw_star(ax, (x, y), size, facecolor, edgecolor, zorder=zorder)


# =============================================================================
# Main plotting
# =============================================================================
def main():
    fig = plt.figure(constrained_layout=True, figsize=FIGSIZE)
    subfigs = fig.subfigures(
        2, 2,
        wspace=0.0,
        width_ratios=[2, 2],
        height_ratios=[4, 4.5],
    )

    ax_img_top = subfigs[0, 0].subplots(1, 1)
    ax_img_side = subfigs[0, 1].subplots(1, 1)
    ax_energy = subfigs[1, 0].subplots(1, 1)
    ax_scheme = subfigs[1, 1].subplots(1, 1)

    # -------------------------------------------------------------------------
    # (a) B3(7) top view image
    # -------------------------------------------------------------------------
    im1 = plt.imread(IMG_TOP)
    ax_img_top.imshow(im1)
    hide_axes_for_image(ax_img_top)
    ax_img_top.annotate("(a)", xy=(-0.06, 0.94), xycoords="axes fraction")
    ax_img_top.annotate(r"$\mathit{B}_{3}(7)$", xy=(0.16, 0.94), xycoords="axes fraction")
    ax_img_top.annotate("1.14 THz", xy=(0.58, 0.94), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (b) B3(7) side view image
    # -------------------------------------------------------------------------
    im2 = plt.imread(IMG_SIDE)
    ax_img_side.imshow(im2)
    hide_axes_for_image(ax_img_side)
    ax_img_side.annotate("(b)", xy=(-0.10, 0.94), xycoords="axes fraction")

    # -------------------------------------------------------------------------
    # (c) Coupled energy curves: E(Q_B3(7), Q_A(18)) vs Q_A(18)
    # -------------------------------------------------------------------------
    if not os.path.exists(H5_COUPLING):
        raise FileNotFoundError(f"Cannot find coupling file: {H5_COUPLING}")

    with h5py.File(H5_COUPLING, "r") as f:
        q_list = f[H5_Q_PATH][()]
        total_energy = f[H5_E_PATH][()]

    # Your original script reverses the second axis (Q_A direction) for plotting
    total_energy = total_energy[:, ::-1]

    # Plot only selected curves (indices 2..4)
    for j, idx in enumerate(CURVE_IDXS):
        ax_energy.plot(
            q_list,
            total_energy[idx],
            linestyle=CURVE_STYLES[j],
            lw=CURVE_LW,
            color=CURVE_COLORS[j],
            label=CURVE_LABELS[j],
        )

    ax_energy.annotate("(c)", xy=(-0.42, 1.00), xycoords="axes fraction")
    ax_energy.set_xlabel(C_XLABEL)
    ax_energy.set_ylabel(C_YLABEL, labelpad=-0.5)
    ax_energy.set_xlim(q_list[0], q_list[-1])
    ax_energy.set_ylim(*C_YLIM)
    ax_energy.set_yticks(C_YTICKS)
    ax_energy.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------------------
    # (d) Schematic: tetramerization pattern and IR/Raman-induced distortions
    # -------------------------------------------------------------------------
    # Chain sites (CDW) on the left block (x ~ 1..2)
    cdw_positions = [
        (1, 1), (1, 1.8), (1, 3), (1, 4.2),
        (2, 1), (2, 2.2), (2, 3), (2, 3.8),
    ]
    cdw_positions_even = [(1, 1.8), (1, 4.2), (2, 2.2), (2, 3.8)]
    cdw_positions_odd = [(1, 1), (1, 3), (2, 1), (2, 3)]

    # Raman-driven configuration on the right block (x ~ 4..5)
    raman_positions = [
        (4, 1), (4, 1.8), (4, 3), (4, 4.2),
        (5, 1), (5, 2.2), (5, 3), (5, 3.8),
    ]

    # Circles: use lower zorder/alpha for "even" sites (matches your original intent)
    draw_circles(ax_scheme, cdw_positions_odd, color="orange", radius=CIRCLE_R, zorder=2, alpha=1.0)
    draw_circles(ax_scheme, cdw_positions_even, color="orange", radius=CIRCLE_R, zorder=0.5, alpha=1.0)
    draw_circles(ax_scheme, raman_positions, color="orange", radius=CIRCLE_R, zorder=2, alpha=1.0)

    # Horizontal dashed guides
    for y in range(1, 5):
        ax_scheme.axhline(y=y, color="black", linestyle="--", linewidth=1.5, zorder=0)

    # IR arrows on CDW (left block)
    cdw_ir_arrows = [
        (cdw_positions[2][0], cdw_positions[2][1], 0, -0.5),
        (cdw_positions[0][0], cdw_positions[0][1], 0, +0.5),
        (cdw_positions[4][0], cdw_positions[4][1], 0, -0.5),
        (cdw_positions[6][0], cdw_positions[6][1], 0, +0.5),
    ]

    # Raman arrows on the right block
    raman_arrows = [
        (raman_positions[3][0], raman_positions[3][1], 0, -0.5),
        (raman_positions[1][0], raman_positions[1][1], 0, +0.5),
        (raman_positions[5][0], raman_positions[5][1], 0, -0.5),
        (raman_positions[7][0], raman_positions[7][1], 0, +0.5),
    ]

    for (x, y, dx, dy) in cdw_ir_arrows + raman_arrows:
        ax_scheme.arrow(
            x, y, dx, dy,
            head_width=0.3,
            head_length=0.2,
            width=0.1,
            fc=ARROW_FC,
            ec=ARROW_EC,
            zorder=1,
        )

    # Stars: short/long bonds (left block)
    short_star_cdw = [cdw_positions[0], cdw_positions[6]]
    long_star_cdw = [cdw_positions[2], cdw_positions[4]]

    # Stars: short/long bonds (right block)
    short_star_raman = [raman_positions[0], raman_positions[6]]
    long_star_raman = [raman_positions[2], raman_positions[4]]

    draw_stars(ax_scheme, short_star_cdw, STAR_SIZE, STAR_SHORT_COLOR, STAR_EDGE, zorder=4)
    draw_stars(ax_scheme, long_star_cdw, STAR_SIZE, STAR_LONG_COLOR, STAR_EDGE, zorder=4)
    draw_stars(ax_scheme, short_star_raman, STAR_SIZE, STAR_SHORT_COLOR, STAR_EDGE, zorder=4)
    draw_stars(ax_scheme, long_star_raman, STAR_SIZE, STAR_LONG_COLOR, STAR_EDGE, zorder=4)

    # Labels
    ax_scheme.text(1.0, 5.0, "I", ha="center", va="top")
    ax_scheme.text(2.0, 5.0, "II", ha="center", va="top")
    ax_scheme.text(4.0, 5.0, "I", ha="center", va="top")
    ax_scheme.text(5.0, 5.0, "II", ha="center", va="top")

    ax_scheme.text(1.5, -0.0, r"$\mathit{B}_{3}(7)$", ha="center", va="top")
    ax_scheme.text(4.5, -0.0, r"$\mathit{A}(18)$", ha="center", va="top")

    ax_scheme.set_xlim(0, 6)
    ax_scheme.set_ylim(-0.5, 5.0)
    ax_scheme.set_aspect("equal")
    ax_scheme.annotate("(d)", xy=(-0.10, 1.04), xycoords="axes fraction")
    ax_scheme.axis("off")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.savefig(OUTPUT_FIG, dpi=DPI)
    plt.show()


if __name__ == "__main__":
    main()

