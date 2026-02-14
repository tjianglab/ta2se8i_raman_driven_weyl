#!/usr/bin/env python3
"""
Figure 1 plotting script.

This script generates Fig. 1 (panels a–c) used in the manuscript:
(a) Top-view eigenvector visualization of the CDW amplitude mode A(18).
(b) Side-view eigenvector visualization with chain/atom labels.
(c) Schematic of Ta-chain tetramerization patterns and the A(18)-induced modulation.

Inputs
------
- refine_v1_35x40x72ppi_MODE18_TOP.png
- refine_v1_35x45x72ppi_MODE18_SIDE.png

Output
------
- FIG_1.png (dpi=600)

Notes
-----
- The top row (a,b) displays pre-rendered images (e.g., from VESTA/povray/etc.).
- The bottom row (c) is drawn programmatically with circles (atoms/sites), arrows (phonon distortion),
  and stars (short/long Ta–Ta bonds) as a schematic.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches


# =============================================================================
# Global plotting style
# =============================================================================
# Custom color cycle (Okabe–Ito-like palette + black), normalized to [0,1]
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
# Helper functions for the schematic (panel c)
# =============================================================================
def draw_circles(ax, positions, color="orange", radius=0.3, zorder=2):
    """Draw filled circles at (x,y) positions."""
    for x, y in positions:
        ax.add_artist(plt.Circle((x, y), radius, edgecolor=color, facecolor=color, zorder=zorder))


def create_five_pointed_star(ax, center, size=0.2, facecolor="red", edgecolor="black", zorder=4):
    """
    Draw a five-pointed star polygon.

    Implementation detail:
    - Use 10 vertices (outer/inner alternating) to form a 5-point star.
    - Rotate so one point is facing upward.
    """
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]  # 10 points
    angles += np.pi / 2                          # rotate: top point upwards
    pts = np.vstack((np.cos(angles), np.sin(angles))).T
    pts[1::2] *= 0.5                             # inner points scaled down
    pts = center + size * pts

    star = patches.Polygon(pts, closed=True, facecolor=facecolor, edgecolor='none', zorder=zorder)
    ax.add_patch(star)


def draw_stars(ax, positions, size=0.2, color="red", zorder=4):
    """Place five-point stars at given (x,y) positions."""
    for x, y in positions:
        create_five_pointed_star(ax, center=np.array([x, y]), size=size, facecolor=color, zorder=zorder)


def add_axis_triad_labels(ax, b_xy, c_xy, a_xy, fontsize=8):
    """Annotate crystallographic axis arrows (b, c, a) as text labels on an image panel."""
    ax.annotate(r"$\overrightarrow{\mathbf{b}}$", xy=b_xy, xycoords="axes fraction",
                ha="center", va="center", fontsize=fontsize, clip_on=False)
    ax.annotate(r"$\overrightarrow{\mathbf{c}}$", xy=c_xy, xycoords="axes fraction",
                ha="center", va="center", fontsize=fontsize, clip_on=False)
    ax.annotate(r"$\overrightarrow{\mathbf{a}}$", xy=a_xy, xycoords="axes fraction",
                ha="center", va="center", fontsize=fontsize, clip_on=False)


def hide_spines_ticks(ax):
    """Remove ticks and spines (useful for image panels)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)


# =============================================================================
# Main figure layout
# =============================================================================
def main():
    # --- Figure canvas ---
    fig = plt.figure(constrained_layout=True, figsize=(4, 4.5))

    # Two stacked subfigures: top for (a,b), bottom for (c)
    subfigs = fig.subfigures(2, 1, wspace=0.0, height_ratios=[2.5, 2.0])

    # (a,b) -> two columns; (c) -> single axis
    axtop = subfigs[0].subplots(1, 2, width_ratios=[1.0, 1.1])
    axbottom = subfigs[1].subplots(1, 1)

    # =============================================================================
    # Panel (a): top-view mode visualization (image)
    # =============================================================================
    im1 = plt.imread("refine_v1_35x40x72ppi_MODE18_TOP.png")
    axtop[0].imshow(im1)
    hide_spines_ticks(axtop[0])

    # Panel tag + descriptive labels
    axtop[0].annotate("(a)", xy=(-0.05, 1.05), xycoords="axes fraction")
    axtop[0].annotate("CDW amplitude mode", xy=(0.10, 1.05), xycoords="axes fraction")
    axtop[0].annotate(r"$\mathit{A}(18)$", xy=(0.15, 0.95), xycoords="axes fraction")
    axtop[0].annotate("2.51 THz", xy=(0.60, 0.95), xycoords="axes fraction")

    # Element legend text (assumes your image already has colored arrows)
    axtop[0].annotate(r"Ta", xy=(0.57, -0.02), fontsize=8, xycoords="axes fraction")
    axtop[0].annotate(r"Se", xy=(0.70, -0.02), fontsize=8, xycoords="axes fraction")
    axtop[0].annotate(r"I",  xy=(0.85, -0.02), fontsize=8, xycoords="axes fraction")

    # Crystallographic axes labels
    add_axis_triad_labels(
        axtop[0],
        b_xy=(0.10, 0.05),
        c_xy=(0.25, 0.10),
        a_xy=(0.40, 0.05),
        fontsize=8,
    )

    # =============================================================================
    # Panel (b): side-view mode visualization (image)
    # =============================================================================
    im2 = plt.imread("refine_v1_35x45x72ppi_MODE18_SIDE.png")
    axtop[1].imshow(im2)
    hide_spines_ticks(axtop[1])

    axtop[1].annotate("(b)", xy=(-0.00, 1.00), xycoords="axes fraction")

    # Chain labels
    axtop[1].annotate("Chain I",  xy=(0.15, 1.01), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Chain II", xy=(0.60, 1.01), fontsize=8, xycoords="axes fraction")

    # Atom labels (as placed in your original script)
    axtop[1].annotate("Ta$_\\mathrm{I}$(1)", xy=(0.32, 0.85), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{I}$(2)", xy=(0.32, 0.68), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{I}$(3)", xy=(0.32, 0.51), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{I}$(4)", xy=(0.32, 0.34), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate(r"Ta$_{\mathrm{I}}$(1$^{\prime}$)", xy=(0.33, 0.19),
                      fontsize=8, xycoords="axes fraction")

    axtop[1].annotate("Ta$_\\mathrm{II}$(1)", xy=(0.78, 0.85), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{II}$(2)", xy=(0.78, 0.68), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{II}$(3)", xy=(0.78, 0.51), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate("Ta$_\\mathrm{II}$(4)", xy=(0.78, 0.34), fontsize=8, xycoords="axes fraction")
    axtop[1].annotate(r"Ta$_{\mathrm{II}}$(1$^{\prime}$)", xy=(0.78, 0.19),
                      fontsize=8, xycoords="axes fraction")

    add_axis_triad_labels(
        axtop[1],
        b_xy=(0.30, 0.02),
        c_xy=(0.55, 0.15),
        a_xy=(0.70, 0.02),
        fontsize=8,
    )

    # =============================================================================
    # Panel (c): schematic (programmatic drawing)
    # =============================================================================
    # Site positions (x,y) for NON-CDW, CDW, and A(18)-distorted CDW
    non_cdw_positions = [(1, 1), (1, 2), (1, 3), (1, 4),
                         (2, 1), (2, 2), (2, 3), (2, 4)]

    cdw_positions = [(4, 1), (4, 1.8), (4, 3), (4, 4.2),
                     (5, 1), (5, 2.2), (5, 3), (5, 3.8)]

    raman_cdw_positions = [(7, 1), (7, 1.8), (7, 3), (7, 4.2),
                           (8, 1), (8, 2.2), (8, 3), (8, 3.8)]

    # Draw sites (circles)
    draw_circles(axbottom, non_cdw_positions, color="orange", radius=0.3, zorder=2)
    draw_circles(axbottom, cdw_positions,      color="orange", radius=0.3, zorder=2)
    draw_circles(axbottom, raman_cdw_positions, color="orange", radius=0.3, zorder=2)

    # Reference guide lines (dashed horizontal lines)
    for y in range(1, 5):
        axbottom.axhline(y=y, color="black", linestyle="--", linewidth=1.5, zorder=0)

    # Arrows indicating A(18)-like displacement pattern (picked from your original indices)
    # Format: (x_start, y_start, dx, dy)
    raman_phonon_arrows = [
        (raman_cdw_positions[3][0], cdw_positions[3][1], 0.0, -0.5),
        (raman_cdw_positions[1][0], cdw_positions[1][1], 0.0, +0.5),
        (raman_cdw_positions[5][0], cdw_positions[5][1], 0.0, -0.5),
        (raman_cdw_positions[7][0], cdw_positions[7][1], 0.0, +0.5),
    ]

    for x, y, dx, dy in raman_phonon_arrows:
        axbottom.arrow(x, y, dx, dy,
                       head_width=0.2, head_length=0.2, width=0.1,
                       fc="blue", ec="blue", zorder=1)

    # Stars: encode short/long Ta–Ta bonds (or any highlighted motif)
    short_star_cdw = [cdw_positions[0], cdw_positions[6]]
    long_star_cdw  = [cdw_positions[2], cdw_positions[4]]

    raman_short_star = [raman_cdw_positions[0], raman_cdw_positions[6]]
    raman_long_star  = [raman_cdw_positions[2], raman_cdw_positions[4]]

    draw_stars(axbottom, short_star_cdw, size=0.2, color="red", zorder=4)
    draw_stars(axbottom, long_star_cdw,  size=0.2, color="green", zorder=4)
    draw_stars(axbottom, raman_short_star, size=0.2, color="red", zorder=4)
    draw_stars(axbottom, raman_long_star,  size=0.2, color="green", zorder=4)

    # Column labels (I/II) and group labels
    axbottom.text(1.0, 5.0, "I",  ha="center", va="top")
    axbottom.text(2.0, 5.0, "II", ha="center", va="top")

    axbottom.text(4.0, 5.0, "I",  ha="center", va="top")
    axbottom.text(5.0, 5.0, "II", ha="center", va="top")

    axbottom.text(7.0, 5.0, "I",  ha="center", va="top")
    axbottom.text(8.0, 5.0, "II", ha="center", va="top")

    axbottom.text(1.5, 0.5, "NON-CDW",             ha="center", va="top")
    axbottom.text(4.5, 0.5, "CDW",                 ha="center", va="top")
    axbottom.text(7.5, 0.5, r"$\mathit{A}(18)$ on CDW", ha="center", va="top")

    # Final axes formatting for schematic
    axbottom.set_xlim(-0.5, 9.5)
    axbottom.set_ylim(0.5, 5.0)
    axbottom.set_aspect("equal")
    axbottom.annotate("(c)", xy=(-0.04, 0.94), xycoords="axes fraction")
    axbottom.axis("off")

    # =============================================================================
    # Save & show
    # =============================================================================
    plt.savefig("FIG_1.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()

