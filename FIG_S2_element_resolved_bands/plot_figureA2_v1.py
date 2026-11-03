import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Global style (your palette)
# =========================
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

# =========================
# Helper functions
# =========================
def load_pband_elements_dat(path, fermi_level=0.0):
    """
    Read PBAND_ELEMENTS_SOC.dat-like file:
      col0: k-distance
      col1: energy
      col2+: projection weights for elements
    Returns: k_dist, energy_shifted, proj_weights
    """
    data = np.loadtxt(path)
    k_dist = data[:, 0]
    energy = data[:, 1] - fermi_level
    proj_weights = data[:, 2:]
    return k_dist, energy, proj_weights


def plot_row(ax_row, dat_path, element_list, panel_tag, q_text,
             vmin=0.0, vmax=1.0, cmap="hot_r", fermi_level=0.0,
             ylim=(-1, 1)):
    """
    Plot one row (3 subplots) of projected bands for given dat file.
    Returns the last scatter handle (for colorbar).
    """
    k_dist, energy, proj_weights = load_pband_elements_dat(dat_path, fermi_level=fermi_level)

    sc = None
    for i, element in enumerate(element_list):
        weights = proj_weights[:, i]

        sc = ax_row[i].scatter(
            k_dist, energy,
            c=weights,
            cmap=cmap,
            s=5,
            alpha=0.7,
            vmin=vmin, vmax=vmax
        )

        ax_row[i].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax_row[i].grid(True, alpha=0.3)

        ax_row[i].set_xlim(np.min(k_dist), np.max(k_dist))
        ax_row[i].set_ylim(*ylim)

        # show y tick labels only on first column
        if i != 0:
            ax_row[i].set_yticklabels([])

        # only show Γ and Z at ends
        ax_row[i].set_xticks([np.min(k_dist), np.max(k_dist)])
        ax_row[i].set_xticklabels([r"$\Gamma$", r"$\mathrm{Z}$"])

        # element label inside panel
        ax_row[i].annotate(element, xy=(0.70, 0.80), xycoords='axes fraction')

    # left axis y label + ticks
    ax_row[0].set_ylabel(r"$E-E_f$ (eV)", labelpad=0.0)
    ax_row[0].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # panel annotation
    ax_row[0].annotate(panel_tag, xy=(-0.80, 1.00), xycoords='axes fraction')
    ax_row[0].annotate(q_text,    xy=(-0.80, 0.84), xycoords='axes fraction')

    return sc


def add_row_colorbar(fig, sc, ax_row, ticks=(0.0, 0.5, 1.0), pad=0.02):
    """
    Add one colorbar for a whole row of axes.
    """
    cbar = fig.colorbar(
        sc,
        ax=ax_row,
        location='right',
        pad=pad
    )
    cbar.set_ticks(list(ticks))
    return cbar


# =========================
# Main
# =========================
if __name__ == "__main__":

    # ---- user settings ----
    element_list = ['Ta', 'Se', 'I']
    vmin, vmax = 0.0, 1.0
    cmap = "hot_r"
    fermi_level = 0.0
    ylim = (-1, 1)

    # input files
    dat_top = "Q_0.0_BANDS_WITH_WEIGHT/PBAND_ELEMENTS_SOC.dat"   # Q=0.0
    dat_bot = "Q_2.5_BANDS_WITH_WEIGHT/PBAND_ELEMENTS_SOC.dat"   # Q=2.5

    # ---- figure / axes ----
    fig = plt.figure(constrained_layout=True, figsize=(4, 4))
    axs = fig.subplots(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # Row 1: Q=0.0
    sc0 = plot_row(
        ax_row=axs[0, :],
        dat_path=dat_top,
        element_list=element_list,
        panel_tag="(a)",
        q_text=r"$Q$=0.0",
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        fermi_level=fermi_level,
        ylim=ylim
    )
    add_row_colorbar(fig, sc0, axs[0, :], ticks=(0.0, 0.5, 1.0), pad=0.02)

    # Row 2: Q=2.5
    sc1 = plot_row(
        ax_row=axs[1, :],
        dat_path=dat_bot,
        element_list=element_list,
        panel_tag="(b)",
        q_text=r"$Q$=2.5",
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        fermi_level=fermi_level,
        ylim=ylim
    )
    add_row_colorbar(fig, sc1, axs[1, :], ticks=(0.0, 0.5, 1.0), pad=0.02)

    plt.savefig("FIG_A2_v1.png", dpi=600)
    plt.show()

