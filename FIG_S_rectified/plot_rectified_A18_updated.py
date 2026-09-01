import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
#
# Energy normalization:
#     meV/atom
#
# Normal-coordinate convention:
#     22-atom primitive CDW cell
#
# Sign convention:
#     Positive Q_R is the A(18) direction that suppresses the
#     Ta-chain tetramerization and the CDW gap.
# ============================================================

f_R = 2.52          # THz, frequency of A(18)
N_at = 22           # atoms in the primitive CDW cell

# Anharmonic coefficients in the updated Q_R convention.
# Terms odd in Q_R change sign under Q_R -> -Q_R.
c2 = -0.180
c21 = -0.080
d2 = -0.019
d22 = 0.009


# ============================================================
# Harmonic coefficient
#
# K_R_cell is the harmonic coefficient per 22-atom cell.
# Dividing by 22 converts it to the same meV/atom
# normalization used for the fitted anharmonic coefficients.
# ============================================================

K_R_cell = 4.09 * f_R**2
K_R = K_R_cell / N_at

print(
    f"K_R_cell = {K_R_cell:.4f} "
    "meV/(amu A^2) per 22-atom cell"
)
print(
    f"K_R      = {K_R:.6f} "
    "meV atom^-1/(amu A^2)"
)


# ============================================================
# Peak B3(7) amplitudes
#
# The nonlinear PES was sampled up to |Q_IR| = 2.
# Therefore, the figure is restricted to Q0 <= 2.
# ============================================================

Q0_array = np.linspace(0.0, 2.0, 401)


# ============================================================
# Lowest-order quasi-static solution
#
# Q_IR(t) = Q0 cos(Omega_IR t)
# <Q_IR^2> = Q0^2 / 2
#
# Q_R_ind = -(c21 / 2 K_R) Q0^2
# ============================================================

Q_lowest = -(c21 / (2.0 * K_R)) * Q0_array**2


# ============================================================
# Small-Q_R solution including d22
#
# Q_R_ind =
# -(c21/2) Q0^2 / (K_R + d22 Q0^2)
# ============================================================

Q_d22 = (
    -0.5 * c21 * Q0_array**2
    / (K_R + d22 * Q0_array**2)
)


# ============================================================
# Complete quasi-static solution
#
# Solve:
#
# d2 Q_R^3
# + c2 Q_R^2
# + (K_R + d22 Q0^2) Q_R
# + (c21/2) Q0^2
# = 0
#
# The selected solution must:
#   1. be real;
#   2. correspond to a local minimum;
#   3. remain continuously connected to Q_R = 0 at Q0 = 0.
# ============================================================

def local_curvature(q_r, q0):
    """Curvature of the cycle-averaged potential at Q_R."""
    return (
        K_R
        + d22 * q0**2
        + 2.0 * c2 * q_r
        + 3.0 * d2 * q_r**2
    )


def calculate_complete_branch(q0_values):
    """Follow the stable quasi-static branch connected to Q_R = 0."""
    q_complete = []
    previous_root = 0.0

    for q0 in q0_values:
        polynomial_coefficients = [
            d2,
            c2,
            K_R + d22 * q0**2,
            0.5 * c21 * q0**2,
        ]

        roots = np.roots(polynomial_coefficients)

        # Keep only numerically real roots.
        real_roots = roots[
            np.abs(roots.imag) < 1.0e-9
        ].real

        # A stable stationary solution must have positive curvature.
        stable_roots = [
            q_r
            for q_r in real_roots
            if local_curvature(q_r, q0) > 0.0
        ]

        if len(stable_roots) == 0:
            q_complete.append(np.nan)
            continue

        # Select the stable root closest to the previous point.
        selected_root = min(
            stable_roots,
            key=lambda q_r: abs(q_r - previous_root),
        )

        q_complete.append(selected_root)
        previous_root = selected_root

    return np.asarray(q_complete)


Q_complete = calculate_complete_branch(Q0_array)


# ============================================================
# Representative drive amplitude Q0 = 2
# ============================================================

Q0 = 2.0

Q_lowest_0 = (
    -(c21 / (2.0 * K_R)) * Q0**2
)

Q_d22_0 = (
    -0.5 * c21 * Q0**2
    / (K_R + d22 * Q0**2)
)

# Calculate the full branch from zero to Q0 to preserve continuity.
Q0_short = np.linspace(0.0, Q0, 401)
Q_complete_0 = calculate_complete_branch(Q0_short)[-1]

print()
print(
    f"At Q0 = {Q0:.1f} A sqrt(amu):"
)
print(
    f"Lowest-order result      = "
    f"{Q_lowest_0:+.4f} A sqrt(amu)"
)
print(
    f"Including d22            = "
    f"{Q_d22_0:+.4f} A sqrt(amu)"
)
print(
    f"Complete quasi-static    = "
    f"{Q_complete_0:+.4f} A sqrt(amu)"
)


# ============================================================
# Main plot
# ============================================================

fig, ax = plt.subplots(figsize=(5.2, 4.0))


# ------------------------------------------------------------
# Positive A(18) amplitudes relevant for gap suppression
# ------------------------------------------------------------

ax.axhspan(
    2.0,
    2.5,
    color="0.88",
    zorder=0,
)

ax.text(
    0.08,
    2.25,
    r"$A(18)$ amplitudes relevant to gap suppression",
    fontsize=8,
    va="center",
    color="0.30",
)


# ------------------------------------------------------------
# Quasi-static curves
# ------------------------------------------------------------

ax.plot(
    Q0_array,
    Q_lowest,
    linestyle=":",
    linewidth=1.7,
    color="0.45",
    label=r"Lowest order",
)

ax.plot(
    Q0_array,
    Q_d22,
    linestyle="--",
    linewidth=1.7,
    color="0.25",
    label=r"Including $d_{22}$",
)

ax.plot(
    Q0_array,
    Q_complete,
    linestyle="-",
    linewidth=2.0,
    color="k",
    label=r"Complete quasi-static solution",
)


# ------------------------------------------------------------
# Representative drive amplitude
# ------------------------------------------------------------

ax.axvline(
    Q0,
    color="0.55",
    linestyle=":",
    linewidth=1.0,
)

ax.plot(
    Q0,
    Q_complete_0,
    marker="o",
    markersize=5,
    color="k",
    zorder=5,
)

ax.axhline(
    0.0,
    color="k",
    linewidth=0.7,
)


# ------------------------------------------------------------
# Axis labels and limits
# ------------------------------------------------------------

ax.set_xlabel(
    r"Peak $B_3(7)$ amplitude $Q_0$ "
    r"($\mathrm{\AA}\sqrt{\mathrm{amu}}$)"
)

ax.set_ylabel(
    r"$\overline{Q_R^{\mathrm{ind}}}$ "
    r"($\mathrm{\AA}\sqrt{\mathrm{amu}}$)"
)

ax.set_xlim(0.0, 2.15)
ax.set_ylim(-0.02, 2.65)

ax.tick_params(
    direction="in",
    top=True,
    right=True,
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    fontsize=7.5,
    loc="center left",
    bbox_to_anchor=(0.01, 0.50),
    frameon=False,
)


# ============================================================
# Inset: zoom of the positive rectified response
# ============================================================

axins = ax.inset_axes(
    [0.52, 0.13, 0.43, 0.33]
)

axins.plot(
    Q0_array,
    Q_lowest,
    linestyle=":",
    linewidth=1.3,
    color="0.45",
)

axins.plot(
    Q0_array,
    Q_d22,
    linestyle="--",
    linewidth=1.3,
    color="0.25",
)

axins.plot(
    Q0_array,
    Q_complete,
    linestyle="-",
    linewidth=1.5,
    color="k",
)

axins.axvline(
    Q0,
    color="0.55",
    linestyle=":",
    linewidth=0.8,
)

axins.plot(
    Q0,
    Q_complete_0,
    marker="o",
    markersize=4,
    color="k",
    zorder=5,
)

axins.axhline(
    0.0,
    color="k",
    linewidth=0.5,
)

axins.set_xlim(0.0, 2.05)
axins.set_ylim(-0.005, 0.145)

axins.set_title(
    r"Zoom of the rectified response",
    fontsize=7,
    pad=5,
)

axins.annotate(
    rf"$Q_0=2.0$: "
    rf"$\overline{{Q_R^{{\rm ind}}}}"
    rf"={Q_complete_0:+.3f}$",
    xy=(Q0, Q_complete_0),
    xytext=(0.47, 0.095),
    fontsize=6.5,
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.7,
        color="0.25",
    ),
)

axins.tick_params(
    direction="in",
    top=True,
    right=True,
    labelsize=6.5,
)


# ============================================================
# Save figure
# ============================================================

plt.tight_layout()

plt.savefig(
    "FIG_S_rectified.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "FIG_S_rectified.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
