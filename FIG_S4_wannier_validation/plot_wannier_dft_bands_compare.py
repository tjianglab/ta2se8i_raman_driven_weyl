#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASE_DIR = Path(__file__).resolve().parent


def find_line(path, keyword):
    with open(path, "r") as handle:
        for line in handle:
            if keyword in line:
                return line
    raise ValueError(f"Cannot find {keyword!r} in {path}")


def read_outcar_fermi(path):
    line = find_line(path, "E-fermi")
    return float(line.split()[2])


def read_doscar_fermi(path):
    with open(path, "r") as handle:
        for _ in range(5):
            next(handle)
        return float(next(handle).split()[3])


def read_num_wann(path):
    line = find_line(path, "num_wann")
    match = re.search(r"num_wann\s*=?\s*(\d+)", line)
    if not match:
        raise ValueError(f"Cannot parse num_wann from {path}: {line}")
    return int(match.group(1))


def read_wannier_xtics(path):
    line = find_line(path, "set xtics")
    matches = re.findall(
        r'"([^"]+)"\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)',
        line,
    )
    if not matches:
        raise ValueError(f"Cannot parse set xtics from {path}: {line}")

    labels = [label.strip() for label, _ in matches]
    positions = np.array([float(value) for _, value in matches], dtype=float)
    return labels, positions


def pretty_label(label):
    label = label.strip()
    if label == "G":
        return r"$\Gamma$"
    return label


REFERENCE_XTIC_LABELS = [
    r"$\Gamma$",
    "Z",
    r"$B_0|D_0$",
    "X",
    r"$G_0|H_0$",
    "Y",
    r"$C_0|A_0$",
    r"$Z|Y$",
    r"$\Gamma$",
    r"$X/\Gamma$",
    "L",
]


def display_xtic_labels(xtic_labels):
    if len(xtic_labels) == len(REFERENCE_XTIC_LABELS):
        return REFERENCE_XTIC_LABELS
    return [pretty_label(label) for label in xtic_labels]


def read_wannier_bands(base_dir, fermi):
    n_kpoints = int((base_dir / "wannier90_band.kpt").read_text().split()[0])
    n_bands = read_num_wann(base_dir / "wannier90.win")

    raw = np.loadtxt(base_dir / "wannier90_band.dat")
    expected_rows = n_kpoints * n_bands
    if raw.shape[0] != expected_rows:
        raise ValueError(
            "Wannier band data size mismatch: "
            f"got {raw.shape[0]} rows, expected {expected_rows} "
            f"({n_bands} bands x {n_kpoints} k-points)"
        )

    raw = raw.reshape(n_bands, n_kpoints, 2)
    kdist = raw[0, :, 0]
    bands = raw[:, :, 1] - fermi
    return kdist, bands


def read_dft_bands(eigenval_path, fermi):
    lines = eigenval_path.read_text().splitlines()
    _, n_kpoints, n_bands = [int(value) for value in lines[5].split()[:3]]

    bands = np.empty((n_bands, n_kpoints), dtype=float)
    idx = 6
    for ik in range(n_kpoints):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        idx += 1

        for ib in range(n_bands):
            parts = lines[idx].split()
            bands[ib, ik] = float(parts[1]) - fermi
            idx += 1

    return bands


def read_kpoints_per_segment(kpoints_path):
    lines = kpoints_path.read_text().splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError(f"Cannot parse points per segment from {kpoints_path}")


def build_dft_kdist(n_kpoints, xtic_positions, points_per_segment):
    n_segments = len(xtic_positions) - 1
    if n_segments * points_per_segment != n_kpoints:
        points_per_segment = n_kpoints // n_segments
        if n_segments * points_per_segment != n_kpoints:
            raise ValueError(
                "DFT k-point count cannot be divided into the Wannier path: "
                f"{n_kpoints} k-points, {n_segments} segments"
            )

    pieces = [
        np.linspace(xtic_positions[i], xtic_positions[i + 1], points_per_segment)
        for i in range(n_segments)
    ]
    return np.concatenate(pieces), points_per_segment


def build_dft_segments_with_dense_gz(
    full_dft_bands,
    dense_gz_bands,
    xtic_positions,
    full_points_per_segment,
):
    n_segments = len(xtic_positions) - 1
    expected_full_kpoints = n_segments * full_points_per_segment
    if full_dft_bands.shape[1] != expected_full_kpoints:
        raise ValueError(
            "Full seekpath DFT k-point count does not match path segmentation: "
            f"got {full_dft_bands.shape[1]}, expected {expected_full_kpoints} "
            f"({n_segments} segments x {full_points_per_segment} points)"
        )
    if dense_gz_bands.shape[0] != full_dft_bands.shape[0]:
        raise ValueError(
            "Dense Gamma-Z and full seekpath DFT have different band counts: "
            f"{dense_gz_bands.shape[0]} vs {full_dft_bands.shape[0]}"
        )

    segments = []
    dense_gz_kdist = np.linspace(
        xtic_positions[0], xtic_positions[1], dense_gz_bands.shape[1]
    )
    segments.append((dense_gz_kdist, dense_gz_bands))

    for iseg in range(1, n_segments):
        start = iseg * full_points_per_segment
        end = (iseg + 1) * full_points_per_segment
        segment_kdist = np.linspace(
            xtic_positions[iseg],
            xtic_positions[iseg + 1],
            full_points_per_segment,
        )
        segments.append((segment_kdist, full_dft_bands[:, start:end]))

    return segments


def plot_comparison(
    output_path,
    xtic_labels,
    xtic_positions,
    wannier_kdist,
    wannier_bands,
    dft_segments,
    ylim,
    title,
    highlight_dft,
    highlight_wannier,
):
    fig, ax = plt.subplots(figsize=(8.5, 3.5), constrained_layout=True)

    n_segments = len(xtic_positions) - 1
    dft_color = "0.15"
    wannier_color = "#d1495b"

    for dft_kdist, dft_segment_bands in dft_segments:
        for band in dft_segment_bands:
            ax.plot(
                dft_kdist,
                band,
                color=dft_color,
                lw=0.65,
                alpha=0.78,
                zorder=2,
            )

    for band in wannier_bands:
        ax.plot(
            wannier_kdist,
            band,
            color=wannier_color,
            lw=0.65,
            ls="--",
            alpha=0.72,
            zorder=3,
        )

    for ib in highlight_dft:
        for dft_kdist, dft_segment_bands in dft_segments:
            if 0 <= ib < dft_segment_bands.shape[0]:
                ax.plot(
                    dft_kdist,
                    dft_segment_bands[ib],
                    color="#005f73",
                    lw=1.25,
                    alpha=0.95,
                    zorder=5,
                )

    for ib in highlight_wannier:
        if 0 <= ib < wannier_bands.shape[0]:
            ax.plot(
                wannier_kdist,
                wannier_bands[ib],
                color="#ee9b00",
                lw=1.25,
                ls="--",
                alpha=0.95,
                zorder=6,
            )

    for xpos in xtic_positions:
        ax.axvline(xpos, lw=0.55, color="0.55", alpha=0.8, zorder=1)
    ax.axhline(0, lw=0.7, color="0.1", ls=":", zorder=1)

    ax.set_xlim(float(xtic_positions[0]), float(xtic_positions[-1]))
    ax.set_ylim(*ylim)
    ax.set_xticks(xtic_positions)
    ax.set_xticklabels(display_xtic_labels(xtic_labels), fontsize=9)
    ax.set_ylabel(r"$E-E_F$ (eV)")
    ax.set_title(title)

    dft_proxy = plt.Line2D([0], [0], color=dft_color, lw=1.2, label="DFT EIGENVAL")
    wannier_proxy = plt.Line2D(
        [0], [0], color=wannier_color, lw=1.2, ls="--", label="Wannier90"
    )
    ax.legend(handles=[dft_proxy, wannier_proxy], loc="upper right", frameon=False)

    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Wannier90 interpolated bands and raw DFT EIGENVAL bands."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-prefix", default="WANNIER_DFT_BANDS_COMPARE")
    parser.add_argument("--full-ylim", nargs=2, type=float, default=(-8.0, 8.0))
    parser.add_argument("--zoom-ylim", nargs=2, type=float, default=(-1.0, 1.0))
    parser.add_argument("--dense-gz-dir", type=Path, default=None)
    # parser.add_argument("--highlight-dft", nargs="*", type=int, default=[129, 130])
    # parser.add_argument("--highlight-wannier", nargs="*", type=int, default=[97, 98])
    args = parser.parse_args()

    base_dir = args.base_dir
    dft_dir = base_dir / "NSCF_FULL_SEEKPATH"
    dense_gz_dir = args.dense_gz_dir or base_dir / "NSCF_10GZ"

    wannier_fermi = read_outcar_fermi(base_dir / "OUTCAR")
    dft_fermi = read_doscar_fermi(base_dir / "DOSCAR")
    dense_gz_fermi = read_doscar_fermi(dense_gz_dir / "DOSCAR")
    common_fermi = dft_fermi

    xtic_labels, xtic_positions = read_wannier_xtics(base_dir / "wannier90_band.gnu")
    wannier_kdist, wannier_bands = read_wannier_bands(base_dir, common_fermi)
    full_dft_bands = read_dft_bands(dft_dir / "EIGENVAL", common_fermi)
    dense_gz_bands = read_dft_bands(dense_gz_dir / "EIGENVAL", common_fermi)
    dft_points_per_segment = read_kpoints_per_segment(dft_dir / "KPOINTS")
    dft_segments = build_dft_segments_with_dense_gz(
        full_dft_bands,
        dense_gz_bands,
        xtic_positions,
        dft_points_per_segment,
    )

    print(f"Wannier Fermi from OUTCAR: {wannier_fermi:.8f} eV")
    print(f"DFT Fermi from DOSCAR:     {dft_fermi:.8f} eV")
    print(f"NSCF_10GZ Fermi from DOSCAR: {dense_gz_fermi:.8f} eV")
    print(f"Using common Fermi:        {common_fermi:.8f} eV")
    print(f"Wannier bands: {wannier_bands.shape[0]}, k-points: {wannier_bands.shape[1]}")
    print(
        f"Full DFT bands: {full_dft_bands.shape[0]}, "
        f"k-points: {full_dft_bands.shape[1]}"
    )
    print(
        f"Dense Gamma-Z DFT bands: {dense_gz_bands.shape[0]}, "
        f"k-points: {dense_gz_bands.shape[1]}"
    )
    print(
        "DFT points per segment after alignment: "
        f"Gamma-Z={dense_gz_bands.shape[1]}, others={dft_points_per_segment}"
    )

    output_dir = Path.cwd()
    full_output = output_dir / f"{args.output_prefix}_FULL.png"
    zoom_output = output_dir / f"{args.output_prefix}_ZOOM.png"

    plot_comparison(
        full_output,
        xtic_labels,
        xtic_positions,
        wannier_kdist,
        wannier_bands,
        dft_segments,
        tuple(args.full_ylim),
        "DFT vs Wannier90 band structure (dense Gamma-Z)",
        [],
        [],
    )
    plot_comparison(
        zoom_output,
        xtic_labels,
        xtic_positions,
        wannier_kdist,
        wannier_bands,
        dft_segments,
        tuple(args.zoom_ylim),
        r"DFT vs Wannier90 band structure near $E_F$ (dense Gamma-Z)",
        [],
        [],
    )

    print(f"Saved {full_output}")
    print(f"Saved {zoom_output}")


if __name__ == "__main__":
    main()
