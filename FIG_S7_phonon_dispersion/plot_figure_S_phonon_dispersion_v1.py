#!/usr/bin/env python3

from pathlib import Path
import gzip
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError:
    if os.environ.get("PHONOPY_BAND_PLOT_RETRY") != "1":
        os.environ["PHONOPY_BAND_PLOT_RETRY"] = "1"
        os.execvp("conda", ["conda", "run", "-n", "py310mac", "python", __file__])
    raise


# ===================== You can change these settings =====================
BAND_YAML = "band.yaml.gz"
OUTPUT_PNG = "FIG_S_phonon_dispersion.png"

DPI = 600
FIGSIZE = (6.5, 3.5)
YMIN = None
YMAX = None

LINE_COLOR = "black"
LINE_WIDTH = 0.8
ZERO_LINE = True
Y_GRID = True

# Apply these replacements first. They are simultaneous, so swaps are safe.
# Labels may be written as D0 or D_0 in band.yaml; both are normalized to D0.
LABEL_REPLACEMENTS = {
    "T": "Z",
    "D0": "B0",
    "B0": "D0",
    "Z": "X",
    "A0": "G0",
    "C0": "H0",
    "H0": "C0",
    "G0": "A0",
}

# Plotting order AFTER LABEL_REPLACEMENTS has been applied.
# Each inner list is one continuous branch. Different branches meet at the
# same x position but are NOT connected by a band line.
#
# This reproduces:
# Gamma-Z-B0 | D0-X-G0 | H0-Y-C0 | A0-Z | Y-Gamma-X / Gamma-L
PLOT_BRANCHES = [
    ["GAMMA", "Z", "B0"],
    ["D0", "X", "G0"],
    ["H0", "Y", "C0"],
    ["A0", "Z"],
    ["Y", "GAMMA", "X"],
    ["GAMMA", "L"],
]

# Use equal horizontal width for every high-symmetry-path segment, matching
# the electronic-band figure. Set False to retain each segment's q-distance.
EQUAL_SEGMENT_WIDTH = True
SEGMENT_WIDTH = 1.0

# Default separator between disconnected branches, e.g. B0|D0 and Z|Y.
DEFAULT_BRANCH_SEPARATOR = "|"

# Optional separator overrides. The electronic-band figure uses X/Gamma.
BRANCH_SEPARATOR_OVERRIDES = {
    ("X", "GAMMA"): "/",
}
# ========================================================================


def canonical_label(label):
    """Convert a YAML/LaTeX label to a plain matching key."""
    label = str(label).strip()
    label = re.sub(r"\$(.*?)\$", r"\1", label)
    label = label.replace(r"\mathrm{", "").replace("}", "")
    label = re.sub(r"\\+Gamma", "GAMMA", label)
    label = label.replace("Γ", "GAMMA")
    label = label.replace("_", "")
    label = label.replace("{", "").replace("}", "")
    label = label.replace(" ", "")
    return label.upper() if label.upper() == "GAMMA" else label


def replaced_label(label):
    """Normalize a label, then apply exactly one simultaneous replacement."""
    key = canonical_label(label)
    return canonical_label(LABEL_REPLACEMENTS.get(key, key))


def display_label(label):
    """Format a canonical label for Matplotlib."""
    label = canonical_label(label)
    if label == "GAMMA":
        return r"$\Gamma$"

    match = re.fullmatch(r"([A-Za-z]+)(\d+)", label)
    if match:
        base, subscript = match.groups()
        return rf"${base}_{{{subscript}}}$"

    return label


def read_phonopy_band_yaml(filename):
    distances = []
    frequencies = []
    one_q_freqs = None
    segment_nqpoint = []
    labels = []
    section = None

    number = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?")

    opener = gzip.open if str(filename).endswith(".gz") else open

    with opener(filename, "rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()

            if text == "segment_nqpoint:":
                section = "segment"
                continue
            if text == "labels:":
                section = "labels"
                continue
            if text == "phonon:":
                section = "phonon"
                continue

            if section == "segment" and text.startswith("- "):
                segment_nqpoint.append(int(text.split()[1]))
                continue

            if section == "labels" and text.startswith("- ["):
                found = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
                found = [a or b for a, b in found]
                if len(found) >= 2:
                    labels.append([found[0], found[1]])
                continue

            if "distance:" in text:
                if one_q_freqs is not None:
                    frequencies.append(one_q_freqs)
                one_q_freqs = []
                distances.append(float(text.split("distance:", 1)[1]))
                continue

            if one_q_freqs is not None and "frequency:" in text:
                match = number.search(text.split("frequency:", 1)[1])
                if match:
                    one_q_freqs.append(float(match.group(0)))

    if one_q_freqs is not None:
        frequencies.append(one_q_freqs)

    distances = np.asarray(distances, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    if not segment_nqpoint:
        raise ValueError("No segment_nqpoint section was found in band.yaml.")
    if len(labels) < len(segment_nqpoint):
        raise ValueError(
            f"band.yaml contains {len(segment_nqpoint)} segments but only "
            f"{len(labels)} endpoint-label pairs."
        )
    if sum(segment_nqpoint) != len(distances):
        raise ValueError(
            "The sum of segment_nqpoint does not match the number of q-points: "
            f"{sum(segment_nqpoint)} != {len(distances)}."
        )
    if frequencies.ndim != 2 or len(frequencies) != len(distances):
        raise ValueError("The frequency array has an unexpected shape.")

    return distances, frequencies, segment_nqpoint, labels


def split_into_segments(distances, frequencies, segment_nqpoint, labels):
    """Create independently reorderable path segments."""
    segments = []
    start_index = 0

    for index, npoint in enumerate(segment_nqpoint):
        end_index = start_index + npoint
        raw_start, raw_end = labels[index]

        segment_distances = distances[start_index:end_index].copy()
        segment_frequencies = frequencies[start_index:end_index].copy()

        # Local distance beginning at zero; this also handles restarted
        # distances between disconnected branches.
        local_distance = segment_distances - segment_distances[0]
        if local_distance[-1] < 0:
            local_distance = -local_distance

        segments.append(
            {
                "index": index,
                "start": replaced_label(raw_start),
                "end": replaced_label(raw_end),
                "distance": local_distance,
                "frequency": segment_frequencies,
                "used": False,
            }
        )
        start_index = end_index

    return segments


def find_segment(segments, wanted_start, wanted_end):
    """Find an unused segment, reversing it when necessary."""
    wanted_start = canonical_label(wanted_start)
    wanted_end = canonical_label(wanted_end)

    for segment in segments:
        if segment["used"]:
            continue

        if segment["start"] == wanted_start and segment["end"] == wanted_end:
            segment["used"] = True
            return segment["distance"].copy(), segment["frequency"].copy()

        if segment["start"] == wanted_end and segment["end"] == wanted_start:
            segment["used"] = True
            old_distance = segment["distance"]
            reversed_distance = old_distance[-1] - old_distance[::-1]
            reversed_frequency = segment["frequency"][::-1].copy()
            return reversed_distance, reversed_frequency

    available = [
        f"{s['start']}-{s['end']}"
        for s in segments
        if not s["used"]
    ]
    raise ValueError(
        f"Could not find the requested segment {wanted_start}-{wanted_end} "
        "after applying LABEL_REPLACEMENTS.\n"
        f"Unused available segments: {', '.join(available)}"
    )


def add_tick(tick_positions, tick_labels, x, label, separator=None):
    """Add a tick or merge a disconnected-branch label at the same x."""
    formatted = display_label(label)

    if tick_positions and abs(x - tick_positions[-1]) < 1.0e-10:
        if formatted != tick_labels[-1]:
            tick_labels[-1] += (separator or DEFAULT_BRANCH_SEPARATOR) + formatted
    else:
        tick_positions.append(x)
        tick_labels.append(formatted)


def build_reordered_plot_data(segments):
    plot_segments = []
    tick_positions = []
    tick_labels = []
    current_x = 0.0
    previous_branch_end = None

    for branch_index, raw_branch in enumerate(PLOT_BRANCHES):
        branch = [canonical_label(label) for label in raw_branch]
        if len(branch) < 2:
            raise ValueError(f"PLOT_BRANCHES entry {branch_index} has fewer than two labels.")

        if branch_index == 0:
            add_tick(tick_positions, tick_labels, current_x, branch[0])
        else:
            separator = BRANCH_SEPARATOR_OVERRIDES.get(
                (previous_branch_end, branch[0]),
                DEFAULT_BRANCH_SEPARATOR,
            )
            add_tick(
                tick_positions,
                tick_labels,
                current_x,
                branch[0],
                separator=separator,
            )

        for path_start, path_end in zip(branch[:-1], branch[1:]):
            local_distance, frequency = find_segment(
                segments, path_start, path_end
            )

            if EQUAL_SEGMENT_WIDTH:
                if len(local_distance) == 1:
                    x_values = np.array([current_x])
                    width = 0.0
                else:
                    width = float(SEGMENT_WIDTH)
                    x_values = np.linspace(current_x, current_x + width, len(local_distance))
            else:
                width = float(local_distance[-1])
                if width <= 0.0:
                    width = float(SEGMENT_WIDTH)
                    x_values = np.linspace(current_x, current_x + width, len(local_distance))
                else:
                    x_values = current_x + local_distance

            plot_segments.append((x_values, frequency, path_start, path_end))
            current_x += width
            add_tick(tick_positions, tick_labels, current_x, path_end)

        previous_branch_end = branch[-1]

    unused = [s for s in segments if not s["used"]]
    if unused:
        print("Warning: these band.yaml segments were not plotted:")
        for segment in unused:
            print(f"  segment {segment['index']}: {segment['start']}-{segment['end']}")

    return plot_segments, tick_positions, tick_labels


def main():
    distances, frequencies, segment_nqpoint, labels = read_phonopy_band_yaml(BAND_YAML)
    segments = split_into_segments(distances, frequencies, segment_nqpoint, labels)
    plot_segments, xticks, xlabels = build_reordered_plot_data(segments)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for x_values, segment_frequencies, _, _ in plot_segments:
        for iband in range(segment_frequencies.shape[1]):
            ax.plot(
                x_values,
                segment_frequencies[:, iband],
                color=LINE_COLOR,
                lw=LINE_WIDTH,
            )

    for x in xticks:
        ax.axvline(x, color="0.75", lw=0.6, zorder=0)

    if ZERO_LINE:
        ax.axhline(0.0, color="0.75", lw=0.8, ls="--", zorder=0)

    ax.set_xlim(xticks[0], xticks[-1])
    if YMIN is not None or YMAX is not None:
        ax.set_ylim(YMIN, YMAX)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Frequency (THz)")
    ax.tick_params(axis="x", length=0)

    if Y_GRID:
        ax.grid(axis="y", color="0.9", lw=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    print(f"Wrote {Path(OUTPUT_PNG).resolve()}")


if __name__ == "__main__":
    main()
