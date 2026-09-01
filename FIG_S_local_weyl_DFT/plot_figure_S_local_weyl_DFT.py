#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

EIGENVAL = Path("EIGENVAL")
OUTCAR = Path("OUTCAR")
OUTPUT = Path("FIG_S_local_weyl_DFT.png")

def read_efermi(path):
    text = path.read_text(errors="ignore")
    values = re.findall(r"E-fermi\s*:\s*([-\d.]+)", text)
    if not values:
        raise ValueError("Could not find E-fermi in OUTCAR")
    return float(values[-1])

def read_eigenval(path):
    lines = path.read_text().splitlines()
    nelect, nkpts, nbands = map(int, lines[5].split()[:3])
    kcoords = []
    bands = np.zeros((nbands, nkpts))
    idx = 6

    for ik in range(nkpts):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        parts = lines[idx].split()
        kcoords.append([float(x) for x in parts[:3]])
        idx += 1

        for ib in range(nbands):
            parts = lines[idx].split()
            bands[ib, ik] = float(parts[1])
            idx += 1

    return np.asarray(kcoords), bands

efermi = read_efermi(OUTCAR)
kcoords, bands = read_eigenval(EIGENVAL)

nsegments = 4
points_per_segment = len(kcoords) // nsegments
segments = [
    (i * points_per_segment, (i + 1) * points_per_segment)
    for i in range(nsegments)
]

kdist = np.zeros(len(kcoords))
segment_bounds = [0.0]
segment_centers = []
current = 0.0

for start, end in segments:
    kdist[start] = current
    for i in range(start + 1, end):
        current += np.linalg.norm(kcoords[i] - kcoords[i - 1])
        kdist[i] = current
    segment_centers.append(kdist[start + points_per_segment // 2])
    segment_bounds.append(current)

fig, ax = plt.subplots(figsize=(8.5, 3.5))

for band in bands:
    shifted = band - efermi
    for start, end in segments:
        ax.plot(kdist[start:end], shifted[start:end], linewidth=0.9)

for xpos in segment_bounds:
    ax.axvline(xpos, linewidth=0.9, alpha=0.55)

for label, xpos in zip(["W1", "W2", "W3", "W4"], segment_centers):
    ax.axvline(xpos, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.text(xpos, 0.024, label, ha="center", va="top",
            fontsize=13, fontweight="bold")

ax.axhline(0.0, linestyle="--", linewidth=0.9, alpha=0.8)
ax.set_xlim(kdist.min(), kdist.max())
ax.set_ylim(-0.05, 0.05)
ax.set_ylabel(r"$E-E_F$ (eV)")
ax.set_xticks(segment_centers)
ax.set_xticklabels(["W1", "W2", "W3", "W4"])
ax.set_title("Direct-DFT bands along local paths through four Weyl points")
ax.tick_params(direction="in", top=True, right=True)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
