Supplementary Fig. S4: SOC-DFT vs Wannier-interpolated bands for the A(18)-distorted structure.

Workflow
1. SCF_WANNIER/   VASP SCF run (with LWANNIER90) on the distorted structure
                  (POSCAR, INCAR, KPOINTS) + Wannier90 input (wannier90.win).
2. wannier90.win, wannier90_band.{dat,kpt,gnu}   Wannier90 band interpolation (148 Wannier functions, spinor).
3. NSCF_FULL_SEEKPATH/  NSCF DFT bands along the full high-symmetry path (INCAR, KPOINTS, EIGENVAL, DOSCAR).
   NSCF_10GZ/           NSCF DFT bands on a dense Gamma-Z line.
4. python plot_wannier_dft_bands_compare.py  ->  WANNIER_DFT_BANDS_COMPARE_{FULL,ZOOM}.png
   (requires numpy, matplotlib; run inside this folder)

POTCAR files are NOT included (VASP license). PAW potentials used:
PAW_PBE Ta 17Jan2003, PAW_PBE Se 06Sep2000, PAW_PBE I 08Apr2002.
