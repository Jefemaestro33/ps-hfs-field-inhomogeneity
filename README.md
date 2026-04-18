# Magnetic-field-inhomogeneity bias in indirect Zeeman measurements of the positronium hyperfine interval

Supporting code for the manuscript:

> Zermeño, E. D. (2026). *Magnetic-field-inhomogeneity bias in indirect Zeeman measurements of the positronium hyperfine interval.* Manuscript in preparation.

This repository contains the simulation code that reproduces all numerical results reported in the manuscript, including the central parameter-free shift of the Mills measurement, the Ritter parameter scan, and the Ishida negative control.

---

## Overview

The code computes the systematic bias in the extracted positronium (Ps) ground-state hyperfine interval arising from propagation of sub-ppm magnetic-field inhomogeneities through the nonlinear Breit–Rabi inversion. The calculation uses published experimental parameters only, with no free fit parameters, and is applied to three generations of precision measurements:

- **Mills & Bearman (1975)** — thin TM₁₁₀ cavity, conventional electromagnet
- **Ritter et al. (1984)** — Varian V4012A electromagnet, post-shimming
- **Ishida et al. (2014)** — superconducting solenoid, 1.5 ppm RMS uniformity

### Headline numbers reproduced by this code

| Quantity | Value |
|---|---|
| Sensitivity coefficient at Mills operating point | dΔ_HFS / dB_res ≈ 4.5 × 10⁵ MHz/T |
| Mills systematic bias (parameter-free) | δΔ = −0.22 ± 0.10 MHz |
| Ritter post-shimming range (c₂ = 3–5 ppm) | δΔ ∈ [−0.18, −0.30] MHz |
| Ishida SC solenoid (uniform Ps) | δΔ = +0.30 MHz |
| Ishida SC solenoid (non-thermalized, λ = 30 mm) | δΔ = −0.09 MHz |
| Lorentzian fit residual | < 10⁻⁴ of peak height |
| Monte Carlo seed-to-seed variation (20 seeds) | ±0.001 MHz |

---

## Requirements

- Python 3.8 or later
- `numpy`
- `scipy` (uses `optimize.brentq`, `optimize.curve_fit`, `interpolate.interp1d`, `special.j1`, `special.jvp`)
- `matplotlib` (for figure generation only)

Install the dependencies with:

```bash
pip install numpy scipy matplotlib
```

Typical runtime on a modern laptop: under 5 minutes for the full main pipeline.

---

## Usage

All scripts are self-contained and produce console output plus JSON / PNG files in the working directory.

### Main simulation (Mills + Ritter + Ishida)

```bash
cd paper1_v4
python run_corrected_simulation.py
```

Output: `corrected_results.json` containing δΔ_HFS for all three experiments and the Mills Ps-weighted mean field offset.

### Systematic-effect verifications

```bash
python fase2_investigations.py
```

Runs three independent cross-checks:
1. **TM₁₁₀ microwave-mode weighting** — confirms the 6.6 % correction (−0.014 MHz).
2. **Non-Hermitian singlet-annihilation cancellation** — demonstrates that the raw ~75 MHz shift cancels to < 0.001 MHz under baseline subtraction.
3. **Pressure-dependence curve** — propagates σ_r(n) ∝ 1/√n through a linear density extrapolation to verify survival of the bias.

### Regenerate manuscript figures

```bash
python generate_figures.py
```

Produces `fig1_zeeman_eigenvalues.png`, `resonance_curves_corrected.png`, and `ritter_c2_scan.png` matching Figs. 1–3 of the manuscript.

### Module self-tests

Each physics module can be invoked directly and runs its own verification suite:

```bash
python breit_rabi.py              # Verifies B=0 eigenvalues, m=±1 degeneracy, round-trip inversion
python magnetic_field_models.py   # Verifies uniform/quadrupolar/solenoid statistics
python ps_distributions.py        # Verifies sampling within cavity bounds
```

---

## Repository structure

| File | Purpose | Paper reference |
|---|---|---|
| `paper1_v4/01_paper_draft.tex` | LaTeX manuscript (REVTeX 4-2, PRA format) | — |
| `paper1_v4/02_paper_refs.bib` | Bibliography | — |
| `paper1_v4/breit_rabi.py` | 4×4 Hamiltonian diagonalization, transition frequency, Breit–Rabi inversion, cubic interpolator | §II.A, Eqs. (1)–(3) |
| `paper1_v4/magnetic_field_models.py` | Uniform, quadrupolar, combined, and finite-solenoid field models | §III.B (Mills), §IV (Ritter), §V (Ishida) |
| `paper1_v4/ps_distributions.py` | Ps spatial distributions (uniform, non-thermalized exponential, Gaussian core) | §III.B |
| `paper1_v4/zeeman_resonance.py` | Resonance-curve simulation, Lorentzian fit, HFS extraction | §III.C |
| `paper1_v4/run_corrected_simulation.py` | Main driver: Mills + Ritter scan + Ishida | Tables I, II |
| `paper1_v4/fase2_investigations.py` | TM₁₁₀, non-Hermitian, pressure-survival checks | §VII (ii)–(iv) |
| `paper1_v4/generate_figures.py` | Produces all three manuscript figures | Figs. 1–3 |
| `paper1_v4/corrected_results.json` | Cached numerical results from the main pipeline | Tables I, II |

---

## Reproducibility map

| Manuscript value | Script | Location in output |
|---|---|---|
| Mills: δΔ = −0.22 MHz | `run_corrected_simulation.py` | `corrected_results.json` → `mills_measured.delta_hfs_corrected` |
| Mills: ⟨δB/B₀⟩ = +0.52 ppm | same | `mills_measured.mean_dB_ppm` |
| Ritter (c₂ = 3 ppm): −0.18 MHz | same | `ritter_scan["3"].delta_hfs_corrected` |
| Ritter (c₂ = 5 ppm): −0.30 MHz | same | `ritter_scan["5"].delta_hfs_corrected` |
| Ritter (c₂ = 20 ppm): −1.19 MHz | same | `ritter_scan["20"].delta_hfs_corrected` |
| Ishida (uniform Ps): +0.30 MHz | same | `ishida_uniform` |
| Ishida (λ = 30 mm): −0.09 MHz | same | `ishida_nontherm` |
| Amplification factor 4.5 × 10⁵ MHz/T | `breit_rabi.py` | derived analytically in `transition_frequency_fast` |
| TM₁₁₀ correction: −0.014 MHz (6.6 %) | `fase2_investigations.py` | stdout block `#13` |
| Non-Hermitian residual < 0.001 MHz | `fase2_investigations.py` | stdout block `#15` |
| Pressure-survival curve | `fase2_investigations.py` | stdout block `#17` |
| Lorentzian fit residual < 10⁻⁴ | `generate_figures.py` | lower panel of `resonance_curves_corrected.png` |

---

## Physical constants

The code uses CODATA 2018 values (essentially identical to CODATA 2022 at the precision relevant here):

- g-factor of the free electron: `g_e = 2.002 319 304 362 56`
- Bohr magneton in frequency units: `μ_B / h = 13 996.244 93 MHz/T`
- QED theoretical HFS interval: `Δ_HFS^QED = 203 391.69 ± 0.41 MHz` (Adkins, Cassidy, Pérez-Ríos 2022 compilation)

All three are declared in `breit_rabi.py`.

---

## Citation

If you use this code, please cite the manuscript once it is available:

```bibtex
@article{Zermeno2026,
  author  = {Zerme\~no, Ernest Darell},
  title   = {Magnetic-field-inhomogeneity bias in indirect Zeeman measurements
             of the positronium hyperfine interval},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

This entry will be updated with the final journal reference upon publication.

---

## License

The code in this repository is released under the MIT License. See [LICENSE](LICENSE) for the full text.

The manuscript itself is subject to the policy of the journal in which it is published.

---

## Contact

Ernest Darell Zermeño
Universidad Panamericana, Guadalajara, México
`0244552@up.edu.mx`

Questions, bug reports, or reproducibility issues are welcome via the GitHub [Issues](https://github.com/Jefemaestro33/ps-hfs-field-inhomogeneity/issues) tab.
