#!/usr/bin/env python3
"""
Corrected simulation with real experimental dimensions and measured gradients.

Mills 1975: R_cav=56.25mm, L_cav=12.7mm, B₀=0.925T, ν_MW=3253 MHz
            Gradients measured by NMR: d²B/dr²=-0.21 m⁻², d²B/dz²=+0.44 m⁻²
            Ps distribution: Gaussian σ=2.44mm (transverse), uniform L=12.7mm (axial)

Ritter 1984: R_cav=76.8mm, L_cav=38.1mm, B₀=0.79T, ν_MW=2384 MHz
             Varian V4012A electromagnet, gap=66.7mm, post-shimming ~0.5 ppm σ
             Ps confined to R≈12.7mm, uniform in z

Ishida 2014: R_cav=64mm, L_cav=100mm, B₀=0.866T, ν_MW=2856.6 MHz
             SC solenoid, 0.9 ppm RMS residual
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import brentq

from breit_rabi import (
    transition_frequency, transition_frequency_fast, extract_hfs,
    build_interpolator, DELTA_HFS, G_E, MU_B,
)
from magnetic_field_models import UniformField, QuadrupolarField, FiniteSolenoid
from ps_distributions import UniformPs, NonThermalizedPs
from zeeman_resonance import simulate_resonance, lorentzian

t_start = time.time()
np.random.seed(42)
N_ATOMS = 500_000
FIG_DIR = "./"

# ============================================================
# NEW FIELD AND DISTRIBUTION MODELS
# ============================================================

class MeasuredGradientField:
    """Field from NMR-measured gradients: B(r,z) = B0*(1 + d2Br*r²/2 + d2Bz*z²/2)."""
    def __init__(self, B0, d2Br, d2Bz, name="Measured gradients"):
        self.B0 = B0
        self.d2Br = d2Br  # m⁻², relative fraction
        self.d2Bz = d2Bz  # m⁻²
        self.name = name

    def B_field(self, r, z):
        return self.B0 * (1 + 0.5 * self.d2Br * r**2 + 0.5 * self.d2Bz * z**2)

    def statistics(self, r, z):
        B = self.B_field(r, z)
        delta = (B - self.B0) / self.B0
        std = np.std(delta)
        return {
            'mean_ppm': np.mean(delta) * 1e6,
            'rms_ppm': np.sqrt(np.mean(delta**2)) * 1e6,
            'max_ppm': np.max(np.abs(delta)) * 1e6,
            'std_ppm': std * 1e6,
        }


class MillsDistribution:
    """Mills Ps: Gaussian σ=2.44mm transverse, uniform L=12.7mm axial."""
    def __init__(self):
        self.name = "Mills (gauss σ=2.44mm, L=12.7mm)"
        self.sigma_r = 2.44e-3
        self.L = 12.7e-3

    def sample(self, N, R_cav=None, L_cav=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = rng.normal(0, self.sigma_r, N)
        y = rng.normal(0, self.sigma_r, N)
        r = np.sqrt(x**2 + y**2)
        z = rng.uniform(-self.L / 2, self.L / 2, N)
        return r, z


class RitterDistribution:
    """Ritter Ps: uniform in cylinder R=12.7mm, L=38.1mm."""
    def __init__(self):
        self.name = "Ritter (uniform R=12.7mm, L=38.1mm)"
        self.R = 12.7e-3
        self.L = 38.1e-3

    def sample(self, N, R_cav=None, L_cav=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        r = self.R * np.sqrt(rng.uniform(0, 1, N))
        z = rng.uniform(-self.L / 2, self.L / 2, N)
        return r, z


# ============================================================
# HELPER
# ============================================================

def find_B_resonance(nu_mw):
    return brentq(lambda B: transition_frequency_fast(B) - nu_mw, 0.1, 2.0)


def make_B_scan(B_center, gamma_nu=5.0, N=200):
    dB = 1e-6
    dnu = (transition_frequency_fast(B_center + dB) -
           transition_frequency_fast(B_center - dB)) / (2 * dB)
    gB = gamma_nu / dnu
    return np.linspace(B_center - 8 * gB, B_center + 8 * gB, N)


def run_with_baseline(magnet, ps_dist, R_cav, L_cav, nu_mw, gamma, label=""):
    """Run simulation and subtract uniform-field baseline."""
    B_center = find_B_resonance(nu_mw)
    B_scan = make_B_scan(B_center, gamma)

    # Baseline
    res_bl = simulate_resonance(
        UniformField(magnet.B0), ps_dist, R_cav, L_cav,
        nu_mw, B_scan, gamma, N_atoms=N_ATOMS, rng=np.random.default_rng(42))
    baseline = res_bl['delta_hfs_shift']

    # Signal
    res = simulate_resonance(
        magnet, ps_dist, R_cav, L_cav,
        nu_mw, B_scan, gamma, N_atoms=N_ATOMS, rng=np.random.default_rng(42))

    corrected = res['delta_hfs_shift'] - baseline

    # Field statistics over the Ps distribution
    r_s, z_s = ps_dist.sample(N_ATOMS, R_cav, L_cav, rng=np.random.default_rng(42))
    stats = magnet.statistics(r_s, z_s)

    return {
        'label': label or f"{magnet.name} / {ps_dist.name}",
        'delta_hfs_raw': res['delta_hfs_shift'],
        'baseline': baseline,
        'delta_hfs_corrected': corrected,
        'B_res': res['B_res'],
        'B_res_baseline': res_bl['B_res'],
        'mean_dB_ppm': stats['mean_ppm'],
        'rms_dB_ppm': stats['rms_ppm'],
        'signal': res['signal'],
        'signal_bl': res_bl['signal'],
        'B0_scan': B_scan,
        'fit_params': res['fit_params'],
        'fit_params_bl': res_bl['fit_params'],
    }


# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

# Mills 1975
mills_field = MeasuredGradientField(
    B0=0.925, d2Br=-0.21, d2Bz=+0.44,
    name="Mills measured gradients")
mills_dist = MillsDistribution()
MILLS_R = 56.25e-3
MILLS_L = 12.7e-3
MILLS_NU = 3253.0
MILLS_GAMMA = 5.0

# Ritter 1984
RITTER_R = 76.8e-3
RITTER_L = 38.1e-3
RITTER_R_REF = 66.7e-3 / 2  # half gap
RITTER_NU = 2384.0
RITTER_GAMMA = 5.0
ritter_dist = RitterDistribution()

# Ishida 2014
ISHIDA_R = 64e-3
ISHIDA_L = 100e-3
ISHIDA_NU = 2856.6
ISHIDA_GAMMA = 3.0

# ============================================================
# VERIFICATIONS
# ============================================================

print("=" * 72)
print("CORRECTED SIMULATION — REAL EXPERIMENTAL DIMENSIONS")
print("=" * 72)

B_mills = find_B_resonance(MILLS_NU)
B_ritter = find_B_resonance(RITTER_NU)
B_ishida = find_B_resonance(ISHIDA_NU)

print(f"\n--- Verification: resonance fields ---")
print(f"  Mills:  B_res = {B_mills:.4f} T  (expected ~0.925)")
print(f"  Ritter: B_res = {B_ritter:.4f} T  (expected ~0.79)")
print(f"  Ishida: B_res = {B_ishida:.4f} T  (expected ~0.866)")

nu_check_m = transition_frequency_fast(0.925)
nu_check_r = transition_frequency_fast(0.79)
nu_check_i = transition_frequency_fast(0.866)
print(f"\n--- Verification: transition frequencies ---")
print(f"  ν(0.925 T) = {nu_check_m:.1f} MHz  (Mills used 3253)")
print(f"  ν(0.790 T) = {nu_check_r:.1f} MHz  (Ritter used 2384)")
print(f"  ν(0.866 T) = {nu_check_i:.1f} MHz  (Ishida used 2856.6)")

# Check <δB/B₀> for Mills field + Mills distribution
r_test, z_test = mills_dist.sample(N_ATOMS, MILLS_R, MILLS_L,
                                    rng=np.random.default_rng(42))
stats_mills = mills_field.statistics(r_test, z_test)
print(f"\n--- Verification: Mills field statistics over Mills Ps ---")
print(f"  <δB/B₀> = {stats_mills['mean_ppm']:.3f} ppm  (expected ~+0.3 to +0.6)")
print(f"  σ(δB/B₀) = {stats_mills['rms_ppm']:.3f} ppm")

# ============================================================
# PART 0: BASELINE — REPRODUCE PAPER NUMBERS
# ============================================================

print(f"\n{'='*72}")
print("PART 0: BASELINE — paper dimensions (R=100mm, L=200mm)")
print(f"{'='*72}")

# Original paper used B₀=0.783T, ν_MW=2338 MHz
B_paper = find_B_resonance(2338.0)
paper_c10_unif = run_with_baseline(
    QuadrupolarField(B_paper, c2_ppm=10, R_ref=0.15),
    UniformPs(), 0.10, 0.20, 2338.0, 5.0,
    label="Paper: c₂=10, uniform, R=100mm")

paper_c10_nont = run_with_baseline(
    QuadrupolarField(B_paper, c2_ppm=10, R_ref=0.15),
    NonThermalizedPs(30), 0.10, 0.20, 2338.0, 5.0,
    label="Paper: c₂=10, λ=30mm, R=100mm")

print(f"  c₂=10 ppm, uniform:  δΔ_HFS = {paper_c10_unif['delta_hfs_corrected']:.4f} MHz "
      f"(paper: -0.31)")
print(f"  c₂=10 ppm, λ=30mm:   δΔ_HFS = {paper_c10_nont['delta_hfs_corrected']:.4f} MHz "
      f"(paper: -1.20)")

# ============================================================
# PART 1: MILLS — MEASURED GRADIENTS
# ============================================================

print(f"\n{'='*72}")
print("PART 1: MILLS 1975 — measured NMR gradients")
print(f"{'='*72}")

res_mills_real = run_with_baseline(
    mills_field, mills_dist, MILLS_R, MILLS_L, MILLS_NU, MILLS_GAMMA,
    label="A) Mills gradients + Mills dist.")

res_mills_unif = run_with_baseline(
    mills_field, UniformPs(), MILLS_R, MILLS_L, MILLS_NU, MILLS_GAMMA,
    label="B) Mills gradients + uniform")

print(f"  A) Mills grad + Mills dist:  <δB>={res_mills_real['mean_dB_ppm']:.3f} ppm, "
      f"δΔ_HFS = {res_mills_real['delta_hfs_corrected']:.4f} MHz")
print(f"  B) Mills grad + uniform:     <δB>={res_mills_unif['mean_dB_ppm']:.3f} ppm, "
      f"δΔ_HFS = {res_mills_unif['delta_hfs_corrected']:.4f} MHz")
print(f"     (NOTE: uniform Ps fills full cavity R=56mm where gradients are huge — not physical)")

# ============================================================
# PART 2: RITTER — QUADRUPOLAR SCAN
# ============================================================

print(f"\n{'='*72}")
print("PART 2: RITTER 1984 — quadrupolar c₂ scan")
print(f"{'='*72}")

B_ritter_nom = find_B_resonance(RITTER_NU)
c2_ritter_vals = [2, 3, 5, 8, 10, 15, 20]
ritter_results = {}

print(f"\n  {'c₂ (ppm)':>10} {'<δB> (ppm)':>12} {'δΔ_HFS (MHz)':>14} {'Dist':>30}")
print("  " + "-" * 70)

for c2 in c2_ritter_vals:
    mag = QuadrupolarField(B_ritter_nom, c2_ppm=c2, R_ref=RITTER_R_REF)
    res = run_with_baseline(
        mag, ritter_dist, RITTER_R, RITTER_L, RITTER_NU, RITTER_GAMMA,
        label=f"Ritter c₂={c2}")
    ritter_results[c2] = res
    print(f"  {c2:>10} {res['mean_dB_ppm']:>12.3f} {res['delta_hfs_corrected']:>14.4f} "
          f"{'Ritter dist (R=12.7,L=38.1)':>30}")

# Ritter with non-thermalized Ps — use Ritter cavity dimensions
# Note: NonThermalizedPs uses R_cav and L_cav passed to sample().
# We must pass Ritter's actual Ps confinement region (R=12.7mm, L=38.1mm).
print()
for lam in [30, 50]:
    mag = QuadrupolarField(B_ritter_nom, c2_ppm=5, R_ref=RITTER_R_REF)
    dist_nt = NonThermalizedPs(lambda_mm=lam)
    # Pass Ritter Ps region dimensions, not full cavity
    R_ps = 12.7e-3
    L_ps = 38.1e-3
    res = run_with_baseline(
        mag, dist_nt, R_ps, L_ps, RITTER_NU, RITTER_GAMMA,
        label=f"Ritter c₂=5, λ={lam}mm")
    ritter_results[f"c2=5_lam={lam}"] = res
    print(f"  {'5':>10} {res['mean_dB_ppm']:>12.3f} {res['delta_hfs_corrected']:>14.4f} "
          f"{f'NonTherm λ={lam}mm (R=12.7mm)':>30}")

# ============================================================
# PART 3: ISHIDA
# ============================================================

print(f"\n{'='*72}")
print("PART 3: ISHIDA 2014 — superconductor verification")
print(f"{'='*72}")

B_ishida_nom = find_B_resonance(ISHIDA_NU)
ishida_mag = FiniteSolenoid(B_ishida_nom, bore_radius=0.4, length=2.0,
                            residual_rms_ppm=0.9, name="Ishida SC")

res_ishida_unif = run_with_baseline(
    ishida_mag, UniformPs(), ISHIDA_R, ISHIDA_L, ISHIDA_NU, ISHIDA_GAMMA,
    label="Ishida SC + uniform")

res_ishida_nont = run_with_baseline(
    ishida_mag, NonThermalizedPs(30), ISHIDA_R, ISHIDA_L, ISHIDA_NU, ISHIDA_GAMMA,
    label="Ishida SC + λ=30mm")

print(f"  Uniform:  δΔ_HFS = {res_ishida_unif['delta_hfs_corrected']:.4f} MHz")
print(f"  λ=30mm:   δΔ_HFS = {res_ishida_nont['delta_hfs_corrected']:.4f} MHz")
ok_ish = max(abs(res_ishida_unif['delta_hfs_corrected']),
             abs(res_ishida_nont['delta_hfs_corrected'])) < 0.2
print(f"  |δΔ_HFS| < 0.2 MHz: [{'PASS' if ok_ish else 'FAIL'}]")

# ============================================================
# SUMMARY TABLES
# ============================================================

print(f"\n{'='*72}")
print("TABLE 1: PAPER ORIGINAL vs CORRECTED DIMENSIONS")
print(f"{'='*72}")
print(f"\n  {'Config':<45} {'δΔ_HFS (MHz)':>14}")
print("  " + "-" * 62)
print(f"  {'Paper: c₂=10, uniform, R=100mm':<45} {paper_c10_unif['delta_hfs_corrected']:>14.4f}")
print(f"  {'Paper: c₂=10, λ=30mm, R=100mm':<45} {paper_c10_nont['delta_hfs_corrected']:>14.4f}")
print(f"  {'Mills measured + Mills dist':<45} {res_mills_real['delta_hfs_corrected']:>14.4f}")
print(f"  {'Ritter c₂=10, Ritter dist':<45} {ritter_results[10]['delta_hfs_corrected']:>14.4f}")

print(f"\n{'='*72}")
print("TABLE 2: CORRECTED RESULTS — ALL CONFIGURATIONS")
print(f"{'='*72}")
print(f"\n  {'Config':<45} {'<δB>(ppm)':>10} {'δΔ_HFS(MHz)':>12}")
print("  " + "-" * 70)
print(f"  {'A) Mills grad + Mills dist':<45} "
      f"{res_mills_real['mean_dB_ppm']:>10.3f} {res_mills_real['delta_hfs_corrected']:>12.4f}")
print(f"  {'B) Mills grad + uniform':<45} "
      f"{res_mills_unif['mean_dB_ppm']:>10.3f} {res_mills_unif['delta_hfs_corrected']:>12.4f}")
for c2 in c2_ritter_vals:
    r = ritter_results[c2]
    print(f"  {f'C) Ritter c₂={c2} + Ritter dist':<45} "
          f"{r['mean_dB_ppm']:>10.3f} {r['delta_hfs_corrected']:>12.4f}")
for lam in [30, 50]:
    r = ritter_results[f"c2=5_lam={lam}"]
    print(f"  {f'D) Ritter c₂=5 + λ={lam}mm':<45} "
          f"{r['mean_dB_ppm']:>10.3f} {r['delta_hfs_corrected']:>12.4f}")
print(f"  {'F) Ishida SC + uniform':<45} "
      f"{res_ishida_unif['mean_dB_ppm']:>10.3f} {res_ishida_unif['delta_hfs_corrected']:>12.4f}")
print(f"  {'G) Ishida SC + λ=30mm':<45} "
      f"{res_ishida_nont['mean_dB_ppm']:>10.3f} {res_ishida_nont['delta_hfs_corrected']:>12.4f}")

print(f"\n{'='*72}")
print("TABLE 3: CONFRONTATION WITH EXPERIMENTAL DISCREPANCIES")
print(f"{'='*72}")

dHFS_mills = res_mills_real['delta_hfs_corrected']
# For Ritter, use c₂=3-5 ppm as most plausible post-shimming range
dHFS_ritter_lo = ritter_results[3]['delta_hfs_corrected']
dHFS_ritter_hi = ritter_results[5]['delta_hfs_corrected']
dHFS_ishida = res_ishida_unif['delta_hfs_corrected']

print(f"\n  {'Experiment':<14} {'Δ_HFS':>12} {'QED':>12} {'Disc.':>8} "
      f"{'δΔ_HFS B-inhom':>16} {'Residual':>10} {'%expl':>8}")
print("  " + "-" * 85)
print(f"  {'Mills 1975':<14} {'203387.5':>12} {'203391.69':>12} {'-4.19':>8} "
      f"{dHFS_mills:>16.3f} {-4.19 - dHFS_mills:>10.3f} "
      f"{abs(dHFS_mills)/4.19*100:>7.1f}%")
print(f"  {'Ritter 1984':<14} {'203389.10':>12} {'203391.69':>12} {'-2.59':>8} "
      f"{dHFS_ritter_lo:>7.3f} to {dHFS_ritter_hi:>6.3f} "
      f"{-2.59 - (dHFS_ritter_lo+dHFS_ritter_hi)/2:>10.3f} "
      f"{abs((dHFS_ritter_lo+dHFS_ritter_hi)/2)/2.59*100:>7.1f}%")
print(f"  {'Ishida 2014':<14} {'203394.2':>12} {'203391.69':>12} {'+2.51':>8} "
      f"{dHFS_ishida:>16.4f} {2.51 - dHFS_ishida:>10.3f} "
      f"{'~0':>8}")

# ============================================================
# FIGURES
# ============================================================

print(f"\nGenerating figures...")

# --- Fig A: Field map for Mills ---
fig_a, ax_a = plt.subplots(figsize=(8, 4))
r_grid = np.linspace(0, 15e-3, 200)  # 0 to 15 mm
z_grid = np.linspace(-8e-3, 8e-3, 200)  # ±8 mm
R_mg, Z_mg = np.meshgrid(r_grid, z_grid)
B_map = mills_field.B_field(R_mg, Z_mg)
dB_ppm = (B_map - mills_field.B0) / mills_field.B0 * 1e6

pcm = ax_a.pcolormesh(R_mg * 1e3, Z_mg * 1e3, dB_ppm, cmap='RdBu_r',
                        shading='gouraud')
# Cavity outline
from matplotlib.patches import Rectangle as Rect
cav_rect = Rect((0, -MILLS_L / 2 * 1e3), MILLS_R * 1e3, MILLS_L * 1e3,
                linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
ax_a.add_patch(cav_rect)
# Ps distribution contours (1σ, 2σ)
theta = np.linspace(0, 2 * np.pi, 100)
for nsig in [1, 2]:
    r_sig = nsig * 2.44  # mm
    ax_a.plot(r_sig * np.ones_like(theta[:50]),
              np.linspace(-MILLS_L / 2 * 1e3, MILLS_L / 2 * 1e3, 50),
              'g-', linewidth=1.5, alpha=0.7)
ax_a.text(2.44 + 0.3, 0, r'$1\sigma$', color='green', fontsize=9)
ax_a.text(4.88 + 0.3, 0, r'$2\sigma$', color='green', fontsize=9)

plt.colorbar(pcm, ax=ax_a, label=r'$\delta B/B_0$ (ppm)')
ax_a.set_xlabel('$r$ (mm)', fontsize=12)
ax_a.set_ylabel('$z$ (mm)', fontsize=12)
ax_a.set_title('Mills 1975: measured field map with Ps distribution', fontsize=13)
ax_a.set_xlim(0, 15)
fig_a.tight_layout()
fig_a.savefig(FIG_DIR + "field_map_mills.png", dpi=200)
print("  Fig A: field_map_mills.png")

# --- Fig B: Resonance curves ---
fig_b, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(10, 8))

# Mills panel
B_mT = res_mills_real['B0_scan'] * 1e3
sig = res_mills_real['signal']
sig_bl = res_mills_real['signal_bl']
ax_b1.plot(B_mT, sig / sig.max(), 'b-', lw=1.5, label='Non-uniform field')
ax_b1.plot(B_mT, sig_bl / sig_bl.max(), 'g:', lw=2, label='Uniform field')
fit = lorentzian(res_mills_real['B0_scan'], *res_mills_real['fit_params'])
ax_b1.plot(B_mT, fit / sig.max(), 'r--', lw=2, alpha=0.7, label='Lorentzian fit')
ax_b1.axvline(res_mills_real['B_res'] * 1e3, color='blue', ls=':', alpha=0.4)
ax_b1.axvline(res_mills_real['B_res_baseline'] * 1e3, color='green', ls=':', alpha=0.4)
ax_b1.text(0.02, 0.95,
           f"$\\delta\\Delta_\\mathrm{{HFS}} = {res_mills_real['delta_hfs_corrected']:.4f}$ MHz",
           transform=ax_b1.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_b1.set_ylabel('Normalized signal', fontsize=12)
ax_b1.set_title('Mills 1975: measured gradients + real Ps distribution', fontsize=13)
ax_b1.legend(fontsize=10)
ax_b1.grid(True, alpha=0.3)

# Ritter panel (c₂=5 ppm)
res_r5 = ritter_results[5]
B_mT_r = res_r5['B0_scan'] * 1e3
sig_r = res_r5['signal']
sig_bl_r = res_r5['signal_bl']
ax_b2.plot(B_mT_r, sig_r / sig_r.max(), 'b-', lw=1.5, label='$c_2 = 5$ ppm')
ax_b2.plot(B_mT_r, sig_bl_r / sig_bl_r.max(), 'g:', lw=2, label='Uniform field')
fit_r = lorentzian(res_r5['B0_scan'], *res_r5['fit_params'])
ax_b2.plot(B_mT_r, fit_r / sig_r.max(), 'r--', lw=2, alpha=0.7, label='Lorentzian fit')
ax_b2.text(0.02, 0.95,
           f"$\\delta\\Delta_\\mathrm{{HFS}} = {res_r5['delta_hfs_corrected']:.4f}$ MHz",
           transform=ax_b2.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_b2.set_xlabel('$B_0$ (mT)', fontsize=12)
ax_b2.set_ylabel('Normalized signal', fontsize=12)
ax_b2.set_title('Ritter 1984: $c_2 = 5$ ppm + real Ps distribution', fontsize=13)
ax_b2.legend(fontsize=10)
ax_b2.grid(True, alpha=0.3)

fig_b.tight_layout()
fig_b.savefig(FIG_DIR + "resonance_curves_corrected.png", dpi=200)
print("  Fig B: resonance_curves_corrected.png")

# --- Fig C: Ritter c₂ scan ---
fig_c, ax_c = plt.subplots(figsize=(9, 6))
c2_arr = np.array(c2_ritter_vals)
dHFS_arr = np.array([ritter_results[c2]['delta_hfs_corrected'] for c2 in c2_ritter_vals])
ax_c.plot(c2_arr, dHFS_arr, 'bo-', lw=2, markersize=7, label='Ritter dist. (uniform)')

# Non-thermalized points at c₂=5
for lam in [30, 50]:
    r = ritter_results[f"c2=5_lam={lam}"]
    ax_c.plot(5, r['delta_hfs_corrected'], 's', markersize=10,
              label=f'$\\lambda = {lam}$ mm (at $c_2=5$)')

ax_c.axhline(-2.59, color='red', ls='--', lw=2, alpha=0.5,
             label='Ritter discrepancy ($-2.59$ MHz)')
ax_c.axhline(0, color='gray', ls='-', lw=0.5)
ax_c.axvspan(2, 5, alpha=0.1, color='green', label='Post-shimming range')
ax_c.axvspan(8, 15, alpha=0.1, color='orange', label='Pre-shimming range')

ax_c.set_xlabel('$c_2$ (ppm)', fontsize=13)
ax_c.set_ylabel(r'$\delta\Delta_\mathrm{HFS}$ (MHz)', fontsize=13)
ax_c.set_title('Ritter 1984: HFS shift vs quadrupolar coefficient\n'
               '(real cavity dimensions)', fontsize=14)
ax_c.legend(fontsize=9, loc='lower left')
ax_c.grid(True, alpha=0.3)
ax_c.set_xlim(0, 22)
fig_c.tight_layout()
fig_c.savefig(FIG_DIR + "ritter_c2_scan.png", dpi=200)
print("  Fig C: ritter_c2_scan.png")

# --- Fig D: Sensitivity map for Ritter dimensions ---
print("  Computing sensitivity grid for Ritter (15×15)...")
c2_grid = np.linspace(2, 25, 15)
lam_grid = np.linspace(10, 60, 15)
grid_ritter = np.zeros((len(lam_grid), len(c2_grid)))

for i, lam in enumerate(lam_grid):
    for j, c2 in enumerate(c2_grid):
        mag = QuadrupolarField(B_ritter_nom, c2_ppm=c2, R_ref=RITTER_R_REF)
        dist = NonThermalizedPs(lambda_mm=lam)
        res = run_with_baseline(
            mag, dist, RITTER_R, RITTER_L, RITTER_NU, RITTER_GAMMA,
            label=f"grid {c2},{lam}")
        grid_ritter[i, j] = res['delta_hfs_corrected']
    if (i + 1) % 5 == 0:
        print(f"    Row {i+1}/{len(lam_grid)}")

fig_d, ax_d = plt.subplots(figsize=(9, 7))
C2G, LAMG = np.meshgrid(c2_grid, lam_grid)
vmin_d = min(grid_ritter.min(), -0.01)
vmax_d = max(grid_ritter.max(), 0.01)
vcenter_d = 0 if vmin_d < 0 < vmax_d else (vmin_d + vmax_d) / 2
# Ensure ascending order
if vmin_d >= vcenter_d or vcenter_d >= vmax_d:
    vcenter_d = (vmin_d + vmax_d) / 2
pcm_d = ax_d.pcolormesh(C2G, LAMG, grid_ritter, cmap='RdBu_r', shading='gouraud',
                          norm=TwoSlopeNorm(vmin=vmin_d, vcenter=vcenter_d, vmax=vmax_d))
cs_d = ax_d.contour(C2G, LAMG, grid_ritter,
                     levels=sorted([-2.59 - 0.74, -2.59, -2.59 + 0.74]),
                     colors=['gray', 'black', 'gray'],
                     linewidths=[1.5, 3, 1.5], linestyles=['--', '-', '--'])
ax_d.clabel(cs_d, fmt='%.2f', fontsize=9)
plt.colorbar(pcm_d, ax=ax_d, label=r'$\delta\Delta_\mathrm{HFS}$ (MHz)')
ax_d.set_xlabel('$c_2$ (ppm)', fontsize=13)
ax_d.set_ylabel(r'$\lambda$ (mm)', fontsize=13)
ax_d.set_title('Sensitivity map — Ritter dimensions\n'
               '($R_{\\rm cav}=76.8$ mm, $L_{\\rm cav}=38.1$ mm)', fontsize=14)
ax_d.text(0.98, 0.02, 'Ritter discrepancy: $-2.59 \\pm 0.74$ MHz',
          transform=ax_d.transAxes, fontsize=10, ha='right', va='bottom',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
fig_d.tight_layout()
fig_d.savefig(FIG_DIR + "sensitivity_corrected.png", dpi=200)
print("  Fig D: sensitivity_corrected.png")

# --- Fig E: Comparison old vs corrected ---
fig_e, ax_e = plt.subplots(figsize=(10, 5))
labels_e = ['c₂=10\nuniform\nR=100mm', 'c₂=10\nλ=30mm\nR=100mm',
            'Mills\nmeasured\ngradients', 'Ritter\nc₂=5\nreal dims',
            'Ritter\nc₂=10\nreal dims']
old_vals = [paper_c10_unif['delta_hfs_corrected'],
            paper_c10_nont['delta_hfs_corrected'],
            0, 0, 0]  # no old values for new configs
new_vals = [paper_c10_unif['delta_hfs_corrected'],
            paper_c10_nont['delta_hfs_corrected'],
            res_mills_real['delta_hfs_corrected'],
            ritter_results[5]['delta_hfs_corrected'],
            ritter_results[10]['delta_hfs_corrected']]

x_pos = np.arange(len(labels_e))
ax_e.bar(x_pos, new_vals, color=['steelblue'] * 2 + ['coral'] * 3, width=0.6)
ax_e.set_xticks(x_pos)
ax_e.set_xticklabels(labels_e, fontsize=9)
ax_e.set_ylabel(r'$\delta\Delta_\mathrm{HFS}$ (MHz)', fontsize=12)
ax_e.set_title('Comparison: paper dimensions vs corrected', fontsize=13)
ax_e.axhline(0, color='gray', lw=0.5)
ax_e.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(new_vals):
    ax_e.text(i, v - 0.02 if v < 0 else v + 0.02, f'{v:.3f}', ha='center',
              fontsize=9, fontweight='bold')
fig_e.tight_layout()
fig_e.savefig(FIG_DIR + "comparison_old_vs_corrected.png", dpi=200)
print("  Fig E: comparison_old_vs_corrected.png")

plt.close('all')

# ============================================================
# SAVE DATA
# ============================================================

output = {
    'mills_measured': {
        'delta_hfs_corrected': res_mills_real['delta_hfs_corrected'],
        'mean_dB_ppm': res_mills_real['mean_dB_ppm'],
    },
    'mills_uniform': {
        'delta_hfs_corrected': res_mills_unif['delta_hfs_corrected'],
        'mean_dB_ppm': res_mills_unif['mean_dB_ppm'],
    },
    'ritter_scan': {c2: {
        'delta_hfs_corrected': ritter_results[c2]['delta_hfs_corrected'],
        'mean_dB_ppm': ritter_results[c2]['mean_dB_ppm'],
    } for c2 in c2_ritter_vals},
    'ritter_nontherm': {lam: {
        'delta_hfs_corrected': ritter_results[f"c2=5_lam={lam}"]['delta_hfs_corrected'],
    } for lam in [30, 50]},
    'ishida_uniform': res_ishida_unif['delta_hfs_corrected'],
    'ishida_nontherm': res_ishida_nont['delta_hfs_corrected'],
    'sensitivity_grid_ritter': {
        'c2_vals': c2_grid.tolist(),
        'lam_vals': lam_grid.tolist(),
        'grid': grid_ritter.tolist(),
    },
}
with open(FIG_DIR + "corrected_results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nData saved: corrected_results.json")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.0f} s ({elapsed/60:.1f} min)")
print("=" * 72)
