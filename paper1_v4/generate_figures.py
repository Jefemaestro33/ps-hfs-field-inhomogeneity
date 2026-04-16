#!/usr/bin/env python3
"""
Generate all figures for paper v4.

Fig 1: Zeeman eigenvalues vs B
Fig 2: Mills resonance curves + residuals panel
Fig 3: Ritter c₂ scan
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from breit_rabi import (
    eigenvalues, eigensystem, _identify_states,
    transition_frequency_fast, extract_hfs, build_interpolator,
    DELTA_HFS, G_E, MU_B,
)
from magnetic_field_models import UniformField, QuadrupolarField, FiniteSolenoid
from ps_distributions import UniformPs, NonThermalizedPs
from zeeman_resonance import simulate_resonance, lorentzian

np.random.seed(42)
N_ATOMS = 500_000

# ============================================================
# Common setup
# ============================================================

class MeasuredGradientField:
    def __init__(self, B0, d2Br, d2Bz, name="Mills measured"):
        self.B0 = B0; self.d2Br = d2Br; self.d2Bz = d2Bz; self.name = name
    def B_field(self, r, z):
        return self.B0 * (1 + 0.5*self.d2Br*r**2 + 0.5*self.d2Bz*z**2)

class MillsDistribution:
    name = "Mills"
    def __init__(self):
        self.sigma_r = 2.44e-3; self.L = 12.7e-3
    def sample(self, N, R_cav=None, L_cav=None, rng=None):
        if rng is None: rng = np.random.default_rng()
        x = rng.normal(0, self.sigma_r, N)
        y = rng.normal(0, self.sigma_r, N)
        return np.sqrt(x**2 + y**2), rng.uniform(-self.L/2, self.L/2, N)

class RitterDistribution:
    name = "Ritter"
    def __init__(self):
        self.R = 12.7e-3; self.L = 38.1e-3
    def sample(self, N, R_cav=None, L_cav=None, rng=None):
        if rng is None: rng = np.random.default_rng()
        return self.R*np.sqrt(rng.uniform(0,1,N)), rng.uniform(-self.L/2, self.L/2, N)

def mills_weight(r, z):
    return np.maximum(1.0 - 140.0*np.abs(z), 0.0)

def find_B_res(nu_mw):
    return brentq(lambda B: transition_frequency_fast(B) - nu_mw, 0.1, 2.0)

def make_scan(B_center, gamma=5.0, N=200):
    dB = 1e-6
    dnu = (transition_frequency_fast(B_center+dB) - transition_frequency_fast(B_center-dB))/(2*dB)
    gB = gamma/dnu
    return np.linspace(B_center - 8*gB, B_center + 8*gB, N)


# ============================================================
# FIGURE 1: Zeeman eigenvalues
# ============================================================
print("Generating Fig 1: Zeeman eigenvalues...")

B_range = np.linspace(0, 1.5, 500)
E_all = np.zeros((500, 4))
for i, B in enumerate(B_range):
    E_all[i] = eigenvalues(B)

fig1, ax1 = plt.subplots(figsize=(8, 5))
D = DELTA_HFS

labels = [r'$E_-$ (mostly singlet)', r'$E_{m=-1} = \Delta/4$',
          r'$E_{m=+1} = \Delta/4$', r'$E_+$ (mostly triplet $m=0$)']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for j in range(4):
    ax1.plot(B_range, E_all[:, j]/1e3, color=colors[j], label=labels[j], linewidth=1.5)

B_mills = 0.925
E_mills = eigenvalues(B_mills)
ids = _identify_states(B_mills)
E_plus = E_mills[ids['+']]
E_m1 = E_mills[ids['m+1']]

ax1.annotate('', xy=(B_mills, E_plus/1e3), xytext=(B_mills, E_m1/1e3),
             arrowprops=dict(arrowstyle='<->', color='#9C27B0', lw=2))
ax1.text(B_mills + 0.04, (E_plus + E_m1)/2/1e3, r'$\nu_\mathrm{trans}$',
         fontsize=12, color='#9C27B0', va='center')

ax1.axvline(B_mills, color='gray', ls='--', alpha=0.5, lw=0.8)
ax1.text(B_mills, -180, f'$B_0 = {B_mills}$ T', fontsize=9, ha='center',
         color='gray')

ax1.set_xlabel('Magnetic field $B$ (T)', fontsize=13)
ax1.set_ylabel('Energy (GHz)', fontsize=13)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_xlim(0, 1.5)
ax1.tick_params(labelsize=11)
fig1.tight_layout()
fig1.savefig('fig1_zeeman_eigenvalues.png', dpi=200, bbox_inches='tight')
plt.close(fig1)
print("  → fig1_zeeman_eigenvalues.png")


# ============================================================
# FIGURE 2: Mills resonance curves + residuals
# ============================================================
print("Generating Fig 2: Mills resonance + residuals...")

mills_field = MeasuredGradientField(0.925, -0.21, +0.44)
mills_dist = MillsDistribution()
B_center = find_B_res(3253.0)
B_scan = make_scan(B_center, 5.0, 300)

res_sig = simulate_resonance(
    mills_field, mills_dist, 56.25e-3, 12.7e-3,
    3253.0, B_scan, 5.0, N_atoms=N_ATOMS,
    rng=np.random.default_rng(42), weight_fn=mills_weight)

res_bl = simulate_resonance(
    UniformField(0.925), mills_dist, 56.25e-3, 12.7e-3,
    3253.0, B_scan, 5.0, N_atoms=N_ATOMS,
    rng=np.random.default_rng(42), weight_fn=mills_weight)

delta_corrected = res_sig['delta_hfs_shift'] - res_bl['delta_hfs_shift']

# Convert B_scan to mT offset from center for readability
B_offset = (B_scan - B_center) * 1e6  # μT

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1],
                                   sharex=True, gridspec_kw={'hspace': 0.05})

# Upper panel: signal + fit + baseline
ax2a.plot(B_offset, res_sig['signal'], 'b-', linewidth=1.5, label='Non-uniform field')
ax2a.plot(B_offset, res_sig['fit_curve'], 'r--', linewidth=1.2, label='Lorentzian fit')
ax2a.plot(B_offset, res_bl['signal'], 'g-', linewidth=1.0, alpha=0.7, label='Uniform control')
ax2a.set_ylabel('Signal (arb. units)', fontsize=12)
ax2a.legend(fontsize=10, loc='upper right')
ax2a.text(0.03, 0.92, f'$\\delta\\Delta_\\mathrm{{HFS}} = {delta_corrected:.3f}$ MHz',
          transform=ax2a.transAxes, fontsize=11, va='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2a.tick_params(labelsize=10)

# Lower panel: residuals
signal_range = res_sig['signal'].max() - res_sig['signal'].min()
residuals_rel = res_sig['residuals'] / signal_range
ax2b.plot(B_offset, residuals_rel * 1e4, 'b-', linewidth=0.8)
ax2b.axhline(0, color='gray', ls='-', lw=0.5)
ax2b.set_xlabel(f'$B - B_\\mathrm{{res}}$ (μT)', fontsize=12)
ax2b.set_ylabel(r'Residuals ($\times 10^{-4}$)', fontsize=11)
ax2b.set_ylim(-2, 2)
ax2b.tick_params(labelsize=10)
max_res = np.max(np.abs(residuals_rel))
ax2b.text(0.03, 0.85, f'Max $|\\mathrm{{res}}| = {max_res:.1e}$',
          transform=ax2b.transAxes, fontsize=9, va='top')

fig2.tight_layout()
fig2.savefig('resonance_curves_corrected.png', dpi=200, bbox_inches='tight')
plt.close(fig2)
print("  → resonance_curves_corrected.png")


# ============================================================
# FIGURE 3: Ritter c₂ scan
# ============================================================
print("Generating Fig 3: Ritter c₂ scan...")

RITTER_R = 76.8e-3
RITTER_L = 38.1e-3
RITTER_R_REF = 66.7e-3 / 2
RITTER_NU = 2384.0
B_ritter = find_B_res(RITTER_NU)
ritter_dist = RitterDistribution()

c2_range = np.linspace(1, 25, 25)
deltas_ritter = []

for c2 in c2_range:
    mag = QuadrupolarField(B_ritter, c2_ppm=c2, R_ref=RITTER_R_REF)
    B_scan_r = make_scan(B_ritter, 5.0, 200)

    res_bl_r = simulate_resonance(
        UniformField(B_ritter), ritter_dist, RITTER_R, RITTER_L,
        RITTER_NU, B_scan_r, 5.0, N_atoms=N_ATOMS,
        rng=np.random.default_rng(42))
    res_r = simulate_resonance(
        mag, ritter_dist, RITTER_R, RITTER_L,
        RITTER_NU, B_scan_r, 5.0, N_atoms=N_ATOMS,
        rng=np.random.default_rng(42))

    deltas_ritter.append(res_r['delta_hfs_shift'] - res_bl_r['delta_hfs_shift'])

deltas_ritter = np.array(deltas_ritter)

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(c2_range, deltas_ritter, 'b-o', markersize=4, linewidth=1.5)

# Shading
ax3.axvspan(2, 5, alpha=0.15, color='green', label='Post-shimming (2–5 ppm)')
ax3.axvspan(8, 25, alpha=0.10, color='orange', label='Pre-shimming (>8 ppm)')

# Reference line: Ritter discrepancy
ax3.axhline(-2.59, color='gray', ls='--', lw=1.0, alpha=0.7)
ax3.text(24, -2.45, 'Ritter discrepancy ($-2.59$ MHz)', fontsize=9,
         ha='right', color='gray')

ax3.set_xlabel('Quadrupolar coefficient $c_2$ (ppm)', fontsize=13)
ax3.set_ylabel('$\\delta\\Delta_\\mathrm{HFS}$ (MHz)', fontsize=13)
ax3.legend(fontsize=10, loc='lower left')
ax3.tick_params(labelsize=11)
ax3.set_xlim(0, 26)

fig3.tight_layout()
fig3.savefig('ritter_c2_scan.png', dpi=200, bbox_inches='tight')
plt.close(fig3)
print("  → ritter_c2_scan.png")

print("\nAll figures generated.")
