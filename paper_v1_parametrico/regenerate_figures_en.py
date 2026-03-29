#!/usr/bin/env python3
"""Regenerate the 3 paper figures with all text in English."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.optimize import brentq

from breit_rabi import (
    eigenvalues, eigensystem, transition_frequency_fast,
    _identify_states, DELTA_HFS, G_E, MU_B,
)
from zeeman_resonance import lorentzian

# Load Paso 1 results
with open('paso1_results.json') as f:
    r1 = json.load(f)

# Load sensitivity data
with open('sensitivity_data.json') as f:
    sd = json.load(f)

B_CENTER = brentq(lambda B: transition_frequency_fast(B) - 2338.0, 0.5, 1.2)
BASELINE = r1['baselines']['mills_ritter']

# ================================================================
# FIG 1: Zeeman eigenvalues
# ================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))
B_range = np.linspace(0, 1.2, 2000)
evals_all = np.array([eigenvalues(B) for B in B_range])
labels_ev = [r'$E_-$ (singlet)', r'$E_{m=-1}$', r'$E_{m=+1}$', r'$E_+$ (triplet $m\!=\!0$)']
colors_ev = ['blue', 'green', 'orange', 'red']
for k in range(4):
    ax1.plot(B_range, evals_all[:, k] / 1e3, color=colors_ev[k],
             linewidth=2, label=labels_ev[k])

ax1.axvline(B_CENTER, color='gray', ls='--', alpha=0.5,
            label=f'$B_\\mathrm{{res}}$ Mills ({B_CENTER:.3f} T)')

# Mark transition arrow
evals_mark = eigenvalues(B_CENTER)
ids_mark = _identify_states(B_CENTER)
E_top = evals_mark[ids_mark['+']] / 1e3
E_m1 = evals_mark[ids_mark['m+1']] / 1e3
ax1.annotate('', xy=(B_CENTER + 0.03, E_top), xytext=(B_CENTER + 0.03, E_m1),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1.text(B_CENTER + 0.05, (E_top + E_m1) / 2,
         r'$\nu_\mathrm{trans} = 2338$ MHz',
         fontsize=10, color='purple', va='center')

ax1.set_xlabel('$B$ (T)', fontsize=13)
ax1.set_ylabel('Energy (GHz)', fontsize=13)
ax1.set_title('Zeeman eigenvalues of the positronium ground state', fontsize=14)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('fig1_zeeman_eigenvalues.png', dpi=150)
print('Fig 1 saved')
plt.close(fig1)

# ================================================================
# FIG 2: Resonance fit
# ================================================================
# Retrieve data from paso1_results
key_10 = 'c₂=10 ppm_Ps uniforme (termalizado)'
key_ctrl = 'Control_Ps uniforme (termalizado)'
res_10 = r1['mills_ritter'][key_10]
res_ctrl = r1['mills_ritter'][key_ctrl]

# We need the actual resonance curves. Re-run the two simulations.
from magnetic_field_models import UniformField, QuadrupolarField
from ps_distributions import UniformPs
from zeeman_resonance import simulate_resonance

dB = 1e-6
dnu = (transition_frequency_fast(B_CENTER + dB) - transition_frequency_fast(B_CENTER - dB)) / (2 * dB)
gB = 5.0 / dnu
B_scan = np.linspace(B_CENTER - 8 * gB, B_CENTER + 8 * gB, 300)

rng = np.random.default_rng(42)
res_ctrl_full = simulate_resonance(
    UniformField(B_CENTER, "Control"), UniformPs(),
    0.10, 0.20, 2338.0, B_scan, 5.0, N_atoms=500_000, rng=rng)

rng = np.random.default_rng(42)
res_10_full = simulate_resonance(
    QuadrupolarField(B_CENTER, c2_ppm=10, R_ref=0.15), UniformPs(),
    0.10, 0.20, 2338.0, B_scan, 5.0, N_atoms=500_000, rng=rng)

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                   sharex=True)

B_mT = B_scan * 1e3
sig_10 = res_10_full['signal']
sig_ctrl = res_ctrl_full['signal']
sig_10_n = sig_10 / sig_10.max()
sig_ctrl_n = sig_ctrl / sig_ctrl.max()

ax2a.plot(B_mT, sig_ctrl_n, 'g-', lw=1.5, alpha=0.7, label='Uniform field')
ax2a.plot(B_mT, sig_10_n, 'b-', lw=1.5, label='$c_2 = 10$ ppm')

fit_10 = lorentzian(B_scan, *res_10_full['fit_params'])
fit_10_n = fit_10 / sig_10.max()
ax2a.plot(B_mT, fit_10_n, 'r--', lw=2, alpha=0.7, label='Lorentzian fit')

ax2a.axvline(res_ctrl_full['B_res'] * 1e3, color='green', ls=':', alpha=0.5)
ax2a.axvline(res_10_full['B_res'] * 1e3, color='blue', ls=':', alpha=0.5)

dB_shift = (res_10_full['B_res'] - res_ctrl_full['B_res']) * 1e6
delta_hfs_shift = res_10_full['delta_hfs_shift']
ax2a.text(0.02, 0.95,
          f'$\\delta B_\\mathrm{{res}} = {dB_shift:.2f}\\;\\mu$T\n'
          f'$\\delta\\Delta_\\mathrm{{HFS}} = {delta_hfs_shift:.4f}$ MHz',
          transform=ax2a.transAxes, fontsize=11, va='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2a.set_ylabel('Normalized signal', fontsize=12)
ax2a.set_title('Zeeman resonance — Mills/Ritter, $c_2 = 10$ ppm, uniform Ps', fontsize=13)
ax2a.legend(fontsize=10)
ax2a.grid(True, alpha=0.3)

residuals = sig_10_n - fit_10_n
ax2b.plot(B_mT, residuals * 1e3, 'b-', lw=1)
ax2b.axhline(0, color='gray', ls='--')
ax2b.set_xlabel('$B_0$ (mT)', fontsize=12)
ax2b.set_ylabel(r'Residuals ($\times 10^3$)', fontsize=12)
ax2b.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig('fig2_resonance_fit.png', dpi=150)
print('Fig 2 saved')
plt.close(fig2)

# ================================================================
# FIG 3: Sensitivity map (c2, lambda)
# ================================================================
c2_vals = np.array(sd['c2_vals'])
lam_vals = np.array(sd['lam_vals'])
grid1 = np.array(sd['grid1_quadrupolar'])

DISCREPANCY = -3.04
DISC_LO, DISC_HI = -3.83, -2.25
C2_PLAUS = (8, 30)
LAM_PLAUS = (15, 50)
C2, LAM = np.meshgrid(c2_vals, lam_vals)

fig3, ax3 = plt.subplots(figsize=(9, 7))
pcm = ax3.pcolormesh(C2, LAM, grid1, cmap='RdBu_r', shading='gouraud',
                      norm=TwoSlopeNorm(vmin=grid1.min(), vcenter=DISCREPANCY / 2, vmax=0))

cs = ax3.contour(C2, LAM, grid1, levels=[DISC_LO, DISCREPANCY, DISC_HI],
                  colors=['gray', 'black', 'gray'],
                  linewidths=[1.5, 3, 1.5], linestyles=['--', '-', '--'])
ax3.clabel(cs, fmt='%.2f', fontsize=9)

rect = Rectangle((C2_PLAUS[0], LAM_PLAUS[0]),
                 C2_PLAUS[1] - C2_PLAUS[0], LAM_PLAUS[1] - LAM_PLAUS[0],
                 linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.15)
ax3.add_patch(rect)
ax3.text(C2_PLAUS[0] + 1, LAM_PLAUS[1] - 3, 'Plausible\nregion',
         fontsize=10, color='green', fontweight='bold')

ax3.set_xlabel('$c_2$ (ppm)', fontsize=13)
ax3.set_ylabel(r'$\lambda$ — thermalization length (mm)', fontsize=13)
ax3.set_title('HFS shift from magnetic field inhomogeneity\n'
              '(quadrupolar, Mills/Ritter cavity)', fontsize=14)
plt.colorbar(pcm, ax=ax3, label=r'$\delta\Delta_\mathrm{HFS}$ (MHz)')

ax3.text(0.98, 0.02, f'Exp. discrepancy: ${DISCREPANCY} \\pm 0.79$ MHz',
         transform=ax3.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

fig3.tight_layout()
fig3.savefig('sensitivity_c2_lambda.png', dpi=200)
print('Fig 3 saved')
plt.close(fig3)

print('All figures regenerated in English.')
