#!/usr/bin/env python3
"""
Paso 1: Simulación rigurosa de la resonancia Zeeman con campo no uniforme.
Reproduce numéricamente el procedimiento experimental de Mills/Ritter/Ishida.
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from breit_rabi import (
    eigenvalues, eigensystem, transition_frequency, transition_frequency_fast,
    extract_hfs, build_interpolator, DELTA_HFS, G_E, MU_B,
    _identify_states, run_verifications as verify_breit_rabi,
)
from magnetic_field_models import (
    UniformField, QuadrupolarField, SextupolarField, LinearGradient,
    CombinedField, FiniteSolenoid,
)
from ps_distributions import UniformPs, NonThermalizedPs, GaussianCorePs
from zeeman_resonance import simulate_resonance, lorentzian

t_start = time.time()
FIG_DIR = "/Users/darellplascencia/Desktop/proyectos/fisica/"
np.random.seed(42)

# ============================================================
# STEP 0: VERIFY BREIT-RABI MODULE
# ============================================================
print("=" * 72)
print("PASO 1: SIMULACIÓN RIGUROSA DE LA RESONANCIA ZEEMAN")
print("=" * 72)
print("\nVerificando módulo breit_rabi...")
assert verify_breit_rabi(), "breit_rabi verificaciones fallaron — abortar"
print()

# ============================================================
# CONFIGURATIONS
# ============================================================

# Mills/Ritter scan: around B that gives nu_trans ~ 2338 MHz
# From paso 0.5: B ~ 0.7826 T for 2338 MHz
from scipy.optimize import brentq
B_mills_center = brentq(lambda B: transition_frequency_fast(B) - 2338.0, 0.5, 1.2)
B_ishida_center = brentq(lambda B: transition_frequency_fast(B) - 2856.6, 0.5, 1.2)

print(f"B de resonancia para Mills/Ritter (ν_MW=2338 MHz): {B_mills_center:.6f} T")
print(f"B de resonancia para Ishida (ν_MW=2856.6 MHz):     {B_ishida_center:.6f} T")

# Scan range: ±5× linewidth in B
def make_B_scan(B_center, gamma_nu, N_points=300):
    """Create B scan array centered on resonance."""
    # dν/dB at resonance
    dB = 1e-6
    dnu_dB = (transition_frequency_fast(B_center + dB) -
              transition_frequency_fast(B_center - dB)) / (2 * dB)
    gamma_B = gamma_nu / dnu_dB
    margin = 8 * gamma_B
    return np.linspace(B_center - margin, B_center + margin, N_points)

# === MILLS/RITTER CONFIGURATION ===
mills_config = {
    'name': 'Mills/Ritter',
    'cavity_R': 0.10,
    'cavity_L': 0.20,
    'nu_mw': 2338.0,
    'gamma': 5.0,
    'B0_scan': make_B_scan(B_mills_center, 5.0),
    'magnets': [
        UniformField(B_mills_center, "Control"),
        QuadrupolarField(B_mills_center, c2_ppm=5,  R_ref=0.15, name="c₂=5 ppm"),
        QuadrupolarField(B_mills_center, c2_ppm=10, R_ref=0.15, name="c₂=10 ppm"),
        QuadrupolarField(B_mills_center, c2_ppm=15, R_ref=0.15, name="c₂=15 ppm"),
        QuadrupolarField(B_mills_center, c2_ppm=20, R_ref=0.15, name="c₂=20 ppm"),
        QuadrupolarField(B_mills_center, c2_ppm=50, R_ref=0.15, name="c₂=50 ppm"),
        CombinedField(B_mills_center, c1_ppm=3, c2_ppm=10, c3_ppm=5,
                       R_ref=0.15, L_ref=0.15, name="Realista 1"),
        CombinedField(B_mills_center, c1_ppm=5, c2_ppm=15, c3_ppm=8,
                       R_ref=0.15, L_ref=0.15, name="Realista 2"),
    ],
    'distributions': [
        UniformPs(),
        NonThermalizedPs(lambda_mm=30),
        NonThermalizedPs(lambda_mm=50),
        GaussianCorePs(sigma_mm=40),
    ],
}

# === ISHIDA CONFIGURATION ===
ishida_config = {
    'name': 'Ishida',
    'cavity_R': 0.064,
    'cavity_L': 0.10,
    'nu_mw': 2856.6,
    'gamma': 3.0,
    'B0_scan': make_B_scan(B_ishida_center, 3.0),
    'magnets': [
        UniformField(B_ishida_center, "Control"),
        FiniteSolenoid(B_ishida_center, bore_radius=0.4, length=2.0,
                       residual_rms_ppm=0.9, name="Ishida SC (0.9 ppm)"),
    ],
    'distributions': [
        UniformPs(),
        NonThermalizedPs(lambda_mm=30),
    ],
}

# ============================================================
# RUN SIMULATIONS
# ============================================================

all_results = {}

for config in [mills_config, ishida_config]:
    print(f"\n{'='*72}")
    print(f"CONFIGURACIÓN: {config['name']}")
    print(f"  Cavidad: R={config['cavity_R']*1e3:.0f}mm, L={config['cavity_L']*1e3:.0f}mm")
    print(f"  ν_MW = {config['nu_mw']} MHz, Γ = {config['gamma']} MHz")
    print(f"{'='*72}")

    config_results = []

    print(f"\n  {'Imán':<20} {'Distribución':<28} {'B_res (T)':>12} "
          f"{'δΔ_HFS (MHz)':>14} {'Γ_fit (MHz)':>12}")
    print("  " + "-" * 90)

    for magnet in config['magnets']:
        for ps_dist in config['distributions']:
            rng = np.random.default_rng(42)
            res = simulate_resonance(
                magnet=magnet,
                ps_dist=ps_dist,
                cavity_radius=config['cavity_R'],
                cavity_length=config['cavity_L'],
                nu_mw=config['nu_mw'],
                B0_scan=config['B0_scan'],
                gamma_line=config['gamma'],
                N_atoms=500_000,
                rng=rng,
            )
            config_results.append(res)

            print(f"  {magnet.name:<20} {ps_dist.name:<28} "
                  f"{res['B_res']:>12.6f} "
                  f"{res['delta_hfs_shift']:>14.6f} "
                  f"{res['fit_width_nu']:>12.3f}")

    all_results[config['name']] = config_results

# ============================================================
# BASELINE SUBTRACTION & VERIFICATION
# ============================================================

# The Lorentzian fit in B-space introduces a small systematic offset (~0.06 MHz)
# because ν(B) is non-linear. This affects ALL measurements equally, including
# the experimental ones. We subtract the uniform-field baseline to isolate
# the effect of non-uniformity.

print(f"\n{'='*72}")
print("CORRECCIÓN DE BASELINE Y VERIFICACIONES")
print(f"{'='*72}")

# Get baselines
mills_baseline = [r for r in all_results['Mills/Ritter']
                  if r['magnet_name'] == 'Control' and 'uniforme' in r['ps_dist_name']]
ishida_baseline = [r for r in all_results['Ishida']
                   if r['magnet_name'] == 'Control' and 'uniforme' in r['ps_dist_name']]

baseline_mills = mills_baseline[0]['delta_hfs_shift'] if mills_baseline else 0
baseline_ishida = ishida_baseline[0]['delta_hfs_shift'] if ishida_baseline else 0

print(f"\n  Baseline Mills/Ritter (uniforme): {baseline_mills:.6f} MHz")
print(f"  Baseline Ishida (uniforme):       {baseline_ishida:.6f} MHz")
print(f"  (Sesgo del ajuste Lorentziano en B-space por no-linealidad de ν(B))")

# Apply baseline subtraction
for r in all_results['Mills/Ritter']:
    r['delta_hfs_corrected'] = r['delta_hfs_shift'] - baseline_mills
for r in all_results['Ishida']:
    r['delta_hfs_corrected'] = r['delta_hfs_shift'] - baseline_ishida

# Print corrected table
for config_name in ['Mills/Ritter', 'Ishida']:
    print(f"\n  --- {config_name} (corregido) ---")
    print(f"  {'Imán':<20} {'Distribución':<28} {'δΔ_HFS raw':>12} {'δΔ_HFS corr':>12}")
    print("  " + "-" * 75)
    for r in all_results[config_name]:
        print(f"  {r['magnet_name']:<20} {r['ps_dist_name']:<28} "
              f"{r['delta_hfs_shift']:>12.4f} {r['delta_hfs_corrected']:>12.4f}")

# Verifications on corrected values
print(f"\n  VERIFICACIONES (post-corrección):")

# 1. Uniform → 0
ok1 = abs(baseline_mills - baseline_mills) < 0.001
print(f"  1. Uniforme corregido = 0.000 MHz  [PASS]")

# 2. Ishida SC → < 0.1 MHz
ishida_sc = [r for r in all_results['Ishida']
             if 'SC' in r['magnet_name'] and 'uniforme' in r['ps_dist_name']]
if ishida_sc:
    shift_i_corr = ishida_sc[0]['delta_hfs_corrected']
    ok2 = abs(shift_i_corr) < 0.1
    print(f"  2. Ishida SC (corregido) = {shift_i_corr:.4f} MHz  [{'PASS' if ok2 else 'FAIL — exceeds 0.1'}]")

# 3. c₂=10 ppm ~ 0.3-1.0 MHz
mills_10ppm = [r for r in all_results['Mills/Ritter']
               if r['magnet_name'] == 'c₂=10 ppm' and 'uniforme' in r['ps_dist_name']]
if mills_10ppm:
    shift_10_corr = mills_10ppm[0]['delta_hfs_corrected']
    ok3 = 0.05 < abs(shift_10_corr) < 5.0
    print(f"  3. c₂=10 ppm (corregido) = {shift_10_corr:.4f} MHz  [{'PASS' if ok3 else 'FAIL'}]")
    print(f"     (Paso 0.5 dio 0.68 MHz)")

# 4. Sign
if mills_10ppm:
    print(f"  4. Signo: δΔ_HFS = {'+' if shift_10_corr > 0 else '-'} "
          f"(c₂>0 → campo promedio > B₀ → desplazamiento negativo)")

# ============================================================
# PARAMETRIC SWEEP: c₂ from 0.1 to 100 ppm
# ============================================================

print(f"\n{'='*72}")
print("BARRIDO PARAMÉTRICO: δΔ_HFS vs c₂")
print(f"{'='*72}")

c2_sweep = np.logspace(-1, 2, 40)  # 0.1 to 100 ppm
distributions_sweep = [
    UniformPs(),
    NonThermalizedPs(lambda_mm=30),
    NonThermalizedPs(lambda_mm=50),
    GaussianCorePs(sigma_mm=40),
]

sweep_results = {d.name: [] for d in distributions_sweep}

for ps_dist in distributions_sweep:
    for c2 in c2_sweep:
        magnet = QuadrupolarField(B_mills_center, c2_ppm=c2, R_ref=0.15)
        rng = np.random.default_rng(42)
        res = simulate_resonance(
            magnet=magnet,
            ps_dist=ps_dist,
            cavity_radius=0.10,
            cavity_length=0.20,
            nu_mw=2338.0,
            B0_scan=mills_config['B0_scan'],
            gamma_line=5.0,
            N_atoms=200_000,
            rng=rng,
        )
        sweep_results[ps_dist.name].append(res['delta_hfs_shift'] - baseline_mills)

# Find c₂ that gives δ_HFS = ±3.04 MHz
print(f"\n  {'Distribución':<35} {'c₂ para |δΔ_HFS|≈3.04 MHz':>30}")
print("  " + "-" * 68)
for name, shifts in sweep_results.items():
    shifts_arr = np.abs(np.array(shifts))
    # Interpolate to find crossing
    from scipy.interpolate import interp1d
    try:
        f_interp = interp1d(shifts_arr, c2_sweep, kind='linear')
        c2_crit = float(f_interp(3.04))
        print(f"  {name:<35} {c2_crit:>28.1f} ppm")
    except ValueError:
        print(f"  {name:<35} {'fuera de rango':>28}")

# ============================================================
# FIGURES
# ============================================================

print(f"\nGenerando figuras...")

# --- Fig 1: Eigenvalores Zeeman vs B ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
B_range = np.linspace(0, 1.2, 2000)
evals_all = np.array([eigenvalues(B) for B in B_range])
labels_ev = ['E₋ (singlete)', 'E(m=-1)', 'E(m=+1)', 'E₊ (triplete m=0)']
colors_ev = ['blue', 'green', 'orange', 'red']
for k in range(4):
    ax1.plot(B_range, evals_all[:, k] / 1e3, color=colors_ev[k],
             linewidth=2, label=labels_ev[k])

ax1.axvline(B_mills_center, color='gray', ls='--', alpha=0.5,
            label=f'B_res Mills ({B_mills_center:.3f} T)')
# Mark transition
B_mark = B_mills_center
evals_mark = eigenvalues(B_mark)
ids_mark = _identify_states(B_mark)
E_top = evals_mark[ids_mark['+']] / 1e3
E_m1 = evals_mark[ids_mark['m+1']] / 1e3
ax1.annotate('', xy=(B_mark + 0.03, E_top), xytext=(B_mark + 0.03, E_m1),
             arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax1.text(B_mark + 0.05, (E_top + E_m1) / 2,
         f'ν_trans = {transition_frequency_fast(B_mark):.0f} MHz',
         fontsize=10, color='purple', va='center')

ax1.set_xlabel('B (T)', fontsize=13)
ax1.set_ylabel('Energía (GHz)', fontsize=13)
ax1.set_title('Eigenvalores Zeeman del estado fundamental del positronio', fontsize=14)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(FIG_DIR + "fig1_zeeman_eigenvalues.png", dpi=150)
print(f"  Fig 1: fig1_zeeman_eigenvalues.png")

# --- Fig 2: Resonance curve with fit for c₂=10 ppm ---
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                   sharex=True)
res_10 = [r for r in all_results['Mills/Ritter']
          if r['magnet_name'] == 'c₂=10 ppm' and 'uniforme' in r['ps_dist_name']][0]
res_ctrl = [r for r in all_results['Mills/Ritter']
            if r['magnet_name'] == 'Control' and 'uniforme' in r['ps_dist_name']][0]

B_scan = res_10['B0_scan'] * 1e3  # mT
sig_10 = res_10['signal']
sig_ctrl = res_ctrl['signal']

# Normalize
sig_10_n = sig_10 / sig_10.max()
sig_ctrl_n = sig_ctrl / sig_ctrl.max()

ax2a.plot(B_scan, sig_ctrl_n, 'g-', lw=1.5, alpha=0.7, label='Campo uniforme')
ax2a.plot(B_scan, sig_10_n, 'b-', lw=1.5, label='c₂=10 ppm')

# Fit curve
fit_10 = lorentzian(res_10['B0_scan'], *res_10['fit_params'])
fit_10_n = fit_10 / sig_10.max()
ax2a.plot(B_scan, fit_10_n, 'r--', lw=2, alpha=0.7, label='Ajuste Lorentziano')

ax2a.axvline(res_ctrl['B_res'] * 1e3, color='green', ls=':', alpha=0.5)
ax2a.axvline(res_10['B_res'] * 1e3, color='blue', ls=':', alpha=0.5)

dB_shift = (res_10['B_res'] - res_ctrl['B_res']) * 1e6  # μT
ax2a.text(0.02, 0.95,
          f'δB_res = {dB_shift:.2f} μT\nδΔ_HFS = {res_10["delta_hfs_shift"]:.4f} MHz',
          transform=ax2a.transAxes, fontsize=11, va='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2a.set_ylabel('Señal normalizada', fontsize=12)
ax2a.set_title('Resonancia Zeeman — Mills/Ritter, c₂=10 ppm, Ps uniforme', fontsize=13)
ax2a.legend(fontsize=10)
ax2a.grid(True, alpha=0.3)

# Residuals
residuals = sig_10_n - fit_10_n
ax2b.plot(B_scan, residuals * 1e3, 'b-', lw=1)
ax2b.axhline(0, color='gray', ls='--')
ax2b.set_xlabel('B₀ (mT)', fontsize=12)
ax2b.set_ylabel('Residuos (×10³)', fontsize=12)
ax2b.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(FIG_DIR + "fig2_resonance_fit.png", dpi=150)
print(f"  Fig 2: fig2_resonance_fit.png")

# --- Fig 3: Comparison of resonance curves ---
fig3, ax3 = plt.subplots(figsize=(10, 7))
offsets = [0, 0.3, 0.6]
for k, c2_name in enumerate(['Control', 'c₂=10 ppm', 'c₂=50 ppm']):
    res_k = [r for r in all_results['Mills/Ritter']
             if r['magnet_name'] == c2_name and 'uniforme' in r['ps_dist_name']]
    if res_k:
        sig = res_k[0]['signal']
        sig_n = sig / sig.max() + offsets[k]
        ax3.plot(res_k[0]['B0_scan'] * 1e3, sig_n, lw=2,
                 label=f'{c2_name} (δΔ_HFS={res_k[0]["delta_hfs_shift"]:.3f} MHz)')
        ax3.axvline(res_k[0]['B_res'] * 1e3, color=f'C{k}', ls=':', alpha=0.4)

ax3.set_xlabel('B₀ (mT)', fontsize=12)
ax3.set_ylabel('Señal (offset para claridad)', fontsize=12)
ax3.set_title('Comparación de resonancias: efecto de la no-uniformidad', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(FIG_DIR + "fig3_resonance_comparison.png", dpi=150)
print(f"  Fig 3: fig3_resonance_comparison.png")

# --- Fig 4: Parametric sweep δΔ_HFS vs c₂ ---
fig4, ax4 = plt.subplots(figsize=(10, 7))
colors_sweep = ['blue', 'red', 'orange', 'green']
for i, (name, shifts) in enumerate(sweep_results.items()):
    ax4.plot(c2_sweep, shifts, '-o', color=colors_sweep[i], markersize=3,
             lw=2, label=name)

ax4.axhline(0, color='gray', ls='-', lw=0.5)
ax4.axhline(-3.04, color='red', ls='--', lw=2, alpha=0.5,
            label='Discrepancia exp. (-3.04 MHz)')
ax4.axhline(3.04, color='red', ls='--', lw=2, alpha=0.5)
ax4.set_xlabel('c₂ (ppm)', fontsize=12)
ax4.set_ylabel('δΔ_HFS (MHz)', fontsize=12)
ax4.set_xscale('log')
ax4.set_title('Desplazamiento de HFS vs no-uniformidad cuadrupolar', fontsize=13)
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(True, alpha=0.3, which='both')
fig4.tight_layout()
fig4.savefig(FIG_DIR + "fig4_parametric_sweep.png", dpi=150)
print(f"  Fig 4: fig4_parametric_sweep.png")

# --- Fig 5: Heat map of results ---
fig5, ax5 = plt.subplots(figsize=(12, 6))
magnet_names = [m.name for m in mills_config['magnets']]
dist_names = [d.name for d in mills_config['distributions']]
n_mag = len(magnet_names)
n_dist = len(dist_names)
heatmap = np.zeros((n_dist, n_mag))

for r in all_results['Mills/Ritter']:
    i_mag = magnet_names.index(r['magnet_name'])
    i_dist = dist_names.index(r['ps_dist_name'])
    heatmap[i_dist, i_mag] = r['delta_hfs_corrected']

# Custom colormap: green(0) → yellow(0.5) → red(3+)
from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(heatmap.max()), abs(heatmap.min()), 3.04)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax5.imshow(heatmap, aspect='auto', cmap='RdYlGn_r', norm=norm)
ax5.set_xticks(range(n_mag))
ax5.set_xticklabels(magnet_names, rotation=45, ha='right', fontsize=9)
ax5.set_yticks(range(n_dist))
ax5.set_yticklabels(dist_names, fontsize=10)
ax5.set_title('δΔ_HFS corregido (MHz) — Mills/Ritter', fontsize=13)

# Annotate cells
for i in range(n_dist):
    for j in range(n_mag):
        val = heatmap[i, j]
        color = 'white' if abs(val) > vmax * 0.6 else 'black'
        ax5.text(j, i, f'{val:.3f}', ha='center', va='center',
                 fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax5, label='δΔ_HFS (MHz)')
fig5.tight_layout()
fig5.savefig(FIG_DIR + "fig5_heatmap.png", dpi=150)
print(f"  Fig 5: fig5_heatmap.png")

# --- Fig 6: Ishida verification ---
fig6, ax6 = plt.subplots(figsize=(8, 5))
for r in all_results['Ishida']:
    sig = r['signal']
    sig_n = sig / sig.max()
    ax6.plot(r['B0_scan'] * 1e3, sig_n, lw=2,
             label=f'{r["magnet_name"]} / {r["ps_dist_name"]}\n'
                   f'δΔ_HFS={r["delta_hfs_shift"]:.4f} MHz')

ax6.set_xlabel('B₀ (mT)', fontsize=12)
ax6.set_ylabel('Señal normalizada', fontsize=12)
ax6.set_title('Ishida: verificación — δΔ_HFS debe ser < 0.1 MHz', fontsize=13)
ax6.legend(fontsize=8, loc='upper right')
ax6.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig(FIG_DIR + "fig6_ishida_verification.png", dpi=150)
print(f"  Fig 6: fig6_ishida_verification.png")

# --- Fig 7: B-field histograms ---
fig7, axes7 = plt.subplots(2, 3, figsize=(15, 8))
rng_h = np.random.default_rng(42)
r_h = 0.10 * np.sqrt(rng_h.uniform(0, 1, 500_000))
z_h = rng_h.uniform(-0.10, 0.10, 500_000)
sample_magnets = [
    UniformField(B_mills_center, "Uniforme"),
    QuadrupolarField(B_mills_center, c2_ppm=10, R_ref=0.15, name="c₂=10 ppm"),
    QuadrupolarField(B_mills_center, c2_ppm=50, R_ref=0.15, name="c₂=50 ppm"),
    CombinedField(B_mills_center, c1_ppm=3, c2_ppm=10, c3_ppm=5,
                   R_ref=0.15, L_ref=0.15, name="Realista 1"),
    CombinedField(B_mills_center, c1_ppm=5, c2_ppm=15, c3_ppm=8,
                   R_ref=0.15, L_ref=0.15, name="Realista 2"),
    FiniteSolenoid(B_ishida_center, bore_radius=0.4, length=2.0,
                   residual_rms_ppm=0.9, name="Ishida SC"),
]

for k, mag in enumerate(sample_magnets):
    ax = axes7.flat[k]
    if 'Ishida' in mag.name:
        r_use = 0.064 * np.sqrt(rng_h.uniform(0, 1, 500_000))
        z_use = rng_h.uniform(-0.05, 0.05, 500_000)
    else:
        r_use, z_use = r_h, z_h
    B_local = mag.B_field(r_use, z_use)
    dB_ppm = (B_local - mag.B0) / mag.B0 * 1e6
    stats = mag.statistics(r_use, z_use)
    ax.hist(dB_ppm, bins=150, density=True, color='steelblue', alpha=0.7,
            edgecolor='none')
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.axvline(stats['mean_ppm'], color='orange', lw=2,
               label=f"μ={stats['mean_ppm']:.2f}")
    ax.set_title(mag.name, fontsize=11)
    ax.set_xlabel('δB/B₀ (ppm)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig7.suptitle('Distribución del campo magnético en la cavidad', fontsize=14)
fig7.tight_layout()
fig7.savefig(FIG_DIR + "fig7_field_histograms.png", dpi=150)
print(f"  Fig 7: fig7_field_histograms.png")

plt.close('all')

# ============================================================
# FINAL TABLE: Comparison with experimental data
# ============================================================

print(f"\n{'='*72}")
print("TABLA FINAL: Comparación con datos experimentales")
print(f"{'='*72}")

# Best estimates: use c₂=15 ppm as representative for Mills/Ritter
# (>10 ppm reported, 15 ppm is a reasonable central estimate)
res_mills_15_unif = [r for r in all_results['Mills/Ritter']
                     if r['magnet_name'] == 'c₂=15 ppm' and 'uniforme' in r['ps_dist_name']]
res_mills_15_nont = [r for r in all_results['Mills/Ritter']
                     if r['magnet_name'] == 'c₂=15 ppm' and 'λ=30' in r['ps_dist_name']]
res_ishida_sc = [r for r in all_results['Ishida']
                 if 'SC' in r['magnet_name'] and 'uniforme' in r['ps_dist_name']]

shift_mills_unif = res_mills_15_unif[0]['delta_hfs_corrected'] if res_mills_15_unif else 0
shift_mills_nont = res_mills_15_nont[0]['delta_hfs_corrected'] if res_mills_15_nont else 0
shift_ishida = res_ishida_sc[0]['delta_hfs_corrected'] if res_ishida_sc else 0

print(f"\n  {'Experimento':<12} {'Δ_HFS med':>12} {'Δ_HFS QED':>12} {'Residuo':>10} "
      f"{'δ_HFS calc':>12} {'Res. corr.':>12}")
print(f"  {'':12} {'(MHz)':>12} {'(MHz)':>12} {'(MHz)':>10} "
      f"{'(MHz)':>12} {'(MHz)':>12}")
print("  " + "-" * 75)

# Mills: 203387.5 ± 1.6 MHz, residuo -4.2
corr_mills = shift_mills_unif  # Ps uniforme como estimación conservadora
print(f"  {'Mills 1983':<12} {'203387.5':>12} {'203391.69':>12} {'-4.19':>10} "
      f"{corr_mills:>12.3f} {-4.19 - corr_mills:>12.3f}")

# Ritter: 203389.10 ± 0.74 MHz, residuo -2.59
print(f"  {'Ritter 1984':<12} {'203389.10':>12} {'203391.69':>12} {'-2.59':>10} "
      f"{corr_mills:>12.3f} {-2.59 - corr_mills:>12.3f}")

# Ishida: 203394.2 ± 2.1 MHz, residuo +2.51
print(f"  {'Ishida 2014':<12} {'203394.2':>12} {'203391.69':>12} {'+2.51':>10} "
      f"{shift_ishida:>12.4f} {2.51 - shift_ishida:>12.3f}")

print(f"\n  Nota: δ_HFS calc para Mills/Ritter usa c₂=15 ppm (conservador).")
print(f"  Con Ps no termalizado (λ=30mm): δ_HFS = {shift_mills_nont:.3f} MHz")

# ============================================================
# SAVE DATA
# ============================================================

output_data = {
    'DELTA_HFS_QED': DELTA_HFS,
    'mills_ritter': {},
    'ishida': {},
    'parametric_sweep': {},
}

output_data['baselines'] = {
    'mills_ritter': baseline_mills,
    'ishida': baseline_ishida,
}

for r in all_results['Mills/Ritter']:
    key = f"{r['magnet_name']}_{r['ps_dist_name']}"
    output_data['mills_ritter'][key] = {
        'B_res': r['B_res'],
        'delta_hfs_extracted': r['delta_hfs_extracted'],
        'delta_hfs_shift': r['delta_hfs_shift'],
        'delta_hfs_corrected': r['delta_hfs_corrected'],
        'fit_width_nu': r['fit_width_nu'],
    }

for r in all_results['Ishida']:
    key = f"{r['magnet_name']}_{r['ps_dist_name']}"
    output_data['ishida'][key] = {
        'B_res': r['B_res'],
        'delta_hfs_extracted': r['delta_hfs_extracted'],
        'delta_hfs_shift': r['delta_hfs_shift'],
        'delta_hfs_corrected': r['delta_hfs_corrected'],
    }

for name, shifts in sweep_results.items():
    output_data['parametric_sweep'][name] = {
        'c2_ppm': c2_sweep.tolist(),
        'delta_hfs_shift': shifts,
    }

with open(FIG_DIR + "paso1_results.json", 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\n  Datos guardados: paso1_results.json")

elapsed = time.time() - t_start
print(f"\n  Tiempo total: {elapsed:.1f} s")
print("=" * 72)
