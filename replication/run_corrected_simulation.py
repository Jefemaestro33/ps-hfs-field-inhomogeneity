#!/usr/bin/env python3
"""
Corrected simulation (v4) with detection weighting w(z) = 1 - b|z|.

Mills 1975: R_cav=56.25mm, L_cav=12.7mm, B₀=0.925T, ν_MW=3253 MHz
            NMR gradients: d²B/dr²=-0.21 m⁻², d²B/dz²=+0.44 m⁻²
            Ps: Gaussian σ=2.44mm (transverse), uniform (axial)
            Detection weight: w(z) = 1 - 1.4cm⁻¹|z|

Ritter 1984: R_cav=76.8mm, L_cav=38.1mm, B₀=0.79T, ν_MW=2384 MHz
             Varian V4012A, post-shimming ~0.5 ppm σ
             Ps confined to R≈12.7mm, uniform in z

Ishida 2014: R_cav=64mm, L_cav=100mm, B₀=0.866T, ν_MW=2856.6 MHz
             SC solenoid, 0.9 ppm RMS residual
"""

import numpy as np
import json
import time
from scipy.optimize import brentq

from breit_rabi import (
    transition_frequency_fast, extract_hfs,
    DELTA_HFS, G_E, MU_B,
)
from magnetic_field_models import UniformField, QuadrupolarField, FiniteSolenoid
from ps_distributions import UniformPs, NonThermalizedPs
from zeeman_resonance import simulate_resonance

t_start = time.time()
np.random.seed(42)
N_ATOMS = 500_000

# ============================================================
# FIELD AND DISTRIBUTION MODELS
# ============================================================

class MeasuredGradientField:
    """Field from NMR-measured gradients: B(r,z) = B0*(1 + d2Br*r²/2 + d2Bz*z²/2)."""
    def __init__(self, B0, d2Br, d2Bz, name="Measured gradients"):
        self.B0 = B0
        self.d2Br = d2Br
        self.d2Bz = d2Bz
        self.name = name

    def B_field(self, r, z):
        return self.B0 * (1 + 0.5 * self.d2Br * r**2 + 0.5 * self.d2Bz * z**2)

    def statistics(self, r, z, weights=None):
        B = self.B_field(r, z)
        delta = (B - self.B0) / self.B0
        if weights is not None:
            w = np.maximum(weights, 0)
            mean_ppm = np.average(delta, weights=w) * 1e6
            rms_ppm = np.sqrt(np.average(delta**2, weights=w)) * 1e6
        else:
            mean_ppm = np.mean(delta) * 1e6
            rms_ppm = np.sqrt(np.mean(delta**2)) * 1e6
        return {'mean_ppm': mean_ppm, 'rms_ppm': rms_ppm}


class MillsDistribution:
    """Mills Ps: Gaussian σ=2.44mm transverse, uniform axial."""
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
# MILLS DETECTION WEIGHT
# ============================================================

def mills_detection_weight(r, z):
    """w(z) = 1 - b|z|, b = 1.4 cm⁻¹ = 140 m⁻¹."""
    return np.maximum(1.0 - 140.0 * np.abs(z), 0.0)


# ============================================================
# HELPERS
# ============================================================

def find_B_resonance(nu_mw):
    return brentq(lambda B: transition_frequency_fast(B) - nu_mw, 0.1, 2.0)


def make_B_scan(B_center, gamma_nu=5.0, N=200):
    dB = 1e-6
    dnu = (transition_frequency_fast(B_center + dB) -
           transition_frequency_fast(B_center - dB)) / (2 * dB)
    gB = gamma_nu / dnu
    return np.linspace(B_center - 8 * gB, B_center + 8 * gB, N)


def run_with_baseline(magnet, ps_dist, R_cav, L_cav, nu_mw, gamma,
                      label="", weight_fn=None):
    """Run simulation and subtract uniform-field baseline."""
    B_center = find_B_resonance(nu_mw)
    B_scan = make_B_scan(B_center, gamma)

    res_bl = simulate_resonance(
        UniformField(magnet.B0), ps_dist, R_cav, L_cav,
        nu_mw, B_scan, gamma, N_atoms=N_ATOMS,
        rng=np.random.default_rng(42), weight_fn=weight_fn)
    baseline = res_bl['delta_hfs_shift']

    res = simulate_resonance(
        magnet, ps_dist, R_cav, L_cav,
        nu_mw, B_scan, gamma, N_atoms=N_ATOMS,
        rng=np.random.default_rng(42), weight_fn=weight_fn)

    corrected = res['delta_hfs_shift'] - baseline

    r_s, z_s = ps_dist.sample(N_ATOMS, R_cav, L_cav, rng=np.random.default_rng(42))
    if weight_fn is not None:
        w = weight_fn(r_s, z_s)
        stats = magnet.statistics(r_s, z_s, weights=w) if hasattr(magnet, 'statistics') and 'weights' in magnet.statistics.__code__.co_varnames else magnet.statistics(r_s, z_s)
        # Compute weighted mean manually for safety
        B_vals = magnet.B_field(r_s, z_s)
        delta_vals = (B_vals - magnet.B0) / magnet.B0
        w = np.maximum(w, 0)
        mean_ppm = np.average(delta_vals, weights=w) * 1e6
    else:
        B_vals = magnet.B_field(r_s, z_s)
        delta_vals = (B_vals - magnet.B0) / magnet.B0
        mean_ppm = np.mean(delta_vals) * 1e6

    return {
        'label': label or f"{magnet.name} / {ps_dist.name}",
        'delta_hfs_corrected': corrected,
        'mean_dB_ppm': mean_ppm,
        'B_res': res['B_res'],
        'signal': res['signal'],
        'signal_bl': res_bl['signal'],
        'B0_scan': B_scan,
        'fit_params': res['fit_params'],
        'residuals': res['residuals'],
        'residuals_bl': res_bl['residuals'],
    }


# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

mills_field = MeasuredGradientField(
    B0=0.925, d2Br=-0.21, d2Bz=+0.44,
    name="Mills measured gradients")
mills_dist = MillsDistribution()
MILLS_R = 56.25e-3
MILLS_L = 12.7e-3
MILLS_NU = 3253.0
MILLS_GAMMA = 5.0

RITTER_R = 76.8e-3
RITTER_L = 38.1e-3
RITTER_R_REF = 66.7e-3 / 2
RITTER_NU = 2384.0
RITTER_GAMMA = 5.0
ritter_dist = RitterDistribution()

ISHIDA_R = 64e-3
ISHIDA_L = 100e-3
ISHIDA_NU = 2856.6
ISHIDA_GAMMA = 3.0

# ============================================================
# MILLS — WITH DETECTION WEIGHTING
# ============================================================

print("=" * 72)
print("MILLS 1975 — measured NMR gradients + detection weight w(z)")
print("=" * 72)

res_mills = run_with_baseline(
    mills_field, mills_dist, MILLS_R, MILLS_L, MILLS_NU, MILLS_GAMMA,
    label="Mills gradients + w(z)", weight_fn=mills_detection_weight)

print(f"  ⟨δB/B₀⟩ = {res_mills['mean_dB_ppm']:.3f} ppm")
print(f"  δΔ_HFS  = {res_mills['delta_hfs_corrected']:.4f} MHz")
print(f"  (Paper: -0.22 MHz, ⟨δB⟩ = +0.52 ppm)")

# ============================================================
# RITTER — QUADRUPOLAR SCAN
# ============================================================

print(f"\n{'='*72}")
print("RITTER 1984 — quadrupolar c₂ scan")
print(f"{'='*72}")

B_ritter_nom = find_B_resonance(RITTER_NU)
c2_vals = [2, 3, 5, 8, 10, 15, 20]
ritter_results = {}

for c2 in c2_vals:
    mag = QuadrupolarField(B_ritter_nom, c2_ppm=c2, R_ref=RITTER_R_REF)
    res = run_with_baseline(
        mag, ritter_dist, RITTER_R, RITTER_L, RITTER_NU, RITTER_GAMMA,
        label=f"Ritter c₂={c2}")
    ritter_results[c2] = res
    print(f"  c₂={c2:>3} ppm: ⟨δB⟩={res['mean_dB_ppm']:>8.3f} ppm, "
          f"δΔ={res['delta_hfs_corrected']:>8.4f} MHz")

# ============================================================
# ISHIDA — SC SOLENOID
# ============================================================

print(f"\n{'='*72}")
print("ISHIDA 2014 — superconductor verification")
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

# ============================================================
# SAVE CORRECTED JSON
# ============================================================

output = {
    "mills_measured": {
        "delta_hfs_corrected": res_mills['delta_hfs_corrected'],
        "mean_dB_ppm": res_mills['mean_dB_ppm'],
        "note": "With detection weight w(z) = 1 - 1.4cm⁻¹|z|"
    },
    "ritter_scan": {
        str(c2): {
            "delta_hfs_corrected": ritter_results[c2]['delta_hfs_corrected'],
            "mean_dB_ppm": ritter_results[c2]['mean_dB_ppm'],
        } for c2 in c2_vals
    },
    "ishida_uniform": res_ishida_unif['delta_hfs_corrected'],
    "ishida_nontherm": res_ishida_nont['delta_hfs_corrected'],
}

with open("corrected_results.json", "w") as f:
    json.dump(output, f, indent=2)

elapsed = time.time() - t_start
print(f"\n{'='*72}")
print(f"Done in {elapsed:.1f}s. Results saved to corrected_results.json")
print(f"{'='*72}")
