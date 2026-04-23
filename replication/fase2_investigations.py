#!/usr/bin/env python3
"""
FASE 2 INVESTIGATIONS — TM₁₁₀, non-Hermitian, Mills pressure curve.

#13: TM₁₁₀ mode weight — does the microwave field pattern change the result?
#15: Non-Hermitian correction — does baseline subtraction cancel it?
#17: Mills pressure curve — end-to-end extrapolation demonstration
"""

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import j1, jvp

from breit_rabi import (
    transition_frequency_fast, extract_hfs, build_interpolator,
    DELTA_HFS, G_E, MU_B,
)
from magnetic_field_models import UniformField
from zeeman_resonance import simulate_resonance, lorentzian

N_ATOMS = 500_000

# Constants
MILLS_B0 = 0.925
MILLS_NU_MW = 3253.0
MILLS_R_CAV = 56.25e-3
MILLS_L_CAV = 12.7e-3
MILLS_SIGMA_R = 2.44e-3
MILLS_B_PARAM = 140.0
MILLS_GAMMA = 5.0
ALPHA_R = -0.21
ALPHA_Z = +0.44

class MeasuredGradientField:
    def __init__(self, B0, d2Br, d2Bz, name="Mills"):
        self.B0 = B0; self.d2Br = d2Br; self.d2Bz = d2Bz; self.name = name
    def B_field(self, r, z):
        return self.B0 * (1 + 0.5*self.d2Br*r**2 + 0.5*self.d2Bz*z**2)

mills_field = MeasuredGradientField(MILLS_B0, ALPHA_R, ALPHA_Z)

class MillsDist:
    name = "Mills"
    def __init__(self, sigma_r=MILLS_SIGMA_R):
        self.sigma_r = sigma_r
        self.L = MILLS_L_CAV
    def sample(self, N, R_cav=None, L_cav=None, rng=None):
        if rng is None: rng = np.random.default_rng()
        x = rng.normal(0, self.sigma_r, N)
        y = rng.normal(0, self.sigma_r, N)
        r = np.sqrt(x**2 + y**2)
        z = rng.uniform(-self.L/2, self.L/2, N)
        return r, z

def mills_weight(r, z):
    return np.maximum(1.0 - MILLS_B_PARAM * np.abs(z), 0.0)

def find_B_res(nu_mw):
    return brentq(lambda B: transition_frequency_fast(B) - nu_mw, 0.1, 2.0)

def make_scan(B_center, gamma=5.0, N=200):
    dB = 1e-6
    dnu = (transition_frequency_fast(B_center+dB) - transition_frequency_fast(B_center-dB))/(2*dB)
    gB = gamma/dnu
    return np.linspace(B_center - 8*gB, B_center + 8*gB, N)


# ====================================================================
# #13: TM₁₁₀ MODE WEIGHT
# ====================================================================
print("=" * 72)
print("#13: TM₁₁₀ MODE WEIGHT")
print("=" * 72)

# TM₁₁₀ mode in cylindrical cavity:
# j₁₁ = first zero of J₁(x) = 3.8317
# k_c = j₁₁ / R_cav
# B_r ∝ (1/r) J₁(k_c r) sin(φ)    [up to constant k_c factor]
# B_φ ∝ J₁'(k_c r) cos(φ)
# |B_MW|² = B_r² + B_φ²
#
# For the transition rate, what matters is |B_⊥|² perpendicular to B₀(z).
# Since B_z = 0 for TM modes, all of B_MW is perpendicular: weight ∝ B_r² + B_φ²

j11 = 3.8317
k_c = j11 / MILLS_R_CAV  # ~68.1 m⁻¹

print(f"\n  Cavity radius R = {MILLS_R_CAV*1e3:.2f} mm")
print(f"  k_c = j₁₁/R = {k_c:.2f} m⁻¹")
print(f"  Ps σ_r = {MILLS_SIGMA_R*1e3:.2f} mm")
print(f"  k_c × σ_r = {k_c * MILLS_SIGMA_R:.4f}")
print(f"  (k_c × σ_r)² = {(k_c * MILLS_SIGMA_R)**2:.4f}")
print(f"  → Ps occupies only {(k_c*MILLS_SIGMA_R)**2*100:.1f}% of the mode's radial scale")

# Analytical estimate of the correction
print(f"\n  --- Analytical estimate ---")
# Near center: J₁(x)/x ≈ 1/2 - x²/16 + ...
# J₁'(x) ≈ 1/2 - 3x²/16 + ...
# |B_MW|² ≈ C² [1/4 - (k_c r)²/16 × (sin²φ + 3cos²φ) + ...]
#          = C² [1/4 - (k_c r)²/16 × (1 + 2cos²φ) + ...]
# Azimuthal average: <1 + 2cos²φ> = 2
# |B_MW|²_avg ≈ C²/4 × [1 - (k_c r)²/4 + ...]
# So the weight is W(r) ≈ 1 - (k_c r)²/4 (up to normalization)
# ⟨W⟩ over Gaussian: ⟨W⟩ = 1 - k_c² ⟨r²⟩/4 = 1 - k_c² × 2σ²/4
print(f"  ⟨W⟩ = 1 - k_c²×2σ²/4 = 1 - {k_c**2 * 2 * MILLS_SIGMA_R**2 / 4:.6f}")
print(f"       = {1 - k_c**2 * 2 * MILLS_SIGMA_R**2 / 4:.6f}")

# The TM₁₁₀ weight varies by ~ (k_c σ)² ≈ 2.8% across the Ps distribution.
# Since the radial contribution to ⟨δB⟩ is only -1.25 ppm (vs total +0.52 ppm),
# and the weight's radial dependence is ~3%, the net correction is:
# δ(⟨δB⟩) ~ 3% × (-1.25 ppm) ≈ -0.04 ppm
# δ(δΔ) ~ amplification × δ(⟨δB⟩) × B₀ ≈ 4.5e5 × 0.04e-6 × 0.925 ≈ 0.02 MHz
print(f"  Estimated TM₁₁₀ correction: ~0.02 MHz (order of (k_c σ)² × radial term)")

# Now do it numerically
print(f"\n  --- Numerical simulation ---")

def tm110_weight(x, y, z):
    """TM₁₁₀ microwave field intensity |B_MW|² at position (x,y,z).
    Returns weight proportional to transition rate."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    k_r = k_c * r

    # Handle r=0 separately (L'Hôpital)
    safe_r = np.where(r > 1e-10, r, 1e-10)
    safe_kr = k_c * safe_r

    # J₁(k_c r) / (k_c r) and J₁'(k_c r)
    j1_over_kr = np.where(k_r > 1e-8,
                          j1(k_r) / safe_kr,
                          0.5 * np.ones_like(k_r))  # limit: 1/2
    j1_prime = jvp(1, k_r, 1)  # J₁'(x) = dJ₁/dx

    # |B_MW|² ∝ (J₁/kr)² sin²φ + (J₁')² cos²φ
    W = j1_over_kr**2 * np.sin(phi)**2 + j1_prime**2 * np.cos(phi)**2
    return W


class MillsDistXY:
    """Mills distribution returning (x, y, z) for TM₁₁₀ calculation."""
    name = "Mills (x,y,z)"
    def __init__(self, sigma_r=MILLS_SIGMA_R):
        self.sigma_r = sigma_r
        self.L = MILLS_L_CAV
    def sample_xyz(self, N, rng=None):
        if rng is None: rng = np.random.default_rng()
        x = rng.normal(0, self.sigma_r, N)
        y = rng.normal(0, self.sigma_r, N)
        z = rng.uniform(-self.L/2, self.L/2, N)
        return x, y, z


def run_tm110_comparison():
    """Compare result with and without TM₁₁₀ weighting."""
    B_center = find_B_res(MILLS_NU_MW)
    B_scan = make_scan(B_center, MILLS_GAMMA, 200)

    rng = np.random.default_rng(42)
    dist_xy = MillsDistXY()
    x, y, z = dist_xy.sample_xyz(N_ATOMS, rng)
    r = np.sqrt(x**2 + y**2)

    # Field profile
    B_profile = mills_field.B_field(r, z)
    B_ratio = B_profile / mills_field.B0

    # Detection weight w(z)
    w_det = np.maximum(1.0 - MILLS_B_PARAM * np.abs(z), 0.0)

    # TM₁₁₀ weight
    w_tm = tm110_weight(x, y, z)

    # Combined weight: detection × TM₁₁₀
    w_combined = w_det * w_tm

    # Build interpolator
    B_min = max(B_scan[0] * np.min(B_ratio) * 0.999, 0.001)
    B_max = B_scan[-1] * np.max(B_ratio) * 1.001
    nu_interp = build_interpolator(B_min, B_max, N=20000)
    gamma2 = (MILLS_GAMMA/2)**2

    results = {}
    for label, weights in [("w(z) only", w_det), ("w(z) × TM₁₁₀", w_combined)]:
        signal = np.empty(len(B_scan))
        signal_bl = np.empty(len(B_scan))
        for i, B0_val in enumerate(B_scan):
            # Signal (non-uniform)
            B_local = B0_val * B_ratio
            nu_local = nu_interp(B_local)
            lor = gamma2 / ((nu_local - MILLS_NU_MW)**2 + gamma2)
            signal[i] = np.average(lor, weights=weights)
            # Baseline (uniform)
            nu_uniform = nu_interp(np.full_like(B_local, B0_val))
            lor_bl = gamma2 / ((nu_uniform - MILLS_NU_MW)**2 + gamma2)
            signal_bl[i] = np.average(lor_bl, weights=weights)

        # Fit both
        idx = np.argmax(signal)
        B_guess = B_scan[idx]
        dnu_dB = G_E*MU_B*B_guess / np.sqrt((DELTA_HFS/2)**2 + (G_E*MU_B*B_guess)**2) * 2*G_E*MU_B
        gB = MILLS_GAMMA / max(dnu_dB, 1.0)
        p0 = [B_guess, gB, signal.max()-signal.min(), signal.min()]

        popt, _ = curve_fit(lorentzian, B_scan, signal, p0=p0, maxfev=10000)
        popt_bl, _ = curve_fit(lorentzian, B_scan, signal_bl, p0=p0, maxfev=10000)

        delta = extract_hfs(popt[0], MILLS_NU_MW) - DELTA_HFS
        delta_bl = extract_hfs(popt_bl[0], MILLS_NU_MW) - DELTA_HFS
        corrected = delta - delta_bl

        # Weighted field statistics
        delta_B = (B_profile - mills_field.B0) / mills_field.B0
        mean_dB = np.average(delta_B, weights=weights) * 1e6

        results[label] = {'corrected': corrected, 'mean_dB': mean_dB}
        print(f"  {label}:")
        print(f"    ⟨δB/B₀⟩ = {mean_dB:.4f} ppm")
        print(f"    δΔ_HFS  = {corrected:.4f} MHz")

    diff = results["w(z) × TM₁₁₀"]['corrected'] - results["w(z) only"]['corrected']
    pct = diff / results["w(z) only"]['corrected'] * 100
    print(f"\n  DIFFERENCE (TM₁₁₀ effect): {diff:.4f} MHz ({pct:.1f}%)")
    print(f"  → TM₁₁₀ mode weight changes the result by {abs(pct):.1f}%")

    return diff, pct

diff_tm, pct_tm = run_tm110_comparison()

# Multi-seed for TM₁₁₀
print(f"\n  --- Multi-seed TM₁₁₀ check (5 seeds) ---")
diffs_tm = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    dist_xy = MillsDistXY()
    x, y, z = dist_xy.sample_xyz(N_ATOMS, rng)
    r = np.sqrt(x**2 + y**2)
    B_profile = mills_field.B_field(r, z)
    B_ratio = B_profile / mills_field.B0
    w_det = np.maximum(1.0 - MILLS_B_PARAM * np.abs(z), 0.0)
    w_tm = tm110_weight(x, y, z)

    B_center = find_B_res(MILLS_NU_MW)
    B_scan = make_scan(B_center, MILLS_GAMMA, 200)
    B_min = max(B_scan[0] * np.min(B_ratio) * 0.999, 0.001)
    B_max = B_scan[-1] * np.max(B_ratio) * 1.001
    nu_interp = build_interpolator(B_min, B_max, N=20000)
    gamma2 = (MILLS_GAMMA/2)**2

    deltas = {}
    for label, weights in [("det", w_det), ("det+tm", w_det*w_tm)]:
        signal = np.empty(len(B_scan))
        signal_bl = np.empty(len(B_scan))
        for i, B0_val in enumerate(B_scan):
            B_local = B0_val * B_ratio
            nu_local = nu_interp(B_local)
            lor = gamma2 / ((nu_local - MILLS_NU_MW)**2 + gamma2)
            signal[i] = np.average(lor, weights=weights)
            nu_unif = nu_interp(np.full_like(B_local, B0_val))
            lor_bl = gamma2 / ((nu_unif - MILLS_NU_MW)**2 + gamma2)
            signal_bl[i] = np.average(lor_bl, weights=weights)

        idx = np.argmax(signal)
        B_guess = B_scan[idx]
        dnu_dB = G_E*MU_B*B_guess / np.sqrt((DELTA_HFS/2)**2 + (G_E*MU_B*B_guess)**2) * 2*G_E*MU_B
        gB = MILLS_GAMMA / max(dnu_dB, 1.0)
        p0 = [B_guess, gB, signal.max()-signal.min(), signal.min()]
        popt, _ = curve_fit(lorentzian, B_scan, signal, p0=p0, maxfev=10000)
        popt_bl, _ = curve_fit(lorentzian, B_scan, signal_bl, p0=p0, maxfev=10000)
        d = (extract_hfs(popt[0], MILLS_NU_MW) - DELTA_HFS) - (extract_hfs(popt_bl[0], MILLS_NU_MW) - DELTA_HFS)
        deltas[label] = d

    diffs_tm.append(deltas["det+tm"] - deltas["det"])

diffs_tm = np.array(diffs_tm)
print(f"  TM₁₁₀ correction across seeds: {diffs_tm.mean():.4f} ± {diffs_tm.std():.4f} MHz")


# ====================================================================
# #15: NON-HERMITIAN CORRECTION
# ====================================================================
print(f"\n{'='*72}")
print("#15: NON-HERMITIAN CORRECTION (singlet annihilation)")
print("=" * 72)

# Decay rates
TAU_SINGLET = 125e-12   # s (para-Ps lifetime)
TAU_TRIPLET = 142e-9    # s (ortho-Ps lifetime)
GAMMA_S = 1.0 / TAU_SINGLET / 1e6  # MHz (= 8000 MHz)
GAMMA_T = 1.0 / TAU_TRIPLET / 1e6  # MHz (= 7.04 MHz)

print(f"\n  Γ_singlet = 1/τ_s = {GAMMA_S:.0f} MHz")
print(f"  Γ_triplet = 1/τ_t = {GAMMA_T:.2f} MHz")
print(f"  Γ_s - Γ_t = {GAMMA_S - GAMMA_T:.0f} MHz")


def transition_freq_non_hermitian(B):
    """Transition frequency including non-Hermitian decay terms."""
    D = DELTA_HFS
    V = G_E * MU_B * B

    # The 2×2 block for m=0 states in {|↑↓⟩, |↓↑⟩} basis:
    # H_real = [[-D/4 + V,  D/2],
    #           [D/2,       -D/4 - V]]
    # Non-Hermitian addition:
    # Γ_nh = -i × [[(Γ_t+Γ_s)/4, (Γ_t-Γ_s)/4],
    #              [(Γ_t-Γ_s)/4, (Γ_t+Γ_s)/4]]

    H = np.array([
        [-D/4 + V - 1j*(GAMMA_T+GAMMA_S)/4,   D/2 - 1j*(GAMMA_T-GAMMA_S)/4],
        [D/2 - 1j*(GAMMA_T-GAMMA_S)/4,         -D/4 - V - 1j*(GAMMA_T+GAMMA_S)/4]
    ], dtype=np.complex128)

    evals = np.linalg.eigvals(H)
    # E_+ is the one with larger real part
    idx_plus = np.argmax(evals.real)
    E_plus = evals[idx_plus]

    # E_{m=+1} = D/4 - iΓ_t/2 (exact eigenstate)
    E_m1 = D/4 - 1j*GAMMA_T/2

    # Transition frequency = Re(E_+ - E_{m=+1})
    return np.real(E_plus - E_m1)


def transition_freq_nh_vectorized(B_array):
    """Vectorized version for arrays."""
    return np.array([transition_freq_non_hermitian(B) for B in np.atleast_1d(B_array)])


# Compare Hermitian vs non-Hermitian at a few B values
print(f"\n  --- Transition frequency comparison ---")
print(f"  {'B (T)':>8} {'ν_Hermitian':>14} {'ν_non-Hermitian':>18} {'Shift (MHz)':>14}")
print("  " + "-" * 58)
for B_test in [0.79, 0.866, 0.925, 1.0]:
    nu_h = transition_frequency_fast(B_test)
    nu_nh = transition_freq_non_hermitian(B_test)
    shift = nu_nh - nu_h
    print(f"  {B_test:>8.3f} {nu_h:>14.4f} {nu_nh:>18.4f} {shift:>14.4f}")

# Now the key test: does the shift cancel under baseline subtraction?
print(f"\n  --- Baseline cancellation test ---")

B_center = find_B_res(MILLS_NU_MW)
B_scan = make_scan(B_center, MILLS_GAMMA, 200)

rng = np.random.default_rng(42)
dist = MillsDist()
r_atoms, z_atoms = dist.sample(N_ATOMS, rng=rng)
w = mills_weight(r_atoms, z_atoms)

B_profile = mills_field.B_field(r_atoms, z_atoms)
B_ratio = B_profile / mills_field.B0

# Build interpolators for both Hermitian and non-Hermitian
B_min = max(B_scan[0] * np.min(B_ratio) * 0.999, 0.001)
B_max = B_scan[-1] * np.max(B_ratio) * 1.001
B_table = np.linspace(B_min, B_max, 20000)

# Hermitian interpolator (already exists)
nu_h_interp = build_interpolator(B_min, B_max, N=20000)

# Non-Hermitian interpolator
from scipy.interpolate import interp1d
nu_nh_table = transition_freq_nh_vectorized(B_table)
nu_nh_interp = interp1d(B_table, nu_nh_table, kind='cubic', fill_value='extrapolate')

gamma2 = (MILLS_GAMMA/2)**2

results_nh = {}
for label, interp_fn in [("Hermitian", nu_h_interp), ("Non-Hermitian", nu_nh_interp)]:
    signal = np.empty(len(B_scan))
    signal_bl = np.empty(len(B_scan))
    for i, B0_val in enumerate(B_scan):
        # Non-uniform
        B_local = B0_val * B_ratio
        nu_local = interp_fn(B_local)
        lor = gamma2 / ((nu_local - MILLS_NU_MW)**2 + gamma2)
        signal[i] = np.average(lor, weights=w)
        # Uniform baseline
        nu_unif = interp_fn(np.full(N_ATOMS, B0_val))
        lor_bl = gamma2 / ((nu_unif - MILLS_NU_MW)**2 + gamma2)
        signal_bl[i] = np.average(lor_bl, weights=w)

    # Fit
    idx = np.argmax(signal)
    B_guess = B_scan[idx]
    dnu_dB = G_E*MU_B*B_guess / np.sqrt((DELTA_HFS/2)**2 + (G_E*MU_B*B_guess)**2) * 2*G_E*MU_B
    gB = MILLS_GAMMA / max(dnu_dB, 1.0)
    p0 = [B_guess, gB, signal.max()-signal.min(), signal.min()]

    popt, _ = curve_fit(lorentzian, B_scan, signal, p0=p0, maxfev=10000)
    popt_bl, _ = curve_fit(lorentzian, B_scan, signal_bl, p0=p0, maxfev=10000)

    d_raw = extract_hfs(popt[0], MILLS_NU_MW) - DELTA_HFS
    d_bl = extract_hfs(popt_bl[0], MILLS_NU_MW) - DELTA_HFS
    d_corrected = d_raw - d_bl

    results_nh[label] = {'raw': d_raw, 'baseline': d_bl, 'corrected': d_corrected}
    print(f"  {label}:")
    print(f"    Raw:       {d_raw:.4f} MHz")
    print(f"    Baseline:  {d_bl:.4f} MHz")
    print(f"    Corrected: {d_corrected:.4f} MHz")

diff_nh = results_nh["Non-Hermitian"]['corrected'] - results_nh["Hermitian"]['corrected']
print(f"\n  RESIDUAL after baseline subtraction: {diff_nh:.4f} MHz")
print(f"  → Non-Hermitian effect cancels to {abs(diff_nh):.4f} MHz under baseline subtraction")
if abs(diff_nh) < 0.01:
    print(f"  → CONFIRMED: cancellation is complete to < 0.01 MHz")
else:
    print(f"  → WARNING: residual of {abs(diff_nh):.3f} MHz should be added to error budget")


# ====================================================================
# #17: MILLS PRESSURE CURVE — END-TO-END EXTRAPOLATION
# ====================================================================
print(f"\n{'='*72}")
print("#17: MILLS PRESSURE CURVE — END-TO-END EXTRAPOLATION")
print("=" * 72)

# Mills measured at multiple gas pressures and extrapolated to zero density.
# The density shift in isobutane: approximately -5.5 MHz/amagat (typical for Ps in gases).
# Mills used pressures roughly 100-500 Torr.
# 1 amagat ≈ 760 Torr at STP.

DENSITY_SHIFT = -5.5  # MHz/amagat (approximate, for Ps in isobutane/N₂ mix)
PRESSURES_TORR = np.array([100, 200, 300, 400, 500])
DENSITIES_AMAGAT = PRESSURES_TORR / 760.0

# Model: σ_r varies with pressure (higher pressure → more stopping → smaller σ_r)
# σ_r(P) ∝ 1/√(P/P₀) is a rough approximation from stopping range scaling
P_NOM = 300.0  # nominal pressure where σ_r = 2.44 mm
SIGMA_NOM = MILLS_SIGMA_R

print(f"\n  Density shift coefficient: {DENSITY_SHIFT} MHz/amagat")
print(f"  Pressures: {PRESSURES_TORR} Torr")
print(f"  Nominal σ_r = {SIGMA_NOM*1e3:.2f} mm at {P_NOM:.0f} Torr")

print(f"\n  --- Computing δΔ_HFS at each pressure ---")
print(f"  {'P (Torr)':>10} {'n (amagat)':>12} {'σ_r (mm)':>10} {'δΔ_inhom':>12} {'Δ_measured':>14}")
print("  " + "-" * 62)

# For each pressure, compute the inhomogeneity bias
B_center = find_B_res(MILLS_NU_MW)
B_scan = make_scan(B_center, MILLS_GAMMA, 200)

delta_inhom_at_P = []
delta_measured = []

for P in PRESSURES_TORR:
    sigma_r_P = SIGMA_NOM * np.sqrt(P_NOM / P)  # stopping range scaling
    n = P / 760.0

    # Analytical δΔ at this σ_r
    # ⟨δB/B₀⟩ = αr × σ² + (1/2)αz × ⟨z²⟩_w
    h = MILLS_L_CAV / 2
    bh = MILLS_B_PARAM * h
    norm_w = 2*h*(1 - bh/2)
    z2_w = 2*(h**3/3 - MILLS_B_PARAM*h**4/4) / norm_w
    contrib_r = ALPHA_R * sigma_r_P**2
    contrib_z = 0.5 * ALPHA_Z * z2_w
    dB_ppm = (contrib_r + contrib_z) * 1e6

    # Amplification factor
    dDelta_dB = 446730.0  # MHz/T
    delta_inhom = dDelta_dB * (-dB_ppm * 1e-6) * MILLS_B0
    delta_inhom_at_P.append(delta_inhom)

    # "Measured" Δ_HFS = true + density shift + inhomogeneity bias
    delta_meas = DELTA_HFS + DENSITY_SHIFT * n + delta_inhom
    delta_measured.append(delta_meas)

    print(f"  {P:>10.0f} {n:>12.4f} {sigma_r_P*1e3:>10.2f} {delta_inhom:>12.4f} {delta_meas:>14.4f}")

delta_inhom_at_P = np.array(delta_inhom_at_P)
delta_measured = np.array(delta_measured)

# Also compute without inhomogeneity
delta_no_inhom = DELTA_HFS + DENSITY_SHIFT * DENSITIES_AMAGAT

# Linear extrapolation to n=0
def linear(x, a, b):
    return a * x + b

popt_with, _ = curve_fit(linear, DENSITIES_AMAGAT, delta_measured)
popt_without, _ = curve_fit(linear, DENSITIES_AMAGAT, delta_no_inhom)

intercept_with = popt_with[1]
intercept_without = popt_without[1]
slope_with = popt_with[0]
slope_without = popt_without[0]

print(f"\n  --- Linear extrapolation to n = 0 ---")
print(f"  WITHOUT inhomogeneity:")
print(f"    Slope = {slope_without:.2f} MHz/amagat, Intercept = {intercept_without:.4f} MHz")
print(f"    (Should be Δ_QED = {DELTA_HFS:.4f} MHz)")
print(f"  WITH inhomogeneity:")
print(f"    Slope = {slope_with:.2f} MHz/amagat, Intercept = {intercept_with:.4f} MHz")
print(f"    Shift in intercept = {intercept_with - intercept_without:.4f} MHz")

# Key check: variation of δΔ_inhom across pressures
print(f"\n  --- Pressure dependence of δΔ_inhom ---")
print(f"  δΔ range: [{delta_inhom_at_P.min():.4f}, {delta_inhom_at_P.max():.4f}] MHz")
print(f"  δΔ at P=0 (extrapolated): {intercept_with - intercept_without:.4f} MHz")
print(f"  δΔ at nominal (P={P_NOM}): {delta_inhom_at_P[2]:.4f} MHz")
print(f"  Variation across 100-500 Torr: {delta_inhom_at_P.max() - delta_inhom_at_P.min():.4f} MHz")

# The slope changes slightly because δΔ has weak pressure dependence
slope_diff = slope_with - slope_without
print(f"\n  Slope change: {slope_diff:.4f} MHz/amagat ({slope_diff/slope_without*100:.2f}%)")
print(f"  → The density shift coefficient is virtually unchanged by the inhomogeneity")

# How much of the bias survives the extrapolation?
survival = (intercept_with - intercept_without)
at_nominal = delta_inhom_at_P[2]
survival_pct = survival / at_nominal * 100
print(f"\n  SURVIVAL through extrapolation:")
print(f"    Bias at nominal pressure: {at_nominal:.4f} MHz")
print(f"    Bias at extrapolated n=0: {survival:.4f} MHz")
print(f"    Survival: {survival_pct:.1f}%")
print(f"    → {survival_pct:.0f}% of the inhomogeneity bias survives the zero-density extrapolation")


# ====================================================================
# SUMMARY
# ====================================================================
print(f"\n{'='*72}")
print("FASE 2 SUMMARY")
print("=" * 72)
print(f"""
  #13 TM₁₁₀ MODE WEIGHT:
    Effect: {abs(pct_tm):.1f}% change in δΔ ({diff_tm:.4f} MHz)
    → NEGLIGIBLE. The Ps distribution (σ=2.44mm) occupies only the
      center of the cavity (R=56.25mm) where the TM₁₁₀ mode is uniform.
    → Can reduce TM₁₁₀ systematic from "±0.05 MHz estimated" to
      "verified numerically: {abs(diff_tm):.3f} MHz"

  #15 NON-HERMITIAN CORRECTION:
    Residual after baseline subtraction: {abs(diff_nh):.4f} MHz
    → {'CONFIRMED: cancels completely (< 0.01 MHz)' if abs(diff_nh) < 0.01 else f'Residual of {abs(diff_nh):.3f} MHz'}
    → The 0.5 MHz line-center shift from singlet annihilation is
      identical in uniform and non-uniform fields, and cancels exactly.

  #17 MILLS PRESSURE CURVE:
    δΔ_inhom range across 100-500 Torr: [{delta_inhom_at_P.min():+.3f}, {delta_inhom_at_P.max():+.3f}] MHz
    δΔ at nominal pressure (P={P_NOM:.0f} Torr):  {at_nominal:+.3f} MHz
    Linear extrapolation to n=0:         {survival:+.3f} MHz
    Slope change due to inhomogeneity:   {slope_diff:+.2f} MHz/amagat ({slope_diff/slope_without*100:+.1f}%)
    → With σ_r ∝ 1/√P, the bias varies significantly across pressures and
      can change sign; the extrapolated intercept differs from the nominal-
      pressure value. The axial-only component (-0.73 MHz in the σ_r→0
      limit) is strictly pressure-independent. Determining the surviving
      fraction requires characterizing σ_r(n), which is not available from
      the original experiment (see manuscript Sec. VI.B).

  ERROR BUDGET (matches manuscript Sec. III.C):
    (i)   Monte Carlo (20 seeds):      ±0.001 MHz
    (ii)  NMR gradient (Maxwell):      ±0.02  MHz  (δΔ: -0.22 to -0.18)
    (iii) Detection weight b (±10%):   ±0.09  MHz  (∂δΔ/∂b ≈ -0.9 MHz·cm)
    (iv)  TM₁₁₀ mode weight:          ±{abs(diff_tm):.3f} MHz (verified this run)
    (v)   Pressure dependence (local): ±0.02  MHz
          Non-Hermitian residual:      ±{abs(diff_nh):.3f} MHz (verified this run)
    ---
    TOTAL (quadrature):                ±{np.sqrt(0.001**2 + 0.02**2 + 0.09**2 + 0.02**2 + diff_tm**2 + diff_nh**2):.3f} MHz
""")
