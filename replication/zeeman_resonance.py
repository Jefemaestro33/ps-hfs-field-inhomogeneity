"""
zeeman_resonance.py — Zeeman resonance simulation (v4, with detection weighting).

Reproduces the complete experimental procedure:
1. Fix microwave frequency ν_MW
2. Sweep central magnetic field B₀
3. Detect increase in 2γ annihilation rate (resonance signal)
4. Fit Lorentzian to extract B_res
5. Invert Breit-Rabi to extract Δ_HFS
"""

import numpy as np
from scipy.optimize import curve_fit
from breit_rabi import (
    transition_frequency_fast, extract_hfs, build_interpolator,
    DELTA_HFS, G_E, MU_B,
)


def lorentzian(x, x0, gamma, A, offset):
    """Lorentzian with offset: A × (γ/2)² / ((x-x₀)² + (γ/2)²) + offset"""
    return A * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2) + offset


def simulate_resonance(
    magnet,
    ps_dist,
    cavity_radius,
    cavity_length,
    nu_mw,
    B0_scan,
    gamma_line,
    N_atoms=500_000,
    rng=None,
    weight_fn=None,
):
    """
    Simulate the Zeeman resonance curve.

    Parameters
    ----------
    magnet : MagnetModel
    ps_dist : PsDistribution
    cavity_radius, cavity_length : float (meters)
    nu_mw : float (MHz) — fixed microwave frequency
    B0_scan : np.ndarray (Tesla) — B₀ values for the sweep
    gamma_line : float (MHz) — individual resonance linewidth
    N_atoms : int — number of Ps atoms
    rng : np.random.Generator
    weight_fn : callable(r, z) -> array, optional
        Detection weight function. If None, uniform weighting is used.
        For the Mills experiment, this is w(z) = 1 - b|z| with b = 1.4 cm⁻¹.

    Returns
    -------
    dict with: B0_scan, signal, B_res, delta_hfs_extracted,
               delta_hfs_shift, fit_params, fit_width, residuals
    """
    if rng is None:
        rng = np.random.default_rng(42)

    r_atoms, z_atoms = ps_dist.sample(N_atoms, cavity_radius, cavity_length, rng)

    if weight_fn is not None:
        weights = weight_fn(r_atoms, z_atoms)
        weights = np.maximum(weights, 0.0)
    else:
        weights = np.ones(N_atoms)

    B_profile = magnet.B_field(r_atoms, z_atoms)
    B_ratio = B_profile / magnet.B0

    B_min = B0_scan[0] * np.min(B_ratio) * 0.999
    B_max = B0_scan[-1] * np.max(B_ratio) * 1.001
    B_min = max(B_min, 0.001)
    nu_interp = build_interpolator(B_min, B_max, N=20000)

    gamma2 = (gamma_line / 2)**2
    signal = np.empty(len(B0_scan))

    for i, B0_val in enumerate(B0_scan):
        B_local = B0_val * B_ratio
        nu_local = nu_interp(B_local)
        lor = gamma2 / ((nu_local - nu_mw)**2 + gamma2)
        signal[i] = np.average(lor, weights=weights)

    idx_max = np.argmax(signal)
    B_guess = B0_scan[idx_max]
    dnu_dB_approx = (
        G_E * MU_B * B_guess
        / np.sqrt((DELTA_HFS / 2)**2 + (G_E * MU_B * B_guess)**2)
        * 2 * G_E * MU_B
    )
    gamma_B_guess = gamma_line / max(dnu_dB_approx, 1.0)

    try:
        popt, pcov = curve_fit(
            lorentzian, B0_scan, signal,
            p0=[B_guess, gamma_B_guess, signal.max() - signal.min(), signal.min()],
            maxfev=10000,
        )
        B_res = popt[0]
        fit_width_B = abs(popt[1])
        fit_ok = True
    except RuntimeError:
        B_res = B_guess
        fit_width_B = gamma_B_guess
        popt = [B_guess, gamma_B_guess, signal.max(), 0]
        fit_ok = False

    delta_hfs_extracted = extract_hfs(B_res, nu_mw)
    delta_hfs_shift = delta_hfs_extracted - DELTA_HFS

    fit_curve = lorentzian(B0_scan, *popt)
    residuals = signal - fit_curve

    fit_width_nu = fit_width_B * dnu_dB_approx

    return {
        'B0_scan': B0_scan,
        'signal': signal,
        'B_res': B_res,
        'delta_hfs_extracted': delta_hfs_extracted,
        'delta_hfs_shift': delta_hfs_shift,
        'fit_params': popt,
        'fit_width_B': fit_width_B,
        'fit_width_nu': fit_width_nu,
        'fit_ok': fit_ok,
        'magnet_name': magnet.name,
        'ps_dist_name': ps_dist.name,
        'fit_curve': fit_curve,
        'residuals': residuals,
    }
