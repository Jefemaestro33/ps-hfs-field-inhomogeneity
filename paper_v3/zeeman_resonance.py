"""
zeeman_resonance.py — Simulación de la resonancia Zeeman completa.

Reproduce el procedimiento experimental:
1. Fija frecuencia de microondas ν_MW
2. Barre el campo magnético central B₀
3. Detecta aumento de aniquilación 2γ (señal de resonancia)
4. Ajusta Lorentziana para extraer B_res
5. Invierte Breit-Rabi para extraer Δ_HFS
"""

import numpy as np
from scipy.optimize import curve_fit
from breit_rabi import (
    transition_frequency_fast, extract_hfs, build_interpolator,
    DELTA_HFS, G_E, MU_B,
)


def lorentzian(x, x0, gamma, A, offset):
    """Lorentziana con offset: A × (γ/2)² / ((x-x₀)² + (γ/2)²) + offset"""
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
):
    """
    Simula la curva de resonancia del experimento Zeeman.

    Parameters
    ----------
    magnet : MagnetModel
    ps_dist : PsDistribution
    cavity_radius, cavity_length : float (metros)
    nu_mw : float (MHz) — frecuencia de microondas fija
    B0_scan : np.ndarray (Tesla) — valores de B₀ para el barrido
    gamma_line : float (MHz) — ancho de línea de la resonancia individual
    N_atoms : int — número de átomos de Ps
    rng : np.random.Generator

    Returns
    -------
    dict con: B0_scan, signal, B_res, delta_hfs_extracted,
              delta_hfs_shift, fit_params, fit_width
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 1. Muestrear posiciones de Ps
    r_atoms, z_atoms = ps_dist.sample(N_atoms, cavity_radius, cavity_length, rng)

    # 2. Pre-calcular el perfil de campo normalizado del imán
    #    B_local(r,z) = magnet.B_field(r,z) a B₀ = magnet.B0
    #    Cuando barremos B₀, escalamos: B_local = (B₀/magnet.B0) × magnet.B_field(r,z)
    B_profile = magnet.B_field(r_atoms, z_atoms)  # a B₀ nominal
    # Factor de no-uniformidad relativo: B_profile / magnet.B0
    B_ratio = B_profile / magnet.B0  # ~1 + δB/B₀

    # 3. Pre-construir interpolador de ν_trans(B)
    B_min = B0_scan[0] * np.min(B_ratio) * 0.999
    B_max = B0_scan[-1] * np.max(B_ratio) * 1.001
    B_min = max(B_min, 0.001)  # evitar B=0
    nu_interp = build_interpolator(B_min, B_max, N=20000)

    # 4. Construir la curva de resonancia
    gamma2 = (gamma_line / 2)**2
    signal = np.empty(len(B0_scan))

    for i, B0_val in enumerate(B0_scan):
        B_local = B0_val * B_ratio
        nu_local = nu_interp(B_local)
        signal[i] = np.mean(gamma2 / ((nu_local - nu_mw)**2 + gamma2))

    # 5. Ajustar Lorentziana para extraer B_res
    # Valor inicial: B del máximo de la señal
    idx_max = np.argmax(signal)
    B_guess = B0_scan[idx_max]
    # Estimar ancho en B: gamma_line / (dν/dB)
    dnu_dB_approx = G_E * MU_B * B_guess / np.sqrt((DELTA_HFS / 2)**2 + (G_E * MU_B * B_guess)**2) * 2 * G_E * MU_B
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

    # 6. Extraer Δ_HFS
    delta_hfs_extracted = extract_hfs(B_res, nu_mw)
    delta_hfs_shift = delta_hfs_extracted - DELTA_HFS

    # Convertir ancho en B a ancho en frecuencia
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
    }
