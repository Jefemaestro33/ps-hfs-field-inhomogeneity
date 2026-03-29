"""
breit_rabi.py — Física exacta del estado fundamental del positronio en campo magnético.

Diagonalización numérica del Hamiltoniano 4×4 en la base producto
{|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩} donde el primer spin es el electrón y el segundo el positrón.

H = H_HFS + H_Zeeman

H_HFS = (Δ/4)(σ_e · σ_p) — acoplamiento hiperfino
H_Z   = g_e μ_B B (S_ez - S_pz)/ℏ — interacción Zeeman del positronio

En positronio los estados m=±1 del triplete NO se desplazan con B
porque (S_ez - S_pz)|↑↑⟩ = 0 y (S_ez - S_pz)|↓↓⟩ = 0.
Solo se mezclan los estados m=0 (singlete y triplete).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ============================================================
# CONSTANTES (CODATA 2018)
# ============================================================
DELTA_HFS = 203391.69      # MHz — predicción QED de la HFS
G_E = 2.00231930436256     # factor g del electrón
MU_B = 13996.24493         # MHz/T — magnetón de Bohr en unidades de frecuencia (μ_B/h)


# ============================================================
# HAMILTONIANO 4×4
# ============================================================

def hamiltonian_4x4(B):
    """
    Matriz 4×4 del Hamiltoniano del Ps fundamental en campo B (Tesla).
    Base: {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}.
    Retorna en MHz.

    H_HFS en la base producto:
      (↑↑,↑↑) = +Δ/4      (↑↓,↑↓) = -Δ/4     (↓↑,↓↑) = -Δ/4     (↓↓,↓↓) = +Δ/4
      (↑↓,↓↑) = +Δ/2      (↓↑,↑↓) = +Δ/2
    Todos los demás elementos son 0.

    H_Z = g_e μ_B B × diag(0, +1, -1, 0)
    porque (S_ez - S_pz)/ℏ tiene eigenvalores 0, +1, -1, 0 en la base producto.
    """
    D = DELTA_HFS
    V = G_E * MU_B * B  # Zeeman coupling en MHz

    H = np.array([
        [D / 4,       0,      0,      0    ],
        [0,      -D / 4 + V,  D / 2,  0    ],
        [0,       D / 2, -D / 4 - V,  0    ],
        [0,       0,      0,      D / 4    ],
    ], dtype=np.float64)

    return H


def eigenvalues(B):
    """
    4 eigenvalores ordenados de menor a mayor, en MHz.
    Retorna np.ndarray de shape (4,).
    """
    H = hamiltonian_4x4(B)
    evals = np.linalg.eigh(H)[0]
    return np.sort(evals)


def eigensystem(B):
    """
    Eigenvalores y eigenvectores ordenados por eigenvalor creciente.
    Retorna (evals, evecs) donde evecs[:,k] es el eigenvector k-ésimo.
    """
    H = hamiltonian_4x4(B)
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def _identify_states(B):
    """
    Identifica los 4 eigenstates a campo B > 0.

    La estructura de eigenvalores para B > 0:
      E₋  : el más bajo (mayormente singlete)
      E_{m=-1}, E_{m=+1}: casi degenerados en el medio (triplete m=±1, no mezclados)
      E₊  : el más alto (mayormente triplete m=0)

    Los estados m=±1 son |↑↑⟩ y |↓↓⟩ respectivamente, que son
    eigenstates exactos del Hamiltoniano para todo B.
    E(m=+1) = E(m=-1) = +Δ/4 exactamente (no dependen de B).

    Retorna dict con índices en el array de eigenvalores ordenados.
    """
    evals, evecs = eigensystem(B)

    # Los estados m=±1 son |↑↑⟩ (idx 0) y |↓↓⟩ (idx 3) en la base producto.
    # Identificar cuáles eigenstates tienen overlap > 0.99 con estos.
    idx_m_plus1 = None
    idx_m_minus1 = None
    for k in range(4):
        if abs(evecs[0, k])**2 > 0.99:  # |↑↑⟩ component
            idx_m_plus1 = k
        if abs(evecs[3, k])**2 > 0.99:  # |↓↓⟩ component
            idx_m_minus1 = k

    # Los otros dos son E₊ (el más alto de los dos) y E₋ (el más bajo)
    remaining = [k for k in range(4) if k != idx_m_plus1 and k != idx_m_minus1]
    if evals[remaining[0]] > evals[remaining[1]]:
        idx_plus = remaining[0]
        idx_minus = remaining[1]
    else:
        idx_plus = remaining[1]
        idx_minus = remaining[0]

    return {
        'm+1': idx_m_plus1,
        'm-1': idx_m_minus1,
        '+': idx_plus,
        '-': idx_minus,
    }


def transition_frequency(B):
    """
    Frecuencia de transición E₊ - E_{m=+1} en MHz.

    E₊ es el estado más alto (mayormente triplete m=0 mezclado con singlete).
    E_{m=+1} = +Δ/4 (estado |↑↑⟩, exactamente independiente de B).

    A B = 0.8 T debería dar ~2300-2500 MHz.
    """
    evals = eigenvalues(B)
    ids = _identify_states(B)
    return evals[ids['+']] - evals[ids['m+1']]


def transition_frequency_fast(B_array):
    """
    Versión vectorizada: calcula transition_frequency para un array o escalar de B.
    Usa el hecho de que E_{m=+1} = Δ/4 exactamente, y E₊ es el eigenvalor
    más alto del bloque 2×2 {|↑↓⟩, |↓↑⟩}.
    """
    B_arr = np.asarray(B_array, dtype=np.float64)
    D = DELTA_HFS
    V = G_E * MU_B * B_arr

    # El bloque 2×2 de los estados m=0 es:
    # [[-D/4 + V,  D/2],
    #  [D/2,      -D/4 - V]]
    # Eigenvalores: -D/4 ± sqrt((D/2)² + V²)
    E_plus = -D / 4 + np.sqrt((D / 2)**2 + V**2)
    E_m1 = D / 4  # exacto

    result = E_plus - E_m1
    return float(result) if result.ndim == 0 else result


def extract_hfs(B_res, nu_mw):
    """
    Dado B_res (campo de resonancia) y nu_mw (frecuencia de microondas),
    extrae Δ_HFS invirtiendo la relación de Breit-Rabi.

    Resuelve numéricamente: (Δ/2)(√(1 + (2 g_e μ_B B_res / Δ)²) - 1) = nu_mw
    """
    def residual(delta):
        x = 2 * G_E * MU_B * B_res / delta
        return (delta / 2) * (np.sqrt(1 + x**2) - 1) - nu_mw

    return brentq(residual, 50000, 500000)


# ============================================================
# INTERPOLADOR RÁPIDO
# ============================================================

_interp_cache = {}


def build_interpolator(B_min, B_max, N=10000):
    """
    Construye interpolador cúbico de transition_frequency(B).
    Cachea por rango para reutilización.
    """
    key = (B_min, B_max, N)
    if key not in _interp_cache:
        B_table = np.linspace(B_min, B_max, N)
        nu_table = transition_frequency_fast(B_table)
        _interp_cache[key] = interp1d(B_table, nu_table, kind='cubic',
                                       fill_value='extrapolate')
    return _interp_cache[key]


# ============================================================
# VERIFICACIONES
# ============================================================

def run_verifications():
    """Ejecuta todas las verificaciones obligatorias."""
    print("=" * 70)
    print("VERIFICACIONES — breit_rabi.py")
    print("=" * 70)
    all_pass = True

    # 1. B = 0: eigenvalores deben ser -3Δ/4 (singlete) y +Δ/4 (triplete ×3)
    print("\n--- Test 1: Eigenvalores a B = 0 ---")
    evals_0 = eigenvalues(0.0)
    D = DELTA_HFS
    expected = np.sort([-3 * D / 4, D / 4, D / 4, D / 4])
    err = np.max(np.abs(evals_0 - expected))
    ok = err < 1e-6
    print(f"  Eigenvalores: {evals_0}")
    print(f"  Esperados:    {expected}")
    print(f"  Error máximo: {err:.2e} MHz  [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        all_pass = False

    # Splitting = Δ
    splitting = evals_0[-1] - evals_0[0]
    err_split = abs(splitting - D)
    ok_split = err_split < 1e-6
    print(f"  Splitting E_max - E_min = {splitting:.6f} MHz (Δ = {D:.2f})")
    print(f"  Error: {err_split:.2e} MHz  [{'PASS' if ok_split else 'FAIL'}]")
    if not ok_split:
        all_pass = False

    # 2. B = 0.8 T: transition_frequency ~ 2300-2500 MHz
    print("\n--- Test 2: Frecuencia de transición a B = 0.8 T ---")
    B_test = 0.8
    nu = transition_frequency(B_test)
    nu_fast = float(transition_frequency_fast(B_test))
    ok2 = 2300 < nu < 2500
    print(f"  ν_trans(0.8 T) = {nu:.4f} MHz  ({nu/1e3:.4f} GHz)")
    print(f"  ν_trans_fast   = {nu_fast:.4f} MHz")
    print(f"  Rango esperado: 2300-2500 MHz  [{'PASS' if ok2 else 'FAIL'}]")
    err_fast = abs(nu - nu_fast)
    print(f"  Discrepancia fast vs exact: {err_fast:.6f} MHz  "
          f"[{'PASS' if err_fast < 0.01 else 'FAIL'}]")
    if not ok2 or err_fast > 0.01:
        all_pass = False

    # 3. m=±1 casi degenerados a B = 0.8 T
    print("\n--- Test 3: Degeneración de m=±1 ---")
    evals_08 = eigenvalues(0.8)
    ids = _identify_states(0.8)
    E_mp1 = evals_08[ids['m+1']]
    E_mm1 = evals_08[ids['m-1']]
    diff_m = abs(E_mp1 - E_mm1)
    ok3 = diff_m < 1e-6
    print(f"  E(m=+1) = {E_mp1:.6f} MHz")
    print(f"  E(m=-1) = {E_mm1:.6f} MHz")
    print(f"  |E(m=+1) - E(m=-1)| = {diff_m:.2e} MHz  [{'PASS' if ok3 else 'FAIL'}]")
    # Ambos deben ser exactamente Δ/4
    err_m = abs(E_mp1 - D / 4)
    print(f"  E(m=+1) - Δ/4 = {err_m:.2e} MHz  (debe ser ~0)")
    if not ok3:
        all_pass = False

    # 4. Round-trip consistency de extract_hfs
    print("\n--- Test 4: Round-trip extract_hfs ---")
    nu_test = 2338.0  # MHz
    # Encontrar B tal que transition_frequency(B) = nu_test
    B_res_test = brentq(lambda B: transition_frequency_fast(B) - nu_test,
                        0.1, 1.5)
    delta_extracted = extract_hfs(B_res_test, nu_test)
    err_rt = abs(delta_extracted - D)
    ok4 = err_rt < 0.01
    print(f"  ν_mw = {nu_test} MHz → B_res = {B_res_test:.6f} T")
    print(f"  Δ_HFS extraída = {delta_extracted:.6f} MHz")
    print(f"  Error vs QED:    {err_rt:.6f} MHz  [{'PASS' if ok4 else 'FAIL'}]")
    if not ok4:
        all_pass = False

    # 5. Fracción de mezcla singlete-triplete a B = 0.8 T
    print("\n--- Test 5: Mezcla singlete-triplete a B = 0.8 T ---")
    evals_08, evecs_08 = eigensystem(0.8)
    ids = _identify_states(0.8)
    # El estado |+⟩ es evecs_08[:, ids['+']]
    v_plus = evecs_08[:, ids['+']]
    # Componentes: |↑↑⟩=0, |↑↓⟩=1, |↓↑⟩=2, |↓↓⟩=3
    # Singlete = (|↑↓⟩ - |↓↑⟩)/√2, Triplete m=0 = (|↑↓⟩ + |↓↑⟩)/√2
    c_triplet_m0 = (v_plus[1] + v_plus[2]) / np.sqrt(2)
    c_singlet = (v_plus[1] - v_plus[2]) / np.sqrt(2)
    eps_sq = abs(c_singlet)**2
    V = G_E * MU_B * 0.8
    theta = np.arctan2(2 * V, D)
    eps_sq_analytic = np.sin(theta / 2)**2
    err_eps = abs(eps_sq - eps_sq_analytic)
    ok5 = err_eps < 1e-6
    print(f"  Estado |+⟩: componentes en base producto:")
    print(f"    |↑↑⟩: {abs(v_plus[0])**2:.8f}")
    print(f"    |↑↓⟩: {abs(v_plus[1])**2:.8f}")
    print(f"    |↓↑⟩: {abs(v_plus[2])**2:.8f}")
    print(f"    |↓↓⟩: {abs(v_plus[3])**2:.8f}")
    print(f"  Fracción triplete m=0: {abs(c_triplet_m0)**2:.8f}")
    print(f"  Fracción singlete:     {eps_sq:.8f}")
    print(f"  ε² analítico (sin²θ/2): {eps_sq_analytic:.8f}")
    print(f"  Error: {err_eps:.2e}  [{'PASS' if ok5 else 'FAIL'}]")
    if not ok5:
        all_pass = False

    # Resumen
    print("\n" + "=" * 70)
    if all_pass:
        print("TODAS LAS VERIFICACIONES PASARON ✓")
    else:
        print("ALGUNAS VERIFICACIONES FALLARON — revisar antes de continuar")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    run_verifications()
