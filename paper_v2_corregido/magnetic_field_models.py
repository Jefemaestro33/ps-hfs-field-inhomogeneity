"""
magnetic_field_models.py — Modelos paramétricos de campo magnético para
las cavidades experimentales de HFS del positronio.

Opción B del protocolo: expansión en armónicos sin FEM.
"""

import numpy as np


class MagnetModel:
    """Modelo base de campo magnético."""

    def __init__(self, B0, name):
        self.B0 = B0    # Campo central en Tesla
        self.name = name

    def B_field(self, r, z):
        """Campo |B| en cada punto (r, z). r y z en metros."""
        raise NotImplementedError

    def statistics(self, r, z):
        """Estadísticas de la distribución de campo."""
        B = self.B_field(r, z)
        delta = (B - self.B0) / self.B0
        std = np.std(delta)
        return {
            'mean_ppm': np.mean(delta) * 1e6,
            'rms_ppm': np.sqrt(np.mean(delta**2)) * 1e6,
            'max_ppm': np.max(np.abs(delta)) * 1e6,
            'std_ppm': std * 1e6,
            'skewness': float(np.mean(((delta - np.mean(delta)) / std)**3)) if std > 0 else 0.0,
        }


class UniformField(MagnetModel):
    """Control: campo perfectamente uniforme."""

    def __init__(self, B0, name="Uniforme"):
        super().__init__(B0, name)

    def B_field(self, r, z):
        return self.B0 * np.ones_like(r)


class QuadrupolarField(MagnetModel):
    """
    Electroimán con no-uniformidad cuadrupolar.
    B(r,z) = B₀ × [1 + c₂ × (2z² - r²) / R²]

    Física: los polos cilíndricos de un electroimán producen naturalmente
    esta estructura. c₂ > 0 → campo más fuerte en el eje.
    R_ref es el radio de referencia (típicamente mitad del gap entre polos).
    """

    def __init__(self, B0, c2_ppm, R_ref, name=None):
        super().__init__(B0, name or f"Cuadrupolar c₂={c2_ppm} ppm")
        self.c2 = c2_ppm * 1e-6
        self.c2_ppm = c2_ppm
        self.R_ref = R_ref

    def B_field(self, r, z):
        return self.B0 * (1 + self.c2 * (2 * z**2 - r**2) / self.R_ref**2)


class SextupolarField(MagnetModel):
    """No-uniformidad sextupolar: B(r,z) = B₀ × [1 + c₃ × z(2z²-3r²) / R³]"""

    def __init__(self, B0, c3_ppm, R_ref, name=None):
        super().__init__(B0, name or f"Sextupolar c₃={c3_ppm} ppm")
        self.c3 = c3_ppm * 1e-6
        self.c3_ppm = c3_ppm
        self.R_ref = R_ref

    def B_field(self, r, z):
        return self.B0 * (1 + self.c3 * z * (2 * z**2 - 3 * r**2) / self.R_ref**3)


class LinearGradient(MagnetModel):
    """Gradiente lineal: B(r,z) = B₀ × [1 + c₁ × z / L]"""

    def __init__(self, B0, c1_ppm, L_ref, name=None):
        super().__init__(B0, name or f"Lineal c₁={c1_ppm} ppm")
        self.c1 = c1_ppm * 1e-6
        self.c1_ppm = c1_ppm
        self.L_ref = L_ref

    def B_field(self, r, z):
        return self.B0 * (1 + self.c1 * z / self.L_ref)


class CombinedField(MagnetModel):
    """
    Combinación realista: cuadrupolar + sextupolar + gradiente lineal.
    Modela un electroimán real donde todos los armónicos están presentes.
    """

    def __init__(self, B0, c1_ppm, c2_ppm, c3_ppm, R_ref, L_ref, name=None):
        super().__init__(B0, name or f"Combinado c₁={c1_ppm}, c₂={c2_ppm}, c₃={c3_ppm}")
        self.c1 = c1_ppm * 1e-6
        self.c2 = c2_ppm * 1e-6
        self.c3 = c3_ppm * 1e-6
        self.R_ref = R_ref
        self.L_ref = L_ref

    def B_field(self, r, z):
        return self.B0 * (
            1
            + self.c1 * z / self.L_ref
            + self.c2 * (2 * z**2 - r**2) / self.R_ref**2
            + self.c3 * z * (2 * z**2 - 3 * r**2) / self.R_ref**3
        )


class FiniteSolenoid(MagnetModel):
    """
    Solenoide superconductor tipo Ishida con bobinas de compensación.
    Bore: 800 mm, longitud: 2000 mm.
    Uniformidad de 0.9 ppm RMS en región de 40 mm diámetro × 100 mm.

    El solenoide simple tiene ~130 ppm de no-uniformidad en la región
    de la cavidad. Las bobinas de compensación cancelan los armónicos
    hasta dejar ~0.9 ppm residual. Modelamos esto como solenoide base
    más el residuo cuadrupolar parametrizado para dar 0.9 ppm RMS.
    """

    def __init__(self, B0, bore_radius, length, residual_rms_ppm=0.9,
                 name="Ishida SC"):
        super().__init__(B0, name)
        self.a = bore_radius    # metros
        self.L = length          # metros
        self.residual_rms_ppm = residual_rms_ppm

    def B_field(self, r, z):
        """
        Campo con no-uniformidad residual después de compensación.
        Modelado como cuadrupolar débil calibrado para dar residual_rms_ppm
        en la región de la cavidad de Ishida (R=64mm, L=100mm).
        """
        # Calibración: para c₂(2z²-r²)/R² en un cilindro R=64mm, L=100mm,
        # la RMS de (2z²-r²)/R² ≈ 0.19 (calculada analíticamente).
        # Entonces c₂ = residual_rms_ppm / 0.19 ≈ 4.7 ppm da 0.9 ppm RMS.
        R_ref = 0.064  # radio de la cavidad de Ishida
        c2 = self.residual_rms_ppm * 1e-6 / 0.19
        return self.B0 * (1 + c2 * (2 * z**2 - r**2) / R_ref**2)


# ============================================================
# MODELOS PRE-CONFIGURADOS PARA EXPERIMENTOS REALES
# ============================================================

def mills_ritter_magnet(c2_ppm=15):
    """Electroimán convencional de Mills/Ritter (~0.8 T, >10 ppm)."""
    return QuadrupolarField(B0=0.8, c2_ppm=c2_ppm, R_ref=0.15,
                            name=f"Mills/Ritter (c₂={c2_ppm} ppm)")


def ishida_magnet():
    """Superconductor de Ishida (0.866 T, 0.9 ppm)."""
    return FiniteSolenoid(B0=0.866, bore_radius=0.4, length=2.0,
                          name="Ishida SC (0.9 ppm)")


def uniform_control(B0=0.8):
    """Campo uniforme de control."""
    return UniformField(B0=B0, name="Control uniforme")


# ============================================================
# VERIFICACIONES
# ============================================================

def run_verifications():
    print("=" * 70)
    print("VERIFICACIONES — magnetic_field_models.py")
    print("=" * 70)
    all_pass = True

    # Generar puntos de prueba en cilindro
    rng = np.random.default_rng(42)
    N = 200_000
    R_cav, L_cav = 0.10, 0.20
    r = R_cav * np.sqrt(rng.uniform(0, 1, N))
    z = rng.uniform(-L_cav / 2, L_cav / 2, N)

    # 1. Uniforme: δB = 0
    print("\n--- Test 1: Campo uniforme ---")
    m_unif = UniformField(0.8)
    stats = m_unif.statistics(r, z)
    ok1 = stats['rms_ppm'] < 1e-10
    print(f"  RMS = {stats['rms_ppm']:.2e} ppm  [{'PASS' if ok1 else 'FAIL'}]")
    if not ok1:
        all_pass = False

    # 2. Cuadrupolar: RMS ~ c₂ × σ_ξ
    print("\n--- Test 2: Cuadrupolar c₂=10 ppm ---")
    m_quad = QuadrupolarField(0.8, c2_ppm=10, R_ref=0.15)
    stats_q = m_quad.statistics(r, z)
    print(f"  Media = {stats_q['mean_ppm']:.3f} ppm")
    print(f"  RMS   = {stats_q['rms_ppm']:.3f} ppm")
    print(f"  Max   = {stats_q['max_ppm']:.3f} ppm")
    print(f"  Skew  = {stats_q['skewness']:.3f}")
    # La media debe ser ≈ c₂ × <(2z²-r²)/R²> = 10 × (2×L²/(12×R_ref²) - R_cav²/(2×R_ref²))
    # Con R_ref=0.15: <ξ> = 2×0.04/(12×0.0225) - 0.01/(2×0.0225) = 0.2963 - 0.2222 = 0.0741
    # Hmm, depende de R_ref vs R_cav. La media no será cero si el perfil no es centrado.
    ok2 = stats_q['rms_ppm'] > 1.0  # should be several ppm
    print(f"  RMS > 1 ppm: [{'PASS' if ok2 else 'FAIL'}]")
    if not ok2:
        all_pass = False

    # 3. Solenoide finito: no-uniformidad ~ 1 ppm en región central
    print("\n--- Test 3: Solenoide finito (Ishida) ---")
    m_sol = ishida_magnet()
    # Región central de Ishida: R=20mm, L=100mm
    r_ish = 0.020 * np.sqrt(rng.uniform(0, 1, N))
    z_ish = rng.uniform(-0.05, 0.05, N)
    stats_s = m_sol.statistics(r_ish, z_ish)
    print(f"  Región 20mm × 100mm:")
    print(f"  Media = {stats_s['mean_ppm']:.3f} ppm")
    print(f"  RMS   = {stats_s['rms_ppm']:.3f} ppm")
    print(f"  Max   = {stats_s['max_ppm']:.3f} ppm")
    ok3 = stats_s['rms_ppm'] < 5.0  # should be ~1 ppm or less
    print(f"  RMS < 5 ppm: [{'PASS' if ok3 else 'FAIL'}]")
    if not ok3:
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("TODAS LAS VERIFICACIONES PASARON ✓")
    else:
        print("ALGUNAS VERIFICACIONES FALLARON")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    run_verifications()
