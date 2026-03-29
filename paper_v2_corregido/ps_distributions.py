"""
ps_distributions.py — Distribuciones espaciales del positronio en la cavidad.
"""

import numpy as np


class PsDistribution:
    """Distribución espacial del positronio en la cavidad."""

    def __init__(self, name):
        self.name = name

    def sample(self, N, R_cav, L_cav, rng=None):
        """
        Genera N puntos (r, z) según la distribución.
        R_cav: radio de la cavidad (metros)
        L_cav: longitud de la cavidad (metros)
        rng: numpy Generator (para reproducibilidad)
        Retorna: (r, z) arrays de longitud N.
        """
        raise NotImplementedError


class UniformPs(PsDistribution):
    """Ps termalizado, distribución uniforme en la cavidad cilíndrica."""

    def __init__(self):
        super().__init__("Ps uniforme (termalizado)")

    def sample(self, N, R_cav, L_cav, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        r = R_cav * np.sqrt(rng.uniform(0, 1, N))
        z = rng.uniform(-L_cav / 2, L_cav / 2, N)
        return r, z


class NonThermalizedPs(PsDistribution):
    """
    Ps no termalizado, concentrado cerca de la fuente.
    Distribución exponencial desde la pared z = -L/2.
    """

    def __init__(self, lambda_mm=30):
        super().__init__(f"Ps no termalizado (λ={lambda_mm} mm)")
        self.lam = lambda_mm * 1e-3  # metros

    def sample(self, N, R_cav, L_cav, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        r = R_cav * np.sqrt(rng.uniform(0, 1, N))
        # z concentrado cerca de z = -L/2 con rebote en paredes
        z_raw = rng.exponential(self.lam, N)
        z = -L_cav / 2 + np.mod(z_raw, L_cav)
        return r, z


class GaussianCorePs(PsDistribution):
    """Ps concentrado cerca del centro (termalización parcial)."""

    def __init__(self, sigma_mm=40):
        super().__init__(f"Ps gaussiano (σ={sigma_mm} mm)")
        self.sigma = sigma_mm * 1e-3

    def sample(self, N, R_cav, L_cav, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Generar con rechazo
        factor = 4
        r_raw = np.abs(rng.normal(0, self.sigma, N * factor))
        z_raw = rng.normal(0, self.sigma, N * factor)
        mask = (r_raw < R_cav) & (np.abs(z_raw) < L_cav / 2)
        r = r_raw[mask][:N]
        z = z_raw[mask][:N]
        if len(r) < N:
            extra = N - len(r)
            r = np.concatenate([r, R_cav * np.sqrt(rng.uniform(0, 1, extra))])
            z = np.concatenate([z, rng.uniform(-L_cav / 2, L_cav / 2, extra)])
        return r, z


def run_verifications():
    print("=" * 70)
    print("VERIFICACIONES — ps_distributions.py")
    print("=" * 70)
    all_pass = True
    rng = np.random.default_rng(42)
    N = 500_000
    R, L = 0.10, 0.20

    for Dist in [UniformPs(), NonThermalizedPs(30), GaussianCorePs(40)]:
        r, z = Dist.sample(N, R, L, rng)
        ok_r = np.all(r >= 0) and np.all(r <= R)
        ok_z = np.all(z >= -L / 2) and np.all(z <= L / 2)
        ok_n = len(r) == N
        ok = ok_r and ok_z and ok_n
        print(f"\n  {Dist.name}:")
        print(f"    N={len(r)}, r∈[{r.min():.4f}, {r.max():.4f}], "
              f"z∈[{z.min():.4f}, {z.max():.4f}]")
        print(f"    <r>={np.mean(r)*1e3:.2f}mm, <z>={np.mean(z)*1e3:.2f}mm  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if not ok:
            all_pass = False

    print("\n" + "=" * 70)
    print("TODAS LAS VERIFICACIONES PASARON ✓" if all_pass else "FALLOS")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    run_verifications()
