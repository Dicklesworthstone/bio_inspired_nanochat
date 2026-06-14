"""Each thrust is a restriction of the master neural SDE (capstone bead `0642.11.1`).

Makes the correspondence table of `docs/theory/master_sde.md` falsifiable: the master object

        ∇^A z = [ L(z)∇E + M(z)∇S ] dt + σ(z) dW          (★)

restricts to each thrust, and these tests check the restriction actually reduces to the thrust's
reference code (A) or its self-contained signature (E/F/D):

  - A (metriplectic)      σ→0, 𝒜→0   ⟹  the deterministic drift IS metriplectic_integrator.field
  - E (stochastic thermo) σσᵀ=2T·M    ⟹  Gibbs/equipartition stationary covariance T·I
  - F (singular pert.)    ε→0          ⟹  fast var slaved to h₀(y); slow flow → reduced flow
  - D (gauge)             keep ∇^A     ⟹  the covariant step commutes with a structure-group gauge

Run:  pytest tests/test_master_sde.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat import master_sde as ms
from bio_inspired_nanochat.metriplectic_integrator import (
    GAMMA_B,
    GAMMA_C,
    OMEGA,
    L_op,
    M_op,
    field,
    grad_E,
    grad_S,
)

pytestmark = pytest.mark.unit


def _grid(rng, n=40):
    # random core states z = (C, B, h) with h >= 0 (coercive energy shell)
    z = rng.standard_normal((n, 3))
    z[:, 2] = np.abs(z[:, 2])
    return z


# --------------------------------------------------------------------------- #
# A — the deterministic drift is exactly the metriplectic field (σ→0, 𝒜→0)
# --------------------------------------------------------------------------- #
def test_metriplectic_restriction_matches_reference():
    rng = np.random.default_rng(0)
    for z in _grid(rng):
        # the assembled master drift IS the metriplectic field (guards the assembly)
        assert np.allclose(ms.master_drift(z), field(z), atol=1e-12)
        # σ→0 Euler–Maruyama step == the deterministic Euler step of the field
        dt = 0.05
        assert np.allclose(
            ms.euler_maruyama_step(z, dt, sigma_on=False), z + dt * field(z), atol=1e-12
        )
        # the GENERIC degeneracy conditions hold (L∇S = 0, M∇E = 0)
        assert np.allclose(L_op(OMEGA) @ grad_S(z), 0.0, atol=1e-12)
        assert np.allclose(M_op(z, GAMMA_C, GAMMA_B) @ grad_E(z), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# E — fluctuation–dissipation ⟹ Gibbs stationary (σσᵀ = 2T·M, Cov = T·I)
# --------------------------------------------------------------------------- #
def test_thermo_restriction_fdr_holds_pointwise():
    rng = np.random.default_rng(1)
    T = 0.7
    for z in _grid(rng):
        sigma = ms.diffusion_matrix(z, T=T)
        M = M_op(z, GAMMA_C, GAMMA_B)
        assert np.allclose(sigma @ sigma.T, 2.0 * T * M, atol=1e-12), "σσᵀ must equal 2T·M (FDR)"


def test_thermo_restriction_gibbs_equipartition():
    T = 0.5
    cov = ms.dissipative_block_stationary_cov(np.array([GAMMA_C, GAMMA_B]), T=T)
    # FDR on the dissipative block ⟹ stationary covariance is T·I (equipartition / Gibbs ∝ e^{−E/T})
    assert np.allclose(cov, T * np.eye(2), atol=1e-9)


def test_solve_lyapunov_general_nondiagonal():
    # guard the exported solver beyond the diagonal case the Gibbs check exercises: for a general
    # (non-diagonal, non-symmetric) A and symmetric Q, the returned X must satisfy A·X + X·Aᵀ + Q = 0.
    A = np.array([[-1.0, 0.5], [0.2, -2.0]])
    Q = np.array([[1.0, 0.3], [0.3, 2.0]])
    X = ms.solve_lyapunov(A, Q)
    assert np.allclose(A @ X + X @ A.T + Q, 0.0, atol=1e-10)
    assert np.allclose(X, X.T, atol=1e-12), "Lyapunov solution for symmetric Q must be symmetric"


def test_thermo_restriction_gibbs_matches_simulation():
    # corroborate the analytic stationary covariance by simulating the OU dissipative block
    rng = np.random.default_rng(7)
    T, dt, steps = 0.5, 0.02, 120_000
    gamma = np.array([GAMMA_C, GAMMA_B])
    x = np.zeros(2)
    sig = np.sqrt(2.0 * T * gamma)  # diagonal σ for the decoupled block
    acc = np.zeros(2)
    burn = 2000
    for s in range(steps):
        x = x - dt * gamma * x + np.sqrt(dt) * sig * rng.standard_normal(2)
        if s >= burn:
            acc += x * x
    var = acc / (steps - burn)
    assert np.allclose(var, T, rtol=0.1), f"empirical equipartition var≈T failed: {var} vs {T}"


# --------------------------------------------------------------------------- #
# F — the singular limit ε→0 slaves the fast var and recovers the reduced flow
# --------------------------------------------------------------------------- #
def test_slow_manifold_restriction():
    k, a, b = 0.8, 1.0, 0.5
    dt, steps, y0 = 1e-3, 3000, 1.0
    y_red = ms.integrate_reduced(y0, dt, steps, k=k, a=a, b=b)

    errs = {}
    for eps in (0.02, 0.005):
        y_full, x_full = ms.integrate_fast_slow(y0, dt, steps, eps=eps, k=k, a=a, b=b)
        errs[eps] = float(np.max(np.abs(y_full - y_red)))
        # the fast variable is slaved to the critical manifold x = h₀(y) = k·y
        assert np.allclose(x_full[-1], ms.slow_manifold_h0(y_full[-1], k=k), atol=5e-3 + 2 * eps)

    # Fenichel: the reduction error is O(ε) — smaller ε ⟹ closer to the reduced flow
    assert errs[0.005] < errs[0.02]
    assert errs[0.005] < 1e-2


# --------------------------------------------------------------------------- #
# D — the covariant step commutes with a structure-group gauge (gauge covariance)
# --------------------------------------------------------------------------- #
def test_gauge_covariance_commutes():
    rng = np.random.default_rng(3)
    dt = 0.05
    L = L_op(OMEGA)
    for _ in range(10):
        z = rng.standard_normal(3)
        z[2] = abs(z[2])
        M = M_op(z, GAMMA_C, GAMMA_B)
        U = ms.gauge_matrix(float(rng.uniform(-np.pi, np.pi)))

        L_g, M_g = ms.apply_gauge(U, L, M)
        lhs = ms.covariant_drift_step(ms.parallel_transport(z, U), dt, L=L_g, M=M_g)
        rhs = ms.parallel_transport(ms.covariant_drift_step(z, dt, L=L, M=M), U)
        assert np.allclose(lhs, rhs, atol=1e-12), "covariant step must commute with the gauge"

    # the structure-group gauge preserves the metriplectic energy E = ½(C²+B²) + h
    z = np.array([1.3, -0.4, 0.9])
    U = ms.gauge_matrix(0.6)
    assert ms.energy(ms.parallel_transport(z, U)) == pytest.approx(ms.energy(z), abs=1e-12)
