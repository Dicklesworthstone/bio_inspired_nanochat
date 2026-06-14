"""The master neural SDE — capstone reference object (bead `0642.11.1`).

`docs/theory/master_sde.md` writes the synaptic transformer as a single object — a
**stochastic, gauge-covariant, metriplectic neural SDE on a fiber bundle** —

        ∇^A z  =  [ L(z)·∇E(z) + M(z)·∇S(z) ] dt  +  σ(z)·dW ,                 (★)

and reads each theory thrust as a *restriction* of it:

    A (metriplectic)        σ → 0, 𝒜 → 0           the deterministic vertical drift
    E (stochastic thermo)   keep σ, σσᵀ = 2T·M      the vertical diffusion (FDR ↔ A's M)
    F (singular pert.)       1/ε fast-drift, ε → 0   the slow-manifold reduction of the drift
    D (gauge)               keep ∇^A                the horizontal connection (gauge-covariance)

This module is the *executable* side of that note: it builds (★) on the metriplectic calcium core
`z = (C, B, h)` (reusing `metriplectic_integrator` so the A-correspondence is exact, not a re-impl)
and provides the four restriction primitives so the correspondences are **falsifiable**
(`tests/test_master_sde.py`). It is pure-`numpy`, dependency-light, and introduces no runtime
behavior — it is a unifying reading of dynamics already specified by the per-thrust notes/modules.
"""

from __future__ import annotations

import numpy as np

from bio_inspired_nanochat.metriplectic_integrator import (
    GAMMA_B,
    GAMMA_C,
    OMEGA,
    TEMP,
    L_op,
    M_op,
    energy,
    field,
    grad_E,
    grad_S,
)

# --------------------------------------------------------------------------- #
# The master object (★) on the calcium core z = (C, B, h).
# --------------------------------------------------------------------------- #
def master_drift(z: np.ndarray, *, omega: float = OMEGA, gC: float = GAMMA_C, gB: float = GAMMA_B) -> np.ndarray:
    """The vertical drift `L(z)·∇E(z) + M(z)·∇S(z)` of (★) — the deterministic GENERIC field.

    Assembled from the same `L, M, ∇E, ∇S` as `metriplectic_integrator`, so the Thrust-A restriction
    (`σ → 0`, trivial connection) is *exactly* the metriplectic field by construction; the test guards
    that this assembly is correct (equals `metriplectic_integrator.field`).
    """
    z = np.asarray(z, dtype=np.float64)
    return L_op(omega) @ grad_E(z) + M_op(z, gC, gB) @ grad_S(z)


def noise_generator(z: np.ndarray, *, gC: float = GAMMA_C, gB: float = GAMMA_B) -> np.ndarray:
    """The `2×3` generator `G` with `GᵀG = M(z)`: rows `√γ_C·u`, `√γ_B·v` (`u,v` as in `M = γ_C uuᵀ+γ_B vvᵀ`)."""
    z = np.asarray(z, dtype=np.float64)
    C, B = z[0], z[1]
    u = np.array([1.0, 0.0, -C])
    v = np.array([0.0, 1.0, -B])
    return np.stack([np.sqrt(gC) * u, np.sqrt(gB) * v])  # (2, 3)


def diffusion_matrix(z: np.ndarray, *, T: float = TEMP, gC: float = GAMMA_C, gB: float = GAMMA_B) -> np.ndarray:
    """The diffusion `σ(z)` of (★) obeying the fluctuation–dissipation relation `σσᵀ = 2T·M(z)`.

    Built as `σ = √(2T)·Gᵀ` with `GᵀG = M`, so `σσᵀ = 2T·GᵀG = 2T·M` exactly (Thrust E's noise tied to
    Thrust A's dissipation `M` at temperature `T`).
    """
    G = noise_generator(z, gC=gC, gB=gB)
    return np.sqrt(2.0 * T) * G.T  # (3, 2)


def euler_maruyama_step(
    z: np.ndarray, dt: float, *, T: float = TEMP, sigma_on: bool = True,
    rng: np.random.Generator | None = None, omega: float = OMEGA, gC: float = GAMMA_C, gB: float = GAMMA_B,
) -> np.ndarray:
    """One Euler–Maruyama step of (★): `z + drift·dt + σ·√dt·ξ`, `ξ ~ N(0, I₂)`.

    With `sigma_on=False` this is the deterministic Euler step of the metriplectic field (the Thrust-A
    restriction `σ → 0`); with `sigma_on=True` it is the Thrust-E noisy dynamics.
    """
    z = np.asarray(z, dtype=np.float64)
    drift = master_drift(z, omega=omega, gC=gC, gB=gB)
    z_next = z + dt * drift
    if sigma_on:
        if rng is None:
            raise ValueError("euler_maruyama_step needs an rng when sigma_on=True")
        sigma = diffusion_matrix(z, T=T, gC=gC, gB=gB)
        z_next = z_next + np.sqrt(dt) * (sigma @ rng.standard_normal(2))
    return z_next


# --------------------------------------------------------------------------- #
# Thrust E restriction — fluctuation–dissipation ⟹ Gibbs / equipartition.
# --------------------------------------------------------------------------- #
def solve_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve the continuous Lyapunov equation `A·X + X·Aᵀ + Q = 0` for `X` (vectorized Kronecker solve)."""
    A = np.asarray(A, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n = A.shape[0]
    kron = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
    x = np.linalg.solve(kron, -Q.reshape(-1))
    return x.reshape(n, n)


def dissipative_block_stationary_cov(gamma: np.ndarray, *, T: float = TEMP) -> np.ndarray:
    """Stationary covariance of the linear dissipative block `dz = −Γz dt + σ dW`, `Γ = diag(gamma)`,
    under the FDR `σσᵀ = 2T·Γ`. Solves `−Γ·Cov − Cov·Γ + 2T·Γ = 0`; the answer is `T·I` (equipartition
    / Gibbs `∝ e^{−E/T}`) — the Thrust-E signature of (★)'s diffusion.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    Gamma = np.diag(gamma)
    Q = 2.0 * T * Gamma  # σσᵀ
    return solve_lyapunov(-Gamma, Q)


# --------------------------------------------------------------------------- #
# Thrust F restriction — the singular (slow-manifold) limit ε → 0.
# --------------------------------------------------------------------------- #
def slow_manifold_h0(y: float, *, k: float) -> float:
    """The critical-manifold (quasi-steady) value `x = h₀(y) = k·y` the fast variable is slaved to."""
    return k * y


def reduced_flow_step(y: float, dt: float, *, k: float, a: float, b: float) -> float:
    """One step of the Fenichel **reduced** slow flow `dy = (b·k − a)·y dt` (fast slaved to `h₀(y)=k·y`)."""
    return y + dt * (b * slow_manifold_h0(y, k=k) - a * y)


def fast_slow_step(
    x: float, y: float, dt: float, *, eps: float, k: float, a: float, b: float
) -> tuple[float, float]:
    """One step of the two-timescale system `ε·dx = −(x − k·y) dt`, `dy = (b·x − a·y) dt`.

    As `ε → 0` the fast `x` is slaved to `h₀(y) = k·y` (Fenichel) and `y` follows `reduced_flow_step`.
    This is the master drift under the `1/ε` fast-drift scaling (the Thrust-F restriction), in a
    minimal linear form whose slow manifold is explicit.
    """
    x_next = x + (dt / eps) * (-(x - k * y))
    y_next = y + dt * (b * x - a * y)
    return x_next, y_next


def integrate_fast_slow(
    y0: float, dt: float, steps: int, *, eps: float, k: float, a: float, b: float, x0: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the two-timescale system; return `(y_traj, x_traj)` (length `steps+1`)."""
    x = float(k * y0 if x0 is None else x0)
    y = float(y0)
    ys, xs = [y], [x]
    for _ in range(steps):
        x, y = fast_slow_step(x, y, dt, eps=eps, k=k, a=a, b=b)
        ys.append(y)
        xs.append(x)
    return np.array(ys), np.array(xs)


def integrate_reduced(y0: float, dt: float, steps: int, *, k: float, a: float, b: float) -> np.ndarray:
    """Integrate the reduced slow flow; return `y_traj` (length `steps+1`)."""
    y = float(y0)
    ys = [y]
    for _ in range(steps):
        y = reduced_flow_step(y, dt, k=k, a=a, b=b)
        ys.append(y)
    return np.array(ys)


# --------------------------------------------------------------------------- #
# Thrust D restriction — the gauge connection (horizontal transport).
# --------------------------------------------------------------------------- #
def gauge_matrix(theta: float) -> np.ndarray:
    """A structure-group element `U = blkdiag(R(θ), 1)` — an O(2) rotation of the calcium `(C, B)` plane,
    fixing the heat `h`. Orthogonal on `(C,B)` ⟹ preserves the metriplectic energy `E = ½(C²+B²) + h`.
    """
    c, s = np.cos(theta), np.sin(theta)
    U = np.eye(3)
    U[0, 0], U[0, 1] = c, -s
    U[1, 0], U[1, 1] = s, c
    return U


def parallel_transport(z: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Transport the fiber state across the base by the connection's frame change `z ↦ U·z`."""
    return np.asarray(U, dtype=np.float64) @ np.asarray(z, dtype=np.float64)


def apply_gauge(U: np.ndarray, L: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Co-transform the bracket operators under a gauge `U`: `(L, M) ↦ (U L Uᵀ, U M Uᵀ)`."""
    U = np.asarray(U, dtype=np.float64)
    return U @ L @ U.T, U @ M @ U.T


def covariant_drift_step(z: np.ndarray, dt: float, *, L: np.ndarray, M: np.ndarray) -> np.ndarray:
    """One drift step with the bracket operators passed **explicitly**: `z + (L·∇E(z) + M·∇S(z))·dt`.

    Gauge covariance (D): for any structure-group `U`, `step(U z; U L Uᵀ, U M Uᵀ) = U·step(z; L, M)` —
    the master object is form-invariant under fiber-frame changes (verified in the test).
    """
    z = np.asarray(z, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    return z + dt * (L @ grad_E(z) + M @ grad_S(z))


__all__ = [
    "master_drift",
    "noise_generator",
    "diffusion_matrix",
    "euler_maruyama_step",
    "solve_lyapunov",
    "dissipative_block_stationary_cov",
    "slow_manifold_h0",
    "reduced_flow_step",
    "fast_slow_step",
    "integrate_fast_slow",
    "integrate_reduced",
    "gauge_matrix",
    "parallel_transport",
    "apply_gauge",
    "covariant_drift_step",
    # re-exported for the A-restriction test
    "field",
    "energy",
]
