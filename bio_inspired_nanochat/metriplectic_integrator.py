"""Discrete-gradient (structure-preserving) integrator for the metriplectic core (bead 0642.1.2.1).

A naive Euler step destroys the conservation the GENERIC theory guarantees
(`docs/theory/metriplectic.md`): energy drifts and the Lyapunov certificate stops holding for the
*actual* code. This module integrates the metriplectic core `z = (C, B, h)`

        dz/dt = L(z)·∇E(z) + M(z)·∇S(z)

with a **Gonzalez discrete gradient** so that, at the **discrete** level (any step `dt`):

    * energy `E` is conserved EXACTLY (to machine precision), and
    * entropy `S` is monotone non-decreasing,

inheriting the continuous degeneracy `L·∇S = 0`, `M·∇E = 0` step-by-step. The update

        z' = z + dt·[ L(z̄)·∇̄E(z,z') + M(z̄)·∇̄S(z,z') ],   z̄ = (z+z')/2

is implicit; we solve it by a contraction fixed-point iteration. Because this core has a **quadratic**
energy and a **linear** entropy, the Gonzalez discrete gradient coincides with the midpoint gradient,
so `∇̄E = ∇E(z̄)` and the *structural* (pointwise) degeneracy `M(z̄)·∇E(z̄) = 0` makes the discrete
conservation exact — the integrator is the implicit midpoint rule in this case, and reduces to forward
Euler at first order. See `docs/theory/metriplectic.md` §4–§5; tested in
`tests/test_metriplectic_integrator.py`.

Scope (0642.1.2.1): the integrator object itself, operating on the metriplectic core that
`docs/theory/metriplectic.md` reduces the synaptic calcium↔buffer subsystem to. Wiring it into the
live synaptic step behind a toggle + fallback is the compile bead `0642.1.2`; the free-energy
deliberation loop that consumes it is `r00r.1.2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default core parameters (see metriplectic.md §0): ω = reversible calcium↔buffer exchange rate;
# γ_C, γ_B = the dissipative leak rates (1−ρc, 1−ρb).
OMEGA, GAMMA_C, GAMMA_B, TEMP = 1.0, 0.2, 0.1, 0.5


# --------------------------------------------------------------------------- #
# The metriplectic core: generators, operators, functionals.
# --------------------------------------------------------------------------- #
def grad_E(z: np.ndarray) -> np.ndarray:
    """∇E for E(z) = ½(C² + B²) + h."""
    return np.array([z[0], z[1], 1.0])


def grad_S(_z: np.ndarray) -> np.ndarray:
    """∇S for S(z) = h (constant)."""
    return np.array([0.0, 0.0, 1.0])


def L_op(omega: float = OMEGA) -> np.ndarray:
    """Skew Poisson operator: the lossless calcium↔buffer rotation (state-independent)."""
    return np.array([[0.0, omega, 0.0], [-omega, 0.0, 0.0], [0.0, 0.0, 0.0]])


def M_op(z: np.ndarray, gC: float = GAMMA_C, gB: float = GAMMA_B) -> np.ndarray:
    """PSD friction M = γ_C·uuᵀ + γ_B·vvᵀ, u=(1,0,−C), v=(0,1,−B); satisfies M·∇E = 0."""
    C, B = z[0], z[1]
    u = np.array([1.0, 0.0, -C])
    v = np.array([0.0, 1.0, -B])
    return gC * np.outer(u, u) + gB * np.outer(v, v)


def energy(z: np.ndarray) -> float:
    return 0.5 * (z[0] * z[0] + z[1] * z[1]) + z[2]


def entropy(z: np.ndarray) -> float:
    return float(z[2])


def free_energy(z: np.ndarray, T: float = TEMP) -> float:
    return energy(z) - T * entropy(z)


def field(z: np.ndarray, omega=OMEGA, gC=GAMMA_C, gB=GAMMA_B) -> np.ndarray:
    """The continuous metriplectic vector field ż = L∇E + M∇S (for the explicit Euler baseline)."""
    return L_op(omega) @ grad_E(z) + M_op(z, gC, gB) @ grad_S(z)


# --------------------------------------------------------------------------- #
# The Gonzalez discrete gradient.
# --------------------------------------------------------------------------- #
def discrete_gradient(grad, fun, z: np.ndarray, z_next: np.ndarray, *, tol: float = 1e-14) -> np.ndarray:
    """Gonzalez (1996) discrete gradient ∇̄f(z, z') of a scalar `fun` with smooth gradient `grad`.

    Satisfies the two defining properties exactly:
      (directional) (z'−z)·∇̄f = f(z') − f(z),
      (consistency) ∇̄f(z, z) = ∇f(z).
    For a quadratic `fun` the correction term vanishes and ∇̄f = ∇f((z+z')/2) (the midpoint gradient).
    """
    zbar = 0.5 * (z + z_next)
    dz = z_next - z
    g = grad(zbar)
    denom = float(dz @ dz)
    if denom < tol:
        return grad(z)
    correction = (fun(z_next) - fun(z) - float(g @ dz)) / denom
    return g + correction * dz


@dataclass
class StepResult:
    z_next: np.ndarray
    iters: int
    converged: bool


def discrete_gradient_step(
    z: np.ndarray,
    dt: float,
    *,
    omega: float = OMEGA,
    gC: float = GAMMA_C,
    gB: float = GAMMA_B,
    max_iter: int = 100,
    tol: float = 1e-13,
) -> StepResult:
    """One structure-preserving step z' = z + dt·[L(z̄)∇̄E + M(z̄)∇̄S], solved by fixed-point iteration.

    The map is a contraction for `dt` within the stability window (the leaks are dissipative and the
    rotation is bounded), so the iteration converges geometrically.
    """
    z = np.asarray(z, dtype=np.float64)
    z_next = z.copy()  # initial guess: z (≡ forward-Euler seed after one sweep)
    for it in range(1, max_iter + 1):
        zbar = 0.5 * (z + z_next)
        gE = discrete_gradient(grad_E, energy, z, z_next)
        gS = discrete_gradient(grad_S, entropy, z, z_next)
        rhs = z + dt * (L_op(omega) @ gE + M_op(zbar, gC, gB) @ gS)
        if np.max(np.abs(rhs - z_next)) < tol:
            return StepResult(rhs, it, True)
        z_next = rhs
    return StepResult(z_next, max_iter, False)


def integrate(z0: np.ndarray, dt: float, steps: int, **kw) -> np.ndarray:
    """Integrate the metriplectic core for `steps` discrete-gradient steps; return the trajectory."""
    z = np.asarray(z0, dtype=np.float64).copy()
    traj = [z.copy()]
    for _ in range(steps):
        z = discrete_gradient_step(z, dt, **kw).z_next
        traj.append(z.copy())
    return np.array(traj)


def euler_integrate(z0: np.ndarray, dt: float, steps: int, **kw) -> np.ndarray:
    """Forward-Euler baseline (the vg9-style step) for the energy-drift comparison."""
    z = np.asarray(z0, dtype=np.float64).copy()
    traj = [z.copy()]
    for _ in range(steps):
        z = z + dt * field(z, **kw)
        traj.append(z.copy())
    return np.array(traj)
