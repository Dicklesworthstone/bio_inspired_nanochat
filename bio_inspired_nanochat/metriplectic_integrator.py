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


def field(z: np.ndarray, omega=OMEGA, gC=GAMMA_C, gB=GAMMA_B, *, L_fn=L_op, M_fn=M_op) -> np.ndarray:
    """The continuous metriplectic vector field ż = L∇E + M∇S (for the explicit Euler baseline).

    ``L_fn``/``M_fn`` are injectable so the guards (below) can be exercised with a degeneracy-breaking
    operator — the case the deterministic fallback exists for.
    """
    return L_fn(omega) @ grad_E(z) + M_fn(z, gC, gB) @ grad_S(z)


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
    L_fn=L_op,
    M_fn=M_op,
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
        rhs = z + dt * (L_fn(omega) @ gE + M_fn(zbar, gC, gB) @ gS)
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


# --------------------------------------------------------------------------- #
# Runtime monitor + guards + deterministic fallback (beads 0642.1.2.2 / 0642.1.2.3).
# --------------------------------------------------------------------------- #
def degeneracy_residuals(z: np.ndarray, *, L_fn=L_op, M_fn=M_op,
                         omega=OMEGA, gC=GAMMA_C, gB=GAMMA_B) -> tuple[float, float]:
    """`(‖L·∇S‖, ‖M·∇E‖)` — the degeneracy residuals (both 0 for the structural operators)."""
    L, M = L_fn(omega), M_fn(z, gC, gB)
    return float(np.linalg.norm(L @ grad_S(z))), float(np.linalg.norm(M @ grad_E(z)))


@dataclass(frozen=True)
class GuardThresholds:
    """Per-step tolerances for the conservation/entropy/degeneracy guards."""

    eps_E: float = 1e-8    # max |E(z') − E(z)| (energy drift)
    eps_S: float = 1e-10   # entropy production must be ≥ −eps_S
    eps_D: float = 1e-8    # degeneracy residuals ‖L∇S‖, ‖M∇E‖ must be ≤ eps_D


@dataclass
class StepRecord:
    """Auditable per-step monitor record (the runtime stability certificate evidence)."""

    step: int
    E: float
    S: float
    F: float
    entropy_production: float   # S(z') − S(z), should be ≥ −eps_S
    energy_drift: float         # E(z') − E(z), should be ≈ 0
    res_L_gradS: float
    res_M_gradE: float
    used_fallback: bool
    breach: str                 # "" if all guards passed, else which guard tripped


def guarded_step(
    z: np.ndarray, dt: float, step: int, thr: GuardThresholds, *,
    omega=OMEGA, gC=GAMMA_C, gB=GAMMA_B, T=TEMP, L_fn=L_op, M_fn=M_op,
) -> tuple[np.ndarray, StepRecord]:
    """One discrete-gradient step under the guards; revert to the clamped-Euler baseline on a breach.

    Budgeted-mode discipline: a (learned) `L/M` that violates degeneracy, or a step that drifts
    energy or destroys entropy beyond tolerance, must NEVER corrupt training — the step deterministically
    falls back to the safe `vg9` Euler baseline and the event is recorded.
    """
    res_ls, res_me = degeneracy_residuals(z, L_fn=L_fn, M_fn=M_fn, omega=omega, gC=gC, gB=gB)
    z_prop = discrete_gradient_step(z, dt, omega=omega, gC=gC, gB=gB, L_fn=L_fn, M_fn=M_fn).z_next
    d_e = energy(z_prop) - energy(z)
    d_s = entropy(z_prop) - entropy(z)

    breach = ""
    if res_ls > thr.eps_D or res_me > thr.eps_D:
        breach = "degeneracy"
    elif abs(d_e) > thr.eps_E:
        breach = "energy_drift"
    elif d_s < -thr.eps_S:
        breach = "entropy"

    if breach:
        # Deterministic fallback: the clamped-Euler baseline step (vg9.5/vg9.7), the safe default.
        z_next = z + dt * field(z, omega, gC, gB, L_fn=L_fn, M_fn=M_fn)
        used_fallback = True
    else:
        z_next, used_fallback = z_prop, False

    rec = StepRecord(
        step=step, E=energy(z_next), S=entropy(z_next), F=free_energy(z_next, T),
        entropy_production=entropy(z_next) - entropy(z), energy_drift=energy(z_next) - energy(z),
        res_L_gradS=res_ls, res_M_gradE=res_me, used_fallback=used_fallback, breach=breach,
    )
    return z_next, rec


class LyapunovMonitor:
    """Accumulates per-step records and asserts the free-energy Lyapunov obligation holds.

    The auditable evidence for the stability obligation (0642.1.2.2): `F = E − T·S` must be
    non-increasing within tolerance, energy conserved, entropy non-decreasing — logged per step so the
    guarantee is something you can SEE, not just a paper claim. Pair with the structured-logging
    schema (`run_logging.TrainingTelemetry`) to emit one record per step.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self.records: list[StepRecord] = []
        self.tol = tol

    def append(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def free_energy_nonincreasing(self) -> bool:
        f = [r.F for r in self.records]
        return all(f[i + 1] <= f[i] + self.tol for i in range(len(f) - 1))

    def assert_lyapunov(self) -> None:
        if not self.free_energy_nonincreasing():
            bad = next(i for i in range(len(self.records) - 1)
                       if self.records[i + 1].F > self.records[i].F + self.tol)
            raise AssertionError(
                f"free-energy Lyapunov obligation breached at step {self.records[bad].step}: "
                f"F {self.records[bad].F:.6g} -> {self.records[bad + 1].F:.6g}"
            )

    def summary(self) -> dict:
        if not self.records:
            return {"steps": 0}
        return {
            "steps": len(self.records),
            "max_energy_drift": max(abs(r.energy_drift) for r in self.records),
            "min_entropy_production": min(r.entropy_production for r in self.records),
            "max_degeneracy_residual": max(max(r.res_L_gradS, r.res_M_gradE) for r in self.records),
            "n_fallbacks": sum(1 for r in self.records if r.used_fallback),
            "lyapunov_ok": self.free_energy_nonincreasing(),
        }


def run_monitored(
    z0: np.ndarray, dt: float, steps: int, *,
    thresholds: GuardThresholds | None = None, **kw,
) -> tuple[np.ndarray, LyapunovMonitor]:
    """Integrate under the guards + monitor; return the trajectory and the populated monitor."""
    thr = thresholds or GuardThresholds()
    z = np.asarray(z0, dtype=np.float64).copy()
    traj = [z.copy()]
    monitor = LyapunovMonitor()
    for step in range(steps):
        z, rec = guarded_step(z, dt, step, thr, **kw)
        monitor.append(rec)
        traj.append(z.copy())
    return np.array(traj), monitor


# --------------------------------------------------------------------------- #
# Free-energy DELIBERATION + energy-based decoding (reference API, bead r00r.1.1).
# The convergence + halting guarantees are exactly the discrete free-energy Lyapunov property of
# the structure-preserving step (0642.1.1 / 0642.1.2.1). The live per-token decode wiring (toggle +
# fallback) is r00r.1.2; this is the executable spec the design note is written against.
# --------------------------------------------------------------------------- #
@dataclass
class DeliberationResult:
    z: np.ndarray            # the relaxed synaptic state
    iters: int               # deliberation steps actually taken (the effort signal)
    F_final: float           # final free energy (the confidence signal — lower = more self-consistent)
    F_drop: float            # F(z0) − F_final, the total free energy released
    halted_converged: bool   # True if stopped on dF < eps, False if the budget was hit


def deliberate(
    z: np.ndarray, dt: float, *, eps: float = 1e-4, max_iters: int = 64, T: float = TEMP,
    thresholds: GuardThresholds | None = None, **kw,
) -> DeliberationResult:
    """Run extra free-energy-minimization steps on the synaptic state until it self-consistently
    relaxes (``|ΔF| < eps``) or the compute budget ``max_iters`` is hit ("think longer on hard tokens").

    Convergence is guaranteed by Thrust A: the structure-preserving step makes ``F`` monotonically
    non-increasing and bounded below on the compact energy shell (0642.1.1 §5), so ``F`` converges and
    ``|ΔF| → 0`` — the halt always fires within a bounded number of steps. ``eps`` and ``max_iters``
    are the compute-vs-quality knob: smaller ``eps`` / larger budget ⟹ more deliberation.
    """
    thr = thresholds or GuardThresholds()
    z = np.asarray(z, dtype=np.float64).copy()
    f_start = free_energy(z, T)
    f_prev = f_start
    used, converged = 0, False
    for k in range(max_iters):
        z, rec = guarded_step(z, dt, k, thr, T=T, **kw)
        used = k + 1
        if abs(f_prev - rec.F) < eps:
            converged = True
            f_prev = rec.F
            break
        f_prev = rec.F
    return DeliberationResult(z=z, iters=used, F_final=f_prev, F_drop=f_start - f_prev,
                              halted_converged=converged)


def boltzmann_weights(free_energies, kT: float = 1.0) -> np.ndarray:
    """Energy-based (Boltzmann) decoding weights ``p ∝ exp(−F/kT)`` over candidate free energies.

    Lower free energy ⟹ a more self-consistent candidate ⟹ higher probability. ``kT`` is the decoding
    temperature (``kT → 0`` ⟹ argmin-F greedy; large ``kT`` ⟹ uniform). Constraints enter as extra
    additive energy terms in ``F`` (energy-based constrained generation). Numerically stabilized.
    """
    f = np.asarray(free_energies, dtype=np.float64)
    if kT <= 0.0:
        raise ValueError(f"kT must be positive, got {kT}")
    logits = -f / kT
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()
