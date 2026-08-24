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

Scope: the NumPy reference implements `0642.1.2.1`; :func:`torch_guarded_step` compiles that core
into the live synaptic recurrence for `0642.1.2`. The free-energy deliberation loop that consumes
the same structure is `r00r.1.2`. :func:`reversible_l_sequence` is the isolated, fallback-free
pure-`L` reconstruction core for `0642.1.2.6`; it is deliberately not the combined live recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from bio_inspired_nanochat.torch_imports import Tensor, torch

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


@dataclass(frozen=True)
class TorchStepRecord:
    """Vectorized guard evidence emitted by :func:`torch_guarded_step`.

    The tensors retain the leading shape of the live presynaptic state. ``breach_code`` is zero for
    a certified proposal, 1 for a non-finite proposal, 2 for energy drift, 3 for negative entropy
    production, and 4 for leaving the live physical domain. Keeping the evidence on-device avoids
    a synchronizing ``.item()`` in the model's hot path; callers reduce it only when logging.
    """

    energy_drift: Tensor
    entropy_production: Tensor
    free_energy_delta: Tensor
    res_L_gradS: Tensor
    res_M_gradE: Tensor
    fallback_mask: Tensor
    breach_code: Tensor

    def detached(self) -> "TorchStepRecord":
        """Return graph-free evidence suitable for retaining as runtime telemetry."""
        return TorchStepRecord(
            energy_drift=self.energy_drift.detach(),
            entropy_production=self.entropy_production.detach(),
            free_energy_delta=self.free_energy_delta.detach(),
            res_L_gradS=self.res_L_gradS.detach(),
            res_M_gradE=self.res_M_gradE.detach(),
            fallback_mask=self.fallback_mask.detach(),
            breach_code=self.breach_code.detach(),
        )


def _torch_scalar(value: float | Tensor, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _torch_midpoint_proposal(
    calcium: Tensor,
    buffer: Tensor,
    half_dt: Tensor,
    omega: Tensor,
    gC: Tensor,
    gB: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the combined ``L+M`` midpoint proposal and its singularity mask."""
    a11 = 1.0 + half_dt * gC
    a12 = -half_dt * omega
    a21 = half_dt * omega
    a22 = 1.0 + half_dt * gB
    rhs_c = (1.0 - half_dt * gC) * calcium + half_dt * omega * buffer
    rhs_b = -half_dt * omega * calcium + (1.0 - half_dt * gB) * buffer
    determinant = a11 * a22 - a12 * a21
    singular = determinant.abs() <= torch.finfo(calcium.dtype).eps
    safe_determinant = torch.where(singular, torch.ones_like(determinant), determinant)
    inverse_determinant = torch.reciprocal(safe_determinant)
    c_prop = (rhs_c * a22 - a12 * rhs_b) * inverse_determinant
    b_prop = (a11 * rhs_b - rhs_c * a21) * inverse_determinant
    return c_prop, b_prop, determinant, singular


def _torch_l_step_unchecked(
    calcium: Tensor,
    buffer: Tensor,
    heat: Tensor,
    omega: Tensor,
    dt: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the guard-free L-only midpoint map with the live operation order.

    This helper intentionally contains no physical-domain guard: a rotation does not preserve the
    live calcium/buffer quadrant. It is the exact discrete map used by the isolated reversible
    sequence below, not an alternative dispatch for the guarded live recurrence.
    """
    work_dtype = torch.float64 if calcium.dtype == torch.float64 else torch.float32
    c0 = calcium.to(work_dtype)
    b0 = buffer.to(work_dtype)
    h0 = heat.to(work_dtype)
    omega_t = omega.to(dtype=work_dtype, device=calcium.device)
    dt_t = dt.to(dtype=work_dtype, device=calcium.device)
    zero = torch.zeros((), dtype=work_dtype, device=calcium.device)
    c_prop, b_prop, _, _ = _torch_midpoint_proposal(
        c0, b0, 0.5 * dt_t, omega_t, zero, zero
    )
    c_next = c_prop.to(calcium.dtype).to(work_dtype)
    b_next = b_prop.to(buffer.dtype).to(work_dtype)
    mechanical0 = 0.5 * (c0.square() + b0.square())
    mechanical1 = 0.5 * (c_next.square() + b_next.square())
    h_next = (h0 + mechanical0 - mechanical1).to(heat.dtype)
    return c_next.to(calcium.dtype), b_next.to(buffer.dtype), h_next


def cayley_l_step(
    calcium: Tensor,
    buffer: Tensor,
    heat: Tensor,
    *,
    omega: float | Tensor = OMEGA,
    dt: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one guard-free L-only implicit-midpoint/Cayley step.

    The operation order matches the ``gC=gB=0`` proposal in :func:`torch_guarded_step`, including
    live-dtype quantization and the energy-shell update for ``heat``. Callers are responsible for
    restricting this reduced map to fallback-free states.
    """
    if calcium.shape != buffer.shape or calcium.shape != heat.shape:
        raise ValueError("calcium, buffer, and heat must have identical shapes")
    if not calcium.is_floating_point():
        raise TypeError(f"metriplectic state must be floating point, got {calcium.dtype}")
    if buffer.dtype != calcium.dtype or heat.dtype != calcium.dtype:
        raise TypeError("calcium, buffer, and heat must have identical dtypes")
    if buffer.device != calcium.device or heat.device != calcium.device:
        raise ValueError("calcium, buffer, and heat must be on the same device")
    work_dtype = torch.float64 if calcium.dtype == torch.float64 else torch.float32
    work_reference = calcium.to(work_dtype)
    omega_t = _torch_scalar(omega, work_reference)
    dt_t = _torch_scalar(dt, work_reference)
    if dt_t.numel() != 1 or float(dt_t.detach()) <= 0.0:
        raise ValueError(f"dt must be a positive scalar, got shape={tuple(dt_t.shape)}")
    return _torch_l_step_unchecked(calcium, buffer, heat, omega_t, dt_t)


def cayley_l_inverse(
    calcium_next: Tensor,
    buffer_next: Tensor,
    heat_next: Tensor,
    *,
    omega: float | Tensor = OMEGA,
    dt: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximately reconstruct the preceding state with the algebraic Cayley inverse.

    Replacing ``omega`` by ``-omega`` transposes the orthogonal midpoint map. Running the reverse
    step through the same energy-shell closure also reconstructs the heat ledger; blindly returning
    ``heat_next`` would miss live-dtype roundoff booked by the forward shell identity. Float16 and
    bfloat16 are rejected because their forward casts discard too much information for this
    inverse-only policy.
    """
    if calcium_next.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "cayley_l_inverse requires float32 or float64 state until low-precision replay exists"
        )
    return cayley_l_step(
        calcium_next,
        buffer_next,
        heat_next,
        omega=-omega,
        dt=dt,
    )


class _ReversibleLSequence(torch.autograd.Function):
    """First-order reverse-mode implementation saving one terminal L state."""

    @staticmethod
    def forward(
        ctx,
        calcium: Tensor,
        buffer: Tensor,
        heat: Tensor,
        omega: Tensor,
        dt: Tensor,
        steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        c_next, b_next, h_next = calcium.clone(), buffer.clone(), heat.clone()
        for _ in range(steps):
            c_next, b_next, h_next = _torch_l_step_unchecked(
                c_next, b_next, h_next, omega, dt
            )
        ctx.steps = steps
        ctx.save_for_backward(c_next, b_next, h_next, omega, dt)
        return c_next, b_next, h_next

    @staticmethod
    def backward(
        ctx,
        grad_calcium: Tensor | None,
        grad_buffer: Tensor | None,
        grad_heat: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, None]:
        c_next, b_next, h_next, omega, dt = ctx.saved_tensors
        grad_c = torch.zeros_like(c_next) if grad_calcium is None else grad_calcium
        grad_b = torch.zeros_like(b_next) if grad_buffer is None else grad_buffer
        grad_h = torch.zeros_like(h_next) if grad_heat is None else grad_heat
        grad_omega = torch.zeros_like(omega)
        grad_dt = torch.zeros_like(dt)
        omega_leaf = omega.detach().requires_grad_(True)
        dt_leaf = dt.detach().requires_grad_(True)

        for _ in range(ctx.steps):
            with torch.no_grad():
                c_prev, b_prev, h_prev = _torch_l_step_unchecked(
                    c_next, b_next, h_next, -omega, dt
                )
            c_leaf = c_prev.detach().requires_grad_(True)
            b_leaf = b_prev.detach().requires_grad_(True)
            h_leaf = h_prev.detach().requires_grad_(True)
            with torch.enable_grad():
                replay = _torch_l_step_unchecked(
                    c_leaf, b_leaf, h_leaf, omega_leaf, dt_leaf
                )
            local_grads = torch.autograd.grad(
                replay,
                (c_leaf, b_leaf, h_leaf, omega_leaf, dt_leaf),
                grad_outputs=(grad_c, grad_b, grad_h),
                create_graph=False,
            )
            grad_c, grad_b, grad_h = local_grads[:3]
            grad_omega = grad_omega + local_grads[3]
            grad_dt = grad_dt + local_grads[4]
            c_next, b_next, h_next = c_prev, b_prev, h_prev

        return grad_c, grad_b, grad_h, grad_omega, grad_dt, None


def reversible_l_sequence(
    calcium: Tensor,
    buffer: Tensor,
    heat: Tensor,
    *,
    omega: float | Tensor = OMEGA,
    dt: float | Tensor = 1.0,
    steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run ``steps`` Cayley L updates with constant saved activation storage in ``steps``.

    Backward reconstructs one preceding state at a time with :func:`cayley_l_inverse`, replays that
    local step, and immediately consumes its VJP. Only the terminal ``(C, B, heat)`` plus ``omega``
    and ``dt`` are saved. This is a first-order, fallback-free reduced-core primitive; it does not
    claim that the driven, dissipative, guarded live recurrence is reversible.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
    if calcium.shape != buffer.shape or calcium.shape != heat.shape:
        raise ValueError("calcium, buffer, and heat must have identical shapes")
    if calcium.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "reversible_l_sequence requires float32 or float64 state until low-precision replay exists"
        )
    if buffer.dtype != calcium.dtype or heat.dtype != calcium.dtype:
        raise TypeError("calcium, buffer, and heat must have identical dtypes")
    if buffer.device != calcium.device or heat.device != calcium.device:
        raise ValueError("calcium, buffer, and heat must be on the same device")
    omega_t = _torch_scalar(omega, calcium)
    dt_t = _torch_scalar(dt, calcium)
    if dt_t.numel() != 1 or float(dt_t.detach()) <= 0.0:
        raise ValueError(f"dt must be a positive scalar, got shape={tuple(dt_t.shape)}")
    return _ReversibleLSequence.apply(calcium, buffer, heat, omega_t, dt_t, steps)


def torch_guarded_step(
    calcium: Tensor,
    buffer: Tensor,
    heat: Tensor,
    *,
    dt: float = 1.0,
    omega: float | Tensor = OMEGA,
    gC: float | Tensor = GAMMA_C,
    gB: float | Tensor = GAMMA_B,
    temperature: float = TEMP,
    thresholds: GuardThresholds | None = None,
    fallback: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, TorchStepRecord]:
    """Advance a batched live ``(C, B, heat)`` state with guarded implicit midpoint.

    This is the torch-native compilation of :func:`guarded_step` used by the actual presynaptic
    recurrence. For the quadratic energy and linear entropy in the theory note, the Gonzalez
    discrete-gradient update has a closed-form 2×2 solve for ``(C, B)``. The heat coordinate is
    then obtained from the energy-shell identity, avoiding the Python fixed-point loop and keeping
    the operation differentiable and GPU-friendly.

    Guard tolerances are interpreted at the working dtype's machine precision. A breached element
    selects its supplied live clamped-Euler ``fallback`` byte-for-byte. If no fallback is supplied,
    the function constructs the reference clamped forward-Euler step.
    """
    if calcium.shape != buffer.shape or calcium.shape != heat.shape:
        raise ValueError(
            "calcium, buffer, and heat must have identical shapes; "
            f"got {calcium.shape}, {buffer.shape}, and {heat.shape}"
        )
    if not calcium.is_floating_point():
        raise TypeError(f"metriplectic state must be floating point, got {calcium.dtype}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    thr = thresholds or GuardThresholds()
    work_dtype = torch.float64 if calcium.dtype == torch.float64 else torch.float32
    c0 = calcium.to(work_dtype)
    b0 = buffer.to(work_dtype)
    h0 = heat.to(work_dtype)
    dt_t = _torch_scalar(dt, c0)
    omega_t = _torch_scalar(omega, c0)
    gc_t = _torch_scalar(gC, c0)
    gb_t = _torch_scalar(gB, c0)
    half_dt = 0.5 * dt_t

    # Implicit-midpoint 2×2 system:
    #   (1+a*gC) C' - a*w B' = (1-a*gC) C + a*w B
    #    a*w C' + (1+a*gB) B' = -a*w C + (1-a*gB) B
    c_prop, b_prop, determinant, singular = _torch_midpoint_proposal(
        c0, b0, half_dt, omega_t, gc_t, gb_t
    )

    # Quantize C/B to the live state dtype before closing the energy shell. This makes the guard
    # certify the values that are actually persisted (including bf16), not an fp32 proposal that
    # would subsequently round differently.
    c_candidate = c_prop.to(calcium.dtype).to(work_dtype)
    b_candidate = b_prop.to(buffer.dtype).to(work_dtype)

    # The exact shell identity is both cheaper and more accurate than accumulating h with the
    # midpoint dissipation formula. It is algebraically identical for this quadratic core.
    mechanical0 = 0.5 * (c0.square() + b0.square())
    mechanical1 = 0.5 * (c_candidate.square() + b_candidate.square())
    h_candidate = (h0 + mechanical0 - mechanical1).to(heat.dtype).to(work_dtype)
    energy0 = mechanical0 + h0
    energy1 = mechanical1 + h_candidate
    energy_drift = energy1 - energy0
    entropy_production = h_candidate - h0
    free_energy_delta = energy_drift - temperature * entropy_production

    # L·∇S and M·∇E cancel structurally for this parameterization. Record explicit zero tensors so
    # the live telemetry schema is identical to the reference monitor's evidence table.
    res_l_grads = torch.zeros_like(energy_drift)
    res_m_grade = torch.zeros_like(energy_drift)
    finite = (
        torch.isfinite(c_candidate)
        & torch.isfinite(b_candidate)
        & torch.isfinite(h_candidate)
        & torch.isfinite(determinant)
        & ~singular
    )
    scale = torch.maximum(torch.maximum(energy0.abs(), energy1.abs()), torch.ones_like(energy0))
    dtype_tol = 16.0 * torch.finfo(calcium.dtype).eps * scale
    energy_tol = torch.maximum(_torch_scalar(thr.eps_E, energy0), dtype_tol)
    entropy_tol = torch.maximum(_torch_scalar(thr.eps_S, energy0), dtype_tol)
    energy_breach = energy_drift.abs() > energy_tol
    entropy_breach = entropy_production < -entropy_tol
    domain_breach = (
        (c_candidate < 0.0)
        | (b_candidate < 0.0)
        | (b_candidate > 1.0)
        | (h_candidate < -entropy_tol)
    )
    nonfinite_breach = ~finite
    fallback_mask = nonfinite_breach | energy_breach | entropy_breach | domain_breach
    breach_code = torch.zeros_like(energy_drift, dtype=torch.int8)
    breach_code = torch.where(domain_breach, torch.full_like(breach_code, 4), breach_code)
    breach_code = torch.where(entropy_breach, torch.full_like(breach_code, 3), breach_code)
    breach_code = torch.where(energy_breach, torch.full_like(breach_code, 2), breach_code)
    breach_code = torch.where(nonfinite_breach, torch.ones_like(breach_code), breach_code)

    if fallback is None:
        c_fallback = (c0 + dt_t * (omega_t * b0 - gc_t * c0)).clamp_min(0.0)
        b_fallback = (b0 + dt_t * (-omega_t * c0 - gb_t * b0)).clamp(0.0, 1.0)
        h_fallback = h0
    else:
        c_fallback, b_fallback, h_fallback = (
            value.to(work_dtype) for value in fallback
        )
        if (
            c_fallback.shape != calcium.shape
            or b_fallback.shape != calcium.shape
            or h_fallback.shape != calcium.shape
        ):
            raise ValueError("every fallback tensor must match the live state shape")

    c_next = torch.where(fallback_mask, c_fallback, c_candidate).to(calcium.dtype)
    b_next = torch.where(fallback_mask, b_fallback, b_candidate).to(buffer.dtype)
    h_next = torch.where(fallback_mask, h_fallback, h_candidate).to(heat.dtype)
    record = TorchStepRecord(
        energy_drift=energy_drift,
        entropy_production=entropy_production,
        free_energy_delta=free_energy_delta,
        res_L_gradS=res_l_grads,
        res_M_gradE=res_m_grade,
        fallback_mask=fallback_mask,
        breach_code=breach_code,
    )
    return c_next, b_next, h_next, record


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
