"""Timescale-separation gauge + per-coupling separation table (bead 0642.10.1).

The singular-perturbation reductions of Thrust A and Thrust F
(`docs/theory/metriplectic.md`, `docs/theory/singular_perturbation.md`) are valid only where the
fast/slow timescales genuinely separate: `ε_k = τ_fast / τ_slow ≪ 1` at each coupling boundary. This
module **measures** that — the runtime `ε_k` gauge those notes deferred to here — and exports the
**separation table** over the synaptic hierarchy

        calcium ≪ release ≪ fast-weights ≪ slow-weights ≪ structure,

so the assumption can be checked against the real config rather than assumed. The fast end uses the
**fast-Jacobian eigenvalue gap** of the calcium↔buffer subsystem (`cb_spectral_radius`, `yw9.7`),
shared with the cusp ε-gauge (`0642.2.2.3`) and the slow-manifold monitor (`0642.2.2.2`).

`ε_k < eps_max` (default 0.5) flags a boundary as well-separated; a larger ratio means the two
timescales overlap and the reduction across that boundary is only approximate — an honest finding the
gauge surfaces rather than hides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bio_inspired_nanochat.synaptic import SynapticConfig, cb_spectral_radius
from bio_inspired_nanochat.torch_imports import torch

# The synaptic timescale hierarchy, fast → slow (the intended separation order).
HIERARCHY: tuple[str, ...] = ("calcium", "release", "fast_weights", "slow_weights", "structure")


def _efold_steps_from_decay(rho: float) -> float:
    """e-folding time (in steps) of a leaky integrator with per-step retention ``rho`` ∈ (0,1)."""
    rho = min(max(rho, 1e-9), 1.0 - 1e-12)
    return -1.0 / math.log(rho)


def calcium_eigenvalue_gap(cfg: SynapticConfig, n_beta: int = 21) -> float:
    """Fast-Jacobian eigenvalue gap `1 − ρ(M_cb)` of the calcium↔buffer subsystem (the fast end).

    A large gap ⟹ the fast subsystem relaxes quickly (short τ_fast); a gap → 0 (`ρ → 1`) ⟹ marginal,
    no separation. Worst case over the bilinear coupling `β = 1−BUF ∈ [0,1]`.
    """
    rho_c = torch.tensor(math.exp(-1.0 / cfg.tau_c), dtype=torch.float64)
    rho_b = torch.tensor(math.exp(-1.0 / cfg.tau_buf), dtype=torch.float64)
    a_on = torch.tensor(float(cfg.alpha_buf_on), dtype=torch.float64)
    a_off = torch.tensor(float(cfg.alpha_buf_off), dtype=torch.float64)
    betas = torch.linspace(0.0, 1.0, n_beta, dtype=torch.float64)
    rho = float(cb_spectral_radius(rho_c, rho_b, a_on, a_off, betas).max())
    return 1.0 - rho


def coupling_timescales(cfg: SynapticConfig) -> dict[str, float]:
    """Characteristic timescale (in steps) of each level of the synaptic hierarchy, from the config.

    - **calcium**: the calcium↔buffer relaxation time `1/gap` (fast-Jacobian eigenvalue gap), the
      rigorous fast-end measurement (falls back to `τ_c` if the gap underflows).
    - **release**: vesicle-pool refill time `τ_rrp`.
    - **fast_weights**: the eligibility-trace / fast-weight e-folding time from `post_trace_decay`.
    - **slow_weights**: the consolidation time `1/post_slow_lr` (the slow weight has no decay; this is
      the steps to accumulate an O(1) change).
    - **structure**: the MoE-lifecycle period `structural_interval`.
    """
    gap = calcium_eigenvalue_gap(cfg)
    calcium_tau = (1.0 / gap) if gap > 1e-9 else float(cfg.tau_c)
    return {
        "calcium": calcium_tau,
        "release": float(cfg.tau_rrp),
        "fast_weights": _efold_steps_from_decay(cfg.post_trace_decay),
        "slow_weights": 1.0 / max(cfg.post_slow_lr, 1e-12),
        "structure": float(cfg.structural_interval),
    }


@dataclass(frozen=True)
class SeparationRow:
    fast: str
    slow: str
    tau_fast: float
    tau_slow: float
    eps: float          # ε_k = τ_fast / τ_slow (≪ 1 ⟺ well separated)
    separated: bool


def separation_table(cfg: SynapticConfig, *, eps_max: float = 0.5) -> list[SeparationRow]:
    """The per-boundary separation table over the hierarchy (consecutive fast→slow couplings).

    Each row is `ε_k = τ_fast / τ_slow`; `separated = ε_k < eps_max`. Note the table reports the
    *intended* hierarchy order (HIERARCHY), so a row with `ε_k ≥ eps_max` (or > 1) flags a place where
    the configured timescales do not actually separate — a real, honest finding.
    """
    tau = coupling_timescales(cfg)
    rows: list[SeparationRow] = []
    for fast, slow in zip(HIERARCHY[:-1], HIERARCHY[1:]):
        tf, ts = tau[fast], tau[slow]
        eps = tf / ts if ts > 0 else float("inf")
        rows.append(SeparationRow(fast, slow, tf, ts, eps, eps < eps_max))
    return rows


def is_well_separated(cfg: SynapticConfig, *, eps_max: float = 0.5) -> bool:
    """True iff every consecutive boundary in the hierarchy is well-separated (`ε_k < eps_max`)."""
    return all(r.separated for r in separation_table(cfg, eps_max=eps_max))


def render_separation_table(cfg: SynapticConfig, *, eps_max: float = 0.5) -> str:
    """Human-readable separation table (for telemetry / the composition harness 0642.10)."""
    rows = separation_table(cfg, eps_max=eps_max)
    lines = [
        f"timescale-separation table (eps_max={eps_max}; ε_k=τ_fast/τ_slow, <eps_max ⟹ separated)",
        f"  {'fast':<13}{'slow':<14}{'τ_fast':>10}{'τ_slow':>11}{'ε_k':>9}  sep",
    ]
    for r in rows:
        lines.append(
            f"  {r.fast:<13}{r.slow:<14}{r.tau_fast:>10.3g}{r.tau_slow:>11.3g}{r.eps:>9.3g}  "
            f"{'yes' if r.separated else 'NO'}"
        )
    lines.append(f"  calcium fast-Jacobian eigenvalue gap (1−ρ(M_cb)): {calcium_eigenvalue_gap(cfg):.4g}")
    return "\n".join(lines)
