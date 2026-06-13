"""Free-energy deliberation + energy-based decoding — live decode-path wiring (bead `r00r.1.2`).

Implements the live per-token wiring of the design note `docs/theory/free_energy_deliberation.md`
(`r00r.1.1`) on top of the reference `deliberate()` / `boltzmann_weights()` API in
`metriplectic_integrator.py`. Per token, the model's synaptic state is mapped to the metriplectic core
`z = (C, B, h)`, relaxed by extra free-energy-minimization steps ("ponder") until it self-consistently
halts (`|ΔF| < eps`) or a compute budget (`max_iters`, the latency bound) is hit, and the resulting
**effort** (iterations) + **confidence** (final free energy) modulate the decode temperature — the
model commits sharply when the state is self-consistent (easy token) and explores when it is not
(hard token). Energy-based decoding is then the Boltzmann softmax `p ∝ exp(−F/kT)` over the model's
logits at that deliberation-derived temperature.

Convergence is guaranteed by Thrust A (the structure-preserving step makes `F` monotonically
non-increasing and bounded below — `docs/theory/metriplectic.md` §5), so the ponder always halts in a
bounded number of steps; a guard trip inside a step deterministically falls back to clamped Euler. The
whole mechanism is **default-off**: with no `DeliberationController` the engine decodes exactly as
before (single-step). Nothing here mutates the model — deliberation only runs the existing descent
longer and reads the result.

**Scope of this wiring (`r00r.1.2`), stated honestly.** The relaxed state `z` is *read*, not fed back:
its effort/confidence set the **decode temperature** only — the model's logits are unchanged, so this
is a confidence-calibrated temperature controller, not (yet) a state-feedback "ponder" that re-runs the
model on a relaxed state. Because the metriplectic core here is the low-dimensional aggregate
`z = (mean C, mean B, 0)`, the effort and `F` are both monotone functions of the mean presynaptic
calcium magnitude (`F_final = (1−T)·½‖(C,B)‖²`), i.e. a simple "how active is the synapse" difficulty
proxy rather than an independent self-consistency measure. The decode temperature is one scalar per
decoded step (shared across the batch, matching `sample_next_token`'s single-temperature API). The
fuller programme — feeding the relaxed state back into the forward, and per-candidate energy decoding
(`boltzmann_weights` over each relaxed continuation's `F`) — is the downstream `re4e.*` work; this
module is the bounded, default-off substrate it builds on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from bio_inspired_nanochat.metriplectic_integrator import (
    TEMP,
    DeliberationResult,
    deliberate,
    free_energy,
)
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class DeliberationConfig:
    """The compute-vs-quality knobs for per-token deliberation (default-off; see design §5)."""

    enabled: bool = False
    eps: float = 1e-4          # halting threshold on |ΔF| (smaller ⟹ deliberate longer)
    max_iters: int = 64        # per-token compute budget (the worst-case latency bound)
    dt: float = 0.5            # deliberation step size; tuned so typical tokens halt in ~25–55 steps
                               # (effort scales with calcium/difficulty without saturating the budget)
    T: float = TEMP            # free-energy temperature in F = E − T·S
    # Adaptive decode temperature: easy (low-effort) tokens sharpen toward `temp_floor` (commit),
    # hard (budget-hitting) tokens widen toward `temp_ceil` (explore). Bounds the multiplier on the
    # caller's base temperature, so base_temp=0 (greedy) stays greedy and `enabled=False` is identity.
    temp_floor: float = 0.7
    temp_ceil: float = 1.3


@dataclass
class DeliberationRecord:
    """Auditable per-token deliberation trace (F-trajectory + effort; the `eqyk.2` schema)."""

    token_index: int
    effort: int                # iterations actually used (the token-difficulty estimate)
    halted_converged: bool     # True ⟹ self-consistent; False ⟹ budget hit (still "thinking")
    F_initial: float
    F_final: float             # the confidence signal (lower ⟹ more self-consistent)
    F_drop: float              # F_initial − F_final, the free energy released by pondering
    base_temperature: float
    effective_temperature: float
    calcium: float             # the aggregated synaptic state that seeded z
    buffer: float


class DeliberationController:
    """Per-token free-energy deliberation for the engine decode path (bead `r00r.1.2`).

    Stateless across tokens except for the F-trajectory log; safe to reuse across a generation. The
    engine calls `effective_temperature(presyn_state, base_temp, token_index)` once per decoded token;
    when no synaptic state is present (vanilla model) it returns `base_temp` unchanged — the
    deterministic fallback to single-step decode.
    """

    def __init__(self, cfg: DeliberationConfig | None = None) -> None:
        self.cfg = cfg or DeliberationConfig()
        self.records: list[DeliberationRecord] = []

    # -- synaptic-state readout ---------------------------------------------- #
    @staticmethod
    def synaptic_z(presyn_state) -> np.ndarray | None:
        """Map the live presyn state to the metriplectic core `z = (C, B, h)`.

        Aggregates the mean calcium `C` and buffer `B` over layers/heads/edges; `h = 0` so the
        "kinetic" calcium energy `½(C²+B²)` is what relaxes into effort/entropy during the ponder
        (an active, high-calcium token is far from equilibrium ⟹ harder ⟹ more deliberation steps).
        Returns ``None`` when there is no synaptic state (a vanilla model ⟹ fall back to single-step).
        """
        if presyn_state is None:
            return None
        layers = presyn_state if isinstance(presyn_state, list) else [presyn_state]
        cs, bs = [], []
        for st in layers:
            if not isinstance(st, dict):
                continue
            c, b = st.get("C"), st.get("BUF")
            # Only accept finite means: an empty or all-NaN calcium tensor must NOT become a NaN `z`
            # (which would silently drive the latch to the explore-ceiling and log NaN). Skip it, and
            # if no usable calcium remains, fall back to None ⟹ single-step decode (vanilla behavior).
            if c is not None:
                cm = float(torch.as_tensor(c, dtype=torch.float64).mean())
                if np.isfinite(cm):
                    cs.append(cm)
            if b is not None:
                bm = float(torch.as_tensor(b, dtype=torch.float64).mean())
                if np.isfinite(bm):
                    bs.append(bm)
        if not cs:
            return None
        c_mean = float(np.mean(cs))
        b_mean = float(np.mean(bs)) if bs else 0.0
        return np.array([c_mean, b_mean, 0.0], dtype=np.float64)

    # -- the ponder ----------------------------------------------------------- #
    def ponder(self, z: np.ndarray) -> DeliberationResult:
        """Run the bounded free-energy-minimization loop on `z` (the design §1 deliberation loop)."""
        return deliberate(z, self.cfg.dt, eps=self.cfg.eps, max_iters=self.cfg.max_iters, T=self.cfg.T)

    def adaptive_temperature(self, base_temp: float, result: DeliberationResult) -> float:
        """Decode temperature from the deliberation outcome — commit when confident, explore when not.

        Multiplier interpolates `temp_floor → temp_ceil` with the effort fraction `iters/max_iters`:
        an easy token (halts in ~1 step) sharpens toward `temp_floor`; a budget-hitting hard token
        widens toward `temp_ceil`. The multiplier scales the caller's base temperature, so greedy
        (`base_temp == 0`) stays greedy and the bounds keep the effective temperature well-conditioned.
        """
        if base_temp <= 0.0:
            return base_temp  # greedy decode is unaffected (argmax regardless of temperature)
        frac = min(1.0, max(0.0, result.iters / max(1, self.cfg.max_iters)))
        mult = self.cfg.temp_floor + (self.cfg.temp_ceil - self.cfg.temp_floor) * frac
        return base_temp * mult

    def effective_temperature(self, presyn_state, base_temp: float, *, token_index: int | None = None) -> float:
        """The per-token engine hook: ponder the synaptic state and return the decode temperature.

        Falls back to `base_temp` (single-step decode) when deliberation is disabled or there is no
        synaptic state. Logs an auditable F-trajectory record when it ponders.
        """
        if not self.cfg.enabled:
            return base_temp
        z = self.synaptic_z(presyn_state)
        if z is None:
            return base_temp
        res = self.ponder(z)
        temp_eff = self.adaptive_temperature(base_temp, res)
        self.records.append(DeliberationRecord(
            token_index=self._next_index(token_index),
            effort=res.iters,
            halted_converged=res.halted_converged,
            F_initial=float(free_energy(z, self.cfg.T)),
            F_final=res.F_final,
            F_drop=res.F_drop,
            base_temperature=base_temp,
            effective_temperature=temp_eff,
            calcium=float(z[0]),
            buffer=float(z[1]),
        ))
        return temp_eff

    # -- energy-based decoding (Boltzmann) ------------------------------------ #
    @staticmethod
    def boltzmann_token_weights(logits, kT: float = 1.0):
        """Energy-based decode weights `p ∝ exp(−F/kT)` over the last (vocab) axis, with the model
        logits as negative energy (`F = −logit`).

        This is exactly the temperature-`kT` softmax of the logits, normalized **per distribution**
        (the last axis), so it is correct for a single `(vocab,)` logit vector *and* a batch
        `(..., vocab)` of them (each row sums to 1). For a single vector it equals
        `metriplectic_integrator.boltzmann_weights(−logits, kT)`. It composes with
        `effective_temperature`, which supplies a deliberation-derived `kT`. For candidate-level energy
        decoding (score each relaxed continuation by its own `F`), pass those free energies to
        `metriplectic_integrator.boltzmann_weights` directly (the `re4e.3` energy-guided search path).
        """
        if kT <= 0.0:
            raise ValueError(f"kT must be positive, got {kT}")
        t = torch.as_tensor(logits, dtype=torch.float64)
        return torch.softmax(t / kT, dim=-1)

    # -- traces --------------------------------------------------------------- #
    def _next_index(self, token_index: int | None) -> int:
        if token_index is not None:
            return token_index
        idx = len(self.records)
        return idx

    def f_trajectory(self) -> list[dict]:
        """The per-token F-trajectory + effort log (JSONL-ready dicts)."""
        return [asdict(r) for r in self.records]

    def to_jsonl(self) -> list[str]:
        """The per-token F-trajectory as JSONL lines — the detailed-logging artifact (`eqyk.2`)."""
        return [json.dumps(asdict(r), ensure_ascii=False) for r in self.records]

    def write_trajectory(self, path) -> None:
        """Write the per-token F-trajectory artifact to ``path`` as JSONL (one record per token)."""
        from pathlib import Path
        Path(path).write_text("\n".join(self.to_jsonl()) + ("\n" if self.records else ""), encoding="utf-8")

    def summary(self) -> dict:
        if not self.records:
            return {"tokens": 0, "enabled": self.cfg.enabled}
        efforts = [r.effort for r in self.records]
        return {
            "tokens": len(self.records),
            "enabled": self.cfg.enabled,
            "mean_effort": float(np.mean(efforts)),
            "max_effort": int(np.max(efforts)),
            "frac_converged": sum(r.halted_converged for r in self.records) / len(self.records),
            "mean_F_drop": float(np.mean([r.F_drop for r in self.records])),
            "max_budget": self.cfg.max_iters,
        }


def make_controller(cfg: DeliberationConfig | None) -> DeliberationController | None:
    """Build a controller iff deliberation is enabled; else ``None`` (the engine decodes as baseline)."""
    if cfg is None or not cfg.enabled:
        return None
    return DeliberationController(cfg)
