"""ATP-gated adaptive inference levers (bead ``r00r.3.2``).

The difficulty router and exact integer :class:`ATPBudget` live in ``deliberation.py``.  This module
turns their abstract compute units into three concrete runtime controls:

* prefix-depth early exit through a model's ``max_layers`` forward argument;
* runtime ``SynapticMoE.top_k`` selection, restored after the call;
* a difficulty-selected number of stochastic-release Monte Carlo draws.

The minimum path is always valid and is charged first.  If the remaining sequence budget cannot fund
that floor, planning fails before mutating the account; it never returns zero layers, zero experts, or
zero predictive samples.  The feature is default-off: a disabled controller returns the fixed
maximum-compute plan and does not touch the ATP account.  The
:func:`quality_guarded_predict` inference primitive adds the downstream ``r00r.3.3`` safety contract:
atomically reserve enough ATP for both the adaptive attempt and a possible same-token fixed-compute
fallback, serve the adaptive prediction only when its predictive confidence meets a configured
quality floor, and emit one detailed JSONL audit event per token.  Cached decoding may use an
early-exit prefix when the cache itself is allocated
for that fixed prefix depth; changing depth inside one existing cache would create missing-layer
history and is deliberately rejected by the model forward.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

from bio_inspired_nanochat.deliberation import (
    ATPBudget,
    ATPDebitRecord,
    DifficultyRouter,
    TokenDifficulty,
)
from bio_inspired_nanochat.mc_ensemble import MCPrediction, mc_predict
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticMoE
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class AdaptiveComputeConfig:
    """Minimum-safe path and exact ATP prices for optional compute."""

    enabled: bool = False
    min_depth_layers: int = 1
    min_experts: int = 1
    min_mc_samples: int = 1
    max_mc_samples: int = 8
    layer_cost_atp: int = 1
    expert_cost_atp: int = 1
    mc_sample_cost_atp: int = 1

    def __post_init__(self) -> None:
        for name in ("min_depth_layers", "min_experts", "min_mc_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if (
            isinstance(self.max_mc_samples, bool)
            or not isinstance(self.max_mc_samples, int)
            or self.max_mc_samples < self.min_mc_samples
        ):
            raise ValueError(
                "max_mc_samples must be an integer greater than or equal to min_mc_samples"
            )
        for name in ("layer_cost_atp", "expert_cost_atp", "mc_sample_cost_atp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class AdaptiveComputePlan:
    """One token's executable compute allocation and the debits that produced it.

    ``expert_top_k == 0`` is the explicit not-applicable value for a dense model with no
    :class:`SynapticMoE`; such a model receives no fictitious expert debit.
    """

    token_index: int
    difficulty: TokenDifficulty
    depth_layers: int
    expert_top_k: int
    mc_samples: int
    max_depth_layers: int
    max_experts: int
    max_mc_samples: int
    debit_records: tuple[ATPDebitRecord, ...]

    @property
    def compute_units(self) -> int:
        """Simple lever-unit count used for allocation tests, not a FLOP equivalence claim."""
        return self.depth_layers + self.expert_top_k + self.mc_samples

    @property
    def maximum_compute_units(self) -> int:
        return self.max_depth_layers + self.max_experts + self.max_mc_samples


@dataclass(frozen=True)
class QualityFloorConfig:
    """Inference-time acceptance threshold for an adaptive next-token prediction.

    ``min_predictive_confidence`` is the minimum probability assigned to the adaptive
    distribution's top token.  It is a calibrated quality proxy, not a proof that the token is
    correct.  A prediction below the floor is never served from the adaptive path: the runner
    deterministically executes and returns the configured fixed-compute baseline instead.
    """

    min_predictive_confidence: float = 0.5

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_predictive_confidence)
            or not 0.0 <= self.min_predictive_confidence <= 1.0
        ):
            raise ValueError(
                "min_predictive_confidence must be finite and in [0, 1], got "
                f"{self.min_predictive_confidence!r}"
            )


@dataclass(frozen=True)
class GuardedAdaptivePrediction:
    """Auditable result of one quality-guarded adaptive next-token inference."""

    prediction: MCPrediction
    proposed_plan: AdaptiveComputePlan
    executed_plan: AdaptiveComputePlan
    adaptive_confidence: float
    served_confidence: float
    quality_floor: float
    quality_floor_passed: bool
    fallback_used: bool
    fallback_reexecuted: bool
    fallback_reason: str | None
    token_spent_atp: int

    @property
    def attempted_compute_units(self) -> int:
        """All lever units physically attempted, including a discarded cheap prediction."""
        attempted = self.proposed_plan.compute_units
        if self.fallback_reexecuted:
            attempted += self.executed_plan.compute_units
        return attempted

    @property
    def saved_compute_units(self) -> int:
        """Lever-unit savings against one fixed pass; negative means fallback overhead."""
        return self.executed_plan.maximum_compute_units - self.attempted_compute_units


class InsufficientATPError(RuntimeError):
    """The adaptive account cannot fund even the configured minimum-safe path."""


class AdaptiveComputeController:
    """Allocate all three inference levers from difficulty and a sequence-local ATP account."""

    def __init__(
        self,
        config: AdaptiveComputeConfig | None = None,
        *,
        router: DifficultyRouter | None = None,
    ) -> None:
        self.config = config or AdaptiveComputeConfig()
        self.router = router or DifficultyRouter()

    @staticmethod
    def model_capacity(model) -> tuple[int, int]:
        """Return ``(layer_count, safe uniform expert-k ceiling)`` for a supported model."""
        model_config = getattr(model, "config", None)
        depth = getattr(model_config, "n_layer", None)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("model.config.n_layer must be a positive integer")
        expert_counts = [
            int(module.num_experts)
            for module in model.modules()
            if isinstance(module, SynapticMoE)
        ]
        return depth, min(expert_counts, default=0)

    def plan_for_model(
        self,
        logits,
        budget: ATPBudget,
        *,
        model,
        token_index: int,
        free_energy_value: float | None = None,
    ) -> AdaptiveComputePlan:
        max_depth_layers, max_experts = self.model_capacity(model)
        return self.plan(
            logits,
            budget,
            token_index=token_index,
            max_depth_layers=max_depth_layers,
            max_experts=max_experts,
            free_energy_value=free_energy_value,
        )

    def plan(
        self,
        logits,
        budget: ATPBudget,
        *,
        token_index: int,
        max_depth_layers: int,
        max_experts: int,
        free_energy_value: float | None = None,
    ) -> AdaptiveComputePlan:
        """Measure difficulty, then buy optional compute without ever undercutting the safe floor."""
        token_index = ATPBudget._nonnegative_int("token_index", token_index)
        max_depth_layers = ATPBudget._nonnegative_int("max_depth_layers", max_depth_layers)
        max_experts = ATPBudget._nonnegative_int("max_experts", max_experts)
        if max_depth_layers < self.config.min_depth_layers:
            raise ValueError(
                f"max_depth_layers={max_depth_layers} is below min_depth_layers="
                f"{self.config.min_depth_layers}"
            )
        minimum_experts = 0 if max_experts == 0 else self.config.min_experts
        if max_experts and max_experts < minimum_experts:
            raise ValueError(
                f"max_experts={max_experts} is below min_experts={minimum_experts}"
            )

        difficulty = self.router.measure(logits, free_energy_value=free_energy_value)
        if not self.config.enabled:
            return AdaptiveComputePlan(
                token_index=token_index,
                difficulty=difficulty,
                depth_layers=max_depth_layers,
                expert_top_k=max_experts,
                mc_samples=self.config.max_mc_samples,
                max_depth_layers=max_depth_layers,
                max_experts=max_experts,
                max_mc_samples=self.config.max_mc_samples,
                debit_records=(),
            )

        minimum_atp = (
            self.config.min_depth_layers * self.config.layer_cost_atp
            + minimum_experts * self.config.expert_cost_atp
            + self.config.min_mc_samples * self.config.mc_sample_cost_atp
        )
        if budget.remaining_atp < minimum_atp:
            raise InsufficientATPError(
                f"minimum adaptive path costs {minimum_atp} ATP but only "
                f"{budget.remaining_atp} ATP remains"
            )

        starting_record_count = len(budget.records)
        depth_base = budget.debit(
            token_index=token_index,
            action="depth_layer",
            difficulty_score=difficulty.score,
            requested_units=self.config.min_depth_layers,
            unit_cost_atp=self.config.layer_cost_atp,
        )
        expert_base_units = 0
        if minimum_experts:
            expert_base = budget.debit(
                token_index=token_index,
                action="expert",
                difficulty_score=difficulty.score,
                requested_units=minimum_experts,
                unit_cost_atp=self.config.expert_cost_atp,
            )
            expert_base_units = expert_base.granted_units
        mc_base = budget.debit(
            token_index=token_index,
            action="mc_sample",
            difficulty_score=difficulty.score,
            requested_units=self.config.min_mc_samples,
            unit_cost_atp=self.config.mc_sample_cost_atp,
        )
        depth_extra = self._buy_extra(
            budget,
            token_index=token_index,
            action="depth_layer",
            difficulty=difficulty,
            maximum=max_depth_layers - self.config.min_depth_layers,
            unit_cost_atp=self.config.layer_cost_atp,
        )
        expert_extra = self._buy_extra(
            budget,
            token_index=token_index,
            action="expert",
            difficulty=difficulty,
            maximum=max_experts - minimum_experts,
            unit_cost_atp=self.config.expert_cost_atp,
        )
        mc_extra = self._buy_extra(
            budget,
            token_index=token_index,
            action="mc_sample",
            difficulty=difficulty,
            maximum=self.config.max_mc_samples - self.config.min_mc_samples,
            unit_cost_atp=self.config.mc_sample_cost_atp,
        )
        return AdaptiveComputePlan(
            token_index=token_index,
            difficulty=difficulty,
            depth_layers=depth_base.granted_units + depth_extra,
            expert_top_k=expert_base_units + expert_extra,
            mc_samples=mc_base.granted_units + mc_extra,
            max_depth_layers=max_depth_layers,
            max_experts=max_experts,
            max_mc_samples=self.config.max_mc_samples,
            debit_records=tuple(budget.records[starting_record_count:]),
        )

    def _buy_extra(
        self,
        budget: ATPBudget,
        *,
        token_index: int,
        action: str,
        difficulty: TokenDifficulty,
        maximum: int,
        unit_cost_atp: int,
    ) -> int:
        requested_units = self.router.requested_units(
            difficulty,
            min_units=0,
            max_units=maximum,
        )
        if requested_units == 0:
            return 0
        debit = budget.debit(
            token_index=token_index,
            action=action,
            difficulty_score=difficulty.score,
            requested_units=requested_units,
            unit_cost_atp=unit_cost_atp,
        )
        return debit.granted_units


@contextmanager
def temporary_expert_top_k(model, top_k: int) -> Iterator[int]:
    """Apply one runtime expert cap to every SynapticMoE and restore it even after failure."""
    modules = [module for module in model.modules() if isinstance(module, SynapticMoE)]
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 0
        or (top_k == 0 and modules)
    ):
        raise ValueError(
            f"top_k must be positive for a model with SynapticMoE layers, got {top_k!r}"
        )
    prior: list[tuple[SynapticMoE, int]] = []
    for module in modules:
        prior.append((module, module.top_k))
        module.top_k = min(top_k, int(module.num_experts))
    try:
        yield len(prior)
    finally:
        for module, previous_top_k in prior:
            module.top_k = previous_top_k


def adaptive_forward(model, input_ids, plan: AdaptiveComputePlan, **forward_kwargs: Any):
    """Execute the selected depth and expert-k for a standalone model forward."""
    kwargs = dict(forward_kwargs)
    try:
        supports_depth = "max_layers" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        supports_depth = False
    if supports_depth:
        kwargs["max_layers"] = plan.depth_layers
    elif plan.depth_layers != plan.max_depth_layers:
        raise TypeError("model.forward does not expose max_layers for adaptive depth execution")
    with temporary_expert_top_k(model, plan.expert_top_k):
        return model(input_ids, **kwargs)


def adaptive_mc_predict(
    model,
    input_ids,
    plan: AdaptiveComputePlan,
    *,
    temperature: float = 1.0,
) -> MCPrediction:
    """Execute the plan's depth, expert-k, and stochastic-release sample count together."""
    try:
        supports_depth = "max_layers" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        supports_depth = False
    forward_kwargs = {"max_layers": plan.depth_layers} if supports_depth else None
    if not supports_depth and plan.depth_layers != plan.max_depth_layers:
        raise TypeError("model.forward does not expose max_layers for adaptive depth execution")
    with temporary_expert_top_k(model, plan.expert_top_k):
        return mc_predict(
            model,
            input_ids,
            n_samples=plan.mc_samples,
            temperature=temperature,
            forward_kwargs=forward_kwargs,
        )


def _prediction_confidence(prediction: MCPrediction) -> float:
    """Return the single sequence's last-position top-token probability."""
    probabilities = prediction.mean_probs
    if probabilities.ndim != 3 or probabilities.shape[0] != 1 or probabilities.shape[1] < 1:
        raise ValueError(
            "quality-guarded inference requires one non-empty sequence, got predictive shape "
            f"{tuple(probabilities.shape)}"
        )
    return float(probabilities[0, -1].max().item())


def _fixed_compute_cost(
    controller: AdaptiveComputeController,
    *,
    max_depth_layers: int,
    max_experts: int,
) -> int:
    cfg = controller.config
    return (
        max_depth_layers * cfg.layer_cost_atp
        + max_experts * cfg.expert_cost_atp
        + cfg.max_mc_samples * cfg.mc_sample_cost_atp
    )


def _fallback_plan(
    proposed: AdaptiveComputePlan,
    budget: ATPBudget,
    controller: AdaptiveComputeController,
) -> AdaptiveComputePlan:
    """Buy a complete fixed pass after a discarded cheap attempt."""
    cfg = controller.config
    records = list(proposed.debit_records)
    for action, maximum, unit_cost in (
        ("quality_fallback_depth_layer", proposed.max_depth_layers, cfg.layer_cost_atp),
        ("quality_fallback_expert", proposed.max_experts, cfg.expert_cost_atp),
        ("quality_fallback_mc_sample", proposed.max_mc_samples, cfg.mc_sample_cost_atp),
    ):
        if maximum == 0:
            continue
        debit = budget.debit(
            token_index=proposed.token_index,
            action=action,
            difficulty_score=proposed.difficulty.score,
            requested_units=maximum,
            unit_cost_atp=unit_cost,
        )
        if debit.granted_units != maximum:
            raise RuntimeError("reserved fixed-compute fallback ATP was unexpectedly unavailable")
        records.append(debit)
    return replace(
        proposed,
        depth_layers=proposed.max_depth_layers,
        expert_top_k=proposed.max_experts,
        mc_samples=proposed.max_mc_samples,
        debit_records=tuple(records),
    )


def _commit_shadow_plan(
    shadow_plan: AdaptiveComputePlan,
    budget: ATPBudget,
) -> AdaptiveComputePlan:
    """Replay a validated shadow plan into the real sequence account exactly once."""
    starting_record_count = len(budget.records)
    for record in shadow_plan.debit_records:
        debit = budget.debit(
            token_index=record.token_index,
            action=record.action,
            difficulty_score=record.difficulty_score,
            requested_units=record.granted_units,
            unit_cost_atp=record.unit_cost_atp,
        )
        if debit.granted_units != record.granted_units:
            raise RuntimeError("reserved adaptive-attempt ATP was unexpectedly unavailable")
    return replace(
        shadow_plan,
        debit_records=tuple(budget.records[starting_record_count:]),
    )


def _log_guarded_prediction(
    logger: RunLogger | None,
    result: GuardedAdaptivePrediction,
    budget: ATPBudget,
) -> None:
    if logger is None:
        return
    plan = result.executed_plan
    logger.event(
        "adaptive_compute_token",
        step=plan.token_index,
        token_index=plan.token_index,
        difficulty=asdict(plan.difficulty),
        proposed={
            "depth_layers": result.proposed_plan.depth_layers,
            "expert_top_k": result.proposed_plan.expert_top_k,
            "mc_samples": result.proposed_plan.mc_samples,
            "compute_units": result.proposed_plan.compute_units,
        },
        executed={
            "depth_layers": plan.depth_layers,
            "expert_top_k": plan.expert_top_k,
            "mc_samples": plan.mc_samples,
            "compute_units": plan.compute_units,
        },
        maximum={
            "depth_layers": plan.max_depth_layers,
            "expert_top_k": plan.max_experts,
            "mc_samples": plan.max_mc_samples,
            "compute_units": plan.maximum_compute_units,
        },
        adaptive_confidence=result.adaptive_confidence,
        served_confidence=result.served_confidence,
        quality_floor=result.quality_floor,
        quality_floor_passed=result.quality_floor_passed,
        fallback_used=result.fallback_used,
        fallback_reexecuted=result.fallback_reexecuted,
        fallback_reason=result.fallback_reason,
        attempted_compute_units=result.attempted_compute_units,
        saved_compute_units=result.saved_compute_units,
        token_spent_atp=result.token_spent_atp,
        sequence_spent_atp=budget.spent_atp,
        sequence_remaining_atp=budget.remaining_atp,
        debit_records=[asdict(record) for record in plan.debit_records],
    )


def quality_guarded_predict(
    model,
    input_ids,
    routing_logits,
    budget: ATPBudget,
    *,
    controller: AdaptiveComputeController,
    token_index: int,
    quality: QualityFloorConfig | None = None,
    free_energy_value: float | None = None,
    temperature: float = 1.0,
    run_logger: RunLogger | None = None,
) -> GuardedAdaptivePrediction:
    """Run one adaptive prediction and fail closed to fixed compute below its quality floor.

    One ATP account maps to one sequence, so batched inputs are rejected rather than silently sharing
    a budget across rows.  Before any debit, an enabled controller verifies that the current token's
    adaptive attempt plus a complete fixed fallback are affordable.  If the cheap path's confidence
    is below the floor (or non-finite), the complete fallback is additionally debited and executed;
    the discarded attempt remains charged because it consumed physical work.  Thus a fallback can
    never overdraw the hard sequence budget, leave it partially charged, or manufacture savings by
    hiding its failed attempt.

    When adaptive compute is disabled, this is the exact fixed-compute path and retains the existing
    no-debit behavior.  The confidence floor is a calibration contract, not a correctness proof; the
    downstream Pareto evaluation must establish whether a chosen threshold preserves task quality.
    """
    if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
        shape = tuple(getattr(input_ids, "shape", ()))
        raise ValueError(f"quality-guarded inference requires input shape (1, T), got {shape}")
    quality = quality or QualityFloorConfig()
    max_depth_layers, max_experts = controller.model_capacity(model)
    if controller.config.enabled:
        fixed_cost = _fixed_compute_cost(
            controller,
            max_depth_layers=max_depth_layers,
            max_experts=max_experts,
        )
        # First determine whether routing already selects the full path. Such a plan needs no second
        # execution if the guard rejects it. A cheaper plan is recomputed against only the ATP left
        # after reserving one complete fallback, so optional work cannot consume its own safety net.
        probe_budget = ATPBudget(budget.remaining_atp)
        probe_proposed = controller.plan(
            routing_logits,
            probe_budget,
            token_index=token_index,
            max_depth_layers=max_depth_layers,
            max_experts=max_experts,
            free_energy_value=free_energy_value,
        )
        proposed_is_fixed = (
            probe_proposed.depth_layers == probe_proposed.max_depth_layers
            and probe_proposed.expert_top_k == probe_proposed.max_experts
            and probe_proposed.mc_samples == probe_proposed.max_mc_samples
        )
        if proposed_is_fixed:
            shadow_budget = probe_budget
            shadow_proposed = probe_proposed
            required_atp = shadow_budget.spent_atp
        else:
            cfg = controller.config
            minimum_experts = 0 if max_experts == 0 else cfg.min_experts
            minimum_attempt_cost = (
                cfg.min_depth_layers * cfg.layer_cost_atp
                + minimum_experts * cfg.expert_cost_atp
                + cfg.min_mc_samples * cfg.mc_sample_cost_atp
            )
            required_atp = fixed_cost + minimum_attempt_cost
            if budget.remaining_atp < required_atp:
                raise InsufficientATPError(
                    "quality guard's adaptive attempt plus fixed-compute fallback requires at least "
                    f"{required_atp} ATP but only {budget.remaining_atp} ATP remains"
                )
            shadow_budget = ATPBudget(budget.remaining_atp - fixed_cost)
            shadow_proposed = controller.plan(
                routing_logits,
                shadow_budget,
                token_index=token_index,
                max_depth_layers=max_depth_layers,
                max_experts=max_experts,
                free_energy_value=free_energy_value,
            )
            required_atp = shadow_budget.spent_atp + fixed_cost
        if budget.remaining_atp < required_atp:
            raise InsufficientATPError(
                "quality guard's adaptive attempt plus fixed-compute fallback requires "
                f"{required_atp} ATP but only "
                f"{budget.remaining_atp} ATP remains"
            )

    spent_before = budget.spent_atp
    proposed = (
        _commit_shadow_plan(shadow_proposed, budget)
        if controller.config.enabled
        else controller.plan(
            routing_logits,
            budget,
            token_index=token_index,
            max_depth_layers=max_depth_layers,
            max_experts=max_experts,
            free_energy_value=free_energy_value,
        )
    )
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    adaptive_prediction = adaptive_mc_predict(
        model,
        input_ids,
        proposed,
        temperature=temperature,
    )
    adaptive_confidence = _prediction_confidence(adaptive_prediction)

    guard_active = controller.config.enabled
    quality_passed = (
        not guard_active
        or (
            math.isfinite(adaptive_confidence)
            and adaptive_confidence >= quality.min_predictive_confidence
        )
    )
    if quality_passed:
        prediction = adaptive_prediction
        executed = proposed
        fallback_used = False
        fallback_reexecuted = False
        fallback_reason = None
    else:
        fallback_used = True
        fallback_reason = (
            "non_finite_predictive_confidence"
            if not math.isfinite(adaptive_confidence)
            else "predictive_confidence_below_floor"
        )
        proposed_is_fixed = (
            proposed.depth_layers == proposed.max_depth_layers
            and proposed.expert_top_k == proposed.max_experts
            and proposed.mc_samples == proposed.max_mc_samples
        )
        if proposed_is_fixed:
            executed = proposed
            prediction = adaptive_prediction
            fallback_reexecuted = False
        else:
            executed = _fallback_plan(proposed, budget, controller)
            # The failed cheap attempt is observational: a fallback must see the exact RNG stream a
            # fixed-compute call would have seen, not the stream after speculative MC draws.
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            prediction = adaptive_mc_predict(model, input_ids, executed, temperature=temperature)
            fallback_reexecuted = True

    result = GuardedAdaptivePrediction(
        prediction=prediction,
        proposed_plan=proposed,
        executed_plan=executed,
        adaptive_confidence=adaptive_confidence,
        served_confidence=_prediction_confidence(prediction),
        quality_floor=quality.min_predictive_confidence,
        quality_floor_passed=quality_passed,
        fallback_used=fallback_used,
        fallback_reexecuted=fallback_reexecuted,
        fallback_reason=fallback_reason,
        token_spent_atp=budget.spent_atp - spent_before,
    )
    _log_guarded_prediction(run_logger, result, budget)
    return result
