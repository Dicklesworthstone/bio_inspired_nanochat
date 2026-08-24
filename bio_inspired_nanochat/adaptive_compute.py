"""ATP-gated adaptive inference levers (bead ``r00r.3.2``).

The difficulty router and exact integer :class:`ATPBudget` live in ``deliberation.py``.  This module
turns their abstract compute units into three concrete runtime controls:

* prefix-depth early exit through a model's ``max_layers`` forward argument;
* runtime ``SynapticMoE.top_k`` selection, restored after the call;
* a difficulty-selected number of stochastic-release Monte Carlo draws.

The minimum path is always valid and is charged first.  If the remaining sequence budget cannot fund
that floor, planning fails before mutating the account; it never returns zero layers, zero experts, or
zero predictive samples.  The feature is default-off: a disabled controller returns the fixed
maximum-compute plan and does not touch the ATP account.  Quality-floor verification and engine-level
fallback/logging belong to downstream bead ``r00r.3.3``; this module supplies the honest, executable
levers it will guard.  Cached decoding may use an early-exit prefix when the cache itself is allocated
for that fixed prefix depth; changing depth inside one existing cache would create missing-layer
history and is deliberately rejected by the model forward.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from bio_inspired_nanochat.deliberation import (
    ATPBudget,
    ATPDebitRecord,
    DifficultyRouter,
    TokenDifficulty,
)
from bio_inspired_nanochat.mc_ensemble import MCPrediction, mc_predict
from bio_inspired_nanochat.synaptic import SynapticMoE


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
