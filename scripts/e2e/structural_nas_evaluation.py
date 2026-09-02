"""Equal-compute variable-count NAS evaluation for bead ``uta.7``.

The live UTA controller receives routing credit measured from the current batch and
executes a preregistered ``4 → 3 → 5 → 4`` expert-count schedule.  A fixed baseline
keeps four experts.  With equal phase lengths, both methods have exactly the same
cumulative expert width, top-k dispatches, training steps, and dominant MoE matmul
FLOPs; the NAS controller/surgery wall-time overhead is reported separately.

This is a controlled synthetic routed-learning experiment, not a language-model-scale
claim.  Final loss and dead-expert fraction are the directional primary outcomes.
Specialization (routing Gini) is descriptive because either extreme can be unhealthy.
Lifecycle stability is a fail-closed maximum event-spike gate.  A win requires both
primary paired bootstrap intervals to favor NAS; statistically supported harm or a
stability breach is a regression; everything else is an honest null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.results_registry import measurement_regime
from bio_inspired_nanochat.eval_stats import (
    Aggregate,
    PairedResult,
    aggregate,
    paired_comparison,
)
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    append_record,
    make_record,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class StructuralNASEvaluationConfig:
    """Predeclared task, lifecycle schedule, and statistical controls."""

    seeds: tuple[int, ...] = (503, 521, 547, 569, 593, 617, 641, 661)
    n_embd: int = 4
    initial_experts: int = 4
    min_experts: int = 3
    max_experts: int = 5
    top_k: int = 2
    points_per_cluster: int = 6
    train_steps: int = 20
    learning_rate: float = 2e-3
    dead_share_floor: float = 0.05
    reset_health_max: float = 0.05
    split_health_min: float = 0.18
    dormant_logit_bias: float = -12.0
    max_event_loss_spike: float = 0.02
    bootstrap_samples: int = 10_000
    # sx1m / uta.8: which lifecycle health signal drives the events. 'product' is the
    # preregistered util x energy signal with the absolute thresholds above; 'relative' scores
    # utilization against the fair share top_k / num_experts, so its thresholds are in
    # fair-share units (split above 1.5x, reset below 0.05x) and the split threshold above 1.0
    # is the point.
    health_mode: str = "product"

    def validate(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least three unique values")
        if self.n_embd != 4:
            raise ValueError("n_embd must be four for the preregistered routed task")
        if (self.min_experts, self.initial_experts, self.max_experts) != (3, 4, 5):
            raise ValueError("expert bounds must encode the preregistered 4→3→5→4 schedule")
        if not 1 <= self.top_k < self.min_experts:
            raise ValueError("top_k must be positive and smaller than min_experts")
        if self.points_per_cluster < 2:
            raise ValueError("points_per_cluster must be at least two")
        if self.train_steps < 2 or self.train_steps % 2:
            raise ValueError("train_steps must be positive, even, and at least two")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.dead_share_floor) or not 0.0 < self.dead_share_floor < 1.0:
            raise ValueError("dead_share_floor must be finite and strictly between zero and one")
        if not math.isfinite(self.reset_health_max) or not 0.0 <= self.reset_health_max < 1.0:
            raise ValueError("reset_health_max must be finite and in [0, 1)")
        if self.health_mode not in ("product", "relative"):
            raise ValueError("health_mode must be 'product' or 'relative'")
        if not math.isfinite(self.split_health_min) or self.split_health_min <= 0.0:
            raise ValueError("split_health_min must be finite and positive")
        if self.health_mode == "product" and self.split_health_min > 1.0:
            raise ValueError("split_health_min must be in (0, 1] for product health")
        if self.health_mode == "relative" and self.split_health_min <= 1.0:
            raise ValueError("split_health_min must exceed 1.0 (fair-share units) for relative health")
        if not math.isfinite(self.dormant_logit_bias) or self.dormant_logit_bias >= 0.0:
            raise ValueError("dormant_logit_bias must be finite and negative")
        if not math.isfinite(self.max_event_loss_spike) or self.max_event_loss_spike < 0.0:
            raise ValueError("max_event_loss_spike must be finite and non-negative")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")


@dataclass(frozen=True)
class LifecycleEventOutcome:
    """One real resize or compute-matched fixed-baseline placebo."""

    name: str
    step: int
    experts_before: int
    experts_after: int
    routing_shares_before: tuple[float, ...]
    routing_shares_after: tuple[float, ...]
    planned_operations: tuple[dict[str, Any], ...]
    optimizer_synced: bool
    loss_before: float
    loss_after: float
    loss_spike: float
    loss_discontinuity: float


@dataclass(frozen=True)
class StructuralNASSeedOutcome:
    """Replayable metrics and exact work accounting for one method/seed."""

    seed: int
    method: str
    initial_loss: float
    final_loss: float
    dead_expert_fraction: float
    routing_gini: float
    max_event_loss_spike: float
    total_event_loss_discontinuity: float
    final_expert_count: int
    top_k: int
    forward_calls: int
    train_forward_calls: int
    expert_dispatches: int
    router_width_token_units: int
    training_router_width_token_units: int
    moe_matmul_flops: int
    average_expert_count: float
    training_average_expert_count: float
    wall_time_seconds: float
    expert_count_trace: tuple[int, ...]
    events: tuple[LifecycleEventOutcome, ...]


@dataclass(frozen=True)
class StructuralNASMetricComparison:
    nas: Aggregate
    fixed: Aggregate
    paired: PairedResult


@dataclass(frozen=True)
class StructuralNASEvaluationReport:
    schema_version: int
    bead: str
    run_id: str
    protocol_id: str
    scope: str
    config: StructuralNASEvaluationConfig
    outcomes: tuple[StructuralNASSeedOutcome, ...]
    comparisons: dict[str, StructuralNASMetricComparison]
    specialization: dict[str, Aggregate]
    invariants: dict[str, bool]
    verdict: str
    registry_verdict: str
    verdict_reason: str
    report_path: str
    events_path: str
    registry_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_payload({**asdict(self), "measurement_regime": measurement_regime()})

    def assert_not_regression(self) -> None:
        if self.verdict == "regression":
            raise AssertionError(self.verdict_reason)


@dataclass
class _WorkCounter:
    n_embd: int
    top_k: int
    forward_calls: int = 0
    train_forward_calls: int = 0
    expert_dispatches: int = 0
    router_width_token_units: int = 0
    training_router_width_token_units: int = 0
    moe_matmul_flops: int = 0
    expert_count_trace: list[int] = field(default_factory=list)

    def record(self, *, num_experts: int, tokens: int, training: bool) -> None:
        """Count dominant router/expert matmuls; backward is two forward equivalents."""
        multiplier = 3 if training else 1
        self.forward_calls += 1
        self.train_forward_calls += int(training)
        self.expert_dispatches += tokens * self.top_k
        self.router_width_token_units += tokens * num_experts
        if training:
            self.training_router_width_token_units += tokens * num_experts
        router_flops = 2 * tokens * self.n_embd * num_experts
        expert_flops = 4 * tokens * self.top_k * self.n_embd * self.n_embd
        self.moe_matmul_flops += multiplier * (router_flops + expert_flops)
        self.expert_count_trace.append(num_experts)


def protocol_id(config: StructuralNASEvaluationConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _strict_json_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_payload(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _synaptic_config() -> SynapticConfig:
    return SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=False,
        router_contrastive_lr=0.0,
        router_contrastive_push=0.0,
        native_genetics=False,
    )


def _lifecycle_config(config: StructuralNASEvaluationConfig) -> SplitMergeConfig:
    return SplitMergeConfig(
        enabled=True,
        merges_per_call=0,
        splits_per_call=0,
        resets_per_call=0,
        function_preserving=True,
        min_step_interval=0,
        warmup_steps=0,
        ddp_broadcast=False,
        variable_expert_count=True,
        min_experts=config.min_experts,
        max_experts=config.max_experts,
        growth_budget_pct=0.5,
        reset_health_max=config.reset_health_max,
        split_health_min=config.split_health_min,
        health_mode=config.health_mode,
        use_neuroscore=True,
        neuroscore_weight=0.5,
    )


def _configure_initial_model(
    model: SynapticMoE,
    config: StructuralNASEvaluationConfig,
) -> None:
    """Three routed specialists plus one intentionally dormant expert."""
    with torch.no_grad():
        route_directions = torch.tensor(
            (
                (1.0, 1.0, 1.0, 1.0),
                (1.0, -1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0, 1.0),
                (0.0, 0.0, 1.0, -1.0),
            ),
            dtype=model.router.weight.dtype,
            device=model.router.weight.device,
        )
        route_directions = route_directions / route_directions.norm(dim=1, keepdim=True)
        model.router.weight.copy_(0.8 * route_directions)
        model.router_logit_bias.zero_()
        model.router_logit_bias[-1] = config.dormant_logit_bias
        model.router_embeddings.zero_()
        model.router_embeddings[:, : config.n_embd].copy_(route_directions)
        if model.Xi is not None:
            model.Xi.zero_()
        model.fatigue.fill_(1.0 / config.initial_experts)
        model.energy.fill_(1.0)


def _make_model(config: StructuralNASEvaluationConfig, seed: int) -> SynapticMoE:
    torch.manual_seed(seed)
    model = SynapticMoE(
        n_embd=config.n_embd,
        num_experts=config.initial_experts,
        top_k=config.top_k,
        hidden_mult=1,
        cfg=_synaptic_config(),
        dropout=0.0,
    )
    _configure_initial_model(model, config)
    return model


def _sample_task(config: StructuralNASEvaluationConfig, seed: int) -> tuple[Any, Any]:
    centers = torch.tensor(
        (
            (2.0, 2.0, 2.0, 2.0),
            (2.0, -2.0, 2.0, -2.0),
            (-2.0, 2.0, -2.0, 2.0),
        ),
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 8_000)
    x = centers.repeat_interleave(config.points_per_cluster, dim=0)
    x = x + 0.08 * torch.randn(x.shape, generator=generator)
    labels = torch.arange(3).repeat_interleave(config.points_per_cluster)
    target = torch.empty_like(x)
    target[labels == 0] = torch.tanh(0.25 * x[labels == 0])
    target[labels == 1] = torch.tanh(
        0.25 * torch.roll(x[labels == 1], shifts=1, dims=1)
    )
    target[labels == 2] = torch.tanh(-0.20 * x[labels == 2])
    return x.unsqueeze(0), target.unsqueeze(0)


def _routing_shares(indices: Any, num_experts: int) -> tuple[float, ...]:
    counts = torch.bincount(indices.reshape(-1), minlength=num_experts).to(torch.float64)
    shares = counts / counts.sum()
    return tuple(float(value) for value in shares.cpu().tolist())


def _routing_metrics(shares: tuple[float, ...], dead_share_floor: float) -> tuple[float, float]:
    values = np.asarray(shares, dtype=np.float64)
    dead_fraction = float(np.mean(values < dead_share_floor))
    ordered = np.sort(values)
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    gini = float(np.sum((2.0 * ranks - values.size - 1.0) * ordered) / values.size)
    return dead_fraction, gini


def _evaluate(
    model: SynapticMoE,
    x: Any,
    target: Any,
    counter: _WorkCounter,
) -> tuple[float, tuple[float, ...]]:
    model.eval()
    num_experts = int(model.num_experts)
    with torch.no_grad():
        prediction, _ = model(x, update_mem=False)
        loss = torch.nn.functional.mse_loss(prediction, target)
    counter.record(
        num_experts=num_experts,
        tokens=int(x.shape[0] * x.shape[1]),
        training=False,
    )
    return float(loss.item()), _routing_shares(model.last_ctx["indices"], num_experts)


def _train(
    model: SynapticMoE,
    optimizer: Any,
    x: Any,
    target: Any,
    counter: _WorkCounter,
    *,
    steps: int,
) -> None:
    model.train()
    tokens = int(x.shape[0] * x.shape[1])
    for _ in range(steps):
        num_experts = int(model.num_experts)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x, update_mem=False)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        counter.record(num_experts=num_experts, tokens=tokens, training=True)


def _publish_routing_credit(model: SynapticMoE, shares: tuple[float, ...]) -> None:
    """Publish measured assignment share as fatigue and NeuroScore credit fitness."""
    with torch.no_grad():
        score = torch.tensor(shares, dtype=model.fatigue.dtype, device=model.fatigue.device)
        model.fatigue.copy_(score)
        model.energy.fill_(1.0)
        object.__setattr__(model, "last_neuroscore", score.clone())


def _optimizer_matches_model(optimizer: Any, model: SynapticMoE) -> bool:
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    grouped_ids = [id(parameter) for parameter in grouped]
    live_ids = {id(parameter) for parameter in model.parameters()}
    return len(grouped_ids) == len(set(grouped_ids)) and set(grouped_ids) == live_ids


def _event(
    *,
    name: str,
    step: int,
    method: str,
    model: SynapticMoE,
    controller: SplitMergeController | None,
    optimizer: Any,
    x: Any,
    target: Any,
    counter: _WorkCounter,
    logger: RunLogger,
) -> LifecycleEventOutcome:
    experts_before = int(model.num_experts)
    loss_before, shares_before = _evaluate(model, x, target, counter)
    planned: tuple[dict[str, Any], ...] = ()
    if controller is not None:
        _publish_routing_credit(model, shares_before)
        planned = tuple(controller._plan_uta_layer(model, step, 0))
        controller.step(step, optimizer)
    experts_after = int(model.num_experts)
    loss_after, shares_after = _evaluate(model, x, target, counter)
    outcome = LifecycleEventOutcome(
        name=name,
        step=step,
        experts_before=experts_before,
        experts_after=experts_after,
        routing_shares_before=shares_before,
        routing_shares_after=shares_after,
        planned_operations=planned,
        optimizer_synced=_optimizer_matches_model(optimizer, model),
        loss_before=loss_before,
        loss_after=loss_after,
        loss_spike=max(0.0, loss_after - loss_before),
        loss_discontinuity=abs(loss_after - loss_before),
    )
    logger.event("structural_nas_lifecycle", method=method, seed_step=step, **asdict(outcome))
    return outcome


def _run_method(
    config: StructuralNASEvaluationConfig,
    *,
    seed: int,
    method: str,
    initial_state: dict[str, Any],
    x: Any,
    target: Any,
    logger: RunLogger,
) -> StructuralNASSeedOutcome:
    model = _make_model(config, seed)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    controller = (
        SplitMergeController(model, _lifecycle_config(config)) if method == "nas" else None
    )
    counter = _WorkCounter(n_embd=config.n_embd, top_k=config.top_k)
    started = time.perf_counter()
    events: list[LifecycleEventOutcome] = []
    events.append(
        _event(
            name="initial_apoptosis",
            step=0,
            method=method,
            model=model,
            controller=controller,
            optimizer=optimizer,
            x=x,
            target=target,
            counter=counter,
            logger=logger,
        )
    )
    half_steps = config.train_steps // 2
    _train(model, optimizer, x, target, counter, steps=half_steps)
    events.append(
        _event(
            name="midpoint_neurogenesis",
            step=1,
            method=method,
            model=model,
            controller=controller,
            optimizer=optimizer,
            x=x,
            target=target,
            counter=counter,
            logger=logger,
        )
    )
    _train(model, optimizer, x, target, counter, steps=half_steps)
    events.append(
        _event(
            name="final_apoptosis",
            step=2,
            method=method,
            model=model,
            controller=controller,
            optimizer=optimizer,
            x=x,
            target=target,
            counter=counter,
            logger=logger,
        )
    )
    wall_time = time.perf_counter() - started
    final_shares = events[-1].routing_shares_after
    dead_fraction, gini = _routing_metrics(final_shares, config.dead_share_floor)
    tokens = int(x.shape[0] * x.shape[1])
    average_experts = counter.router_width_token_units / (tokens * counter.forward_calls)
    training_average_experts = counter.training_router_width_token_units / (
        tokens * counter.train_forward_calls
    )
    outcome = StructuralNASSeedOutcome(
        seed=seed,
        method=method,
        initial_loss=events[0].loss_before,
        final_loss=events[-1].loss_after,
        dead_expert_fraction=dead_fraction,
        routing_gini=gini,
        max_event_loss_spike=max(event.loss_spike for event in events),
        total_event_loss_discontinuity=sum(event.loss_discontinuity for event in events),
        final_expert_count=int(model.num_experts),
        top_k=model.top_k,
        forward_calls=counter.forward_calls,
        train_forward_calls=counter.train_forward_calls,
        expert_dispatches=counter.expert_dispatches,
        router_width_token_units=counter.router_width_token_units,
        training_router_width_token_units=counter.training_router_width_token_units,
        moe_matmul_flops=counter.moe_matmul_flops,
        average_expert_count=average_experts,
        training_average_expert_count=training_average_experts,
        wall_time_seconds=wall_time,
        expert_count_trace=tuple(counter.expert_count_trace),
        events=tuple(events),
    )
    logger.event("structural_nas_seed_outcome", **asdict(outcome))
    return outcome


def _comparison(
    outcomes: tuple[StructuralNASSeedOutcome, ...],
    field_name: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> StructuralNASMetricComparison:
    nas = {
        outcome.seed: float(getattr(outcome, field_name))
        for outcome in outcomes
        if outcome.method == "nas"
    }
    fixed = {
        outcome.seed: float(getattr(outcome, field_name))
        for outcome in outcomes
        if outcome.method == "fixed"
    }
    paired = paired_comparison(
        nas,
        fixed,
        lower_is_better=True,
        n_boot=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if paired is None:
        raise AssertionError("validated matched seeds did not produce paired statistics")
    return StructuralNASMetricComparison(
        nas=aggregate(list(nas.values())),
        fixed=aggregate(list(fixed.values())),
        paired=paired,
    )


def _finite_outcome(outcome: StructuralNASSeedOutcome) -> bool:
    values = (
        outcome.initial_loss,
        outcome.final_loss,
        outcome.dead_expert_fraction,
        outcome.routing_gini,
        outcome.max_event_loss_spike,
        outcome.total_event_loss_discontinuity,
        outcome.average_expert_count,
        outcome.training_average_expert_count,
        outcome.wall_time_seconds,
    )
    return all(math.isfinite(value) for value in values)


def _invariants(
    config: StructuralNASEvaluationConfig,
    outcomes: tuple[StructuralNASSeedOutcome, ...],
) -> dict[str, bool]:
    by_key = {(outcome.seed, outcome.method): outcome for outcome in outcomes}
    pairs = [
        (by_key[(seed, "nas")], by_key[(seed, "fixed")]) for seed in config.seeds
    ]
    nas_outcomes = [pair[0] for pair in pairs]
    fixed_outcomes = [pair[1] for pair in pairs]
    expected_nas_events = ((4, 3), (3, 5), (5, 4))
    expected_ops = ("shrink", "grow", "shrink")
    return {
        "matched_initial_state": all(
            math.isclose(nas.initial_loss, fixed.initial_loss, rel_tol=0.0, abs_tol=1e-12)
            for nas, fixed in pairs
        ),
        "exact_equal_model_work": all(
            nas.forward_calls == fixed.forward_calls
            and nas.train_forward_calls == fixed.train_forward_calls
            and nas.expert_dispatches == fixed.expert_dispatches
            and nas.router_width_token_units == fixed.router_width_token_units
            and nas.training_router_width_token_units
            == fixed.training_router_width_token_units
            and nas.moe_matmul_flops == fixed.moe_matmul_flops
            for nas, fixed in pairs
        ),
        "equal_average_expert_count": all(
            math.isclose(nas.average_expert_count, 4.0, abs_tol=1e-12)
            and math.isclose(fixed.average_expert_count, 4.0, abs_tol=1e-12)
            and math.isclose(nas.training_average_expert_count, 4.0, abs_tol=1e-12)
            and math.isclose(fixed.training_average_expert_count, 4.0, abs_tol=1e-12)
            for nas, fixed in pairs
        ),
        "routing_credit_drives_4_3_5_4": all(
            tuple((event.experts_before, event.experts_after) for event in outcome.events)
            == expected_nas_events
            and tuple(
                str(event.planned_operations[0]["kind"])
                if len(event.planned_operations) == 1
                else "invalid"
                for event in outcome.events
            )
            == expected_ops
            and all(
                math.isclose(sum(event.routing_shares_before), 1.0, abs_tol=1e-12)
                for event in outcome.events
            )
            for outcome in nas_outcomes
        ),
        "fixed_baseline_stays_at_four": all(
            all(
                event.experts_before == event.experts_after == config.initial_experts
                and not event.planned_operations
                for event in outcome.events
            )
            for outcome in fixed_outcomes
        ),
        "optimizer_param_groups_synced": all(
            all(event.optimizer_synced for event in outcome.events) for outcome in outcomes
        ),
        "fixed_placebos_are_output_identical": all(
            all(event.loss_discontinuity == 0.0 for event in outcome.events)
            for outcome in fixed_outcomes
        ),
        "nas_event_spike_within_gate": all(
            outcome.max_event_loss_spike <= config.max_event_loss_spike
            for outcome in nas_outcomes
        ),
        "all_outcomes_finite": all(_finite_outcome(outcome) for outcome in outcomes),
    }


def _verdict(
    comparisons: dict[str, StructuralNASMetricComparison],
    invariants: dict[str, bool],
) -> tuple[str, str, str]:
    broken = [name for name, holds in invariants.items() if not holds]
    if broken:
        return (
            "regression",
            "invalidated",
            "equal-work/lifecycle/stability contract failed: " + ", ".join(broken),
        )
    primary = (comparisons["final_loss"], comparisons["dead_expert_fraction"])
    regressions = [
        name
        for name in ("final_loss", "dead_expert_fraction")
        if comparisons[name].paired.delta_ci_low > 0.0
    ]
    if regressions:
        return (
            "regression",
            "invalidated",
            "NAS showed a statistically supported primary regression in: "
            + ", ".join(regressions),
        )
    if all(comparison.paired.delta_ci_high < 0.0 for comparison in primary):
        return (
            "win",
            "positive",
            "exact compute and stability controls held; both primary paired bootstrap intervals "
            "favored variable-count NAS",
        )
    return (
        "null",
        "null",
        "exact compute and stability controls held, but both primary paired bootstrap intervals "
        "did not exclude zero in favor of variable-count NAS",
    )


def _append_registry_records(
    report: StructuralNASEvaluationReport,
    registry_path: str,
) -> None:
    paired_notes = "; ".join(
        (
            f"{name}_delta={comparison.paired.mean_delta:.17g},"
            f"ci=[{comparison.paired.delta_ci_low:.17g},"
            f"{comparison.paired.delta_ci_high:.17g}]"
        )
        for name, comparison in report.comparisons.items()
    )
    for outcome in report.outcomes:
        record = make_record(
            "eval",
            {
                "dead_expert_frac": outcome.dead_expert_fraction,
                "moe_gini": outcome.routing_gini,
                "structural_final_loss": outcome.final_loss,
                "structural_event_loss_spike": outcome.max_event_loss_spike,
                "structural_event_loss_discontinuity": (
                    outcome.total_event_loss_discontinuity
                ),
                "total_training_time": outcome.wall_time_seconds,
            },
            run_id=f"{report.run_id}-{outcome.method}-s{outcome.seed}",
            config={**asdict(report.config), "method": outcome.method},
            seed=outcome.seed,
            verdict=(report.registry_verdict if outcome.method == "nas" else None),
            eligible_for_best=False,
            notes=(
                "experiment=structural_nas_evaluation; scope=controlled_synthetic; "
                "exact_equal_moe_flops=true; average_experts=4; "
                f"group_verdict={report.verdict}; {paired_notes}; artifact={report.report_path}"
            ),
        )
        append_record(record, registry_path)


def run_structural_nas_evaluation(
    config: StructuralNASEvaluationConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> StructuralNASEvaluationReport:
    config = config or StructuralNASEvaluationConfig()
    config.validate()
    run_id = uuid.uuid4().hex[:12]
    experiment_protocol_id = protocol_id(config)
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs/e2e/structural_nas_evaluation") / run_id
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix structural NAS artifacts in {output_dir}")
    statistics_path = (
        Path(report_path) if report_path is not None else output_dir / "statistics.json"
    )
    if statistics_path.exists():
        raise FileExistsError(f"refusing to overwrite structural NAS report {statistics_path}")
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path_str = str(registry_path) if registry_path is not None else None

    with RunLogger(
        output_dir,
        name="structural_nas_evaluation",
        run_id=run_id,
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-uta.7",
            "protocol_id": experiment_protocol_id,
            "config": asdict(config),
            "baseline": "fixed four-expert MoE",
            "compute_contract": "equal cumulative expert width and top-k dispatches",
            "scope": "controlled synthetic routed-learning task; not scale evidence",
        },
    ) as logger:
        outcomes_list: list[StructuralNASSeedOutcome] = []
        for seed in config.seeds:
            initial_model = _make_model(config, seed)
            initial_state = {
                name: value.detach().clone()
                for name, value in initial_model.state_dict().items()
            }
            x, target = _sample_task(config, seed)
            outcomes_list.append(
                _run_method(
                    config,
                    seed=seed,
                    method="nas",
                    initial_state=initial_state,
                    x=x,
                    target=target,
                    logger=logger,
                )
            )
            outcomes_list.append(
                _run_method(
                    config,
                    seed=seed,
                    method="fixed",
                    initial_state=initial_state,
                    x=x,
                    target=target,
                    logger=logger,
                )
            )
        outcomes = tuple(outcomes_list)
        comparisons = {
            "final_loss": _comparison(
                outcomes,
                "final_loss",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=0,
            ),
            "dead_expert_fraction": _comparison(
                outcomes,
                "dead_expert_fraction",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=1,
            ),
            "event_loss_spike": _comparison(
                outcomes,
                "max_event_loss_spike",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=2,
            ),
        }
        specialization = {
            method: aggregate(
                [
                    outcome.routing_gini
                    for outcome in outcomes
                    if outcome.method == method
                ]
            )
            for method in ("nas", "fixed")
        }
        invariants = _invariants(config, outcomes)
        verdict, registry_verdict, reason = _verdict(comparisons, invariants)
        report = StructuralNASEvaluationReport(
            schema_version=1,
            bead="bio_inspired_nanochat-uta.7",
            run_id=logger.run_id,
            protocol_id=experiment_protocol_id,
            scope="controlled synthetic routed-learning task; not language-model-scale evidence",
            config=config,
            outcomes=outcomes,
            comparisons=comparisons,
            specialization=specialization,
            invariants=invariants,
            verdict=verdict,
            registry_verdict=registry_verdict,
            verdict_reason=reason,
            report_path=str(statistics_path),
            events_path=str(output_dir / "events.jsonl"),
            registry_path=registry_path_str,
        )
        statistics_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if registry_path_str is not None:
            _append_registry_records(report, registry_path_str)
        logger.event("structural_nas_summary", **report.to_dict())
    return report


def render_report(
    report: StructuralNASEvaluationReport,
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    table = Table(title="Variable-count NAS vs fixed MoE (equal compute)")
    table.add_column("seed", justify="right")
    table.add_column("method")
    table.add_column("final MSE", justify="right")
    table.add_column("dead", justify="right")
    table.add_column("Gini", justify="right")
    table.add_column("max spike", justify="right")
    table.add_column("avg E", justify="right")
    table.add_column("MoE FLOPs", justify="right")
    table.add_column("wall ms", justify="right")
    for outcome in report.outcomes:
        table.add_row(
            str(outcome.seed),
            outcome.method,
            f"{outcome.final_loss:.6g}",
            f"{outcome.dead_expert_fraction:.3f}",
            f"{outcome.routing_gini:.3f}",
            f"{outcome.max_event_loss_spike:.5g}",
            f"{outcome.average_expert_count:.2f}",
            str(outcome.moe_matmul_flops),
            f"{1e3 * outcome.wall_time_seconds:.2f}",
        )
    console.print(table)
    color = {"win": "green", "null": "yellow", "regression": "red"}[report.verdict]
    console.print(f"[{color}]VERDICT: {report.verdict.upper()}[/{color}] — {report.verdict_reason}")
    for name, comparison in report.comparisons.items():
        paired = comparison.paired
        console.print(
            f"{name}: NAS−fixed={paired.mean_delta:.6g}, "
            f"95% bootstrap CI [{paired.delta_ci_low:.6g}, {paired.delta_ci_high:.6g}]"
        )
    console.print(f"Statistical report: {report.report_path}")
    console.print(f"Detailed events: {report.events_path}")
    if report.registry_path is not None:
        console.print(f"Result observations appended to: {report.registry_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate variable-count NAS against a fixed MoE at equal compute"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument(
        "--health-mode", choices=["product", "relative"], default="product",
        help="lifecycle health signal (sx1m); relative uses fair-share thresholds split 1.5 / reset 0.05",
    )
    parser.add_argument("--split-health-min", type=float, default=None)
    args = parser.parse_args(argv)
    config = StructuralNASEvaluationConfig()
    if args.health_mode == "relative":
        config = replace(config, health_mode="relative", split_health_min=1.5)
    if args.split_health_min is not None:
        config = replace(config, split_health_min=args.split_health_min)
    if args.seeds is not None:
        config = replace(config, seeds=tuple(args.seeds))
    if args.train_steps is not None:
        config = replace(config, train_steps=args.train_steps)
    if args.bootstrap_samples is not None:
        config = replace(config, bootstrap_samples=args.bootstrap_samples)
    report = run_structural_nas_evaluation(
        config,
        run_dir=args.run_dir,
        report_path=args.report_path,
        registry_path=args.registry_path,
    )
    render_report(report)
    return 1 if report.verdict == "regression" else 0


if __name__ == "__main__":
    raise SystemExit(main())
