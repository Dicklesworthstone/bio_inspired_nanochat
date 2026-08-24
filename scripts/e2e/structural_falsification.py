"""Matched-seed falsification of topological NAS against the UTA heuristic.

This is a deliberately small, controlled CPU experiment for bead
``bio_inspired_nanochat-0642.5.3.1``.  It is not a scale claim.  Each seed starts from
the same four-expert MoE state and spends the same expert-dispatch budget; only the
structural lifecycle controller differs:

* ``topological`` uses the live spectral, H0-persistence, and optimal-transport
  certificates; and
* ``uta`` uses the existing utilization-times-energy health thresholds.

The predeclared primary outcomes are dead-expert fraction, final prediction MSE, and
the positive part of the immediate event loss change.  Exact certificate, work, and
fallback invariants fail closed to an ``invalidated`` verdict.  If they hold but the
paired bootstrap intervals do not support all three improvements, the honest verdict
is ``null`` rather than a selectively reported success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

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
from bio_inspired_nanochat.structural_geometry import (
    condition_number,
    coverage_signal,
    ot_merge_certificate,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class StructuralFalsificationConfig:
    """Predeclared seeds, work budget, and invariant tolerances."""

    # Held out from the exploratory seeds used to tune the synthetic fixture.
    seeds: tuple[int, ...] = (307, 331, 353, 379, 401, 431, 457, 487)
    n_embd: int = 4
    num_experts: int = 4
    top_k: int = 2
    points_per_cluster: int = 4
    train_steps: int = 12
    learning_rate: float = 2e-4
    dead_share_floor: float = 0.05
    perturb_epsilon: float = 1e-3
    bootstrap_samples: int = 10_000

    def validate(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least three unique values")
        if self.n_embd != 4:
            raise ValueError("n_embd must be four for the preregistered controlled fixture")
        if self.num_experts < 4:
            raise ValueError("num_experts must be at least four")
        if not 1 <= self.top_k < self.num_experts:
            raise ValueError("top_k must be positive and smaller than num_experts")
        if self.points_per_cluster < 2:
            raise ValueError("points_per_cluster must be at least two")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.dead_share_floor) or not 0.0 < self.dead_share_floor < 1.0:
            raise ValueError("dead_share_floor must be finite and strictly between zero and one")
        if not math.isfinite(self.perturb_epsilon) or self.perturb_epsilon <= 0.0:
            raise ValueError("perturb_epsilon must be finite and positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")


@dataclass(frozen=True)
class StructuralSeedOutcome:
    """Replayable measurements for one seed and one lifecycle controller."""

    seed: int
    method: str
    expert_count_before: int
    expert_count_after: int
    top_k: int
    model_forward_calls: int
    model_work_units: int
    lifecycle_mode: str
    lifecycle_action: str
    lifecycle_reason: str
    planned_operations: tuple[dict[str, Any], ...]
    loss_before: float
    loss_after_event: float
    event_loss_spike: float
    event_loss_discontinuity: float
    final_loss: float
    dead_expert_fraction: float
    routing_gini: float
    kappa_bound: float | None
    max_child_condition_number: float | None
    spectral_bound_holds: bool | None
    persistence_ratio: float | None
    persistence_stability_delta: float | None
    persistence_stability_bound: float | None
    persistence_stability_holds: bool | None
    merge_transport_cost: float | None
    merge_naive_cost: float | None
    ot_merge_optimal: bool | None


@dataclass(frozen=True)
class FallbackOutcome:
    """Deterministic identity check for missing topological evidence."""

    seed: int
    fallback_reason: str
    uta_plans_equal: bool
    state_identity: bool
    output_identity: bool


@dataclass(frozen=True)
class StructuralMetricComparison:
    """Treatment/baseline aggregates and their matched-seed comparison."""

    topological: Aggregate
    uta: Aggregate
    paired: PairedResult


@dataclass(frozen=True)
class StructuralFalsificationReport:
    """Strict statistical evidence plus the predeclared verdict."""

    schema_version: int
    bead: str
    run_id: str
    protocol_id: str
    scope: str
    config: StructuralFalsificationConfig
    outcomes: tuple[StructuralSeedOutcome, ...]
    fallback_outcomes: tuple[FallbackOutcome, ...]
    comparisons: dict[str, StructuralMetricComparison]
    invariants: dict[str, bool]
    verdict: str
    verdict_reason: str
    report_path: str
    events_path: str
    registry_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_payload(asdict(self))

    def assert_not_invalidated(self) -> None:
        if self.verdict == "invalidated":
            raise AssertionError(self.verdict_reason)


def protocol_id(config: StructuralFalsificationConfig) -> str:
    """Stable identifier for the predeclared configuration, not a run identity."""
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


def _synaptic_config(*, topological: bool) -> SynapticConfig:
    return SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=False,
        router_contrastive_lr=0.0,
        router_contrastive_push=0.0,
        topological_nas=topological,
        native_genetics=False,
    )


def _lifecycle_config() -> SplitMergeConfig:
    return SplitMergeConfig(
        enabled=True,
        merge_cosine_threshold=0.5,
        merge_health_max=0.25,
        merges_per_call=1,
        split_health_min=0.75,
        splits_per_call=1,
        resets_per_call=0,
        function_preserving=True,
        fp_divergence_noise=0.01,
        min_step_interval=0,
        warmup_steps=0,
        ddp_broadcast=False,
        topological_kappa_target=20.0,
        topological_merge_cost_ratio_max=0.1,
        topological_functional_distance_max=0.2,
        topological_persistence_ratio_threshold=1.2,
        topological_coverage_distance_threshold=0.1,
        topological_max_points=64,
        topological_max_dim=8,
        topological_max_spectral_candidates=4,
        topological_max_exact_merge_candidates=4,
    )


def _configure_controlled_model(model: SynapticMoE) -> None:
    """Install a reproducible redundant/dead pair and two healthy specialists."""
    with torch.no_grad():
        spectral_profiles = (
            (1.0, 0.8, 0.6, 0.5),
            (1.0, 0.8, 0.6, 0.5),
            (1.0, 0.9, 0.8, 0.7),
            (1.0, 1.0, 1.0, 1.0),
        )
        for index, expert in enumerate(model.experts):
            scale = 0.2 * (index + 1)
            for linear in (expert.fc1, expert.fc2):
                linear.w_slow.zero_()
                linear.w_slow.copy_(
                    scale
                    * torch.diag(
                        torch.tensor(
                            spectral_profiles[index],
                            dtype=linear.w_slow.dtype,
                            device=linear.w_slow.device,
                        )
                    ).reshape(
                        linear.w_slow.shape[0], linear.w_slow.shape[1]
                    )
                )
                if linear.w_fast is not None:
                    linear.w_fast.zero_()
                if linear.bias is not None:
                    linear.bias.zero_()

        # Experts 0/1 are deliberately redundant and unhealthy: both controllers
        # see the same obvious merge candidate.
        model.experts[1].load_state_dict(model.experts[0].state_dict())
        model.Xi.zero_()
        model.router_logit_bias.zero_()
        model.fatigue.copy_(
            torch.tensor((0.1, 0.1, 0.8, 1.0), dtype=model.fatigue.dtype)
        )
        model.energy.fill_(1.0)

        route_directions = torch.tensor(
            (
                (1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 1.0),
                (1.0, -1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0, 1.0),
            ),
            dtype=model.router.weight.dtype,
            device=model.router.weight.device,
        )
        route_directions = route_directions / route_directions.norm(dim=1, keepdim=True)
        model.router.weight.copy_(0.7 * route_directions)
        model.router_probe.weight.zero_()
        model.router_probe.weight[: model.router.in_features].copy_(
            torch.eye(
                model.router.in_features,
                dtype=model.router_probe.weight.dtype,
                device=model.router_probe.weight.device,
            )
        )
        model.router_embeddings.zero_()
        model.router_embeddings[:, : model.router.in_features].copy_(route_directions)


def _make_model(config: StructuralFalsificationConfig, seed: int, *, topological: bool) -> SynapticMoE:
    torch.manual_seed(seed)
    model = SynapticMoE(
        config.n_embd,
        config.num_experts,
        config.top_k,
        hidden_mult=1,
        cfg=_synaptic_config(topological=topological),
        dropout=0.0,
    )
    _configure_controlled_model(model)
    return model


def _initial_state(config: StructuralFalsificationConfig, seed: int) -> dict[str, Any]:
    model = _make_model(config, seed, topological=False)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _sample_batch(
    config: StructuralFalsificationConfig, seed: int
) -> tuple[Any, Any]:
    centers = torch.tensor(
        (
            (2.0, 2.0, 2.0, 2.0),
            (2.0, -2.0, 2.0, -2.0),
            (-2.0, 2.0, -2.0, 2.0),
            (-2.0, 2.0, 0.0, 0.0),
        ),
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_000)
    x = centers.repeat_interleave(config.points_per_cluster, dim=0)
    x = x + 0.06 * torch.randn(x.shape, generator=generator)
    x = x.unsqueeze(0)
    target = torch.tanh(0.15 * x)
    return x, target


def _loss_and_routing(model: SynapticMoE, x: Any, target: Any) -> tuple[float, Any]:
    model.eval()
    with torch.no_grad():
        prediction, _ = model(x, update_mem=False)
        loss = torch.nn.functional.mse_loss(prediction, target)
    return float(loss.item()), model.last_ctx["indices"].detach().clone()


def _routing_summary(indices: Any, num_experts: int, dead_share_floor: float) -> tuple[float, float]:
    counts = torch.bincount(indices.reshape(-1), minlength=num_experts).to(torch.float64)
    shares = (counts / counts.sum()).cpu().numpy()
    dead_fraction = float(np.mean(shares < dead_share_floor))
    ordered = np.sort(shares)
    ranks = np.arange(1, num_experts + 1, dtype=np.float64)
    gini = float(np.sum((2.0 * ranks - num_experts - 1.0) * ordered) / num_experts)
    return dead_fraction, gini


def _expert_weight_samples(model: SynapticMoE) -> tuple[np.ndarray, ...]:
    samples: list[np.ndarray] = []
    for expert in model.experts:
        samples.append(
            np.concatenate(
                (
                    expert.fc1.w_slow.detach().cpu().numpy().reshape(-1),
                    expert.fc2.w_slow.detach().cpu().numpy().reshape(-1),
                )
            ).astype(np.float64, copy=False)
        )
    return tuple(samples)


def _persistence_stability(
    points: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> tuple[float, float, bool, float]:
    baseline = coverage_signal(points, ratio_threshold=1.2)
    rng = np.random.default_rng(seed + 20_000)
    directions = rng.normal(size=points.shape)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    perturbation = epsilon * directions / np.maximum(norms, 1e-12)
    shifted = coverage_signal(points + perturbation, ratio_threshold=1.2)
    delta = abs(shifted.max_gap - baseline.max_gap)
    bound = 2.0 * epsilon + 1e-10
    return delta, bound, bool(delta <= bound), baseline.persistence_ratio


def _run_method(
    config: StructuralFalsificationConfig,
    *,
    seed: int,
    method: str,
    initial_state: dict[str, Any],
    x: Any,
    target: Any,
    logger: RunLogger,
) -> StructuralSeedOutcome:
    topological = method == "topological"
    model = _make_model(config, seed, topological=topological)
    model.load_state_dict(initial_state)
    lifecycle_cfg = _lifecycle_config()
    controller = SplitMergeController(
        model,
        lifecycle_cfg,
        event_logger=logger if topological else None,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    expert_count_before = model.num_experts
    loss_before, _ = _loss_and_routing(model, x, target)

    projected_points: np.ndarray | None = None
    pre_event_weights: tuple[np.ndarray, ...] = ()
    if topological:
        projected_points = controller._routing_points(model)
        pre_event_weights = _expert_weight_samples(model)
        planned_operations: tuple[dict[str, Any], ...] = ()
    else:
        planned_operations = tuple(controller._plan_uta_layer(model, seed, 0))

    controller.step(seed, optimizer)
    expert_count_after = model.num_experts
    loss_after, _ = _loss_and_routing(model, x, target)

    lifecycle_mode = "uta"
    lifecycle_action = "+".join(str(op["kind"]) for op in planned_operations) or "noop"
    lifecycle_reason = "health_threshold_plan"
    kappa_bound: float | None = None
    max_child_condition_number: float | None = None
    spectral_bound_holds: bool | None = None
    persistence_ratio: float | None = None
    persistence_delta: float | None = None
    persistence_bound: float | None = None
    persistence_holds: bool | None = None
    merge_transport_cost: float | None = None
    merge_naive_cost: float | None = None
    ot_merge_optimal: bool | None = None

    if topological:
        if not controller.topological_decisions:
            raise AssertionError("topological controller did not emit a decision")
        decision = controller.topological_decisions[-1]
        lifecycle_mode = decision.mode
        lifecycle_action = decision.action
        lifecycle_reason = decision.reason
        planned_operations = (asdict(decision),)
        kappa_bound = decision.kappa_bound
        persistence_ratio = decision.persistence_ratio

        if projected_points is not None:
            persistence_delta, persistence_bound, persistence_holds, measured_ratio = (
                _persistence_stability(
                    projected_points,
                    epsilon=config.perturb_epsilon,
                    seed=seed,
                )
            )
            if persistence_ratio is None:
                persistence_ratio = measured_ratio

        if (
            decision.split_source is not None
            and decision.split_destination is not None
            and decision.kappa_bound is not None
        ):
            child_conditions = (
                condition_number(
                    model.experts[decision.split_source].fc1.w_slow.detach().cpu().numpy()
                ),
                condition_number(
                    model.experts[decision.split_destination].fc1.w_slow.detach().cpu().numpy()
                ),
            )
            max_child_condition_number = max(child_conditions)
            spectral_bound_holds = bool(
                max_child_condition_number <= decision.kappa_bound * (1.0 + 1e-5) + 1e-8
            )
        else:
            spectral_bound_holds = False

        if decision.merge_pair is not None:
            merge_i, merge_j = decision.merge_pair
            certificate = ot_merge_certificate(
                pre_event_weights[merge_i], pre_event_weights[merge_j]
            )
            merge_transport_cost = certificate.transport_cost
            merge_naive_cost = certificate.naive_cost
            ot_merge_optimal = bool(
                certificate.comparator_available
                and certificate.transport_optimal
                and certificate.transport_cost <= certificate.naive_cost + 1e-12
            )
        else:
            ot_merge_optimal = False

    model.train()
    for _ in range(config.train_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x, update_mem=False)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_loss, final_indices = _loss_and_routing(model, x, target)
    dead_fraction, gini = _routing_summary(
        final_indices,
        model.num_experts,
        config.dead_share_floor,
    )
    forward_calls = config.train_steps + 3
    model_work_units = (
        forward_calls * int(x.shape[0]) * int(x.shape[1]) * model.top_k
    )
    outcome = StructuralSeedOutcome(
        seed=seed,
        method=method,
        expert_count_before=expert_count_before,
        expert_count_after=expert_count_after,
        top_k=model.top_k,
        model_forward_calls=forward_calls,
        model_work_units=model_work_units,
        lifecycle_mode=lifecycle_mode,
        lifecycle_action=lifecycle_action,
        lifecycle_reason=lifecycle_reason,
        planned_operations=planned_operations,
        loss_before=loss_before,
        loss_after_event=loss_after,
        event_loss_spike=max(0.0, loss_after - loss_before),
        event_loss_discontinuity=abs(loss_after - loss_before),
        final_loss=final_loss,
        dead_expert_fraction=dead_fraction,
        routing_gini=gini,
        kappa_bound=kappa_bound,
        max_child_condition_number=max_child_condition_number,
        spectral_bound_holds=spectral_bound_holds,
        persistence_ratio=persistence_ratio,
        persistence_stability_delta=persistence_delta,
        persistence_stability_bound=persistence_bound,
        persistence_stability_holds=persistence_holds,
        merge_transport_cost=merge_transport_cost,
        merge_naive_cost=merge_naive_cost,
        ot_merge_optimal=ot_merge_optimal,
    )
    logger.event("structural_seed_outcome", **asdict(outcome))
    return outcome


def _run_fallback(
    config: StructuralFalsificationConfig,
    *,
    seed: int,
    initial_state: dict[str, Any],
    logger: RunLogger,
) -> FallbackOutcome:
    uta_model = _make_model(config, seed, topological=False)
    topological_model = _make_model(config, seed, topological=True)
    uta_model.load_state_dict(initial_state)
    topological_model.load_state_dict(initial_state)
    lifecycle_cfg = _lifecycle_config()
    uta_controller = SplitMergeController(uta_model, lifecycle_cfg)
    topological_controller = SplitMergeController(
        topological_model,
        lifecycle_cfg,
        event_logger=logger,
    )
    uta_plan = uta_controller._plan_uta_layer(uta_model, seed, 0)
    fallback_uta_plan = topological_controller._plan_uta_layer(
        topological_model, seed, 0
    )
    topological_controller.step(seed)
    uta_controller.step(seed)
    decision = topological_controller.topological_decisions[-1]
    uta_state = uta_model.state_dict()
    topological_state = topological_model.state_dict()
    state_identity = uta_state.keys() == topological_state.keys() and all(
        torch.equal(uta_state[name], topological_state[name]) for name in uta_state
    )
    x, _ = _sample_batch(config, seed + 1)
    uta_model.eval()
    topological_model.eval()
    with torch.no_grad():
        uta_output, _ = uta_model(x, update_mem=False)
        topological_output, _ = topological_model(x, update_mem=False)
    outcome = FallbackOutcome(
        seed=seed,
        fallback_reason=decision.reason,
        uta_plans_equal=uta_plan == fallback_uta_plan,
        state_identity=bool(state_identity),
        output_identity=bool(torch.equal(uta_output, topological_output)),
    )
    logger.event("structural_fallback_outcome", **asdict(outcome))
    return outcome


def _comparison(
    outcomes: tuple[StructuralSeedOutcome, ...],
    field: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> StructuralMetricComparison:
    topological = {
        outcome.seed: float(getattr(outcome, field))
        for outcome in outcomes
        if outcome.method == "topological"
    }
    uta = {
        outcome.seed: float(getattr(outcome, field))
        for outcome in outcomes
        if outcome.method == "uta"
    }
    paired = paired_comparison(
        topological,
        uta,
        lower_is_better=True,
        n_boot=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if paired is None:
        raise AssertionError("validated matched seeds did not produce paired statistics")
    return StructuralMetricComparison(
        topological=aggregate(list(topological.values())),
        uta=aggregate(list(uta.values())),
        paired=paired,
    )


def _finite_outcome(outcome: StructuralSeedOutcome) -> bool:
    required = (
        outcome.loss_before,
        outcome.loss_after_event,
        outcome.event_loss_spike,
        outcome.event_loss_discontinuity,
        outcome.final_loss,
        outcome.dead_expert_fraction,
        outcome.routing_gini,
    )
    optional = (
        outcome.kappa_bound,
        outcome.max_child_condition_number,
        outcome.persistence_ratio,
        outcome.persistence_stability_delta,
        outcome.persistence_stability_bound,
        outcome.merge_transport_cost,
        outcome.merge_naive_cost,
    )
    return all(math.isfinite(value) for value in required) and all(
        value is None or math.isfinite(value) for value in optional
    )


def _invariants(
    config: StructuralFalsificationConfig,
    outcomes: tuple[StructuralSeedOutcome, ...],
    fallback_outcomes: tuple[FallbackOutcome, ...],
) -> dict[str, bool]:
    by_key = {(outcome.seed, outcome.method): outcome for outcome in outcomes}
    pairs = [
        (by_key[(seed, "topological")], by_key[(seed, "uta")])
        for seed in config.seeds
    ]
    topological = [pair[0] for pair in pairs]
    return {
        "matched_seed_initial_state": all(
            math.isclose(topo.loss_before, uta.loss_before, rel_tol=0.0, abs_tol=1e-12)
            for topo, uta in pairs
        ),
        "equal_model_work": all(
            topo.model_work_units == uta.model_work_units
            and topo.model_forward_calls == uta.model_forward_calls
            for topo, uta in pairs
        ),
        "fixed_expert_count_and_top_k": all(
            outcome.expert_count_before == config.num_experts
            and outcome.expert_count_after == config.num_experts
            and outcome.top_k == config.top_k
            for outcome in outcomes
        ),
        "topological_merge_split_executed": all(
            outcome.lifecycle_mode == "topological"
            and outcome.lifecycle_action == "merge_split"
            for outcome in topological
        ),
        "spectral_condition_bound_holds": all(
            outcome.spectral_bound_holds is True for outcome in topological
        ),
        "persistence_stability_bound_holds": all(
            outcome.persistence_stability_holds is True for outcome in topological
        ),
        "ot_barycenter_cost_bound_holds": all(
            outcome.ot_merge_optimal is True for outcome in topological
        ),
        "missing_evidence_fallback_is_uta_identity": all(
            outcome.fallback_reason == "missing_routing_points"
            and outcome.uta_plans_equal
            and outcome.state_identity
            and outcome.output_identity
            for outcome in fallback_outcomes
        ),
        "all_outcomes_finite": all(_finite_outcome(outcome) for outcome in outcomes),
    }


def _verdict(
    comparisons: dict[str, StructuralMetricComparison],
    invariants: dict[str, bool],
) -> tuple[str, str]:
    broken = [name for name, holds in invariants.items() if not holds]
    if broken:
        return "invalidated", "exact invariant failure: " + ", ".join(broken)
    regressions = [
        name
        for name, comparison in comparisons.items()
        if comparison.paired.delta_ci_low > 0.0
    ]
    if regressions:
        return (
            "invalidated",
            "topological lifecycle showed a predeclared paired regression in: "
            + ", ".join(regressions),
        )
    if all(comparison.paired.delta_ci_high < 0.0 for comparison in comparisons.values()):
        return (
            "positive",
            "all exact invariants held and every primary matched-seed bootstrap interval favored "
            "the topological lifecycle",
        )
    return (
        "null",
        "all exact invariants held, but not every primary matched-seed bootstrap interval excluded "
        "zero in favor of the topological lifecycle",
    )


def _append_registry_records(
    report: StructuralFalsificationReport,
    registry_path: str,
) -> None:
    comparison_note = "; ".join(
        (
            f"{name}_delta={comparison.paired.mean_delta:.17g},"
            f"ci=[{comparison.paired.delta_ci_low:.17g},"
            f"{comparison.paired.delta_ci_high:.17g}]"
        )
        for name, comparison in report.comparisons.items()
    )
    for outcome in report.outcomes:
        method_verdict = report.verdict if outcome.method == "topological" else None
        record = make_record(
            "eval",
            {
                "dead_expert_frac": outcome.dead_expert_fraction,
                "moe_gini": outcome.routing_gini,
                "structural_final_loss": outcome.final_loss,
                "structural_event_loss_spike": outcome.event_loss_spike,
                "structural_event_loss_discontinuity": outcome.event_loss_discontinuity,
            },
            run_id=f"{report.run_id}-{outcome.method}-s{outcome.seed}",
            config={**asdict(report.config), "method": outcome.method},
            seed=outcome.seed,
            verdict=method_verdict,
            eligible_for_best=False,
            notes=(
                "experiment=structural_falsification; scope=tiny_controlled_synthetic; "
                "equal_model_work=true; fixed_expert_count=true; "
                f"group_verdict={report.verdict}; {comparison_note}; "
                f"artifact={report.report_path}"
            ),
        )
        append_record(record, registry_path)


def run_structural_falsification(
    config: StructuralFalsificationConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> StructuralFalsificationReport:
    """Run the matched comparison and persist strict audit evidence."""
    config = config or StructuralFalsificationConfig()
    config.validate()
    run_id = uuid.uuid4().hex[:12]
    experiment_protocol_id = protocol_id(config)
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs/e2e/structural_falsification") / run_id
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix structural run artifacts in {output_dir}")
    statistics_path = (
        Path(report_path) if report_path is not None else output_dir / "statistics.json"
    )
    if statistics_path.exists():
        raise FileExistsError(f"refusing to overwrite structural report {statistics_path}")
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path_str = str(registry_path) if registry_path is not None else None

    with RunLogger(
        output_dir,
        name="structural_falsification",
        run_id=run_id,
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-0642.5.3.1",
            "protocol_id": experiment_protocol_id,
            "config": asdict(config),
            "named_baseline": "UTA utilization-times-energy health thresholds",
            "scope": "tiny controlled synthetic; not scale evidence",
        },
    ) as logger:
        outcome_list: list[StructuralSeedOutcome] = []
        fallback_list: list[FallbackOutcome] = []
        for seed in config.seeds:
            state = _initial_state(config, seed)
            x, target = _sample_batch(config, seed)
            outcome_list.append(
                _run_method(
                    config,
                    seed=seed,
                    method="topological",
                    initial_state=state,
                    x=x,
                    target=target,
                    logger=logger,
                )
            )
            outcome_list.append(
                _run_method(
                    config,
                    seed=seed,
                    method="uta",
                    initial_state=state,
                    x=x,
                    target=target,
                    logger=logger,
                )
            )
            fallback_list.append(
                _run_fallback(
                    config,
                    seed=seed,
                    initial_state=state,
                    logger=logger,
                )
            )
        outcomes = tuple(outcome_list)
        fallback_outcomes = tuple(fallback_list)
        comparisons = {
            "dead_expert_fraction": _comparison(
                outcomes,
                "dead_expert_fraction",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=0,
            ),
            "final_loss": _comparison(
                outcomes,
                "final_loss",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=1,
            ),
            "event_loss_spike": _comparison(
                outcomes,
                "event_loss_spike",
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=2,
            ),
        }
        invariants = _invariants(config, outcomes, fallback_outcomes)
        verdict, verdict_reason = _verdict(comparisons, invariants)
        report = StructuralFalsificationReport(
            schema_version=1,
            bead="bio_inspired_nanochat-0642.5.3.1",
            run_id=logger.run_id,
            protocol_id=experiment_protocol_id,
            scope="tiny controlled synthetic equal-work comparison; not scale evidence",
            config=config,
            outcomes=outcomes,
            fallback_outcomes=fallback_outcomes,
            comparisons=comparisons,
            invariants=invariants,
            verdict=verdict,
            verdict_reason=verdict_reason,
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
        logger.event("structural_falsification_summary", **report.to_dict())
    return report


def render_report(
    report: StructuralFalsificationReport,
    *,
    console: Console | None = None,
) -> None:
    """Render the seed-level comparison through Rich."""
    console = console or Console()
    table = Table(title="Structural lifecycle falsification")
    table.add_column("seed", justify="right")
    table.add_column("method")
    table.add_column("action")
    table.add_column("final MSE", justify="right")
    table.add_column("event spike", justify="right")
    table.add_column("dead experts", justify="right")
    table.add_column("work", justify="right")
    for outcome in report.outcomes:
        table.add_row(
            str(outcome.seed),
            outcome.method,
            outcome.lifecycle_action,
            f"{outcome.final_loss:.6g}",
            f"{outcome.event_loss_spike:.6g}",
            f"{outcome.dead_expert_fraction:.3f}",
            str(outcome.model_work_units),
        )
    console.print(table)
    color = {"positive": "green", "null": "yellow", "invalidated": "red"}[report.verdict]
    console.print(f"[{color}]VERDICT: {report.verdict.upper()}[/{color}] — {report.verdict_reason}")
    for name, comparison in report.comparisons.items():
        paired = comparison.paired
        console.print(
            f"{name}: topological−UTA={paired.mean_delta:.6g}, "
            f"95% bootstrap CI [{paired.delta_ci_low:.6g}, {paired.delta_ci_high:.6g}]"
        )
    console.print(f"Statistical report: {report.report_path}")
    console.print(f"Detailed events: {report.events_path}")
    if report.registry_path is not None:
        console.print(f"Result observations appended to: {report.registry_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Falsify topological structural plasticity against UTA"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args(argv)
    config = StructuralFalsificationConfig()
    if args.seeds is not None:
        config = replace(config, seeds=tuple(args.seeds))
    if args.train_steps is not None:
        config = replace(config, train_steps=args.train_steps)
    if args.bootstrap_samples is not None:
        config = replace(config, bootstrap_samples=args.bootstrap_samples)
    report = run_structural_falsification(
        config,
        run_dir=args.run_dir,
        report_path=args.report_path,
        registry_path=args.registry_path,
    )
    render_report(report)
    return 1 if report.verdict == "invalidated" else 0


if __name__ == "__main__":
    raise SystemExit(main())
