"""Falsifiable hypothesis generation and immutable pre-registration.

This is the first stage of the AI-neuroscientist flywheel (r00r.2.1). It deliberately does not
run experiments. It freezes the mechanism intervention/control, confirmatory metric, directional
minimum effect, fresh paired seeds, fixed stopping rule, and compute cap before r00r.2.2 can spend
any compute. Exploratory results and interpretability signals affect only proposal priority; the
confirmatory endpoint comes from a static mechanism template and cannot be selected post hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.ablation_registry import MECHANISMS, MechanismFlag
from bio_inspired_nanochat.checkpoint_manager import _git_sha
from bio_inspired_nanochat.metrics_schema import Direction, get_metric
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    read_records,
)

DEFAULT_PREREGISTRY = os.path.join("results", "preregistrations.jsonl")
_INFRASTRUCTURE_MECHANISMS = {
    "flex_attention",
    "native_presyn",
    "native_genetics",
    "recurrence_checkpoint",
}


@dataclass(frozen=True)
class _HypothesisTemplate:
    metric: str
    minimum_effect: float
    effect_scale: str
    harness: str
    rationale: str


_QUALITY = _HypothesisTemplate(
    "eval_bpb",
    0.005,
    "absolute",
    "eval_matrix",
    "The mechanism should improve held-out language modeling at equal tokens and model FLOPs.",
)
_MEMORY = _HypothesisTemplate(
    "niah_accuracy",
    0.02,
    "absolute",
    "eval_matrix",
    "The mechanism should improve held-out associative retrieval at equal compute.",
)
_FORGETTING = _HypothesisTemplate(
    "forgetting_rate",
    0.02,
    "absolute",
    "working_memory_suite",
    "The consolidation mechanism should reduce peak-to-final loss of prior-task accuracy.",
)
_ROUTING = _HypothesisTemplate(
    "dead_expert_frac",
    0.02,
    "absolute",
    "structural_falsification",
    "The routing mechanism should reduce under-used experts at fixed average expert compute.",
)
_CALIBRATION = _HypothesisTemplate(
    "id_ece",
    0.005,
    "absolute",
    "uncertainty_calibration",
    "Release stochasticity should improve held-out calibration at a fixed prediction budget.",
)
_STABILITY = _HypothesisTemplate(
    "integrator_divergence_rate",
    0.05,
    "absolute",
    "metriplectic_stability_curve",
    "The guarded dynamics should reduce divergence across the predeclared stress sweep.",
)
_THROUGHPUT = _HypothesisTemplate(
    "tok_per_sec",
    0.05,
    "relative",
    "attention_backend",
    "The backend should improve end-to-end forward/backward throughput at function parity.",
)
_TROPICAL = _HypothesisTemplate(
    "tropical_exactness_rate",
    0.02,
    "absolute",
    "tropical_falsification",
    "The certified controller should increase exact soft-to-hard selection agreement.",
)

_TEMPLATES: dict[str, _HypothesisTemplate] = {
    "presyn": _MEMORY,
    "hebbian": _MEMORY,
    "metabolism": _ROUTING,
    "genome": _ROUTING,
    "stochastic_release": _CALIBRATION,
    "doc2": _MEMORY,
    "septin_barrier": _QUALITY,
    "bdnf": _FORGETTING,
    "bistable_latch": _FORGETTING,
    "flex_attention": _THROUGHPUT,
    "native_presyn": _THROUGHPUT,
    "native_genetics": _THROUGHPUT,
    "learnable_kinetics": _QUALITY,
    "differentiable_recurrence": _QUALITY,
    "cusp_latch": _FORGETTING,
    "metriplectic_integrator": _STABILITY,
    "recurrence_checkpoint": _THROUGHPUT,
    "topological_nas": _ROUTING,
    "tropical_skeleton": _TROPICAL,
}


@dataclass(frozen=True)
class InterpretabilitySignal:
    """Exploratory signal used only to rank, never to redefine, a hypothesis."""

    mechanism: str
    signal_name: str
    effect_size: float
    confidence: float
    source_id: str

    def __post_init__(self) -> None:
        if self.mechanism not in {flag.mechanism for flag in MECHANISMS}:
            raise ValueError(f"unknown signal mechanism {self.mechanism!r}")
        if not self.signal_name.strip() or not self.source_id.strip():
            raise ValueError("signal_name and source_id must be non-empty")
        if not math.isfinite(self.effect_size):
            raise ValueError("effect_size must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> InterpretabilitySignal:
        return cls(
            mechanism=str(payload["mechanism"]),
            signal_name=str(payload["signal_name"]),
            effect_size=float(payload["effect_size"]),
            confidence=float(payload["confidence"]),
            source_id=str(payload["source_id"]),
        )


@dataclass(frozen=True)
class FixedStoppingRule:
    paired_seed_count: int
    confidence_interval: str
    multiplicity_control: str
    positive_rule: str
    null_rule: str
    invalidation_rule: str
    no_early_efficacy_stop: bool = True

    def __post_init__(self) -> None:
        if self.paired_seed_count < 2:
            raise ValueError("paired_seed_count must be at least two")
        if not self.no_early_efficacy_stop:
            raise ValueError("pre-registrations must forbid early efficacy stopping")


@dataclass(frozen=True)
class ComputeBudget:
    maximum_runs: int
    maximum_tokens_per_run: int
    equal_model_flops: bool = True

    def __post_init__(self) -> None:
        if self.maximum_runs < 2 or self.maximum_runs % 2:
            raise ValueError("maximum_runs must be a positive even number")
        if self.maximum_tokens_per_run < 1:
            raise ValueError("maximum_tokens_per_run must be positive")
        if not self.equal_model_flops:
            raise ValueError("confirmatory interventions must use equal model FLOPs")


@dataclass(frozen=True)
class PreregisteredHypothesis:
    schema_version: int
    hypothesis_id: str
    batch_id: str
    registered_at: str
    git_sha: str | None
    results_snapshot_sha256: str
    mechanism: str
    mechanism_field: str
    proposal_rank: int
    proposal_score: float
    statement: str
    rationale: str
    primary_metric: str
    metric_direction: str
    minimum_effect: float
    effect_scale: str
    harness: str
    control: dict[str, Any]
    intervention: dict[str, Any]
    paired_seeds: tuple[int, ...]
    stopping_rule: FixedStoppingRule
    compute_budget: ComputeBudget
    exploratory_run_ids: tuple[str, ...]
    interpretability_signal_ids: tuple[str, ...]
    status: str = "preregistered"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported preregistration schema version")
        if not self.hypothesis_id.startswith("hyp-") or not self.batch_id.startswith("batch-"):
            raise ValueError("hypothesis_id and batch_id have invalid prefixes")
        if self.status != "preregistered":
            raise ValueError("new hypothesis status must be 'preregistered'")
        if self.proposal_rank < 1 or not math.isfinite(self.proposal_score) or self.proposal_score < 0:
            raise ValueError("proposal rank/score must be finite and positive")
        try:
            snapshot_bytes = bytes.fromhex(self.results_snapshot_sha256)
        except ValueError as exc:
            raise ValueError("results_snapshot_sha256 must be hexadecimal") from exc
        if len(snapshot_bytes) != 32:
            raise ValueError("results_snapshot_sha256 must be a full SHA-256 digest")
        try:
            registered = datetime.fromisoformat(self.registered_at)
        except ValueError as exc:
            raise ValueError("registered_at must be an ISO-8601 timestamp") from exc
        if registered.tzinfo is None:
            raise ValueError("registered_at must include a timezone")
        if self.control == self.intervention:
            raise ValueError("control and intervention must differ")
        if len(self.paired_seeds) != self.stopping_rule.paired_seed_count:
            raise ValueError("paired seed count must match the stopping rule")
        if len(self.paired_seeds) != len(set(self.paired_seeds)):
            raise ValueError("paired seeds must be unique")
        if self.compute_budget.maximum_runs != 2 * len(self.paired_seeds):
            raise ValueError("compute budget must contain exactly one control/intervention pair per seed")
        if self.effect_scale not in {"absolute", "relative"}:
            raise ValueError("effect_scale must be absolute or relative")
        if not math.isfinite(self.minimum_effect) or self.minimum_effect <= 0.0:
            raise ValueError("minimum_effect must be finite and positive")
        metric = get_metric(self.primary_metric)
        if metric is None or metric.direction.value != self.metric_direction:
            raise ValueError("primary metric and registered direction do not match the metric schema")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PreregisteredHypothesis:
        data = dict(payload)
        data["paired_seeds"] = tuple(int(seed) for seed in data["paired_seeds"])
        data["exploratory_run_ids"] = tuple(str(value) for value in data["exploratory_run_ids"])
        data["interpretability_signal_ids"] = tuple(
            str(value) for value in data["interpretability_signal_ids"]
        )
        data["stopping_rule"] = FixedStoppingRule(**data["stopping_rule"])
        data["compute_budget"] = ComputeBudget(**data["compute_budget"])
        return cls(**data)


def results_snapshot_digest(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.exists():
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_preregistrations(path: str | Path = DEFAULT_PREREGISTRY) -> list[PreregisteredHypothesis]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[PreregisteredHypothesis] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.JSONDecoder().decode(line)
                records.append(PreregisteredHypothesis.from_json(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid preregistration at {source}:{line_number}: {exc}") from exc
    return records


def append_preregistration(
    record: PreregisteredHypothesis,
    path: str | Path = DEFAULT_PREREGISTRY,
) -> None:
    destination = Path(path)
    existing = read_preregistrations(destination)
    if any(item.hypothesis_id == record.hypothesis_id for item in existing):
        raise ValueError(f"hypothesis {record.hypothesis_id} is already preregistered")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def _active_value(flag: MechanismFlag) -> Any:
    if flag.default != flag.off_value:
        return flag.default
    if isinstance(flag.off_value, bool):
        return not flag.off_value
    if isinstance(flag.off_value, int):
        return 1
    if isinstance(flag.off_value, float):
        return 1.0
    raise TypeError(f"cannot infer active value for {flag.field}")


def _activation_config(flag: MechanismFlag, by_field: Mapping[str, MechanismFlag]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for prerequisite in flag.requires:
        prerequisite_flag = by_field[prerequisite]
        config.update(_activation_config(prerequisite_flag, by_field))
    config[flag.field] = _active_value(flag)
    return config


def _direct_evidence(flag: MechanismFlag, records: Sequence[RunRecord]) -> tuple[str, ...]:
    needles = (flag.mechanism.casefold(), flag.field.casefold())
    return tuple(
        record.run_id
        for record in records
        if any(needle in record.notes.casefold() for needle in needles)
    )


def _fresh_seeds(records: Sequence[RunRecord], *, seed_start: int, count: int) -> tuple[int, ...]:
    used = {record.seed for record in records if record.seed is not None}
    seeds: list[int] = []
    candidate = seed_start
    while len(seeds) < count:
        if candidate not in used:
            seeds.append(candidate)
        candidate += 1
    return tuple(seeds)


def _metric_direction(metric_name: str) -> str:
    metric = get_metric(metric_name)
    if metric is None or metric.direction == Direction.NEUTRAL:
        raise ValueError(f"hypothesis metric {metric_name!r} must have a directional schema")
    return metric.direction.value


def _canonical_digest(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


def generate_hypotheses(
    records: Sequence[RunRecord],
    *,
    results_digest: str,
    signals: Sequence[InterpretabilitySignal] = (),
    mechanisms: Sequence[MechanismFlag] = MECHANISMS,
    selected_mechanisms: Iterable[str] | None = None,
    limit: int = 3,
    paired_seed_count: int = 8,
    seed_start: int = 10007,
    maximum_tokens_per_run: int = 131_072,
    registered_at: str | None = None,
    include_infrastructure: bool = False,
) -> list[PreregisteredHypothesis]:
    """Generate deterministic, fixed-endpoint proposals from an immutable evidence snapshot."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(results_digest) != 64:
        raise ValueError("results_digest must be a full SHA-256 digest")
    selected = set(selected_mechanisms) if selected_mechanisms is not None else None
    known = {flag.mechanism for flag in mechanisms}
    if selected is not None and not selected <= known:
        raise ValueError(f"unknown selected mechanisms: {sorted(selected - known)}")

    signals_by_mechanism: dict[str, list[InterpretabilitySignal]] = {}
    for signal in signals:
        signals_by_mechanism.setdefault(signal.mechanism, []).append(signal)
    candidates: list[tuple[float, MechanismFlag, tuple[str, ...]]] = []
    for flag in mechanisms:
        if selected is not None and flag.mechanism not in selected:
            continue
        if not include_infrastructure and flag.mechanism in _INFRASTRUCTURE_MECHANISMS:
            continue
        evidence = _direct_evidence(flag, records)
        exploratory_signal = sum(
            min(abs(signal.effect_size), 5.0) * signal.confidence
            for signal in signals_by_mechanism.get(flag.mechanism, [])
        )
        evidence_gap = 2.0 / (1.0 + len(evidence))
        default_claim_risk = 0.5 if flag.default_on else 0.0
        candidates.append((evidence_gap + default_claim_risk + exploratory_signal, flag, evidence))
    candidates.sort(key=lambda item: (-item[0], item[1].mechanism))
    candidates = candidates[:limit]
    if not candidates:
        raise ValueError("no mechanisms remain after filtering")

    timestamp = registered_at or datetime.now(UTC).isoformat()
    seeds = _fresh_seeds(records, seed_start=seed_start, count=paired_seed_count)
    batch_payload = {
        "results_snapshot_sha256": results_digest,
        "mechanisms": [flag.mechanism for _, flag, _ in candidates],
        "paired_seeds": seeds,
        "registered_at": timestamp,
    }
    batch_id = _canonical_digest(batch_payload, prefix="batch")
    by_field = {flag.field: flag for flag in mechanisms}
    stopping_rule = FixedStoppingRule(
        paired_seed_count=paired_seed_count,
        confidence_interval="paired percentile bootstrap; 10,000 resamples; alpha=0.05",
        multiplicity_control=f"Holm correction across {len(candidates)} hypotheses in {batch_id}",
        positive_rule=(
            "After every preregistered seed pair completes, the multiplicity-adjusted 95% paired "
            "confidence interval must exclude zero in the predicted direction and the paired "
            "mean improvement must meet or exceed minimum_effect."
        ),
        null_rule=(
            "After every preregistered seed pair completes, any non-invalidated result that does "
            "not satisfy positive_rule is reported as null; no seed replacement is allowed."
        ),
        invalidation_rule=(
            "Mark invalidated, never positive, if equal-token/equal-FLOP matching fails, a run is "
            "non-finite, a prerequisite toggle is inactive, or the compute cap is exceeded."
        ),
    )
    budget = ComputeBudget(
        maximum_runs=2 * paired_seed_count,
        maximum_tokens_per_run=maximum_tokens_per_run,
    )

    hypotheses: list[PreregisteredHypothesis] = []
    for rank, (score, flag, evidence) in enumerate(candidates, start=1):
        template = _TEMPLATES[flag.mechanism]
        direction = _metric_direction(template.metric)
        intervention = _activation_config(flag, by_field)
        control = {**intervention, flag.field: flag.off_value}
        verb = "increase" if direction == Direction.HIGHER_BETTER.value else "decrease"
        statement = (
            f"At equal tokens and model FLOPs, enabling {flag.mechanism} ({flag.field}) will "
            f"{verb} {template.metric} by at least {template.minimum_effect:g} "
            f"{template.effect_scale} versus its preregistered ablation."
        )
        signal_ids = tuple(
            signal.source_id for signal in signals_by_mechanism.get(flag.mechanism, [])
        )
        identity_payload = {
            "results_snapshot_sha256": results_digest,
            "mechanism": flag.mechanism,
            "metric": template.metric,
            "control": control,
            "intervention": intervention,
            "paired_seeds": seeds,
            "minimum_effect": template.minimum_effect,
            "effect_scale": template.effect_scale,
        }
        hypotheses.append(
            PreregisteredHypothesis(
                schema_version=1,
                hypothesis_id=_canonical_digest(identity_payload, prefix="hyp"),
                batch_id=batch_id,
                registered_at=timestamp,
                git_sha=_git_sha(),
                results_snapshot_sha256=results_digest,
                mechanism=flag.mechanism,
                mechanism_field=flag.field,
                proposal_rank=rank,
                proposal_score=score,
                statement=statement,
                rationale=(
                    f"{template.rationale} Exploratory evidence is disclosed only for ranking: "
                    f"{len(evidence)} prior direct run(s), {len(signal_ids)} interpretability signal(s)."
                ),
                primary_metric=template.metric,
                metric_direction=direction,
                minimum_effect=template.minimum_effect,
                effect_scale=template.effect_scale,
                harness=template.harness,
                control=control,
                intervention=intervention,
                paired_seeds=seeds,
                stopping_rule=stopping_rule,
                compute_budget=budget,
                exploratory_run_ids=evidence,
                interpretability_signal_ids=signal_ids,
            )
        )
    return hypotheses


def _read_signals(path: str | None) -> list[InterpretabilitySignal]:
    if path is None:
        return []
    with Path(path).open(encoding="utf-8") as handle:
        raw_payload = handle.read()
    try:
        payload = json.JSONDecoder().decode(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid interpretability signal JSON at {path}: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise TypeError("interpretability signal input must be a JSON list")
    return [InterpretabilitySignal.from_json(item) for item in payload]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and preregister falsifiable hypotheses")
    parser.add_argument("--results-registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--preregistry", default=DEFAULT_PREREGISTRY)
    parser.add_argument("--signals", default=None)
    parser.add_argument("--mechanism", action="append", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--paired-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=10007)
    parser.add_argument("--maximum-tokens-per-run", type=int, default=131_072)
    parser.add_argument("--registered-at", default=None)
    parser.add_argument("--include-infrastructure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    records = read_records(args.results_registry)
    hypotheses = generate_hypotheses(
        records,
        results_digest=results_snapshot_digest(args.results_registry),
        signals=_read_signals(args.signals),
        selected_mechanisms=args.mechanism,
        limit=args.limit,
        paired_seed_count=args.paired_seeds,
        seed_start=args.seed_start,
        maximum_tokens_per_run=args.maximum_tokens_per_run,
        registered_at=args.registered_at,
        include_infrastructure=args.include_infrastructure,
    )
    if not args.dry_run:
        for hypothesis in hypotheses:
            append_preregistration(hypothesis, args.preregistry)

    table = Table(title="Preregistered hypothesis proposals")
    table.add_column("Rank", justify="right")
    table.add_column("Mechanism")
    table.add_column("Primary metric")
    table.add_column("Prediction")
    table.add_column("ID")
    for hypothesis in hypotheses:
        table.add_row(
            str(hypothesis.proposal_rank),
            hypothesis.mechanism,
            hypothesis.primary_metric,
            hypothesis.statement,
            hypothesis.hypothesis_id,
        )
    Console().print(table)
    destination = "dry run (nothing appended)" if args.dry_run else args.preregistry
    Console().print(f"[green]Registry destination:[/green] {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(_main())
    except (TypeError, ValueError) as exc:
        Console().print(f"[bold red]Pre-registration aborted:[/bold red] {exc}")
        raise SystemExit(2) from exc
