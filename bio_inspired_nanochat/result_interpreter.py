"""Statistical interpretation and append-only mechanism knowledge base.

This is stage three of the AI-neuroscientist flywheel.  It consumes only r00r.2.2 audit records,
requires the complete pre-registered multiplicity family, reuses the 74f.3 paired statistics, and
records an honest ``confirmed`` / ``refuted`` / ``null`` / ``invalidated`` conclusion.  Pairwise
delta correlations are reported as redundancy candidates only; without a factorial joint
intervention they are never presented as causal interactions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.checkpoint_manager import _git_sha
from bio_inspired_nanochat.eval_stats import aggregate, holm_adjust, paired_comparison
from bio_inspired_nanochat.hypothesis_generator import (
    DEFAULT_PREREGISTRY,
    PreregisteredHypothesis,
    read_preregistrations,
    results_snapshot_digest,
)
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    append_record,
    make_record,
    read_records,
)

DEFAULT_KNOWLEDGE_BASE = "results/mechanism_knowledge.jsonl"
InterpretationVerdict = Literal["confirmed", "refuted", "null", "invalidated"]


@dataclass(frozen=True)
class KnowledgeBatch:
    schema_version: int
    knowledge_batch_id: str
    execution_batch_id: str
    preregistration_batch_id: str
    interpreted_at: str
    git_sha: str | None
    source_registry_sha256: str
    source_preregistry_sha256: str
    status: str
    ranking: tuple[str, ...]
    status_counts: dict[str, int]
    entries: tuple[dict[str, Any], ...]
    interactions: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported knowledge schema version")
        if not self.knowledge_batch_id.startswith("knowledge-"):
            raise ValueError("invalid knowledge batch ID")
        if len(self.ranking) != len(self.entries) or len(set(self.ranking)) != len(self.ranking):
            raise ValueError("ranking must contain every hypothesis exactly once")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> KnowledgeBatch:
        data = dict(payload)
        data["ranking"] = tuple(str(value) for value in data["ranking"])
        data["entries"] = tuple(dict(value) for value in data["entries"])
        data["interactions"] = tuple(dict(value) for value in data["interactions"])
        data["status_counts"] = {
            str(key): int(value) for key, value in data["status_counts"].items()
        }
        return cls(**data)


@dataclass(frozen=True)
class _AuditCell:
    record: RunRecord
    hypothesis_id: str | None
    arm: str
    seed: int
    status: str
    source_run_id: str | None


@dataclass
class _Provisional:
    hypothesis: PreregisteredHypothesis
    valid: bool
    invalidation_reasons: list[str]
    source_run_ids: tuple[str, ...]
    control_by_seed: dict[int, float]
    intervention_by_seed: dict[int, float]
    improvements_by_seed: dict[int, float]
    stats: dict[str, Any] | None
    effect_value: float | None
    effect_ci_low: float | None
    effect_ci_high: float | None
    effect_threshold_ratio: float | None


def _strict_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return value


def _canonical_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_strict_json(dict(payload)), sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.JSONDecoder().decode(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {source}:{line_number} must be an object")
            rows.append(value)
    return rows


def read_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_BASE) -> list[KnowledgeBatch]:
    return [KnowledgeBatch.from_json(row) for row in _read_jsonl(path)]


def _append_knowledge(batch: KnowledgeBatch, path: str | Path) -> None:
    destination = Path(path)
    existing = read_knowledge_base(destination)
    if any(item.execution_batch_id == batch.execution_batch_id for item in existing):
        raise ValueError(f"execution batch {batch.execution_batch_id} is already interpreted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_strict_json(batch.to_json()), sort_keys=True, allow_nan=False)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def _validate_batch_ledger(path: str | Path, execution_batch_id: str) -> dict[str, Any]:
    finished = [
        row
        for row in _read_jsonl(path)
        if row.get("event") == "batch_finished"
        and row.get("execution_batch_id") == execution_batch_id
    ]
    if len(finished) != 1:
        raise ValueError(
            f"expected exactly one batch_finished event for {execution_batch_id}; found {len(finished)}"
        )
    report = finished[0].get("report")
    if not isinstance(report, dict) or report.get("status") not in {
        "completed",
        "completed_with_failures",
    }:
        raise ValueError("batch ledger does not contain a completed execution report")
    return report


def _parse_audit_cells(records: Sequence[RunRecord], execution_batch_id: str) -> list[_AuditCell]:
    cells: list[_AuditCell] = []
    prefix = f"{execution_batch_id}-"
    for record in records:
        if not record.run_id.startswith(prefix):
            continue
        try:
            notes = json.JSONDecoder().decode(record.notes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"audit record {record.run_id} has invalid structured notes") from exc
        if not isinstance(notes, dict) or notes.get("orchestrator") != "ai_neuroscientist":
            raise ValueError(f"record {record.run_id} is not an AI-neuroscientist audit record")
        if record.seed is None:
            raise ValueError(f"audit record {record.run_id} has no seed")
        cells.append(
            _AuditCell(
                record=record,
                hypothesis_id=(
                    str(notes["hypothesis_id"])
                    if notes.get("hypothesis_id") is not None
                    else None
                ),
                arm=str(notes.get("arm", "")),
                seed=record.seed,
                status=str(notes.get("status", "")),
                source_run_id=(
                    str(notes["source_run_id"])
                    if notes.get("source_run_id") is not None
                    else None
                ),
            )
        )
    if not cells:
        raise ValueError(f"no audit records found for execution batch {execution_batch_id}")
    return cells


def _build_provisional(
    hypothesis: PreregisteredHypothesis,
    cells: Sequence[_AuditCell],
) -> _Provisional:
    relevant = [cell for cell in cells if cell.hypothesis_id == hypothesis.hypothesis_id]
    expected = {
        (arm, seed)
        for arm in ("control", "intervention")
        for seed in hypothesis.paired_seeds
    }
    grouped: dict[tuple[str, int], list[_AuditCell]] = {}
    for cell in relevant:
        grouped.setdefault((cell.arm, cell.seed), []).append(cell)
    observed = set(grouped)
    reasons: list[str] = []
    if observed != expected:
        reasons.append(
            f"cell coverage mismatch: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )
    duplicates = sorted(key for key, values in grouped.items() if len(values) != 1)
    if duplicates:
        reasons.append(f"duplicate audit cells: {duplicates!r}")

    control: dict[int, float] = {}
    intervention: dict[int, float] = {}
    source_ids: list[str] = []
    for key in sorted(expected & observed):
        values = grouped[key]
        if len(values) != 1:
            continue
        cell = values[0]
        if cell.status != "completed" or cell.record.verdict == "invalidated":
            reasons.append(f"{cell.arm}/seed={cell.seed} status={cell.status}")
            continue
        value = cell.record.metrics.get(hypothesis.primary_metric)
        if value is None or not math.isfinite(value):
            reasons.append(
                f"{cell.arm}/seed={cell.seed} lacks finite {hypothesis.primary_metric}"
            )
            continue
        if cell.source_run_id is None:
            reasons.append(f"{cell.arm}/seed={cell.seed} lacks source_run_id provenance")
            continue
        target = control if cell.arm == "control" else intervention
        target[cell.seed] = float(value)
        source_ids.append(cell.source_run_id)

    valid = not reasons and set(control) == set(hypothesis.paired_seeds) == set(intervention)
    if not valid:
        return _Provisional(
            hypothesis,
            False,
            reasons or ["paired metric coverage is incomplete"],
            tuple(source_ids),
            control,
            intervention,
            {},
            None,
            None,
            None,
            None,
            None,
        )

    lower_better = hypothesis.metric_direction == "lower_better"
    paired = paired_comparison(
        intervention,
        control,
        lower_is_better=lower_better,
        n_boot=10_000,
        seed=0,
    )
    if paired is None:
        return _Provisional(
            hypothesis,
            False,
            ["fewer than two matched pairs; paired inference is impossible"],
            tuple(source_ids),
            control,
            intervention,
            {},
            None,
            None,
            None,
            None,
            None,
        )
    sign = -1.0 if lower_better else 1.0
    improvements = {
        seed: sign * (intervention[seed] - control[seed]) for seed in hypothesis.paired_seeds
    }
    improvement = sign * paired.mean_delta
    improvement_ci_low = (
        -paired.delta_ci_high if lower_better else paired.delta_ci_low
    )
    improvement_ci_high = (
        -paired.delta_ci_low if lower_better else paired.delta_ci_high
    )
    scale = 1.0
    if hypothesis.effect_scale == "relative":
        control_mean = aggregate(list(control.values())).mean
        if abs(control_mean) <= 1.0e-12:
            return _Provisional(
                hypothesis,
                False,
                ["relative effect is undefined because the control mean is zero"],
                tuple(source_ids),
                control,
                intervention,
                improvements,
                None,
                None,
                None,
                None,
                None,
            )
        scale = abs(control_mean)
    effect_value = improvement / scale
    effect_ci_low = improvement_ci_low / scale
    effect_ci_high = improvement_ci_high / scale
    return _Provisional(
        hypothesis,
        True,
        [],
        tuple(source_ids),
        control,
        intervention,
        improvements,
        asdict(paired),
        effect_value,
        effect_ci_low,
        effect_ci_high,
        effect_value / hypothesis.minimum_effect,
    )


def _interaction_evidence(provisionals: Sequence[_Provisional]) -> list[dict[str, Any]]:
    interactions: list[dict[str, Any]] = []
    valid = [item for item in provisionals if item.valid]
    for left_index, left in enumerate(valid):
        for right in valid[left_index + 1 :]:
            seeds = sorted(set(left.improvements_by_seed) & set(right.improvements_by_seed))
            pearson_r: float | None = None
            status = "not_estimable"
            if len(seeds) >= 3:
                left_values = np.asarray(
                    [left.improvements_by_seed[seed] for seed in seeds], dtype=np.float64
                )
                right_values = np.asarray(
                    [right.improvements_by_seed[seed] for seed in seeds], dtype=np.float64
                )
                if left_values.std() > 0.0 and right_values.std() > 0.0:
                    pearson_r = float(np.corrcoef(left_values, right_values)[0, 1])
                    if pearson_r >= 0.8:
                        status = "redundancy_candidate"
                    elif pearson_r <= -0.8:
                        status = "complementarity_candidate"
                    else:
                        status = "weak_or_unclear_association"
                else:
                    status = "not_estimable_constant_deltas"
            interactions.append(
                {
                    "mechanisms": [left.hypothesis.mechanism, right.hypothesis.mechanism],
                    "hypothesis_ids": [
                        left.hypothesis.hypothesis_id,
                        right.hypothesis.hypothesis_id,
                    ],
                    "matched_seeds": seeds,
                    "pearson_improvement_correlation": pearson_r,
                    "association_status": status,
                    "causal_interaction_estimable": False,
                    "causal_interaction_reason": (
                        "No factorial joint-intervention cell was preregistered; correlation can "
                        "flag redundancy/saturation candidates but cannot identify interaction."
                    ),
                }
            )
    return interactions


def interpret_experiment_batch(
    hypotheses: Sequence[PreregisteredHypothesis],
    records: Sequence[RunRecord],
    *,
    execution_batch_id: str,
    batch_ledger_path: str | Path,
    source_registry_path: str | Path,
    source_preregistry_path: str | Path,
    interpreted_at: str | None = None,
) -> KnowledgeBatch:
    """Interpret one complete pre-registered multiplicity family without changing evidence."""
    report = _validate_batch_ledger(batch_ledger_path, execution_batch_id)
    cells = _parse_audit_cells(records, execution_batch_id)
    audit_hypothesis_ids = {cell.hypothesis_id for cell in cells if cell.hypothesis_id is not None}
    by_id = {item.hypothesis_id: item for item in hypotheses}
    unknown = audit_hypothesis_ids - set(by_id)
    if unknown:
        raise ValueError(f"execution batch references unknown hypotheses: {sorted(unknown)}")
    selected = [item for item in hypotheses if item.hypothesis_id in audit_hypothesis_ids]
    if not selected:
        raise ValueError("execution batch contains no hypothesis cells")
    preregistration_ids = {item.batch_id for item in selected}
    if len(preregistration_ids) != 1:
        raise ValueError("execution batch mixes preregistration batches")
    preregistration_batch_id = next(iter(preregistration_ids))
    complete_family = {
        item.hypothesis_id for item in hypotheses if item.batch_id == preregistration_batch_id
    }
    if audit_hypothesis_ids != complete_family:
        raise ValueError(
            "execution batch does not contain the complete preregistered Holm family: "
            f"missing={sorted(complete_family - audit_hypothesis_ids)!r}, "
            f"extra={sorted(audit_hypothesis_ids - complete_family)!r}"
        )
    if int(report.get("spent_runs", -1)) != len(cells):
        raise ValueError("batch ledger spent_runs does not match registry audit records")

    provisionals = [_build_provisional(item, cells) for item in selected]
    raw_t = {
        item.hypothesis.hypothesis_id: (
            float(item.stats["t_p_value"]) if item.valid and item.stats else 1.0
        )
        for item in provisionals
    }
    raw_w = {
        item.hypothesis.hypothesis_id: (
            float(item.stats["wilcoxon_p_value"]) if item.valid and item.stats else 1.0
        )
        for item in provisionals
    }
    adjusted_t = holm_adjust(raw_t)
    adjusted_w = holm_adjust(raw_w)
    interactions = _interaction_evidence(provisionals)

    entries: list[dict[str, Any]] = []
    verdicts: list[InterpretationVerdict] = []
    for item in provisionals:
        hypothesis = item.hypothesis
        if not item.valid:
            verdict: InterpretationVerdict = "invalidated"
            conclusion = "; ".join(item.invalidation_reasons)
        else:
            if (
                item.effect_value is None
                or item.effect_ci_low is None
                or item.effect_ci_high is None
            ):
                raise RuntimeError("valid provisional result is missing effect estimates")
            tests_pass = (
                adjusted_t[hypothesis.hypothesis_id] <= 0.05
                and adjusted_w[hypothesis.hypothesis_id] <= 0.05
            )
            if (
                tests_pass
                and item.effect_ci_low > 0.0
                and item.effect_value >= hypothesis.minimum_effect
            ):
                verdict = "confirmed"
                conclusion = (
                    "The preregistered directional CI, Holm-adjusted paired tests, and minimum "
                    "effect threshold all passed."
                )
            elif tests_pass and item.effect_ci_high < 0.0:
                verdict = "refuted"
                conclusion = "The paired evidence supports an effect opposite to the prediction."
            else:
                verdict = "null"
                conclusion = (
                    "The complete preregistered batch did not satisfy the positive or adverse "
                    "support rule; this is not evidence of equivalence."
                )
        verdicts.append(verdict)
        interaction_refs = [
            index
            for index, evidence in enumerate(interactions)
            if hypothesis.hypothesis_id in evidence["hypothesis_ids"]
        ]
        entry = {
            "knowledge_id": _canonical_id(
                "finding",
                {
                    "execution_batch_id": execution_batch_id,
                    "hypothesis_id": hypothesis.hypothesis_id,
                },
            ),
            "hypothesis_id": hypothesis.hypothesis_id,
            "mechanism": hypothesis.mechanism,
            "mechanism_field": hypothesis.mechanism_field,
            "statement": hypothesis.statement,
            "primary_metric": hypothesis.primary_metric,
            "metric_direction": hypothesis.metric_direction,
            "minimum_effect": hypothesis.minimum_effect,
            "effect_scale": hypothesis.effect_scale,
            "verdict": verdict,
            "conclusion": conclusion,
            "complete_preregistered_pairs": item.valid,
            "expected_seeds": list(hypothesis.paired_seeds),
            "observed_control_seeds": sorted(item.control_by_seed),
            "observed_intervention_seeds": sorted(item.intervention_by_seed),
            "control_aggregate": (
                asdict(aggregate(list(item.control_by_seed.values())))
                if item.control_by_seed
                else None
            ),
            "intervention_aggregate": (
                asdict(aggregate(list(item.intervention_by_seed.values())))
                if item.intervention_by_seed
                else None
            ),
            "paired_statistics": item.stats,
            "improvement": item.effect_value,
            "improvement_ci_low": item.effect_ci_low,
            "improvement_ci_high": item.effect_ci_high,
            "effect_threshold_ratio": item.effect_threshold_ratio,
            "paired_t_p_adjusted_holm": adjusted_t[hypothesis.hypothesis_id],
            "wilcoxon_p_adjusted_holm": adjusted_w[hypothesis.hypothesis_id],
            "multiplicity_family_size": len(provisionals),
            "source_run_ids": list(item.source_run_ids),
            "invalidation_reasons": item.invalidation_reasons,
            "interaction_evidence_indices": interaction_refs,
            "saturation_status": "not_estimable_without_factorial_joint_intervention",
        }
        entries.append(entry)

    order = sorted(
        range(len(entries)),
        key=lambda index: (
            entries[index]["verdict"] != "invalidated",
            entries[index]["effect_threshold_ratio"]
            if entries[index]["effect_threshold_ratio"] is not None
            else float("-inf"),
            entries[index]["mechanism"],
        ),
        reverse=True,
    )
    ranking = tuple(str(entries[index]["hypothesis_id"]) for index in order)
    rank_by_hypothesis = {hypothesis_id: rank for rank, hypothesis_id in enumerate(ranking, 1)}
    for entry in entries:
        entry["contribution_rank"] = rank_by_hypothesis[entry["hypothesis_id"]]

    identity = {
        "execution_batch_id": execution_batch_id,
        "preregistration_batch_id": preregistration_batch_id,
        "source_registry_sha256": results_snapshot_digest(source_registry_path),
        "source_preregistry_sha256": results_snapshot_digest(source_preregistry_path),
    }
    return KnowledgeBatch(
        schema_version=1,
        knowledge_batch_id=_canonical_id("knowledge", identity),
        execution_batch_id=execution_batch_id,
        preregistration_batch_id=preregistration_batch_id,
        interpreted_at=interpreted_at or datetime.now(UTC).isoformat(),
        git_sha=_git_sha(),
        source_registry_sha256=identity["source_registry_sha256"],
        source_preregistry_sha256=identity["source_preregistry_sha256"],
        status="interpreted",
        ranking=ranking,
        status_counts=dict(Counter(verdicts)),
        entries=tuple(entries),
        interactions=tuple(interactions),
    )


def append_interpretation(
    batch: KnowledgeBatch,
    *,
    knowledge_base_path: str | Path,
    registry_path: str | Path,
) -> None:
    """Append one atomic knowledge row and one non-ranking registry decision per hypothesis."""
    records = read_records(str(registry_path))
    interpretation_ids = {
        f"interpret-{batch.execution_batch_id}-{entry['hypothesis_id']}"
        for entry in batch.entries
    }
    duplicates = interpretation_ids & {record.run_id for record in records}
    if duplicates:
        raise ValueError(f"interpretation records already exist: {sorted(duplicates)}")
    _append_knowledge(batch, knowledge_base_path)
    for entry in batch.entries:
        verdict = str(entry["verdict"])
        registry_verdict = (
            "positive"
            if verdict == "confirmed"
            else "invalidated"
            if verdict == "invalidated"
            else "null"
        )
        append_record(
            make_record(
                "eval",
                {},
                run_id=f"interpret-{batch.execution_batch_id}-{entry['hypothesis_id']}",
                config={
                    "knowledge_batch_id": batch.knowledge_batch_id,
                    "execution_batch_id": batch.execution_batch_id,
                    "hypothesis_id": entry["hypothesis_id"],
                },
                notes=json.dumps(
                    {
                        "knowledge_batch_id": batch.knowledge_batch_id,
                        "execution_batch_id": batch.execution_batch_id,
                        "hypothesis_id": entry["hypothesis_id"],
                        "mechanism": entry["mechanism"],
                        "interpretation_verdict": verdict,
                        "contribution_rank": entry["contribution_rank"],
                        "improvement": entry["improvement"],
                        "improvement_ci": [
                            entry["improvement_ci_low"],
                            entry["improvement_ci_high"],
                        ],
                    },
                    sort_keys=True,
                ),
                verdict=registry_verdict,
                eligible_for_best=False,
            ),
            str(registry_path),
        )


def query_knowledge(
    batches: Sequence[KnowledgeBatch],
    *,
    mechanism: str | None = None,
    verdict: str | None = None,
) -> list[dict[str, Any]]:
    """Return newest-first mechanism findings with optional exact filters."""
    rows: list[dict[str, Any]] = []
    for batch in reversed(batches):
        for entry in batch.entries:
            if mechanism is not None and entry["mechanism"] != mechanism:
                continue
            if verdict is not None and entry["verdict"] != verdict:
                continue
            rows.append(
                {
                    **entry,
                    "knowledge_batch_id": batch.knowledge_batch_id,
                    "execution_batch_id": batch.execution_batch_id,
                    "interpreted_at": batch.interpreted_at,
                }
            )
    return rows


def _render(rows: Sequence[Mapping[str, Any]]) -> None:
    table = Table(title="AI-neuroscientist mechanism knowledge")
    table.add_column("Rank", justify="right")
    table.add_column("Mechanism")
    table.add_column("Metric")
    table.add_column("Verdict")
    table.add_column("Improvement", justify="right")
    table.add_column("Execution batch")
    for row in rows:
        improvement = row.get("improvement")
        table.add_row(
            str(row.get("contribution_rank", "—")),
            str(row["mechanism"]),
            str(row["primary_metric"]),
            str(row["verdict"]),
            f"{improvement:+.6g}" if isinstance(improvement, (float, int)) else "—",
            str(row["execution_batch_id"]),
        )
    Console().print(table)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interpret and query AI-neuroscientist results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    interpret = subparsers.add_parser("interpret")
    interpret.add_argument("--execution-batch-id", required=True)
    interpret.add_argument("--preregistry", default=DEFAULT_PREREGISTRY)
    interpret.add_argument("--registry-path", default=DEFAULT_REGISTRY)
    interpret.add_argument("--batch-ledger", required=True)
    interpret.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    interpret.add_argument("--dry-run", action="store_true")

    query = subparsers.add_parser("query")
    query.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    query.add_argument("--mechanism")
    query.add_argument(
        "--verdict", choices=["confirmed", "refuted", "null", "invalidated"]
    )
    args = parser.parse_args(argv)

    if args.command == "query":
        rows = query_knowledge(
            read_knowledge_base(args.knowledge_base),
            mechanism=args.mechanism,
            verdict=args.verdict,
        )
        _render(rows)
        return 0

    hypotheses = read_preregistrations(args.preregistry)
    records = read_records(args.registry_path)
    batch = interpret_experiment_batch(
        hypotheses,
        records,
        execution_batch_id=args.execution_batch_id,
        batch_ledger_path=args.batch_ledger,
        source_registry_path=args.registry_path,
        source_preregistry_path=args.preregistry,
    )
    _render(
        [
            {
                **entry,
                "execution_batch_id": batch.execution_batch_id,
            }
            for entry in batch.entries
        ]
    )
    if args.dry_run:
        Console().print("[yellow]Dry run:[/yellow] knowledge base and registry unchanged")
        return 0
    append_interpretation(
        batch,
        knowledge_base_path=args.knowledge_base,
        registry_path=args.registry_path,
    )
    Console().print(
        f"[green]Appended knowledge batch:[/green] {batch.knowledge_batch_id}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(_main())
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        Console().print(f"[bold red]Interpretation aborted:[/bold red] {exc}")
        raise SystemExit(2) from exc
