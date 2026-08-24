"""Closed-loop, human-reviewed AI-neuroscientist research orchestration.

This module composes the append-only stages delivered by beads r00r.2.1-r00r.2.3:

``preregister -> guarded experiment -> statistical interpretation -> knowledge -> proposal``

The orchestrator does not accept arbitrary commands, alter model code, or launch a proposed
search.  A proposal is an evidence-linked, falsifiable handoff with status
``pending_human_review``.  Only the existing guarded runner may execute experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.checkpoint_manager import _git_sha
from bio_inspired_nanochat.experiment_runner import (
    BatchPlan,
    BatchReport,
    EvalMatrixOptions,
    Executor,
    HardBudget,
    build_batch_plan,
    execute_batch_plan,
)
from bio_inspired_nanochat.hypothesis_generator import (
    PreregisteredHypothesis,
    append_preregistration,
    generate_hypotheses,
    read_preregistrations,
    results_snapshot_digest,
)
from bio_inspired_nanochat.result_interpreter import (
    KnowledgeBatch,
    append_interpretation,
    interpret_experiment_batch,
)
from bio_inspired_nanochat.results_registry import read_records
from bio_inspired_nanochat.run_logging import RunLogger

DEFAULT_CYCLE_ROOT = "runs/ai_neuroscientist_cycles"
DEFAULT_PROPOSAL_REGISTRY = "results/mechanism_proposals.jsonl"
ProposalReviewStatus = Literal["pending_human_review", "approved", "rejected"]


def _strict_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return value


def _canonical_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_strict_json(dict(payload)), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(encoded.encode()).hexdigest()[:20]}"


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
                payload = json.JSONDecoder().decode(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row at {source}:{line_number} must be an object")
            rows.append(payload)
    return rows


def _workspace_path(path: str | Path, *, root: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be a child of the project root")
    return resolved


@dataclass(frozen=True)
class MechanismProposal:
    """Evidence-linked proposal that cannot execute until a human approves it."""

    schema_version: int
    proposal_id: str
    proposed_at: str
    git_sha: str | None
    source_knowledge_batch_id: str
    source_knowledge_id: str
    source_hypothesis_id: str
    source_verdict: str
    title: str
    mechanism: str
    rationale: str
    falsifiable_hypothesis: str
    primary_metric: str
    metric_direction: str
    minimum_effect: float
    control: dict[str, Any]
    intervention: dict[str, Any]
    search_handoff: dict[str, Any]
    review_status: ProposalReviewStatus = "pending_human_review"
    automatic_execution_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.proposal_id.startswith("proposal-"):
            raise ValueError("unsupported mechanism proposal schema or ID")
        if self.review_status not in {"pending_human_review", "approved", "rejected"}:
            raise ValueError("invalid proposal review status")
        if self.automatic_execution_allowed:
            raise ValueError("mechanism proposals may never authorize their own execution")
        if not self.source_knowledge_batch_id.startswith("knowledge-"):
            raise ValueError("proposal must reference a knowledge batch")
        if not self.source_knowledge_id.startswith("finding-"):
            raise ValueError("proposal must reference a knowledge finding")
        if not math.isfinite(self.minimum_effect) or self.minimum_effect <= 0.0:
            raise ValueError("proposal minimum_effect must be finite and positive")
        if self.control == self.intervention:
            raise ValueError("proposal control and intervention must differ")
        if self.search_handoff.get("launch_policy") != "human_approval_required":
            raise ValueError("proposal search handoff must require human approval")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> MechanismProposal:
        return cls(**dict(payload))


@dataclass(frozen=True)
class CycleConfig:
    """Explicit configuration for one bounded research cycle."""

    selected_mechanisms: tuple[str, ...]
    paired_seed_count: int
    seed_start: int
    maximum_tokens_per_run: int
    eval_options: EvalMatrixOptions
    budget: HardBudget
    project_root: str | Path
    run_root: str | Path = DEFAULT_CYCLE_ROOT
    registry_path: str | Path = "results/registry.jsonl"
    preregistry_path: str | Path = "results/preregistrations.jsonl"
    batch_ledger_path: str | Path = "results/experiment_batches.jsonl"
    knowledge_base_path: str | Path = "results/mechanism_knowledge.jsonl"
    proposal_registry_path: str | Path = DEFAULT_PROPOSAL_REGISTRY
    console_logging: bool = True

    def __post_init__(self) -> None:
        if not self.selected_mechanisms or len(set(self.selected_mechanisms)) != len(
            self.selected_mechanisms
        ):
            raise ValueError("selected mechanisms must be non-empty and unique")
        if self.paired_seed_count < 2:
            raise ValueError("paired_seed_count must be at least two")
        if self.maximum_tokens_per_run < self.eval_options.tokens_per_run:
            raise ValueError("per-run cap is smaller than requested train+eval tokens")
        required_runs = 2 * self.paired_seed_count * len(self.selected_mechanisms)
        required_tokens = required_runs * self.eval_options.tokens_per_run
        if self.budget.maximum_runs < required_runs:
            raise ValueError(f"hard run cap must allow the complete family ({required_runs} runs)")
        if self.budget.maximum_total_tokens < required_tokens:
            raise ValueError(
                f"hard token cap must allow the complete family ({required_tokens} tokens)"
            )


@dataclass(frozen=True)
class CyclePreview:
    cycle_id: str
    cycle_dir: str
    hypotheses: tuple[PreregisteredHypothesis, ...]
    plan: BatchPlan


@dataclass(frozen=True)
class CycleReport:
    cycle_id: str
    cycle_dir: str
    status: str
    hypothesis_ids: tuple[str, ...]
    execution_batch_id: str
    knowledge_batch_id: str
    proposal_ids: tuple[str, ...]
    experiment_report: BatchReport
    events_path: str


def read_proposals(path: str | Path = DEFAULT_PROPOSAL_REGISTRY) -> list[MechanismProposal]:
    return [MechanismProposal.from_json(row) for row in _read_jsonl(path)]


def append_proposals(
    proposals: Sequence[MechanismProposal],
    path: str | Path = DEFAULT_PROPOSAL_REGISTRY,
) -> None:
    """Append a proposal group only after validating the entire group for duplicates."""
    if not proposals:
        raise ValueError("at least one proposal is required")
    destination = Path(path)
    existing = read_proposals(destination)
    existing_ids = {item.proposal_id for item in existing}
    proposed_ids = [item.proposal_id for item in proposals]
    if len(set(proposed_ids)) != len(proposed_ids):
        raise ValueError("proposal group contains duplicate IDs")
    duplicates = existing_ids & set(proposed_ids)
    if duplicates:
        raise ValueError(f"proposal IDs already exist: {sorted(duplicates)}")
    existing_sources = {
        (item.source_knowledge_batch_id, item.source_knowledge_id) for item in existing
    }
    duplicate_sources = existing_sources & {
        (item.source_knowledge_batch_id, item.source_knowledge_id) for item in proposals
    }
    if duplicate_sources:
        raise ValueError("a proposal already exists for this knowledge finding")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(_strict_json(proposal.to_json()), sort_keys=True) + "\n")


def query_proposals(
    proposals: Sequence[MechanismProposal],
    *,
    mechanism: str | None = None,
    review_status: str | None = None,
) -> list[MechanismProposal]:
    return [
        proposal
        for proposal in reversed(proposals)
        if (mechanism is None or proposal.mechanism == mechanism)
        and (review_status is None or proposal.review_status == review_status)
    ]


def _doc2_search(entry: Mapping[str, Any]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    title = "Recovery-matched Doc2 facilitation search"
    rationale = (
        "A fixed Doc2 gain was not sufficient in the source experiment. Search the coupled "
        "facilitation, vesicle-priming, and recovery regime before adding another mechanism."
    )
    control = {
        "design": "fixed project defaults",
        "doc2_gain": 0.08,
        "validation": "held-out paired seeds at equal tokens and model FLOPs",
    }
    intervention = {
        "design": "multi-seed constrained operating-regime search",
        "parameters": [
            {"name": "doc2_gain", "lower": 0.0, "upper": 0.5},
            {"name": "syt_slow_kd", "lower": 0.2, "upper": 5.0, "log_scale": True},
            {"name": "prime_rate", "lower": 0.005, "upper": 0.3, "log_scale": True},
            {"name": "nsf_recover", "lower": 0.001, "upper": 0.3, "log_scale": True},
        ],
        "fixed_constraints": {
            "mean_compute": "matched",
            "paired_seeds": True,
            "primary_metric": entry["primary_metric"],
        },
    }
    return title, rationale, control, intervention


def _generic_search(
    entry: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    mechanism = str(entry["mechanism"])
    field = str(entry["mechanism_field"])
    title = f"State-conditioned {mechanism} operating-regime search"
    rationale = (
        f"The source verdict for {mechanism} was {entry['verdict']}. Test whether a bounded, "
        "state-conditioned operating regime is more useful than the single fixed setting."
    )
    control = {
        "design": "source preregistered fixed setting",
        "mechanism_field": field,
        "validation": "held-out paired seeds at equal tokens and model FLOPs",
    }
    intervention = {
        "design": "bounded state-conditioned activation",
        "mechanism_field": field,
        "implementation_required": True,
        "constraints": ["default-off", "bounded state", "no extra model FLOPs"],
    }
    return title, rationale, control, intervention


def propose_new_mechanisms(
    knowledge: KnowledgeBatch,
    *,
    proposed_at: str | None = None,
) -> list[MechanismProposal]:
    """Derive one conservative, falsifiable proposal from an interpretable finding."""
    interpretable = [entry for entry in knowledge.entries if entry["verdict"] != "invalidated"]
    if not interpretable:
        raise ValueError("cannot propose a mechanism from wholly invalidated evidence")
    verdict_order = {"refuted": 0, "null": 1, "confirmed": 2}
    entry = min(
        interpretable,
        key=lambda item: (verdict_order[str(item["verdict"])], int(item["contribution_rank"])),
    )
    mechanism = str(entry["mechanism"])
    if mechanism == "doc2":
        title, rationale, control, intervention = _doc2_search(entry)
    else:
        title, rationale, control, intervention = _generic_search(entry)
    minimum_effect = float(entry["minimum_effect"])
    identity = {
        "source_knowledge_batch_id": knowledge.knowledge_batch_id,
        "source_knowledge_id": entry["knowledge_id"],
        "design": intervention,
    }
    proposal = MechanismProposal(
        schema_version=1,
        proposal_id=_canonical_id("proposal", identity),
        proposed_at=proposed_at or datetime.now(UTC).isoformat(),
        git_sha=_git_sha(),
        source_knowledge_batch_id=knowledge.knowledge_batch_id,
        source_knowledge_id=str(entry["knowledge_id"]),
        source_hypothesis_id=str(entry["hypothesis_id"]),
        source_verdict=str(entry["verdict"]),
        title=title,
        mechanism=mechanism,
        rationale=rationale,
        falsifiable_hypothesis=(
            f"At equal tokens and model FLOPs, {title.lower()} will improve "
            f"{entry['primary_metric']} by at least {minimum_effect:g} {entry['effect_scale']} "
            "over the fixed-setting control on a fresh, preregistered paired-seed family."
        ),
        primary_metric=str(entry["primary_metric"]),
        metric_direction=str(entry["metric_direction"]),
        minimum_effect=minimum_effect,
        control=control,
        intervention=intervention,
        search_handoff={
            "destination": "bio_inspired_nanochat-hea",
            "strategy": "multi_fidelity_multi_seed_cma_es",
            "entrypoint": "scripts.tune_bio_params",
            "confirmation_harness": "scripts.eval_matrix",
            "launch_policy": "human_approval_required",
            "required_review": [
                "scientific rationale",
                "parameter bounds",
                "compute budget",
                "held-out confirmation design",
            ],
        },
    )
    return [proposal]


def preview_research_cycle(
    config: CycleConfig,
    *,
    started_at: str | None = None,
) -> CyclePreview:
    """Build a complete cycle plan without writing files or running a harness."""
    root = Path(config.project_root).resolve()
    registry = _workspace_path(config.registry_path, root=root, label="registry_path")
    run_root = _workspace_path(config.run_root, root=root, label="run_root")
    records = read_records(str(registry))
    timestamp = started_at or datetime.now(UTC).isoformat()
    hypotheses = generate_hypotheses(
        records,
        results_digest=results_snapshot_digest(registry),
        selected_mechanisms=config.selected_mechanisms,
        limit=len(config.selected_mechanisms),
        paired_seed_count=config.paired_seed_count,
        seed_start=config.seed_start,
        maximum_tokens_per_run=config.maximum_tokens_per_run,
        registered_at=timestamp,
    )
    identity = {
        "hypothesis_ids": [item.hypothesis_id for item in hypotheses],
        "results_snapshot_sha256": results_snapshot_digest(registry),
        "eval_options": asdict(config.eval_options),
        "budget": asdict(config.budget),
    }
    cycle_id = _canonical_id("cycle", identity)
    cycle_dir = run_root / cycle_id
    plan = build_batch_plan(
        hypotheses,
        budget=config.budget,
        eval_options=config.eval_options,
        registry_path=registry,
        output_root=cycle_dir / "experiments",
        project_root=root,
        created_at=timestamp,
    )
    return CyclePreview(cycle_id, str(cycle_dir), tuple(hypotheses), plan)


def run_research_cycle(
    config: CycleConfig,
    *,
    executor: Executor | None = None,
    started_at: str | None = None,
) -> CycleReport:
    """Run one complete, bounded cycle and stop at a pending human proposal review."""
    root = Path(config.project_root).resolve()
    preregistry = _workspace_path(config.preregistry_path, root=root, label="preregistry_path")
    registry = _workspace_path(config.registry_path, root=root, label="registry_path")
    ledger = _workspace_path(config.batch_ledger_path, root=root, label="batch_ledger_path")
    knowledge_base = _workspace_path(
        config.knowledge_base_path, root=root, label="knowledge_base_path"
    )
    proposal_registry = _workspace_path(
        config.proposal_registry_path, root=root, label="proposal_registry_path"
    )
    preview = preview_research_cycle(config, started_at=started_at)
    cycle_dir = Path(preview.cycle_dir)
    if cycle_dir.exists():
        raise ValueError(f"refusing to overwrite existing research cycle: {cycle_dir}")
    existing_hypothesis_ids = {
        item.hypothesis_id for item in read_preregistrations(preregistry)
    }
    duplicate_hypotheses = existing_hypothesis_ids & {
        item.hypothesis_id for item in preview.hypotheses
    }
    if duplicate_hypotheses:
        raise ValueError(f"hypotheses already preregistered: {sorted(duplicate_hypotheses)}")

    with RunLogger(
        cycle_dir,
        name="ai_neuroscientist",
        run_id=preview.cycle_id,
        console=config.console_logging,
        provenance={
            "cycle_id": preview.cycle_id,
            "selected_mechanisms": config.selected_mechanisms,
            "budget": asdict(config.budget),
            "eval_options": asdict(config.eval_options),
        },
    ) as logger:
        logger.event("cycle_stage", step=1, stage="preregister", state="started")
        for hypothesis in preview.hypotheses:
            append_preregistration(hypothesis, preregistry)
        logger.event(
            "cycle_stage",
            step=1,
            stage="preregister",
            state="completed",
            hypothesis_ids=[item.hypothesis_id for item in preview.hypotheses],
            preregistration_batch_id=preview.hypotheses[0].batch_id,
            evidence_snapshot_sha256=preview.plan.results_snapshot_sha256,
        )

        logger.event(
            "cycle_stage",
            step=2,
            stage="execute",
            state="started",
            execution_batch_id=preview.plan.execution_batch_id,
            projected_runs=preview.plan.projected_runs,
            projected_tokens=preview.plan.projected_tokens,
        )
        execute_kwargs: dict[str, Any] = {
            "registry_path": registry,
            "batch_ledger_path": ledger,
            "project_root": root,
        }
        if executor is not None:
            execute_kwargs["executor"] = executor
        experiment_report = execute_batch_plan(preview.plan, **execute_kwargs)
        logger.event(
            "cycle_stage",
            step=2,
            stage="execute",
            state="completed",
            status=experiment_report.status,
            spent_runs=experiment_report.spent_runs,
            spent_tokens=experiment_report.spent_tokens,
            receipts=[asdict(item) for item in experiment_report.receipts],
        )

        logger.event("cycle_stage", step=3, stage="interpret", state="started")
        knowledge = interpret_experiment_batch(
            preview.hypotheses,
            read_records(str(registry)),
            execution_batch_id=preview.plan.execution_batch_id,
            batch_ledger_path=ledger,
            source_registry_path=registry,
            source_preregistry_path=preregistry,
        )
        append_interpretation(
            knowledge,
            knowledge_base_path=knowledge_base,
            registry_path=registry,
        )
        logger.event(
            "cycle_stage",
            step=3,
            stage="interpret",
            state="completed",
            knowledge_batch_id=knowledge.knowledge_batch_id,
            status_counts=knowledge.status_counts,
            ranking=knowledge.ranking,
        )

        logger.event("cycle_stage", step=4, stage="propose", state="started")
        proposals = propose_new_mechanisms(knowledge)
        append_proposals(proposals, proposal_registry)
        logger.event(
            "cycle_stage",
            step=4,
            stage="propose",
            state="completed",
            proposals=[proposal.to_json() for proposal in proposals],
            automatic_execution_allowed=False,
            next_action="human_review",
        )
        report = CycleReport(
            cycle_id=preview.cycle_id,
            cycle_dir=str(cycle_dir),
            status="completed_pending_human_review",
            hypothesis_ids=tuple(item.hypothesis_id for item in preview.hypotheses),
            execution_batch_id=preview.plan.execution_batch_id,
            knowledge_batch_id=knowledge.knowledge_batch_id,
            proposal_ids=tuple(item.proposal_id for item in proposals),
            experiment_report=experiment_report,
            events_path=str(logger.events_path),
        )
        logger.event("research_cycle_completed", report=asdict(report))
        return report


def _render_preview(preview: CyclePreview) -> None:
    table = Table(title="AI-neuroscientist cycle preview")
    table.add_column("Cycle")
    table.add_column("Hypotheses", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Output")
    table.add_row(
        preview.cycle_id,
        str(len(preview.hypotheses)),
        str(preview.plan.projected_runs),
        str(preview.plan.projected_tokens),
        preview.cycle_dir,
    )
    Console().print(table)


def _render_proposals(proposals: Sequence[MechanismProposal]) -> None:
    table = Table(title="Human-reviewed mechanism proposals")
    table.add_column("Proposal")
    table.add_column("Mechanism")
    table.add_column("Source verdict")
    table.add_column("Review status")
    table.add_column("Title")
    for proposal in proposals:
        table.add_row(
            proposal.proposal_id,
            proposal.mechanism,
            proposal.source_verdict,
            proposal.review_status,
            proposal.title,
        )
    Console().print(table)


def _cycle_config_from_args(args: argparse.Namespace) -> CycleConfig:
    options = EvalMatrixOptions(
        train_tokens=args.train_tokens,
        eval_tokens=args.eval_tokens,
        checkpoint_dir=args.checkpoint_dir,
        inline_smoke_training=args.inline_smoke_training,
        device_type=args.device_type,
        data=args.data,
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device_batch_size=args.device_batch_size,
        total_batch_size_tokens=args.total_batch_size_tokens,
    )
    return CycleConfig(
        selected_mechanisms=tuple(args.mechanism),
        paired_seed_count=args.paired_seed_count,
        seed_start=args.seed_start,
        maximum_tokens_per_run=args.maximum_tokens_per_run,
        eval_options=options,
        budget=HardBudget(args.maximum_runs, args.maximum_total_tokens, args.maximum_wall_seconds),
        project_root=args.project_root,
        run_root=args.run_root,
        registry_path=args.registry_path,
        preregistry_path=args.preregistry,
        batch_ledger_path=args.batch_ledger,
        knowledge_base_path=args.knowledge_base,
        proposal_registry_path=args.proposal_registry,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded AI-neuroscientist flywheel")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mechanism", action="append", required=True)
    run.add_argument("--paired-seed-count", type=int, required=True)
    run.add_argument("--seed-start", type=int, required=True)
    run.add_argument("--train-tokens", type=int, required=True)
    run.add_argument("--eval-tokens", type=int, required=True)
    run.add_argument("--maximum-tokens-per-run", type=int, required=True)
    run.add_argument("--maximum-runs", type=int, required=True)
    run.add_argument("--maximum-total-tokens", type=int, required=True)
    run.add_argument("--maximum-wall-seconds", type=float, required=True)
    run.add_argument("--checkpoint-dir", default="")
    run.add_argument("--inline-smoke-training", action="store_true")
    run.add_argument("--device-type", choices=["cpu", "cuda", "mps"], default="cpu")
    run.add_argument("--data", choices=["synthetic", "fineweb"], default="synthetic")
    run.add_argument("--sequence-len", type=int, default=32)
    run.add_argument("--vocab-size", type=int, default=64)
    run.add_argument("--n-layer", type=int, default=1)
    run.add_argument("--n-head", type=int, default=2)
    run.add_argument("--n-embd", type=int, default=32)
    run.add_argument("--device-batch-size", type=int, default=1)
    run.add_argument("--total-batch-size-tokens", type=int, default=32)
    run.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    run.add_argument("--run-root", default=DEFAULT_CYCLE_ROOT)
    run.add_argument("--registry-path", default="results/registry.jsonl")
    run.add_argument("--preregistry", default="results/preregistrations.jsonl")
    run.add_argument("--batch-ledger", default="results/experiment_batches.jsonl")
    run.add_argument("--knowledge-base", default="results/mechanism_knowledge.jsonl")
    run.add_argument("--proposal-registry", default=DEFAULT_PROPOSAL_REGISTRY)
    run.add_argument("--execute", action="store_true")

    query = subparsers.add_parser("query")
    query.add_argument("--proposal-registry", default=DEFAULT_PROPOSAL_REGISTRY)
    query.add_argument("--mechanism")
    query.add_argument(
        "--review-status", choices=["pending_human_review", "approved", "rejected"]
    )
    args = parser.parse_args(argv)

    if args.command == "query":
        proposals = query_proposals(
            read_proposals(args.proposal_registry),
            mechanism=args.mechanism,
            review_status=args.review_status,
        )
        _render_proposals(proposals)
        return 0

    config = _cycle_config_from_args(args)
    if not args.execute:
        preview = preview_research_cycle(config)
        _render_preview(preview)
        Console().print("[yellow]Preview only:[/yellow] pass --execute to authorize the cycle")
        return 0
    report = run_research_cycle(config)
    Console().print(
        f"[green]Cycle completed:[/green] {report.cycle_id}; "
        f"proposal(s) awaiting human review: {', '.join(report.proposal_ids)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(_main())
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        Console().print(f"[bold red]AI-neuroscientist aborted:[/bold red] {exc}")
        raise SystemExit(2) from exc
