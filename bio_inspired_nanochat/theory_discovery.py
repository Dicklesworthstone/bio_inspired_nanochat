"""Human-reviewed conjecture discovery with bounded Lean proof feedback.

This module bridges the evidence-linked proposals emitted by :mod:`ai_neuroscientist`
to the formal feedback loop.  It deliberately keeps proposal, review, and proof attempt
as separate immutable records: generated conjectures cannot approve or execute themselves,
and a failed Lean compilation is recorded as ``unresolved`` rather than misreported as a
mathematical refutation.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.ai_neuroscientist import (
    DEFAULT_PROPOSAL_REGISTRY,
    MechanismProposal,
    read_proposals,
)
from bio_inspired_nanochat.checkpoint_manager import _git_sha
from bio_inspired_nanochat.run_logging import RunLogger

DEFAULT_OUTPUT_ROOT = "runs/theory_discovery"
DEFAULT_LEAN_COMMAND = ("rch", "exec", "--", "lake", "env", "lean", "--stdin")
ReviewDecision = Literal["approved", "rejected"]
ConjectureVerdict = Literal["proved", "refuted", "unresolved", "rejected"]
AttemptRoute = Literal["proof", "refutation"]

_BOUNDED_METRIC_TEMPLATE = "bounded_metric_perturbation"
_UNIFORM_IMPROVEMENT_TEMPLATE = "uniform_strict_improvement"


def _strict_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return value


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _strict_json(dict(value)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _canonical_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{_digest(payload)[:20]}"


def _timestamp(value: str, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")


@dataclass(frozen=True)
class FormalConjecture:
    """Evidence-linked conjecture awaiting an independent review decision."""

    schema_version: int
    conjecture_id: str
    proposed_at: str
    git_sha: str | None
    source_proposal_id: str
    source_proposal_sha256: str
    source_mechanism: str
    source_verdict: str
    registered_metric: str
    registered_direction: str
    registered_tolerance: float
    title: str
    mathematical_family: str
    template_id: str
    statement: str
    theorem_id: str
    assumptions: tuple[str, ...]
    attempt_route: AttemptRoute
    review_status: Literal["pending_human_review"] = "pending_human_review"
    automatic_execution_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.conjecture_id.startswith("conjecture-"):
            raise ValueError("unsupported conjecture schema or ID")
        _timestamp(self.proposed_at, "proposed_at")
        if not self.source_proposal_id.startswith("proposal-"):
            raise ValueError("conjecture must reference an AI-neuroscientist proposal")
        try:
            source_digest = bytes.fromhex(self.source_proposal_sha256)
        except ValueError as exc:
            raise ValueError("source proposal digest must be hexadecimal") from exc
        if len(source_digest) != 32:
            raise ValueError("source proposal digest must be a full SHA-256 digest")
        if (
            not math.isfinite(self.registered_tolerance)
            or self.registered_tolerance <= 0
        ):
            raise ValueError("registered tolerance must be finite and positive")
        if self.registered_direction not in {"higher_better", "lower_better"}:
            raise ValueError(
                "registered direction must be higher_better or lower_better"
            )
        if self.template_id not in {
            _BOUNDED_METRIC_TEMPLATE,
            _UNIFORM_IMPROVEMENT_TEMPLATE,
        }:
            raise ValueError("unknown conjecture template")
        if not self.theorem_id.startswith("BioInspiredNanochat.TheoryDiscovery."):
            raise ValueError(
                "theorem must live in the audited theory-discovery namespace"
            )
        if not self.assumptions or any(not item.strip() for item in self.assumptions):
            raise ValueError("conjecture assumptions must be explicit and non-empty")
        if (
            self.review_status != "pending_human_review"
            or self.automatic_execution_allowed
        ):
            raise ValueError("a conjecture cannot approve or execute itself")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> FormalConjecture:
        values = dict(payload)
        values["assumptions"] = tuple(values["assumptions"])
        return cls(**values)


@dataclass(frozen=True)
class ConjectureReview:
    """Explicit review record kept separate from the generated conjecture."""

    schema_version: int
    review_id: str
    conjecture_id: str
    proposal_sha256: str
    decision: ReviewDecision
    reviewer: str
    reviewed_at: str
    notes: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.review_id.startswith("review-"):
            raise ValueError("unsupported conjecture review schema or ID")
        if not self.conjecture_id.startswith("conjecture-"):
            raise ValueError("review must reference a conjecture")
        if len(self.proposal_sha256) != 64:
            raise ValueError("review proposal digest must be a full SHA-256 digest")
        if self.decision not in {"approved", "rejected"}:
            raise ValueError("review decision must be approved or rejected")
        if not self.reviewer.strip() or not self.notes.strip():
            raise ValueError("reviewer and review notes must be non-empty")
        _timestamp(self.reviewed_at, "reviewed_at")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProofBudget:
    """Hard cap for a single compiler attempt."""

    maximum_attempts: int = 1
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.maximum_attempts != 1:
            raise ValueError(
                "theory discovery currently permits exactly one Lean attempt"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Lean timeout must be finite and positive")


@dataclass(frozen=True)
class CompilerOutcome:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    unavailable: bool = False

    def __post_init__(self) -> None:
        if self.timed_out and self.returncode is not None:
            raise ValueError("a timed-out compiler cannot have a return code")


CompilerRunner = Callable[[tuple[str, ...], Path, str, float], CompilerOutcome]


@dataclass(frozen=True)
class TheoryVerdict:
    """Immutable result of the reviewed, bounded formal-feedback attempt."""

    schema_version: int
    verdict_id: str
    conjecture_id: str
    review_id: str
    attempted_at: str
    verdict: ConjectureVerdict
    attempt_route: AttemptRoute
    theorem_id: str
    lean_source_sha256: str
    command: tuple[str, ...]
    attempts_used: int
    timeout_seconds: float
    compiler_returncode: int | None
    timed_out: bool
    compiler_unavailable: bool
    stdout_sha256: str
    stderr_sha256: str
    stdout_tail: str
    stderr_tail: str
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.verdict_id.startswith("verdict-"):
            raise ValueError("unsupported theory verdict schema or ID")
        _timestamp(self.attempted_at, "attempted_at")
        if self.verdict == "rejected":
            if self.attempts_used != 0 or self.compiler_returncode is not None:
                raise ValueError("rejected conjectures must not invoke Lean")
        elif self.attempts_used != 1:
            raise ValueError("attempted conjectures must consume exactly one attempt")
        if self.verdict in {"proved", "refuted"} and self.compiler_returncode != 0:
            raise ValueError(
                "a proved/refuted verdict requires a successful Lean compile"
            )
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("attempt duration must be finite and non-negative")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def propose_formal_conjecture(
    source: MechanismProposal,
    *,
    proposed_at: str | None = None,
) -> FormalConjecture:
    """Derive a deterministic formal conjecture from an evidence-linked proposal."""

    source_payload = source.to_json()
    source_digest = _digest(source_payload)
    if source.source_verdict == "refuted":
        template_id = _UNIFORM_IMPROVEMENT_TEMPLATE
        title = f"Uniform strict-improvement conjecture for {source.mechanism}"
        statement = (
            "Every budget-respecting intervention strictly improves its baseline metric. "
            "The zero-change intervention is an admissible counterexample."
        )
        theorem_id = (
            "BioInspiredNanochat.TheoryDiscovery.uniformStrictImprovement_refuted"
        )
        route: AttemptRoute = "refutation"
        assumptions = (
            "The metric is real-valued.",
            "A zero-change intervention is inside every non-negative perturbation budget.",
        )
    else:
        template_id = _BOUNDED_METRIC_TEMPLATE
        title = f"Metric-robustness conjecture for {source.mechanism}"
        statement = (
            "If the absolute candidate-minus-baseline metric change is at most epsilon, "
            "then the candidate lies in the closed interval [baseline-epsilon, baseline+epsilon]."
        )
        theorem_id = "BioInspiredNanochat.TheoryDiscovery.boundedMetricPerturbation"
        route = "proof"
        assumptions = (
            "The registered metric is represented by a real scalar.",
            "The perturbation certificate bounds the absolute metric change by epsilon.",
            "The runtime must establish the perturbation premise; Lean certifies only the implication.",
        )
    identity = {
        "source_proposal_id": source.proposal_id,
        "source_proposal_sha256": source_digest,
        "template_id": template_id,
        "theorem_id": theorem_id,
    }
    return FormalConjecture(
        schema_version=1,
        conjecture_id=_canonical_id("conjecture", identity),
        proposed_at=proposed_at or datetime.now(UTC).isoformat(),
        git_sha=_git_sha(),
        source_proposal_id=source.proposal_id,
        source_proposal_sha256=source_digest,
        source_mechanism=source.mechanism,
        source_verdict=source.source_verdict,
        registered_metric=source.primary_metric,
        registered_direction=source.metric_direction,
        registered_tolerance=source.minimum_effect,
        title=title,
        mathematical_family="robust optimization and interval analysis",
        template_id=template_id,
        statement=statement,
        theorem_id=theorem_id,
        assumptions=assumptions,
        attempt_route=route,
    )


def review_conjecture(
    conjecture: FormalConjecture,
    *,
    decision: ReviewDecision,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> ConjectureReview:
    """Record an explicit independent decision without mutating the proposal."""

    timestamp = reviewed_at or datetime.now(UTC).isoformat()
    proposal_digest = _digest(conjecture.to_json())
    identity = {
        "conjecture_id": conjecture.conjecture_id,
        "proposal_sha256": proposal_digest,
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
    }
    return ConjectureReview(
        schema_version=1,
        review_id=_canonical_id("review", identity),
        conjecture_id=conjecture.conjecture_id,
        proposal_sha256=proposal_digest,
        decision=decision,
        reviewer=reviewer,
        reviewed_at=timestamp,
        notes=notes,
    )


def lean_source_for(conjecture: FormalConjecture) -> str:
    """Regenerate the audited Lean source from a closed template identifier."""

    if conjecture.template_id == _BOUNDED_METRIC_TEMPLATE:
        return """import Mathlib

namespace BioInspiredNanochat.TheoryDiscovery

/-- An absolute perturbation certificate supplies both directional metric bounds. -/
theorem boundedMetricPerturbation
    (baseline candidate epsilon : ℝ)
    (hbudget : |candidate - baseline| ≤ epsilon) :
    baseline - epsilon ≤ candidate ∧ candidate ≤ baseline + epsilon := by
  rcases abs_le.mp hbudget with ⟨hlower, hupper⟩
  constructor <;> linarith

#print axioms boundedMetricPerturbation

end BioInspiredNanochat.TheoryDiscovery
"""
    if conjecture.template_id == _UNIFORM_IMPROVEMENT_TEMPLATE:
        return """import Mathlib

namespace BioInspiredNanochat.TheoryDiscovery

/-- Zero change refutes uniform strict improvement under a non-strict budget. -/
theorem uniformStrictImprovement_refuted :
    ¬ (∀ baseline candidate epsilon : ℝ,
        0 ≤ epsilon → |candidate - baseline| ≤ epsilon → baseline < candidate) := by
  intro hypothesis
  have impossible := hypothesis 0 0 0 (by norm_num) (by norm_num)
  linarith

#print axioms uniformStrictImprovement_refuted

end BioInspiredNanochat.TheoryDiscovery
"""
    raise ValueError(f"unsupported conjecture template: {conjecture.template_id}")


def _default_compiler_runner(
    command: tuple[str, ...],
    cwd: Path,
    source: str,
    timeout_seconds: float,
) -> CompilerOutcome:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=source,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return CompilerOutcome(None, stderr=str(exc), unavailable=True)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CompilerOutcome(None, stdout=stdout, stderr=stderr, timed_out=True)
    return CompilerOutcome(
        completed.returncode,
        completed.stdout,
        completed.stderr,
        unavailable=completed.returncode == 127,
    )


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tail(value: str, limit: int = 4000) -> str:
    return value[-limit:]


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_strict_json(dict(payload)), handle, sort_keys=True, indent=2)
        handle.write("\n")


def run_formal_feedback(
    conjecture: FormalConjecture,
    review: ConjectureReview,
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    budget: ProofBudget | None = None,
    command: tuple[str, ...] = DEFAULT_LEAN_COMMAND,
    runner: CompilerRunner = _default_compiler_runner,
    attempted_at: str | None = None,
    console_logging: bool = True,
) -> TheoryVerdict:
    """Record and execute one reviewed attempt, failing closed on every mismatch."""

    root = Path(repo_root).resolve()
    package_root = root / "formal/lean"
    if not package_root.is_dir():
        raise ValueError("repo_root must contain formal/lean")
    destination = Path(run_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination == root:
        raise ValueError("run_dir must not be the repository root")
    if destination.exists():
        raise ValueError(
            f"refusing to overwrite existing theory-discovery run: {destination}"
        )
    if review.conjecture_id != conjecture.conjecture_id:
        raise ValueError("review references a different conjecture")
    if not hmac.compare_digest(review.proposal_sha256, _digest(conjecture.to_json())):
        raise ValueError("review does not match the immutable conjecture payload")
    if not command or any(not part for part in command):
        raise ValueError("Lean command must be non-empty")

    effective_budget = budget if budget is not None else ProofBudget()
    timestamp = attempted_at or datetime.now(UTC).isoformat()
    _timestamp(timestamp, "attempted_at")
    source = lean_source_for(conjecture)
    source_digest = _text_digest(source)
    started = time.monotonic()

    with RunLogger(
        destination,
        name="theory_discovery",
        run_id=_canonical_id(
            "theory", {"review_id": review.review_id, "attempted_at": timestamp}
        ),
        console=console_logging,
        provenance={
            "conjecture_id": conjecture.conjecture_id,
            "review_id": review.review_id,
            "source_proposal_id": conjecture.source_proposal_id,
            "budget": asdict(effective_budget),
        },
    ) as logger:
        _write_json_exclusive(destination / "conjecture.json", conjecture.to_json())
        _write_json_exclusive(destination / "review.json", review.to_json())
        logger.event(
            "conjecture_proposed",
            conjecture=conjecture.to_json(),
            automatic_execution_allowed=False,
        )
        logger.event("human_review_recorded", review=review.to_json())

        if review.decision == "rejected":
            outcome = CompilerOutcome(None)
            verdict_name: ConjectureVerdict = "rejected"
            attempts_used = 0
            logger.event(
                "lean_attempt_skipped",
                reason="human_review_rejected",
                attempt_route=conjecture.attempt_route,
            )
        else:
            logger.event(
                "lean_attempt_started",
                attempt=1,
                attempt_route=conjecture.attempt_route,
                theorem_id=conjecture.theorem_id,
                lean_source_sha256=source_digest,
                timeout_seconds=effective_budget.timeout_seconds,
                command=command,
            )
            outcome = runner(
                command, package_root, source, effective_budget.timeout_seconds
            )
            attempts_used = 1
            if outcome.returncode == 0:
                verdict_name = (
                    "proved" if conjecture.attempt_route == "proof" else "refuted"
                )
            else:
                verdict_name = "unresolved"
            logger.event(
                "lean_attempt_completed",
                attempt=1,
                attempt_route=conjecture.attempt_route,
                compiler_returncode=outcome.returncode,
                timed_out=outcome.timed_out,
                compiler_unavailable=outcome.unavailable,
                stdout_sha256=_text_digest(outcome.stdout),
                stderr_sha256=_text_digest(outcome.stderr),
            )

        duration = time.monotonic() - started
        verdict_identity = {
            "conjecture_id": conjecture.conjecture_id,
            "review_id": review.review_id,
            "attempted_at": timestamp,
            "lean_source_sha256": source_digest,
            "verdict": verdict_name,
        }
        verdict = TheoryVerdict(
            schema_version=1,
            verdict_id=_canonical_id("verdict", verdict_identity),
            conjecture_id=conjecture.conjecture_id,
            review_id=review.review_id,
            attempted_at=timestamp,
            verdict=verdict_name,
            attempt_route=conjecture.attempt_route,
            theorem_id=conjecture.theorem_id,
            lean_source_sha256=source_digest,
            command=command,
            attempts_used=attempts_used,
            timeout_seconds=effective_budget.timeout_seconds,
            compiler_returncode=outcome.returncode,
            timed_out=outcome.timed_out,
            compiler_unavailable=outcome.unavailable,
            stdout_sha256=_text_digest(outcome.stdout),
            stderr_sha256=_text_digest(outcome.stderr),
            stdout_tail=_tail(outcome.stdout),
            stderr_tail=_tail(outcome.stderr),
            duration_seconds=duration,
        )
        _write_json_exclusive(destination / "verdict.json", verdict.to_json())
        logger.event("theory_verdict_recorded", verdict=verdict.to_json())
    return verdict


def _select_source(path: str | Path, proposal_id: str | None) -> MechanismProposal:
    proposals = read_proposals(path)
    if not proposals:
        raise ValueError(f"no AI-neuroscientist proposals found in {path}")
    if proposal_id is None:
        return proposals[-1]
    matches = [
        proposal for proposal in proposals if proposal.proposal_id == proposal_id
    ]
    if len(matches) != 1:
        raise ValueError(f"proposal ID not found exactly once: {proposal_id}")
    return matches[0]


def _render_conjecture(conjecture: FormalConjecture) -> None:
    table = Table(title="Human-reviewed theory-discovery proposal")
    table.add_column("Conjecture")
    table.add_column("Source")
    table.add_column("Route")
    table.add_column("Review")
    table.add_column("Statement")
    table.add_row(
        conjecture.conjecture_id,
        conjecture.source_proposal_id,
        conjecture.attempt_route,
        conjecture.review_status,
        conjecture.statement,
    )
    Console().print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-registry", default=DEFAULT_PROPOSAL_REGISTRY)
    parser.add_argument("--proposal-id")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-notes")
    parser.add_argument("--reject", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = _select_source(args.proposal_registry, args.proposal_id)
    conjecture = propose_formal_conjecture(source)
    _render_conjecture(conjecture)
    if not args.execute:
        Console().print(
            "[yellow]Preview only:[/yellow] pass --execute with --reviewer and "
            "--review-notes after independent review"
        )
        return 0
    if not args.reviewer or not args.review_notes:
        Console(stderr=True).print(
            "[bold red]Execution refused:[/bold red] --reviewer and --review-notes are required"
        )
        return 2
    decision: ReviewDecision = "rejected" if args.reject else "approved"
    review = review_conjecture(
        conjecture,
        decision=decision,
        reviewer=args.reviewer,
        notes=args.review_notes,
    )
    run_id = _canonical_id(
        "run",
        {"review_id": review.review_id, "conjecture_id": conjecture.conjecture_id},
    )
    verdict = run_formal_feedback(
        conjecture,
        review,
        repo_root=args.project_root,
        run_dir=Path(args.output_root) / run_id,
        budget=ProofBudget(timeout_seconds=args.timeout_seconds),
    )
    style = "green" if verdict.verdict in {"proved", "refuted"} else "yellow"
    Console().print(
        f"[{style}]Theory verdict:[/{style}] {verdict.verdict} ({verdict.verdict_id})"
    )
    return 0 if verdict.verdict in {"proved", "refuted", "rejected"} else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        Console(stderr=True).print(
            f"[bold red]Theory discovery aborted:[/bold red] {exc}"
        )
        raise SystemExit(2) from exc
