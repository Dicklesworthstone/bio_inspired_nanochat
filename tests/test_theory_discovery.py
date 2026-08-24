"""Conjecture-to-Lean feedback tests for bead re4e.6."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.ai_neuroscientist import MechanismProposal
from bio_inspired_nanochat.run_logging import read_run_events
from bio_inspired_nanochat.theory_discovery import (
    CompilerOutcome,
    ProofBudget,
    lean_source_for,
    propose_formal_conjecture,
    review_conjecture,
    run_formal_feedback,
)


def _mechanism_proposal(*, source_verdict: str = "null") -> MechanismProposal:
    return MechanismProposal(
        schema_version=1,
        proposal_id="proposal-0123456789abcdefabcd",
        proposed_at="2026-08-24T00:00:00+00:00",
        git_sha="a" * 40,
        source_knowledge_batch_id="knowledge-0123456789abcdefabcd",
        source_knowledge_id="finding-0123456789abcdefabcd",
        source_hypothesis_id="hyp-0123456789abcdefabcd",
        source_verdict=source_verdict,
        title="State-conditioned Doc2 operating regime",
        mechanism="doc2",
        rationale="Test a bounded operating regime.",
        falsifiable_hypothesis="The intervention improves held-out retrieval.",
        primary_metric="niah_accuracy",
        metric_direction="higher_better",
        minimum_effect=0.02,
        control={"doc2_gain": 0.08},
        intervention={"doc2_gain": [0.0, 0.5]},
        search_handoff={"launch_policy": "human_approval_required"},
    )


def _review(conjecture, *, decision="approved"):
    return review_conjecture(
        conjecture,
        decision=decision,
        reviewer="Ada Lovelace",
        notes="The statement and assumptions are narrow enough for a bounded formal attempt.",
        reviewed_at="2026-08-24T01:00:00+00:00",
    )


@pytest.mark.unit
def test_proposal_is_deterministic_formalizable_and_cannot_self_execute():
    source = _mechanism_proposal()
    first = propose_formal_conjecture(source, proposed_at="2026-08-24T00:30:00+00:00")
    second = propose_formal_conjecture(source, proposed_at="2026-08-24T00:31:00+00:00")

    assert first.conjecture_id == second.conjecture_id
    assert first.source_proposal_id == source.proposal_id
    assert first.review_status == "pending_human_review"
    assert not first.automatic_execution_allowed
    assert first.attempt_route == "proof"
    assert "runtime must establish" in " ".join(first.assumptions).lower()
    lean = lean_source_for(first)
    assert "theorem boundedMetricPerturbation" in lean
    assert "#print axioms boundedMetricPerturbation" in lean


@pytest.mark.unit
def test_review_mismatch_fails_before_compiler_or_output(tmp_path: Path):
    conjecture = propose_formal_conjecture(
        _mechanism_proposal(), proposed_at="2026-08-24T00:30:00+00:00"
    )
    review = replace(_review(conjecture), proposal_sha256="0" * 64)
    calls = 0

    def forbidden_runner(*_args):
        nonlocal calls
        calls += 1
        return CompilerOutcome(0)

    with pytest.raises(ValueError, match="immutable conjecture"):
        run_formal_feedback(
            conjecture,
            review,
            repo_root=Path(__file__).resolve().parents[1],
            run_dir=tmp_path / "attempt",
            runner=forbidden_runner,
            console_logging=False,
        )
    assert calls == 0
    assert not (tmp_path / "attempt").exists()


@pytest.mark.unit
def test_rejected_review_records_no_attempt(tmp_path: Path):
    conjecture = propose_formal_conjecture(
        _mechanism_proposal(), proposed_at="2026-08-24T00:30:00+00:00"
    )
    calls = 0

    def forbidden_runner(*_args):
        nonlocal calls
        calls += 1
        return CompilerOutcome(0)

    verdict = run_formal_feedback(
        conjecture,
        _review(conjecture, decision="rejected"),
        repo_root=Path(__file__).resolve().parents[1],
        run_dir=tmp_path / "rejected",
        runner=forbidden_runner,
        attempted_at="2026-08-24T02:00:00+00:00",
        console_logging=False,
    )

    assert verdict.verdict == "rejected"
    assert verdict.attempts_used == 0
    assert calls == 0
    events = read_run_events(tmp_path / "rejected")
    assert any(event["event"] == "lean_attempt_skipped" for event in events)


@pytest.mark.unit
def test_compile_failure_is_unresolved_not_refuted(tmp_path: Path):
    conjecture = propose_formal_conjecture(
        _mechanism_proposal(), proposed_at="2026-08-24T00:30:00+00:00"
    )

    verdict = run_formal_feedback(
        conjecture,
        _review(conjecture),
        repo_root=Path(__file__).resolve().parents[1],
        run_dir=tmp_path / "unresolved",
        runner=lambda *_args: CompilerOutcome(1, stderr="unsolved goals"),
        attempted_at="2026-08-24T02:00:00+00:00",
        console_logging=False,
    )

    assert verdict.verdict == "unresolved"
    assert verdict.compiler_returncode == 1
    assert verdict.stderr_tail == "unsolved goals"


@pytest.mark.unit
def test_refuted_source_generates_machine_checked_counterexample_route(tmp_path: Path):
    conjecture = propose_formal_conjecture(
        _mechanism_proposal(source_verdict="refuted"),
        proposed_at="2026-08-24T00:30:00+00:00",
    )
    observed_source = ""

    def successful_runner(_command, _cwd, source, _timeout):
        nonlocal observed_source
        observed_source = source
        return CompilerOutcome(0, stdout="axioms: []")

    verdict = run_formal_feedback(
        conjecture,
        _review(conjecture),
        repo_root=Path(__file__).resolve().parents[1],
        run_dir=tmp_path / "refuted",
        runner=successful_runner,
        attempted_at="2026-08-24T02:00:00+00:00",
        console_logging=False,
    )

    assert conjecture.attempt_route == "refutation"
    assert verdict.verdict == "refuted"
    assert "hypothesis 0 0 0" in observed_source


@pytest.mark.unit
def test_budget_and_existing_run_fail_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        ProofBudget(maximum_attempts=2)
    with pytest.raises(ValueError, match="positive"):
        ProofBudget(timeout_seconds=0)

    conjecture = propose_formal_conjecture(
        _mechanism_proposal(), proposed_at="2026-08-24T00:30:00+00:00"
    )
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        run_formal_feedback(
            conjecture,
            _review(conjecture),
            repo_root=Path(__file__).resolve().parents[1],
            run_dir=run_dir,
            runner=lambda *_args: CompilerOutcome(0),
            console_logging=False,
        )


@pytest.mark.e2e
def test_approved_conjecture_attempt_records_verdict_and_structured_log(tmp_path: Path):
    conjecture = propose_formal_conjecture(
        _mechanism_proposal(), proposed_at="2026-08-24T00:30:00+00:00"
    )
    observed: dict[str, object] = {}

    def lean_runner(command, cwd, source, timeout):
        observed.update(command=command, cwd=cwd, source=source, timeout=timeout)
        return CompilerOutcome(
            0, stdout="'boundedMetricPerturbation' depends on axioms: []"
        )

    run_dir = tmp_path / "proved"
    verdict = run_formal_feedback(
        conjecture,
        _review(conjecture),
        repo_root=Path(__file__).resolve().parents[1],
        run_dir=run_dir,
        budget=ProofBudget(timeout_seconds=45),
        runner=lean_runner,
        attempted_at="2026-08-24T02:00:00+00:00",
        console_logging=False,
    )

    assert verdict.verdict == "proved"
    assert verdict.attempts_used == 1
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[:3] == ("rch", "exec", "--")
    assert observed["cwd"] == Path(__file__).resolve().parents[1] / "formal/lean"
    assert observed["timeout"] == 45
    assert json.loads((run_dir / "verdict.json").read_text())["verdict"] == "proved"
    assert (
        json.loads((run_dir / "review.json").read_text())["reviewer"] == "Ada Lovelace"
    )
    events = read_run_events(run_dir)
    assert [event["event"] for event in events] == [
        "run_start",
        "conjecture_proposed",
        "human_review_recorded",
        "lean_attempt_started",
        "lean_attempt_completed",
        "theory_verdict_recorded",
        "run_end",
    ]
