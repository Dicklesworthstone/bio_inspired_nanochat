"""Full-cycle tests for the bounded AI-neuroscientist flywheel — r00r.2.4."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.ai_neuroscientist import (
    CycleConfig,
    append_proposals,
    preview_research_cycle,
    query_proposals,
    read_proposals,
    run_research_cycle,
)
from bio_inspired_nanochat.experiment_runner import EvalMatrixOptions, HardBudget, ProcessOutcome
from bio_inspired_nanochat.result_interpreter import read_knowledge_base
from bio_inspired_nanochat.results_registry import append_record, make_record, read_records
from bio_inspired_nanochat.run_logging import read_run_events


def _config(tmp_path: Path, *, console_logging: bool = False) -> CycleConfig:
    return CycleConfig(
        selected_mechanisms=("doc2",),
        paired_seed_count=2,
        seed_start=701,
        maximum_tokens_per_run=32,
        eval_options=EvalMatrixOptions(
            train_tokens=16,
            eval_tokens=16,
            inline_smoke_training=True,
            sequence_len=8,
            vocab_size=16,
            n_layer=1,
            n_head=2,
            n_embd=8,
            device_batch_size=1,
            total_batch_size_tokens=8,
        ),
        budget=HardBudget(4, 128, 60),
        project_root=tmp_path,
        run_root="cycles",
        registry_path="registry.jsonl",
        preregistry_path="preregistrations.jsonl",
        batch_ledger_path="batches.jsonl",
        knowledge_base_path="knowledge.jsonl",
        proposal_registry_path="proposals.jsonl",
        console_logging=console_logging,
    )


def _successful_executor(registry: Path):
    def execute(command, _cwd, _timeout):
        seed = int(command[command.index("--seed") + 1])
        preset = command[command.index("--preset") + 1]
        value = {"bio_no_doc2": 0.5, "bio_all": 0.6}[preset]
        append_record(
            make_record(
                "eval",
                {"niah_accuracy": value},
                run_id=f"source-{preset}-{seed}",
                config={"preset": preset},
                seed=seed,
            ),
            str(registry),
        )
        return ProcessOutcome(0, "ok", "")

    return execute


@pytest.mark.e2e
def test_full_cycle_preregisters_runs_interprets_proposes_and_logs(tmp_path: Path):
    config = _config(tmp_path)
    report = run_research_cycle(
        config,
        executor=_successful_executor(tmp_path / "registry.jsonl"),
        started_at="2026-08-24T00:00:00+00:00",
    )

    assert report.status == "completed_pending_human_review"
    assert report.experiment_report.status == "completed"
    assert report.experiment_report.spent_runs == 4
    assert report.experiment_report.spent_tokens == 128
    assert len(read_records(str(tmp_path / "registry.jsonl"))) == 9
    knowledge = read_knowledge_base(tmp_path / "knowledge.jsonl")
    assert len(knowledge) == 1
    assert knowledge[0].entries[0]["verdict"] == "null"
    proposals = read_proposals(tmp_path / "proposals.jsonl")
    assert len(proposals) == 1
    assert proposals[0].review_status == "pending_human_review"
    assert not proposals[0].automatic_execution_allowed
    assert proposals[0].search_handoff["destination"] == "bio_inspired_nanochat-hea"

    events = read_run_events(report.cycle_dir)
    stages = [
        (event["step"], event["stage"], event["state"])
        for event in events
        if event["event"] == "cycle_stage"
    ]
    assert stages == [
        (1, "preregister", "started"),
        (1, "preregister", "completed"),
        (2, "execute", "started"),
        (2, "execute", "completed"),
        (3, "interpret", "started"),
        (3, "interpret", "completed"),
        (4, "propose", "started"),
        (4, "propose", "completed"),
    ]
    assert any(event["event"] == "research_cycle_completed" for event in events)
    assert events[-1]["event"] == "run_end"


@pytest.mark.unit
def test_preview_is_non_mutating_and_exactly_budgeted(tmp_path: Path):
    config = _config(tmp_path)
    preview = preview_research_cycle(
        config, started_at="2026-08-24T00:00:00+00:00"
    )

    assert preview.cycle_id.startswith("cycle-")
    assert preview.plan.projected_runs == 4
    assert preview.plan.projected_tokens == 128
    assert [cell.preset for cell in preview.plan.cells] == [
        "bio_no_doc2",
        "bio_no_doc2",
        "bio_all",
        "bio_all",
    ]
    assert not any(tmp_path.iterdir())


@pytest.mark.unit
def test_cycle_refuses_duplicate_or_existing_output_before_experiment(tmp_path: Path):
    config = _config(tmp_path)
    preview = preview_research_cycle(
        config, started_at="2026-08-24T00:00:00+00:00"
    )
    Path(preview.cycle_dir).mkdir(parents=True)
    attempts = 0

    def forbidden_executor(_command, _cwd, _timeout):
        nonlocal attempts
        attempts += 1
        return ProcessOutcome(0)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_research_cycle(
            config,
            executor=forbidden_executor,
            started_at="2026-08-24T00:00:00+00:00",
        )
    assert attempts == 0
    assert not (tmp_path / "preregistrations.jsonl").exists()


@pytest.mark.unit
def test_cycle_config_rejects_incomplete_family_budget(tmp_path: Path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="complete family"):
        replace(config, budget=HardBudget(3, 128, 60))
    with pytest.raises(ValueError, match="complete family"):
        replace(config, budget=HardBudget(4, 127, 60))


@pytest.mark.unit
def test_proposal_is_falsifiable_queryable_and_append_only(tmp_path: Path):
    config = _config(tmp_path)
    report = run_research_cycle(
        config,
        executor=_successful_executor(tmp_path / "registry.jsonl"),
        started_at="2026-08-24T00:00:00+00:00",
    )
    proposals = read_proposals(tmp_path / "proposals.jsonl")

    assert proposals[0].proposal_id in report.proposal_ids
    assert "at least 0.02 absolute" in proposals[0].falsifiable_hypothesis
    assert query_proposals(
        proposals, mechanism="doc2", review_status="pending_human_review"
    ) == proposals
    with pytest.raises(ValueError, match="already"):
        append_proposals(proposals, tmp_path / "proposals.jsonl")


@pytest.mark.unit
def test_invalidated_knowledge_cannot_seed_speculative_mechanism(tmp_path: Path):
    config = _config(tmp_path)

    def failing_executor(_command, _cwd, _timeout):
        return ProcessOutcome(9, "", "controlled")

    with pytest.raises(ValueError, match="wholly invalidated"):
        run_research_cycle(
            config,
            executor=failing_executor,
            started_at="2026-08-24T00:00:00+00:00",
        )
    knowledge = read_knowledge_base(tmp_path / "knowledge.jsonl")
    assert knowledge[0].entries[0]["verdict"] == "invalidated"
    assert not (tmp_path / "proposals.jsonl").exists()
    events_path = next((tmp_path / "cycles").glob("*/events.jsonl"))
    events = [
        json.JSONDecoder().decode(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "run_error" for event in events)
