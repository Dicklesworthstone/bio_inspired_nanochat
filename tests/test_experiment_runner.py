"""Hard-budget and provenance tests for the preregistered experiment runner."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.experiment_runner import (
    EvalMatrixOptions,
    HardBudget,
    ProcessOutcome,
    build_batch_plan,
    config_to_ablation_preset,
    execute_batch_plan,
)
from bio_inspired_nanochat.hypothesis_generator import generate_hypotheses, results_snapshot_digest
from bio_inspired_nanochat.results_registry import append_record, make_record, read_records


def _hypothesis(registry: Path):
    return generate_hypotheses(
        [],
        results_digest=results_snapshot_digest(registry),
        selected_mechanisms=["doc2"],
        limit=1,
        paired_seed_count=2,
        seed_start=101,
        maximum_tokens_per_run=64,
        registered_at="2026-08-24T00:00:00+00:00",
    )[0]


def _options() -> EvalMatrixOptions:
    return EvalMatrixOptions(
        train_tokens=32,
        eval_tokens=32,
        inline_smoke_training=True,
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=16,
        device_batch_size=1,
        total_batch_size_tokens=8,
    )


@pytest.mark.unit
def test_plan_maps_frozen_arms_to_existing_ablation_presets_and_exact_pairs(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    plan = build_batch_plan(
        [hypothesis],
        budget=HardBudget(4, 256, 60),
        eval_options=_options(),
        registry_path=registry,
        output_root=tmp_path / "runs",
        project_root=tmp_path,
        created_at="2026-08-24T00:00:01+00:00",
    )

    assert plan.projected_runs == 4
    assert plan.projected_tokens == 256
    assert [cell.preset for cell in plan.cells] == [
        "bio_no_doc2",
        "bio_no_doc2",
        "bio_all",
        "bio_all",
    ]
    assert {(cell.arm, cell.seed) for cell in plan.cells} == {
        ("control", 101),
        ("control", 102),
        ("intervention", 101),
        ("intervention", 102),
    }
    assert all("scripts.eval_matrix" in cell.command for cell in plan.cells)
    assert all(cell.command[0] for cell in plan.cells)


@pytest.mark.unit
def test_plan_fails_closed_on_any_run_token_snapshot_or_harness_violation(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    with pytest.raises(ValueError, match="requires 4 runs"):
        build_batch_plan(
            [hypothesis],
            budget=HardBudget(3, 256, 60),
            eval_options=_options(),
            registry_path=registry,
            output_root=tmp_path / "runs",
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="requires 256 tokens"):
        build_batch_plan(
            [hypothesis],
            budget=HardBudget(4, 255, 60),
            eval_options=_options(),
            registry_path=registry,
            output_root=tmp_path / "runs",
            project_root=tmp_path,
        )

    registry.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after pre-registration"):
        build_batch_plan(
            [hypothesis],
            budget=HardBudget(4, 256, 60),
            eval_options=_options(),
            registry_path=registry,
            output_root=tmp_path / "runs",
            project_root=tmp_path,
        )

    wrong_harness = replace(hypothesis, harness="invented_shell_harness")
    registry.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not an approved existing runner"):
        build_batch_plan(
            [wrong_harness],
            budget=HardBudget(4, 256, 60),
            eval_options=_options(),
            registry_path=registry,
            output_root=tmp_path / "runs",
            project_root=tmp_path,
        )


@pytest.mark.unit
def test_effective_config_mapping_supports_genome_and_rejects_freeform_configs():
    assert config_to_ablation_preset({"xi_dim": 0}) == "bio_no_genome"
    assert config_to_ablation_preset({"doc2_gain": 0.0, "enable_presyn": True}) == "bio_no_doc2"
    assert config_to_ablation_preset({"doc2_gain": 0.08, "enable_presyn": True}) == "bio_all"
    with pytest.raises(ValueError, match="not representable"):
        config_to_ablation_preset({"doc2_gain": 0.04})
    with pytest.raises(ValueError, match="unknown SynapticConfig"):
        config_to_ablation_preset({"shell_command": "anything"})


@pytest.mark.unit
def test_execution_audits_every_attempt_and_never_exceeds_budget(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    ledger = tmp_path / "batch.jsonl"
    plan = build_batch_plan(
        [hypothesis],
        budget=HardBudget(4, 256, 60),
        eval_options=_options(),
        registry_path=registry,
        output_root=tmp_path / "runs",
        project_root=tmp_path,
    )

    def successful_existing_harness(command, _cwd, _timeout):
        seed = int(command[command.index("--seed") + 1])
        preset = command[command.index("--preset") + 1]
        append_record(
            make_record(
                "eval",
                {"niah_accuracy": 0.5 if preset == "bio_no_doc2" else 0.75},
                run_id=f"source-{preset}-{seed}",
                config={"preset": preset},
                seed=seed,
            ),
            str(registry),
        )
        return ProcessOutcome(0, "ok", "")

    report = execute_batch_plan(
        plan,
        registry_path=registry,
        batch_ledger_path=ledger,
        project_root=tmp_path,
        executor=successful_existing_harness,
    )

    assert report.status == "completed"
    assert report.spent_runs == plan.budget.maximum_runs == 4
    assert report.spent_tokens == plan.budget.maximum_total_tokens == 256
    records = read_records(str(registry))
    audit = [record for record in records if record.run_id.startswith(plan.execution_batch_id)]
    assert len(audit) == 4
    assert all(record.git_sha and record.config_hash and not record.eligible_for_best for record in audit)
    notes = [json.JSONDecoder().decode(record.notes) for record in audit]
    assert {note["hypothesis_id"] for note in notes} == {hypothesis.hypothesis_id}
    assert {note["source_run_id"] for note in notes} == {
        "source-bio_no_doc2-101",
        "source-bio_no_doc2-102",
        "source-bio_all-101",
        "source-bio_all-102",
    }
    events = [
        json.JSONDecoder().decode(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "batch_planned",
        "cell_finished",
        "cell_finished",
        "cell_finished",
        "cell_finished",
        "batch_finished",
    ]


@pytest.mark.unit
def test_failures_and_timeouts_are_invalidated_and_registry_stamped(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    plan = build_batch_plan(
        [hypothesis],
        budget=HardBudget(4, 256, 60),
        eval_options=_options(),
        registry_path=registry,
        output_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    attempts = 0

    def failing_existing_harness(_command, _cwd, _timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.TimeoutExpired(cmd="fixed", timeout=1)
        return ProcessOutcome(7, "", "controlled failure")

    report = execute_batch_plan(
        plan,
        registry_path=registry,
        batch_ledger_path=tmp_path / "batch.jsonl",
        project_root=tmp_path,
        executor=failing_existing_harness,
    )

    assert report.status == "completed_with_failures"
    assert [item.status for item in report.receipts] == [
        "timed_out",
        "failed",
        "failed",
        "failed",
    ]
    records = read_records(str(registry))
    assert len(records) == 4
    assert all(record.verdict == "invalidated" and not record.eligible_for_best for record in records)


@pytest.mark.unit
def test_execution_refuses_overwrite_or_duplicate_batch(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    plan = build_batch_plan(
        [hypothesis],
        budget=HardBudget(4, 256, 60),
        eval_options=_options(),
        registry_path=registry,
        output_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    Path(plan.output_root).mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        execute_batch_plan(
            plan,
            registry_path=registry,
            batch_ledger_path=tmp_path / "batch.jsonl",
            project_root=tmp_path,
        )


@pytest.mark.unit
def test_execution_revalidates_allowlisted_command_before_writing(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypothesis = _hypothesis(registry)
    plan = build_batch_plan(
        [hypothesis],
        budget=HardBudget(4, 256, 60),
        eval_options=_options(),
        registry_path=registry,
        output_root=tmp_path / "runs",
        project_root=tmp_path,
    )
    tampered_cell = replace(plan.cells[0], command=("/bin/sh", "-c", "anything"))
    tampered_plan = replace(plan, cells=(tampered_cell, *plan.cells[1:]))

    with pytest.raises(ValueError, match="allowlisted Python interpreter"):
        execute_batch_plan(
            tampered_plan,
            registry_path=registry,
            batch_ledger_path=tmp_path / "batch.jsonl",
            project_root=tmp_path,
        )
    assert not Path(plan.output_root).exists()
    assert not (tmp_path / "batch.jsonl").exists()
