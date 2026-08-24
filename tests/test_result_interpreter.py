"""Stats-backed, append-only mechanism knowledge tests — bead r00r.2.3."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.hypothesis_generator import generate_hypotheses, results_snapshot_digest
from bio_inspired_nanochat.result_interpreter import (
    append_interpretation,
    interpret_experiment_batch,
    query_knowledge,
    read_knowledge_base,
)
from bio_inspired_nanochat.results_registry import append_record, make_record, read_records


def _hypotheses(registry: Path, *, mechanisms=("doc2",), seed_count: int = 8):
    return generate_hypotheses(
        [],
        results_digest=results_snapshot_digest(registry),
        selected_mechanisms=list(mechanisms),
        limit=len(mechanisms),
        paired_seed_count=seed_count,
        seed_start=301,
        registered_at="2026-08-24T00:00:00+00:00",
    )


def _append_audit(
    registry: Path,
    *,
    execution_batch_id: str,
    hypothesis_id: str,
    metric: str,
    arm: str,
    seed: int,
    value: float,
    status: str = "completed",
):
    append_record(
        make_record(
            "eval",
            {metric: value} if status == "completed" else {},
            run_id=f"{execution_batch_id}-{hypothesis_id}-{arm}-s{seed}",
            config={"hypothesis_id": hypothesis_id, "arm": arm, "seed": seed},
            seed=seed,
            notes=json.dumps(
                {
                    "orchestrator": "ai_neuroscientist",
                    "hypothesis_id": hypothesis_id,
                    "arm": arm,
                    "status": status,
                    "source_run_id": f"source-{hypothesis_id}-{arm}-{seed}",
                }
            ),
            verdict="invalidated" if status != "completed" else None,
            eligible_for_best=False,
        ),
        str(registry),
    )


def _ledger(path: Path, execution_batch_id: str, spent_runs: int, status="completed"):
    path.write_text(
        json.dumps(
            {
                "event": "batch_finished",
                "execution_batch_id": execution_batch_id,
                "report": {"status": status, "spent_runs": spent_runs},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _complete_batch(
    registry: Path,
    hypotheses,
    execution_batch_id: str,
    *,
    effect_by_mechanism: dict[str, float],
):
    for hypothesis in hypotheses:
        for offset, seed in enumerate(hypothesis.paired_seeds):
            control = 0.4 + 0.01 * offset
            effect = effect_by_mechanism[hypothesis.mechanism]
            intervention = control + effect
            _append_audit(
                registry,
                execution_batch_id=execution_batch_id,
                hypothesis_id=hypothesis.hypothesis_id,
                metric=hypothesis.primary_metric,
                arm="control",
                seed=seed,
                value=control,
            )
            _append_audit(
                registry,
                execution_batch_id=execution_batch_id,
                hypothesis_id=hypothesis.hypothesis_id,
                metric=hypothesis.primary_metric,
                arm="intervention",
                seed=seed,
                value=intervention,
            )


@pytest.mark.unit
def test_complete_supported_batch_is_confirmed_ranked_and_registry_logged(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    preregistry = tmp_path / "prereg.jsonl"
    preregistry.write_text("snapshot\n", encoding="utf-8")
    hypotheses = _hypotheses(registry)
    execution = "exec-confirmed"
    _complete_batch(registry, hypotheses, execution, effect_by_mechanism={"doc2": 0.1})
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 16)

    batch = interpret_experiment_batch(
        hypotheses,
        read_records(str(registry)),
        execution_batch_id=execution,
        batch_ledger_path=ledger,
        source_registry_path=registry,
        source_preregistry_path=preregistry,
        interpreted_at="2026-08-24T00:01:00+00:00",
    )

    entry = batch.entries[0]
    assert entry["verdict"] == "confirmed"
    assert entry["complete_preregistered_pairs"]
    assert entry["improvement"] == pytest.approx(0.1)
    assert entry["contribution_rank"] == 1
    assert batch.ranking == (hypotheses[0].hypothesis_id,)

    knowledge = tmp_path / "knowledge.jsonl"
    append_interpretation(batch, knowledge_base_path=knowledge, registry_path=registry)
    assert read_knowledge_base(knowledge) == [batch]
    interpretation = [
        record for record in read_records(str(registry)) if record.run_id.startswith("interpret-")
    ]
    assert len(interpretation) == 1
    assert interpretation[0].verdict == "positive"
    assert not interpretation[0].eligible_for_best
    with pytest.raises(ValueError, match="already"):
        append_interpretation(batch, knowledge_base_path=knowledge, registry_path=registry)


@pytest.mark.unit
def test_complete_equal_smoke_is_honest_null_not_equivalence(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    preregistry = tmp_path / "prereg.jsonl"
    preregistry.write_text("snapshot\n", encoding="utf-8")
    hypotheses = _hypotheses(registry, seed_count=2)
    execution = "exec-null"
    _complete_batch(registry, hypotheses, execution, effect_by_mechanism={"doc2": 0.0})
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 4)

    batch = interpret_experiment_batch(
        hypotheses,
        read_records(str(registry)),
        execution_batch_id=execution,
        batch_ledger_path=ledger,
        source_registry_path=registry,
        source_preregistry_path=preregistry,
    )

    assert batch.entries[0]["verdict"] == "null"
    assert "not evidence of equivalence" in batch.entries[0]["conclusion"]


@pytest.mark.unit
def test_missing_or_failed_cell_invalidates_without_fake_statistics(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    preregistry = tmp_path / "prereg.jsonl"
    preregistry.write_text("snapshot\n", encoding="utf-8")
    hypotheses = _hypotheses(registry, seed_count=2)
    hypothesis = hypotheses[0]
    execution = "exec-invalid"
    for seed in hypothesis.paired_seeds:
        _append_audit(
            registry,
            execution_batch_id=execution,
            hypothesis_id=hypothesis.hypothesis_id,
            metric=hypothesis.primary_metric,
            arm="control",
            seed=seed,
            value=0.5,
        )
    _append_audit(
        registry,
        execution_batch_id=execution,
        hypothesis_id=hypothesis.hypothesis_id,
        metric=hypothesis.primary_metric,
        arm="intervention",
        seed=hypothesis.paired_seeds[0],
        value=0.6,
        status="failed",
    )
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 3, status="completed_with_failures")

    batch = interpret_experiment_batch(
        hypotheses,
        read_records(str(registry)),
        execution_batch_id=execution,
        batch_ledger_path=ledger,
        source_registry_path=registry,
        source_preregistry_path=preregistry,
    )

    entry = batch.entries[0]
    assert entry["verdict"] == "invalidated"
    assert entry["paired_statistics"] is None
    assert entry["invalidation_reasons"]


@pytest.mark.unit
def test_interactions_are_candidates_not_causal_claims_and_queryable(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    preregistry = tmp_path / "prereg.jsonl"
    preregistry.write_text("snapshot\n", encoding="utf-8")
    hypotheses = _hypotheses(registry, mechanisms=("doc2", "bdnf"), seed_count=8)
    # BDNF is lower-better, so a negative raw intervention delta is a positive improvement.
    effects = {"doc2": 0.08, "bdnf": -0.04}
    execution = "exec-interactions"
    _complete_batch(registry, hypotheses, execution, effect_by_mechanism=effects)
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 32)

    batch = interpret_experiment_batch(
        hypotheses,
        read_records(str(registry)),
        execution_batch_id=execution,
        batch_ledger_path=ledger,
        source_registry_path=registry,
        source_preregistry_path=preregistry,
    )

    assert len(batch.interactions) == 1
    assert not batch.interactions[0]["causal_interaction_estimable"]
    assert "factorial" in batch.interactions[0]["causal_interaction_reason"]
    assert {entry["contribution_rank"] for entry in batch.entries} == {1, 2}
    rows = query_knowledge([batch], mechanism="doc2", verdict="confirmed")
    assert len(rows) == 1 and rows[0]["mechanism"] == "doc2"


@pytest.mark.unit
def test_incomplete_preregistered_holm_family_is_rejected(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    preregistry = tmp_path / "prereg.jsonl"
    preregistry.write_text("snapshot\n", encoding="utf-8")
    hypotheses = _hypotheses(registry, mechanisms=("doc2", "bdnf"), seed_count=2)
    execution = "exec-subset"
    _complete_batch(
        registry,
        [hypotheses[0]],
        execution,
        effect_by_mechanism={hypotheses[0].mechanism: 0.1},
    )
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 4)

    with pytest.raises(ValueError, match="complete preregistered Holm family"):
        interpret_experiment_batch(
            hypotheses,
            read_records(str(registry)),
            execution_batch_id=execution,
            batch_ledger_path=ledger,
            source_registry_path=registry,
            source_preregistry_path=preregistry,
        )


@pytest.mark.unit
def test_mixed_preregistration_batch_is_detected(tmp_path: Path):
    registry = tmp_path / "registry.jsonl"
    hypotheses = _hypotheses(registry, mechanisms=("doc2", "bdnf"), seed_count=2)
    altered = [hypotheses[0], replace(hypotheses[1], batch_id="batch-other")]
    execution = "exec-mixed"
    _complete_batch(
        registry,
        altered,
        execution,
        effect_by_mechanism={"doc2": 0.1, "bdnf": -0.1},
    )
    ledger = tmp_path / "batches.jsonl"
    _ledger(ledger, execution, 8)

    with pytest.raises(ValueError, match="mixes preregistration batches"):
        interpret_experiment_batch(
            altered,
            read_records(str(registry)),
            execution_batch_id=execution,
            batch_ledger_path=ledger,
            source_registry_path=registry,
            source_preregistry_path=tmp_path / "prereg.jsonl",
        )
