"""Hypothesis generation and immutable preregistration — bead r00r.2.1."""

from __future__ import annotations

import json

import pytest

from bio_inspired_nanochat.hypothesis_generator import (
    InterpretabilitySignal,
    append_preregistration,
    generate_hypotheses,
    read_preregistrations,
    results_snapshot_digest,
)
from bio_inspired_nanochat.results_registry import RunRecord


def _record(run_id: str, *, seed: int, notes: str = "") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        harness="eval",
        metrics={"eval_bpb": 1.0},
        seed=seed,
        notes=notes,
    )


@pytest.mark.unit
def test_generator_freezes_endpoint_controls_seeds_and_stopping_rule(tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text('{"existing":"snapshot"}\n', encoding="utf-8")
    records = [_record("old-presyn", seed=10007, notes="mechanism=presyn")]

    hypotheses = generate_hypotheses(
        records,
        results_digest=results_snapshot_digest(registry),
        selected_mechanisms=["presyn", "bdnf"],
        limit=2,
        paired_seed_count=4,
        seed_start=10007,
        registered_at="2026-08-24T00:00:00+00:00",
    )

    assert [item.mechanism for item in hypotheses] == ["bdnf", "presyn"]
    assert all(item.primary_metric and item.statement for item in hypotheses)
    assert all(item.control != item.intervention for item in hypotheses)
    assert all(item.paired_seeds == (10008, 10009, 10010, 10011) for item in hypotheses)
    assert all(item.stopping_rule.no_early_efficacy_stop for item in hypotheses)
    assert all(item.compute_budget.maximum_runs == 8 for item in hypotheses)
    assert hypotheses[1].exploratory_run_ids == ("old-presyn",)


@pytest.mark.unit
def test_signal_changes_priority_but_not_preregistered_endpoint():
    digest = "0" * 64
    baseline = generate_hypotheses(
        [],
        results_digest=digest,
        selected_mechanisms=["bdnf", "presyn"],
        limit=2,
        registered_at="2026-08-24T00:00:00+00:00",
    )
    signaled = generate_hypotheses(
        [],
        results_digest=digest,
        signals=[
            InterpretabilitySignal(
                mechanism="presyn",
                signal_name="lesion_effect",
                effect_size=3.0,
                confidence=1.0,
                source_id="probe-17",
            )
        ],
        selected_mechanisms=["bdnf", "presyn"],
        limit=2,
        registered_at="2026-08-24T00:00:00+00:00",
    )

    assert baseline[0].mechanism == "bdnf"  # deterministic alphabetical tie break
    assert signaled[0].mechanism == "presyn"
    baseline_presyn = next(item for item in baseline if item.mechanism == "presyn")
    signaled_presyn = next(item for item in signaled if item.mechanism == "presyn")
    assert signaled_presyn.proposal_score > baseline_presyn.proposal_score
    assert signaled_presyn.primary_metric == baseline_presyn.primary_metric == "niah_accuracy"
    assert signaled_presyn.metric_direction == baseline_presyn.metric_direction
    assert signaled_presyn.minimum_effect == baseline_presyn.minimum_effect
    assert "probe-17" in signaled_presyn.interpretability_signal_ids
    assert len(signaled_presyn.interpretability_signal_ids) == 1

    repeated_later = generate_hypotheses(
        [],
        results_digest=digest,
        selected_mechanisms=["bdnf", "presyn"],
        limit=2,
        registered_at="2026-08-25T00:00:00+00:00",
    )
    assert [item.hypothesis_id for item in repeated_later] == [
        item.hypothesis_id for item in baseline
    ]


@pytest.mark.unit
def test_preregistry_is_append_only_duplicate_safe_and_validated(tmp_path):
    path = tmp_path / "preregistrations.jsonl"
    hypothesis = generate_hypotheses(
        [],
        results_digest="1" * 64,
        selected_mechanisms=["metriplectic_integrator"],
        limit=1,
        registered_at="2026-08-24T00:00:00+00:00",
    )[0]

    append_preregistration(hypothesis, path)
    assert read_preregistrations(path) == [hypothesis]
    with pytest.raises(ValueError, match="already preregistered"):
        append_preregistration(hypothesis, path)

    path.write_text(path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"preregistrations\.jsonl:2"):
        read_preregistrations(path)


@pytest.mark.unit
def test_preregistration_json_contains_explicit_prediction_and_fixed_rule(tmp_path):
    path = tmp_path / "preregistrations.jsonl"
    hypothesis = generate_hypotheses(
        [],
        results_digest="2" * 64,
        selected_mechanisms=["stochastic_release"],
        limit=1,
        registered_at="2026-08-24T00:00:00+00:00",
    )[0]
    append_preregistration(hypothesis, path)

    payload = json.JSONDecoder().decode(path.read_text(encoding="utf-8"))
    assert payload["primary_metric"] == "id_ece"
    assert payload["metric_direction"] == "lower_better"
    assert payload["minimum_effect"] == 0.005
    assert bool(payload["stopping_rule"]["no_early_efficacy_stop"])
    assert bool(payload["compute_budget"]["equal_model_flops"])
