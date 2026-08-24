"""Equal-compute variable-count NAS evaluation tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.results_registry import read_records
from scripts.e2e.structural_nas_evaluation import (
    StructuralNASEvaluationConfig,
    protocol_id,
    run_structural_nas_evaluation,
)


def _fast_config(**changes) -> StructuralNASEvaluationConfig:
    base = replace(
        StructuralNASEvaluationConfig(),
        seeds=(11, 13, 17),
        train_steps=6,
        bootstrap_samples=200,
    )
    return replace(base, **changes)


@pytest.mark.e2e
def test_variable_count_nas_is_exactly_compute_matched_and_auditable(tmp_path):
    registry_path = tmp_path / "results" / "registry.jsonl"
    report = run_structural_nas_evaluation(
        _fast_config(),
        run_dir=tmp_path / "run",
        registry_path=registry_path,
    )

    assert report.verdict in {"win", "null", "regression"}
    assert report.registry_verdict in {"positive", "null", "invalidated"}
    assert len(report.outcomes) == 6
    assert all(report.invariants.values())
    assert report.comparisons["dead_expert_fraction"].paired.delta_ci_high < 0.0

    nas = [outcome for outcome in report.outcomes if outcome.method == "nas"]
    fixed = [outcome for outcome in report.outcomes if outcome.method == "fixed"]
    for treatment, baseline in zip(nas, fixed):
        assert treatment.seed == baseline.seed
        assert treatment.initial_loss == pytest.approx(baseline.initial_loss, abs=1e-12)
        assert treatment.forward_calls == baseline.forward_calls == 12
        assert treatment.train_forward_calls == baseline.train_forward_calls == 6
        assert treatment.expert_dispatches == baseline.expert_dispatches
        assert treatment.router_width_token_units == baseline.router_width_token_units
        assert (
            treatment.training_router_width_token_units
            == baseline.training_router_width_token_units
        )
        assert treatment.moe_matmul_flops == baseline.moe_matmul_flops
        assert treatment.average_expert_count == pytest.approx(4.0)
        assert treatment.training_average_expert_count == pytest.approx(4.0)
        assert treatment.final_expert_count == baseline.final_expert_count == 4
        assert treatment.dead_expert_fraction < baseline.dead_expert_fraction
        assert treatment.max_event_loss_spike <= report.config.max_event_loss_spike
        assert [
            (event.experts_before, event.experts_after) for event in treatment.events
        ] == [(4, 3), (3, 5), (5, 4)]
        assert [
            event.planned_operations[0]["kind"] for event in treatment.events
        ] == ["shrink", "grow", "shrink"]
        assert all(event.optimizer_synced for event in treatment.events)
        assert all(event.experts_before == event.experts_after == 4 for event in baseline.events)
        assert all(event.loss_discontinuity == 0.0 for event in baseline.events)

    report_text = (tmp_path / "run" / "statistics.json").read_text(encoding="utf-8")
    assert "NaN" not in report_text and "Infinity" not in report_text
    payload = json.loads(report_text)
    assert payload["schema_version"] == 1
    assert payload["bead"] == "bio_inspired_nanochat-uta.7"
    assert payload["scope"].endswith("not language-model-scale evidence")

    event_names = {
        json.loads(line)["event"]
        for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert {
        "structural_nas_lifecycle",
        "structural_nas_seed_outcome",
        "structural_nas_summary",
    } <= event_names

    records = read_records(str(registry_path))
    assert len(records) == 6
    assert all(not record.eligible_for_best for record in records)
    assert {record.verdict for record in records} == {None, report.registry_verdict}
    for record in records:
        assert {
            "dead_expert_frac",
            "moe_gini",
            "structural_final_loss",
            "structural_event_loss_spike",
            "structural_event_loss_discontinuity",
            "total_training_time",
        } == record.metrics.keys()
        assert "exact_equal_moe_flops=true" in record.notes


@pytest.mark.e2e
def test_structural_nas_evaluation_refuses_nonempty_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_structural_nas_evaluation(_fast_config(), run_dir=run_dir)

    with pytest.raises(FileExistsError, match="refusing to mix"):
        run_structural_nas_evaluation(_fast_config(), run_dir=run_dir)


@pytest.mark.unit
def test_committed_structural_nas_report_is_machine_auditable():
    config = StructuralNASEvaluationConfig()
    artifact_path = (
        Path(__file__).parents[1]
        / "results"
        / f"structural_nas_evaluation_{protocol_id(config)}.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["protocol_id"] == protocol_id(config)
    assert payload["verdict"] in {"win", "null", "regression"}
    assert payload["registry_verdict"] in {"positive", "null", "invalidated"}
    assert len(payload["outcomes"]) == 2 * len(config.seeds)
    assert all(payload["invariants"].values())
    assert payload["scope"].endswith("not language-model-scale evidence")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_fast_config(seeds=(11, 13)), "at least three unique"),
        (_fast_config(n_embd=5), "n_embd must be four"),
        (_fast_config(initial_experts=5), "4→3→5→4"),
        (_fast_config(top_k=3), "smaller than min_experts"),
        (_fast_config(train_steps=3), "positive, even"),
        (_fast_config(dead_share_floor=0.0), "strictly between zero and one"),
        (_fast_config(dormant_logit_bias=0.0), "finite and negative"),
        (_fast_config(max_event_loss_spike=-1.0), "finite and non-negative"),
        (_fast_config(bootstrap_samples=0), "bootstrap_samples must be positive"),
    ],
)
def test_structural_nas_evaluation_rejects_invalid_controls(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()
