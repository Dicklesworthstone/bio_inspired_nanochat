"""Controlled equal-work falsification of topological structural plasticity."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bio_inspired_nanochat.results_registry import read_records
from scripts.e2e.structural_falsification import (
    StructuralFalsificationConfig,
    protocol_id,
    run_structural_falsification,
)


def _fast_config(**changes) -> StructuralFalsificationConfig:
    base = replace(
        StructuralFalsificationConfig(),
        seeds=(11, 13, 17),
        train_steps=2,
        bootstrap_samples=200,
    )
    return replace(base, **changes)


@pytest.mark.e2e
def test_structural_falsification_is_equal_work_and_auditable(tmp_path):
    registry_path = tmp_path / "results" / "registry.jsonl"
    report = run_structural_falsification(
        _fast_config(),
        run_dir=tmp_path / "run",
        registry_path=registry_path,
    )

    report.assert_not_invalidated()
    assert report.verdict == "null"
    assert len(report.outcomes) == 6
    assert len(report.fallback_outcomes) == 3
    assert all(report.invariants.values())
    assert report.comparisons["dead_expert_fraction"].paired.mean_delta == 0.0
    assert report.comparisons["final_loss"].paired.delta_ci_high < 0.0
    assert report.comparisons["event_loss_spike"].paired.delta_ci_high < 0.0

    topological = [
        outcome for outcome in report.outcomes if outcome.method == "topological"
    ]
    uta = [outcome for outcome in report.outcomes if outcome.method == "uta"]
    for treatment, baseline in zip(topological, uta):
        assert treatment.seed == baseline.seed
        assert treatment.loss_before == pytest.approx(baseline.loss_before, abs=1e-12)
        assert treatment.model_work_units == baseline.model_work_units
        assert treatment.model_forward_calls == baseline.model_forward_calls
        assert treatment.expert_count_after == baseline.expert_count_after == 4
        assert treatment.top_k == baseline.top_k == 2
        assert treatment.lifecycle_mode == "topological"
        assert treatment.lifecycle_action == "merge_split"
        assert treatment.spectral_bound_holds
        assert treatment.persistence_stability_holds
        assert treatment.ot_merge_optimal
        assert treatment.max_child_condition_number <= treatment.kappa_bound

    for fallback in report.fallback_outcomes:
        assert fallback.fallback_reason == "missing_routing_points"
        assert fallback.uta_plans_equal
        assert fallback.state_identity
        assert fallback.output_identity

    report_text = (tmp_path / "run" / "statistics.json").read_text(encoding="utf-8")
    assert "NaN" not in report_text and "Infinity" not in report_text
    payload = json.loads(report_text)
    assert payload["schema_version"] == 1
    assert payload["bead"] == "bio_inspired_nanochat-0642.5.3.1"
    assert payload["scope"].endswith("not scale evidence")

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_names = {event["event"] for event in events}
    assert {
        "topological_nas",
        "structural_seed_outcome",
        "structural_fallback_outcome",
        "structural_falsification_summary",
    } <= event_names

    records = read_records(str(registry_path))
    assert len(records) == 6
    assert all(not record.eligible_for_best for record in records)
    assert {record.verdict for record in records} == {None, "null"}
    for record in records:
        assert {
            "dead_expert_frac",
            "moe_gini",
            "structural_final_loss",
            "structural_event_loss_spike",
            "structural_event_loss_discontinuity",
        } == record.metrics.keys()
        assert "scope=tiny_controlled_synthetic" in record.notes


@pytest.mark.e2e
def test_structural_falsification_refuses_nonempty_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_structural_falsification(_fast_config(), run_dir=run_dir)

    with pytest.raises(FileExistsError, match="refusing to mix"):
        run_structural_falsification(_fast_config(), run_dir=run_dir)


@pytest.mark.unit
def test_committed_structural_report_is_machine_auditable():
    config = StructuralFalsificationConfig()
    artifact_path = (
        Path(__file__).parents[1]
        / "results"
        / f"structural_falsification_{protocol_id(config)}.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["protocol_id"] == protocol_id(config)
    assert payload["verdict"] in {"positive", "null"}
    assert len(payload["outcomes"]) == 2 * len(config.seeds)
    assert len(payload["fallback_outcomes"]) == len(config.seeds)
    assert all(payload["invariants"].values())
    assert payload["scope"].endswith("not scale evidence")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_fast_config(seeds=(11, 13)), "at least three unique"),
        (_fast_config(n_embd=5), "n_embd must be four"),
        (_fast_config(num_experts=3), "at least four"),
        (_fast_config(top_k=4), "smaller than num_experts"),
        (_fast_config(train_steps=0), "train_steps must be positive"),
        (_fast_config(dead_share_floor=0.0), "strictly between zero and one"),
        (_fast_config(perturb_epsilon=0.0), "finite and positive"),
        (_fast_config(bootstrap_samples=0), "bootstrap_samples must be positive"),
    ],
)
def test_structural_falsification_rejects_invalid_controls(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()
