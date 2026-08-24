"""End-to-end falsification of the exact-affine tropical skeleton."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bio_inspired_nanochat.results_registry import read_records
import scripts.e2e.tropical_falsification as tropical_falsification
from scripts.e2e.tropical_falsification import (
    TropicalFalsificationConfig,
    _empirical_adversarial_radius,
    _oracle_argmax,
    _sample_affine_family,
    _verdict,
    run_tropical_falsification,
)


def _fast_config(**changes) -> TropicalFalsificationConfig:
    base = replace(
        TropicalFalsificationConfig(),
        # Exploratory seeds stay separate from the held-out defaults.
        seeds=(11, 23, 37, 53, 71, 89, 107, 131),
        angle_samples=512,
        binary_steps=32,
        interior_trials=96,
        bootstrap_samples=1_000,
    )
    return replace(base, **changes)


@pytest.mark.e2e
def test_multiseed_tropical_falsification_is_positive_and_auditable(tmp_path):
    registry_path = tmp_path / "results" / "registry.jsonl"
    report = run_tropical_falsification(
        _fast_config(),
        run_dir=tmp_path / "run",
        registry_path=registry_path,
    )

    report.assert_positive()
    assert report.exactness.mean == pytest.approx(1.0)
    assert report.radius_safety_margin.ci_low > 0.0
    assert report.convergence_comparison.mean_delta < 0.0
    assert report.convergence_comparison.delta_ci_high < 0.0
    assert report.convergence_comparison.t_p_value < 0.05
    assert report.convergence_comparison.wilcoxon_p_value <= 0.05
    assert report.attribution_comparison.mean_delta < 0.0
    assert report.attribution_comparison.delta_ci_high < 0.0
    assert report.attribution_comparison.t_p_value < 0.05
    assert report.attribution_comparison.wilcoxon_p_value <= 0.05

    for outcome in report.outcomes:
        assert outcome.exactness_rate == pytest.approx(1.0)
        assert outcome.monotonic_convergence and outcome.convergence_bound_satisfied
        assert outcome.low_temperature_readout_l1_error < 1e-12
        assert outcome.certified_radius > report.config.min_certified_radius
        assert outcome.certified_radius <= outcome.empirical_adversarial_radius
        assert 0.0 <= outcome.attack_resolution_error
        assert outcome.attack_resolution_error <= report.config.max_attack_resolution_error
        assert outcome.interior_flips == 0
        assert outcome.tropical_attribution_l1_error == pytest.approx(0.0)
        assert outcome.attention_rollout_attribution_l1_error > 0.0
        assert not outcome.high_temperature_gate_passed
        assert outcome.low_temperature_gate_passed
        assert outcome.high_temperature_fallback_identity
        assert outcome.low_temperature_hard_authorized

    payload_text = (tmp_path / "run" / "statistics.json").read_text(encoding="utf-8")
    assert "NaN" not in payload_text and "Infinity" not in payload_text
    payload = json.loads(payload_text)
    assert payload["bead"] == "bio_inspired_nanochat-0642.6.3.1"
    assert payload["verdict"] == "positive"

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_names = {event["event"] for event in events}
    assert {
        "tropical_temperature_observation",
        "tropical_certificate",
        "tropical_routing_transition",
        "tropical_seed_outcome",
        "tropical_falsification_summary",
    } <= event_names

    records = read_records(str(registry_path))
    assert len(records) == len(report.config.seeds)
    assert {record.seed for record in records} == set(report.config.seeds)
    assert all(record.harness == "eval" for record in records)
    assert all(record.verdict == "positive" for record in records)
    assert all(record.eligible_for_best for record in records)
    assert all("verdict=positive" in record.notes for record in records)
    assert all(
        set(record.metrics)
        == {
            "tropical_exactness_rate",
            "tropical_readout_l1_error",
            "soft_readout_l1_error",
            "tropical_certified_radius",
            "empirical_adversarial_radius",
            "tropical_radius_tightness",
            "tropical_attribution_l1_error",
            "attention_rollout_attribution_l1_error",
        }
        for record in records
    )


@pytest.mark.e2e
def test_tropical_falsification_is_deterministic(tmp_path):
    config = _fast_config(
        seeds=(11, 23, 37, 53, 71, 89),
        angle_samples=256,
        interior_trials=32,
        bootstrap_samples=500,
    )
    first = run_tropical_falsification(config, run_dir=tmp_path / "first")
    second = run_tropical_falsification(config, run_dir=tmp_path / "second")

    assert first.outcomes == second.outcomes
    assert first.exactness == second.exactness
    assert first.radius_ratio == second.radius_ratio
    assert first.convergence_comparison == second.convergence_comparison
    assert first.attribution_comparison == second.attribution_comparison
    assert first.verdict == second.verdict == "positive"


@pytest.mark.unit
def test_tropical_verdict_distinguishes_invalidated_from_underpowered(tmp_path):
    report = run_tropical_falsification(
        _fast_config(seeds=(11, 23, 37, 53, 71, 89), angle_samples=128),
        run_dir=tmp_path / "run",
    )
    forged = replace(
        report.outcomes[0],
        certified_radius=report.outcomes[0].empirical_adversarial_radius + 0.1,
    )
    invalidated, invalidated_reason = _verdict(
        report.config,
        (forged, *report.outcomes[1:]),
        report.convergence_comparison,
        report.attribution_comparison,
    )
    underpowered = replace(
        report.convergence_comparison,
        delta_ci_high=0.1,
        t_p_value=0.5,
        wilcoxon_p_value=0.5,
    )
    null, null_reason = _verdict(
        report.config,
        report.outcomes,
        underpowered,
        report.attribution_comparison,
    )

    assert invalidated == "invalidated"
    assert "certificate exceeded" in invalidated_reason
    assert null == "null"
    assert "did not clear" in null_reason


@pytest.mark.unit
def test_empirical_attack_uses_independent_oracle(monkeypatch):
    config = _fast_config(angle_samples=128)
    family = _sample_affine_family(config, seed=11)
    _, winner_id = _oracle_argmax(family.scores, family.choice_ids)

    def _unexpected_runtime_call(*args, **kwargs):
        raise AssertionError("empirical attack called the runtime argmax")

    monkeypatch.setattr(tropical_falsification, "deterministic_argmax", _unexpected_runtime_call)
    radius = _empirical_adversarial_radius(
        family,
        winner_id=winner_id,
        angle_samples=128,
        binary_steps=16,
        search_radius=config.search_radius,
    )
    assert radius > 0.0


@pytest.mark.unit
def test_tropical_falsification_rejects_nonempty_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_tropical_falsification(_fast_config(), run_dir=run_dir)

    with pytest.raises(FileExistsError, match="refusing to mix"):
        run_tropical_falsification(_fast_config(), run_dir=run_dir)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_fast_config(seeds=(11,)), "at least two unique"),
        (_fast_config(temperatures=(0.1, 0.2)), "strictly decreasing"),
        (_fast_config(input_dimension=3), "must be two"),
        (_fast_config(angle_samples=8), "at least 32"),
        (_fast_config(binary_steps=2), "at least eight"),
        (_fast_config(max_attack_resolution_error=0.0), "must be finite and positive"),
        (_fast_config(attribution_temperature=0.0), "must be finite and positive"),
        (_fast_config(bootstrap_samples=0), "bootstrap_samples must be positive"),
    ],
)
def test_tropical_falsification_rejects_invalid_controls(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()
