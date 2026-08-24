"""E2E falsification of clamped Euler versus the guarded metriplectic recurrence.

The harness uses the live torch integrator over a fixed physical horizon and compares both arms to
the analytic continuous flow. It must reproduce the explicit-Euler stability boundary while the
structure-preserving arm remains certified, fallback-free, and at least as accurate.

Run with:

    uv run python -m pytest tests/test_metriplectic_falsification.py -v
"""

from __future__ import annotations

import json
import math

import pytest

from bio_inspired_nanochat.results_registry import read_records
from scripts.e2e.metriplectic_stability_curve import (
    StabilitySweepConfig,
    run_statistical_stability_sweep,
    run_stability_sweep,
)

pytestmark = pytest.mark.e2e


def test_stability_curve_reproduces_the_predicted_leapfrog(tmp_path):
    report = run_stability_sweep(run_dir=tmp_path)

    report.assert_leapfrog()
    assert report.predicted_baseline_boundary == pytest.approx(0.5)
    assert report.measured_baseline_boundary == pytest.approx(0.5)
    assert report.measured_metriplectic_boundary is None
    proof = report.proof_obligation
    assert proof.verified
    assert proof.max_abs_energy_drift <= 1e-10
    assert proof.min_entropy_production >= -1e-10
    assert proof.max_free_energy_delta <= 1e-10
    assert proof.max_degeneracy_residual <= 1e-10
    assert proof.structural_fallback_count == 0
    assert proof.fallback_injection.verified
    assert proof.fallback_injection.fallback_count == proof.fallback_injection.steps
    assert proof.fallback_injection.max_residual > 1e-10
    assert proof.fallback_injection.every_breach_was_degeneracy
    assert proof.fallback_injection.every_fallback_matched_baseline
    assert proof.fallback_injection.trajectory_finite
    assert proof.fallback_injection.physical_domain

    for point in report.curve:
        assert point.metriplectic.stable
        assert point.metriplectic.finite
        assert point.metriplectic.physical_domain
        assert point.metriplectic.fallback_count == 0
        assert point.metriplectic.max_abs_energy_drift <= 1e-10
        assert point.metriplectic.min_entropy_production >= -1e-10
        assert point.metriplectic.max_free_energy_delta <= 1e-10
        assert point.metriplectic_loss_no_worse

    diverged = [point for point in report.curve if not point.baseline.stable]
    assert [point.step_size for point in diverged] == [0.5, 1.0]
    assert all(point.baseline.finite and point.baseline.physical_domain for point in diverged)
    assert all(
        point.baseline.divergence_reasons == ("free_energy_increase",)
        for point in diverged
    )


def test_stability_curve_emits_complete_strict_json_evidence(tmp_path):
    cfg = StabilitySweepConfig(step_sizes=(0.25, 0.5, 1.0))
    report = run_stability_sweep(cfg, run_dir=tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [event for event in events if event["event"] == "metriplectic_stability_step"]
    points = [
        event for event in events if event["event"] == "metriplectic_stability_curve_point"
    ]
    summaries = [
        event for event in events if event["event"] == "metriplectic_stability_summary"
    ]
    injections = [
        event
        for event in events
        if event["event"] == "metriplectic_fallback_injection_step"
    ]
    expected_steps = 2 * sum(round(cfg.duration / dt) for dt in cfg.step_sizes)
    assert len(steps) == expected_steps
    assert {event["arm"] for event in steps} == {"baseline", "metriplectic"}
    assert len(points) == len(cfg.step_sizes)
    assert len(summaries) == 1 and summaries[0]["leapfrog_reproduced"]
    assert summaries[0]["proof_obligation_verified"]
    assert len(injections) == report.proof_obligation.fallback_injection.steps
    assert all(event["breach"] == "degeneracy" for event in injections)
    assert all(event["used_fallback"] for event in injections)
    assert all(event["fallback_matches_baseline"] for event in injections)
    structural_steps = [event for event in steps if event["arm"] == "metriplectic"]
    assert all(max(event["res_L_gradS"] + event["res_M_gradE"]) <= 1e-10 for event in structural_steps)

    payload = json.loads((tmp_path / "stability_curve.json").read_text(encoding="utf-8"))
    assert payload["bead"] == "bio_inspired_nanochat-0642.1.3.1"
    assert payload["leapfrog_reproduced"]
    assert payload["proof_obligation"]["verified"]
    assert payload["report_path"] == report.report_path


def test_stability_curve_is_deterministic(tmp_path):
    first = run_stability_sweep(run_dir=tmp_path / "first")
    second = run_stability_sweep(run_dir=tmp_path / "second")

    assert first.curve == second.curve
    assert first.predicted_baseline_boundary == second.predicted_baseline_boundary
    assert first.measured_baseline_boundary == second.measured_baseline_boundary
    assert first.measured_metriplectic_boundary == second.measured_metriplectic_boundary
    assert first.leapfrog_reproduced == second.leapfrog_reproduced


def test_multiseed_paired_verdict_and_registry_evidence(tmp_path):
    seeds = (11, 23, 37, 53, 71, 89, 107, 131)
    registry_path = tmp_path / "results" / "registry.jsonl"
    report = run_statistical_stability_sweep(
        seeds=seeds,
        bootstrap_samples=1_000,
        run_dir=tmp_path / "statistics",
        registry_path=registry_path,
    )

    report.assert_positive()
    assert report.stress_step_sizes == (0.5, 1.0)
    assert all(outcome.baseline_boundary == pytest.approx(0.5) for outcome in report.outcomes)
    assert all(outcome.metriplectic_boundary is None for outcome in report.outcomes)

    loss = report.endpoint_loss_comparison
    assert loss.mean_delta < 0.0
    assert loss.delta_ci_high < 0.0
    assert loss.t_p_value < 0.05
    assert loss.wilcoxon_p_value <= 0.05
    assert loss.n_favorable == loss.n_pairs == len(seeds)

    divergence = report.divergence_rate_comparison
    assert divergence.mean_delta == pytest.approx(-1.0)
    assert divergence.delta_ci_low == pytest.approx(-1.0)
    assert divergence.delta_ci_high == pytest.approx(-1.0)
    assert math.isinf(divergence.t_stat)
    assert divergence.t_p_value == pytest.approx(0.0)
    assert divergence.wilcoxon_p_value <= 0.05
    assert divergence.n_favorable == divergence.n_pairs == len(seeds)

    records = read_records(str(registry_path))
    assert len(records) == 2 * len(seeds)
    assert {record.seed for record in records} == set(seeds)
    assert all(record.harness == "eval" for record in records)
    assert all("paired_verdict=positive" in record.notes for record in records)
    assert all(
        set(record.metrics)
        == {"integrator_endpoint_loss", "integrator_divergence_rate"}
        for record in records
    )

    payload = json.loads((tmp_path / "statistics" / "statistics.json").read_text())
    assert payload["verdict"] == "positive"
    assert payload["divergence_rate_comparison"]["t_stat"] is None


def test_multiseed_statistics_are_deterministic(tmp_path):
    seeds = (11, 23, 37, 53, 71, 89)
    first = run_statistical_stability_sweep(
        seeds=seeds,
        bootstrap_samples=500,
        run_dir=tmp_path / "first",
    )
    second = run_statistical_stability_sweep(
        seeds=seeds,
        bootstrap_samples=500,
        run_dir=tmp_path / "second",
    )

    assert first.outcomes == second.outcomes
    assert first.endpoint_loss_comparison == second.endpoint_loss_comparison
    assert first.divergence_rate_comparison == second.divergence_rate_comparison
    assert first.verdict == second.verdict == "positive"


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (StabilitySweepConfig(step_sizes=(0.3,)), "integer multiple"),
        (StabilitySweepConfig(step_sizes=(0.5, 0.25)), "strictly increasing"),
        (
            StabilitySweepConfig(calcium0=(0.8,), buffer0=(0.2, 0.3), heat0=(0.0,)),
            "equally sized",
        ),
    ],
)
def test_stability_config_rejects_invalid_sweeps(cfg, message):
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_statistical_sweep_rejects_invalid_controls(tmp_path):
    with pytest.raises(ValueError, match="at least two unique"):
        run_statistical_stability_sweep(run_dir=tmp_path / "one", seeds=(11,))
    with pytest.raises(ValueError, match="at least two unique"):
        run_statistical_stability_sweep(run_dir=tmp_path / "duplicate", seeds=(11, 11))
    with pytest.raises(ValueError, match="bootstrap_samples must be positive"):
        run_statistical_stability_sweep(run_dir=tmp_path / "bootstrap", bootstrap_samples=0)
