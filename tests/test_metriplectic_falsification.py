"""E2E falsification of clamped Euler versus the guarded metriplectic recurrence.

The harness uses the live torch integrator over a fixed physical horizon and compares both arms to
the analytic continuous flow. It must reproduce the explicit-Euler stability boundary while the
structure-preserving arm remains certified, fallback-free, and at least as accurate.

Run with:

    uv run python -m pytest tests/test_metriplectic_falsification.py -v
"""

from __future__ import annotations

import json

import pytest

from scripts.e2e.metriplectic_stability_curve import (
    StabilitySweepConfig,
    run_stability_sweep,
)

pytestmark = pytest.mark.e2e


def test_stability_curve_reproduces_the_predicted_leapfrog(tmp_path):
    report = run_stability_sweep(run_dir=tmp_path)

    report.assert_leapfrog()
    assert report.predicted_baseline_boundary == pytest.approx(0.5)
    assert report.measured_baseline_boundary == pytest.approx(0.5)
    assert report.measured_metriplectic_boundary is None

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
    expected_steps = 2 * sum(round(cfg.duration / dt) for dt in cfg.step_sizes)
    assert len(steps) == expected_steps
    assert {event["arm"] for event in steps} == {"baseline", "metriplectic"}
    assert len(points) == len(cfg.step_sizes)
    assert len(summaries) == 1 and summaries[0]["leapfrog_reproduced"]

    payload = json.loads((tmp_path / "stability_curve.json").read_text(encoding="utf-8"))
    assert payload["bead"] == "bio_inspired_nanochat-0642.1.3.1"
    assert payload["leapfrog_reproduced"]
    assert payload["report_path"] == report.report_path


def test_stability_curve_is_deterministic(tmp_path):
    first = run_stability_sweep(run_dir=tmp_path / "first")
    second = run_stability_sweep(run_dir=tmp_path / "second")

    assert first.curve == second.curve
    assert first.predicted_baseline_boundary == second.predicted_baseline_boundary
    assert first.measured_baseline_boundary == second.measured_baseline_boundary
    assert first.measured_metriplectic_boundary == second.measured_metriplectic_boundary
    assert first.leapfrog_reproduced == second.leapfrog_reproduced


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
