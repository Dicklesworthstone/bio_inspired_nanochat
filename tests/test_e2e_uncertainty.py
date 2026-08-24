"""End-to-end uncertainty/calibration evidence contract (bead ``eqyk.20``)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from bio_inspired_nanochat.torch_imports import torch
from scripts.e2e.stochastic_thermo_uq import ExperimentConfig
from scripts.e2e.uncertainty_calibration import (
    UncertaintyE2EConfig,
    run_uncertainty_e2e,
)

pytestmark = pytest.mark.e2e


def _small_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        vocab_size=16,
        seq_len=6,
        batch_size=2,
        pool_size=2,
        eval_pool_size=2,
        train_steps=2,
        n_head=1,
        n_embd=16,
        dropout=0.15,
        mc_samples=4,
        ece_bins=4,
        ft_trajectories=30_000,
        ft_min_count=40,
        ft_tolerance=0.35,
        ft_integral_tolerance=0.07,
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_uncertainty_e2e_logs_metrics_actions_and_thermodynamic_evidence(
    tmp_path: Path,
) -> None:
    config = UncertaintyE2EConfig(experiment=_small_experiment())
    report = run_uncertainty_e2e(config, run_dir=tmp_path, verbose=False)

    report.assert_passed()
    assert report.passed
    assert len(report.invariants) == 8

    events_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "validation-summary.json"
    assert events_path.is_file()
    assert summary_path.is_file()

    events = _read_events(events_path)
    prediction_events = [
        event for event in events if event["event"] == "uncertainty_prediction_batch"
    ]
    assert len(prediction_events) == 6
    assert {
        (event["method"], event["split"])
        for event in prediction_events
    } == {
        (method, split)
        for method in ("softmax_entropy", "mc_dropout", "thermo_uq")
        for split in ("id", "ood")
    }
    for event in prediction_events:
        probabilities = torch.tensor(event["predictive_distribution"])
        uncertainty = torch.tensor(event["predictive_entropy"])
        variance = torch.tensor(event["predictive_variance"])
        assert probabilities.shape[:-1] == uncertainty.shape == variance.shape
        assert torch.isfinite(probabilities).all()
        assert torch.isfinite(uncertainty).all()
        assert torch.isfinite(variance).all()
        assert torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(uncertainty),
            atol=1e-5,
            rtol=1e-5,
        )
        assert variance.ge(0.0).all()

    thermo_id = next(
        event
        for event in prediction_events
        if event["method"] == "thermo_uq" and event["split"] == "id"
    )
    assert thermo_id["nonzero_variance_count"] > 0

    method_events = [
        event for event in events if event["event"] == "uncertainty_calibration_method"
    ]
    assert len(method_events) == 3
    for event in method_events:
        assert 0.0 <= event["ece"] <= 1.0
        assert 0.0 <= event["ood_auroc"] <= 1.0
        assert event["calibration_curve"]
        assert event["risk_coverage_curve"]

    action = next(
        event for event in events if event["event"] == "uncertainty_decoding_action"
    )
    assert action["action"] == "abstain"
    assert action["policy"] == "uncertainty_decode_action"
    assert action["abstained"] > 0
    assert action["accepted"] < action["full_accepted"]
    assert action["selected_risk"] <= action["full_risk"]

    thermodynamic = next(
        event
        for event in events
        if event["event"] == "uncertainty_thermodynamic_residual"
    )
    assert math.isfinite(thermodynamic["integral_ft_residual"])
    assert math.isfinite(thermodynamic["max_crooks_residual"])
    evidence = thermodynamic["predictive_thermo_evidence"]
    assert evidence["observed_events"] == evidence["tested_events"] > 0

    validation = json.loads(summary_path.read_text(encoding="utf-8"))
    assert validation["schema_version"] == 1
    assert validation["strict"] is True
    assert validation["validation_suite"] == "bio_inspired_nanochat-eqyk.16"
    assert validation["subsystem"] == "uncertainty_calibration"
    assert validation["bead"] == "bio_inspired_nanochat-eqyk.20"
    assert validation["run_id"] == report.run_id
    assert validation["passed"] is True
    assert all(invariant["passed"] for invariant in validation["invariants"])


@pytest.mark.parametrize("target_coverage", [float("nan"), 0.0, 1.0])
def test_uncertainty_e2e_rejects_invalid_target_coverage(
    target_coverage: float,
) -> None:
    with pytest.raises(ValueError, match="target_coverage"):
        UncertaintyE2EConfig(
            experiment=_small_experiment(),
            target_coverage=target_coverage,
        ).validate()
