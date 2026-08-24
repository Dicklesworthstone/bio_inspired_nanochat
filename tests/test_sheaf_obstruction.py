"""Fixed-sheaf obstruction score and calibrated threshold (r00r.5.1)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from bio_inspired_nanochat.sheaf_obstruction import (
    MVP_CERTIFICATE_KIND,
    fit_obstruction_calibrator,
    measure_sheaf_obstruction,
    reliability_diagram_svg,
)
from scripts.e2e.sheaf_obstruction_calibration import (
    SheafCalibrationConfig,
    run_sheaf_calibration,
)

pytestmark = pytest.mark.unit


def _ring(nodes: int) -> torch.Tensor:
    tail = torch.arange(nodes, dtype=torch.long)
    return torch.stack((tail, torch.roll(tail, shifts=-1)))


def test_consistent_section_has_zero_obstruction_and_honest_provenance() -> None:
    stalks = torch.tensor([[1.0, -0.5], [1.0, -0.5], [1.0, -0.5]])
    result = measure_sheaf_obstruction(stalks, _ring(3))

    assert result.available
    assert result.score == pytest.approx(0.0)
    assert result.quadratic_energy == pytest.approx(0.0)
    assert result.certificate_kind == MVP_CERTIFICATE_KIND
    assert result.h1_certified is False
    assert result.fallback_reason is None


def test_inconsistent_binding_scores_higher_and_is_scale_invariant() -> None:
    stalks = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    result = measure_sheaf_obstruction(stalks, _ring(3))
    scaled = measure_sheaf_obstruction(13.0 * stalks, _ring(3))

    assert result.score > 0.1
    assert result.normalized_residual > 0.0
    assert result.score == pytest.approx(scaled.score, rel=1e-6)
    assert max(result.edge_residual_norms) > 0.0


def test_edge_restrictions_can_certify_nonidentical_local_coordinates() -> None:
    stalks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    identity_result = measure_sheaf_obstruction(stalks, edge_index)

    tail_map = torch.tensor([[[1.0, 0.0]]])
    head_map = torch.tensor([[[0.0, 1.0]]])
    restricted_result = measure_sheaf_obstruction(
        stalks,
        edge_index,
        tail_restrictions=tail_map,
        head_restrictions=head_map,
    )

    assert identity_result.score > 0.0
    assert restricted_result.score == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("edge_index", "edge_weight", "reason"),
    [
        (torch.empty((2, 0), dtype=torch.long), None, "no_edges"),
        (torch.tensor([[0], [1]]), torch.zeros(1), "no_positive_weight_edges"),
    ],
)
def test_unassessable_graph_falls_back_neutrally(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    reason: str,
) -> None:
    result = measure_sheaf_obstruction(
        torch.ones((2, 3)),
        edge_index,
        edge_weight=edge_weight,
    )

    assert not result.available
    assert result.score == 0.0
    assert result.fallback_reason == reason
    assert not result.h1_certified


def test_invalid_graph_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        measure_sheaf_obstruction(torch.ones((2, 2)), torch.zeros((2, 1)))
    with pytest.raises(ValueError, match="outside"):
        measure_sheaf_obstruction(
            torch.ones((2, 2)), torch.tensor([[0], [2]], dtype=torch.long)
        )
    with pytest.raises(ValueError, match="supplied together"):
        measure_sheaf_obstruction(
            torch.ones((2, 2)),
            torch.tensor([[0], [1]], dtype=torch.long),
            tail_restrictions=torch.eye(2).unsqueeze(0),
        )
    with pytest.raises(ValueError, match="finite"):
        measure_sheaf_obstruction(
            torch.tensor([[math.nan, 0.0], [0.0, 1.0]]),
            torch.tensor([[0], [1]], dtype=torch.long),
        )


def test_calibrator_is_monotone_and_respects_false_positive_budget() -> None:
    negatives = [0.005 * index for index in range(1, 21)]
    positives = [0.4 + 0.02 * index for index in range(20)]
    scores = negatives + positives
    labels = [0] * len(negatives) + [1] * len(positives)
    calibrator = fit_obstruction_calibrator(
        scores,
        labels,
        target_false_positive_rate=0.2,
        probability_bins=8,
    )

    predicted = [calibrator.predict_probability(score) for score in sorted(scores)]
    assert predicted == sorted(predicted)
    assert calibrator.calibration_false_positive_rate <= 0.2
    assert calibrator.calibration_true_positive_rate == 1.0
    assert calibrator.threshold_protocol == "negative_only_split_conformal_quantile"
    assert calibrator.h1_certified is False

    evaluation = calibrator.evaluate(scores, labels, reliability_bins=5)
    assert evaluation.false_positive_rate <= 0.2
    assert evaluation.true_positive_rate == 1.0
    assert evaluation.sample_count == 40
    assert sum(point.count for point in evaluation.reliability_bins) == 40


def test_too_few_negatives_for_target_rate_fails_closed() -> None:
    calibrator = fit_obstruction_calibrator(
        [0.01, 0.02, 0.8, 0.9],
        [0, 0, 1, 1],
        target_false_positive_rate=0.1,
        probability_bins=2,
    )

    assert math.isinf(calibrator.threshold)
    assert calibrator.to_dict()["threshold"] is None
    assert not calibrator.is_flagged(1.0)
    assert calibrator.calibration_false_positive_rate == 0.0
    assert calibrator.calibration_true_positive_rate == 0.0


def test_reliability_diagram_is_auditable_svg() -> None:
    calibrator = fit_obstruction_calibrator(
        [0.01, 0.02, 0.03, 0.7, 0.8, 0.9],
        [0, 0, 0, 1, 1, 1],
        target_false_positive_rate=0.5,
        probability_bins=3,
    )
    evaluation = calibrator.evaluate(
        [0.015, 0.025, 0.75, 0.85],
        [0, 0, 1, 1],
        reliability_bins=4,
    )
    svg = reliability_diagram_svg(evaluation)

    assert svg.startswith("<svg")
    assert "Sheaf obstruction reliability" in svg
    assert "ECE=" in svg
    assert svg.count("<circle") == len(evaluation.reliability_bins)


@pytest.mark.e2e
def test_calibration_harness_writes_disjoint_split_evidence(tmp_path) -> None:
    report = run_sheaf_calibration(
        SheafCalibrationConfig(
            seed=7,
            calibration_examples=64,
            evaluation_examples=64,
        ),
        run_dir=tmp_path,
        verbose=False,
    )

    report.assert_passed()
    assert report.evaluation.false_positive_rate <= 0.15
    assert report.evaluation.true_positive_rate >= 0.9
    assert report.calibration["h1_certified"] is False
    assert "not an H^1 certificate" in " ".join(report.limitations)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["evaluation"]["sample_count"] == 64
    assert (tmp_path / "reliability.svg").read_text(encoding="utf-8").startswith("<svg")
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {event["event"] for event in events} >= {
        "sheaf_obstruction_calibration_fit",
        "sheaf_obstruction_reliability",
        "sheaf_obstruction_calibration_verdict",
    }
