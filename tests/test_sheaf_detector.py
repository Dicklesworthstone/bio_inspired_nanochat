"""Unit, fallback, and AUROC tests for Sheaf Hallucination Detector (bead r00r.5)."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from bio_inspired_nanochat.sheaf_binding import SheafConsistencyMonitor
from bio_inspired_nanochat.sheaf_detector import (
    DetectorAction,
    SheafHallucinationDetector,
    log_hallucination_audit,
)


def test_sheaf_hallucination_detector_calibrates_and_flags():
    """SheafHallucinationDetector flags inconsistent bindings above threshold."""
    stalk_dim = 4
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian(
        edges=edges, num_nodes=num_nodes, stalk_dim=stalk_dim
    )

    detector = SheafHallucinationDetector(d_model=stalk_dim, threshold=0.1)

    # Consistent input: zero obstruction
    base = torch.randn(1, 1, stalk_dim)
    x_clean = base.repeat(1, num_nodes, 1)
    rep_clean = detector(x_clean, laplacian)
    assert not rep_clean.is_hallucination
    assert rep_clean.obstruction_score < 0.05

    # Inconsistent / hallucinated input
    x_hallucinated = torch.randn(1, num_nodes, stalk_dim)
    rep_hall = detector(x_hallucinated, laplacian)
    assert rep_hall.is_hallucination
    assert rep_hall.obstruction_score > 0.1


def test_sheaf_hallucination_detector_repair_action():
    """Repair action applies sheaf diffusion to project hallucinated activations to safe set."""
    stalk_dim = 4
    num_nodes = 3
    edges = [(0, 1), (1, 2)]
    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian(
        edges=edges, num_nodes=num_nodes, stalk_dim=stalk_dim
    )

    detector = SheafHallucinationDetector(
        d_model=stalk_dim,
        threshold=0.05,
        action=DetectorAction.REPAIR,
        num_repair_steps=8,
    )

    x_hallucinated = torch.randn(1, num_nodes, stalk_dim)
    rep = detector(x_hallucinated, laplacian)

    assert rep.is_hallucination
    assert rep.repaired_activations is not None

    e_repaired = detector.compute_obstruction_score(rep.repaired_activations, laplacian)
    assert e_repaired < rep.obstruction_score
    assert e_repaired < 0.25 * rep.obstruction_score


def test_sheaf_hallucination_detector_fallback_when_disabled(tmp_path: Path):
    """When disabled or when laplacian is None, detector falls back to safe no-op pass."""
    detector = SheafHallucinationDetector(d_model=4, enabled=False)
    x = torch.randn(1, 3, 4)
    rep = detector(x, laplacian=None)

    assert not rep.is_hallucination
    assert rep.obstruction_score == 0.0
    assert not rep.certificate.is_certified

    # Test audit logging
    log_path = tmp_path / "audit.jsonl"
    log_hallucination_audit(rep, jsonl_path=log_path)
    assert log_path.exists()


def test_detector_uses_whole_token_path_for_batched_activations():
    x = torch.tensor([[1., 2.], [-1., -2.], [1., 2.]])
    detector = SheafHallucinationDetector(2, threshold=0.1, action=DetectorAction.REPAIR)
    single = detector(x)
    batched = detector(x.unsqueeze(0))
    assert single.is_hallucination and batched.is_hallucination
    assert batched.obstruction_score == pytest.approx(single.obstruction_score)
    assert batched.repaired_activations is not None and single.repaired_activations is not None
    torch.testing.assert_close(batched.repaired_activations[0], single.repaired_activations)
    laplacian = SheafConsistencyMonitor.build_sheaf_laplacian([(0, 1), (1, 2)], num_nodes=3, stalk_dim=2)
    for scale in (0.01, 10.0):
        assert detector.compute_obstruction_score(x.unsqueeze(0), scale * laplacian) == pytest.approx(
            single.obstruction_score, abs=1e-6)
    mixed = detector(torch.stack([torch.ones_like(x), x]))
    assert mixed.obstruction_score == pytest.approx(single.obstruction_score / 2)
