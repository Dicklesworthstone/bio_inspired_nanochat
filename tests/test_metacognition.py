"""Tests for the Metacognitive Self-Model Layer (beads `re4e.2`, `re4e.2.1`)."""

import numpy as np
import torch

from bio_inspired_nanochat.metacognition import (
    EpistemicStatus,
    MetacognitionConfig,
    MetacognitiveSelfModel,
    SpanCompetenceReport,
)


def test_epistemic_tri_state_classification():
    """Verify that low energy/obstruction yields KNOWN, and high yields UNKNOWN."""
    model = MetacognitiveSelfModel()

    # 1. High competence (zero energy, zero obstruction, low entropy)
    h_known = torch.ones(4, 32)
    logits_known = torch.zeros(4, 100)
    logits_known[:, 0] = 20.0 # sharp peak -> low entropy
    rep_known = model.assess_span(
        span_tokens=[1, 2, 3, 4],
        hidden_states=h_known,
        logits=logits_known,
        sheaf_obstruction=0.0,
    )
    assert rep_known.status == EpistemicStatus.KNOWN
    assert rep_known.competence_score >= 0.70

    # 2. Low competence / high uncertainty (high energy, high obstruction, high entropy)
    h_unknown = torch.randn(4, 32) * 5.0
    logits_unknown = torch.zeros(4, 100) # uniform -> high entropy
    rep_unknown = model.assess_span(
        span_tokens=[5, 6, 7, 8],
        hidden_states=h_unknown,
        logits=logits_unknown,
        sheaf_obstruction=0.9,
    )
    assert rep_unknown.status == EpistemicStatus.UNKNOWN
    assert rep_unknown.competence_score < 0.35


def test_free_energy_and_sheaf_obstruction_monotone_impact():
    """Verify that increasing free energy and sheaf obstruction strictly decreases competence."""
    model = MetacognitiveSelfModel()
    h = torch.ones(4, 32)
    logits = torch.zeros(4, 50)
    logits[:, 1] = 10.0

    rep1 = model.assess_span([1, 2, 3, 4], h, logits, sheaf_obstruction=0.1)
    rep2 = model.assess_span([1, 2, 3, 4], h, logits, sheaf_obstruction=0.8)

    assert rep1.competence_score > rep2.competence_score


def test_sequence_tiling_span_reports():
    """Verify sequence tiling and report generation across multiple spans."""
    model = MetacognitiveSelfModel(MetacognitionConfig(span_size=3))
    tokens = [10, 11, 12, 13, 14, 15, 16]
    hidden = torch.randn(7, 32)
    logits = torch.randn(7, 64)

    reports = model.assess_sequence(tokens, hidden, logits, span_size=3)
    assert len(reports) == 3
    assert reports[0].span_start == 0 and reports[0].span_end == 3
    assert reports[1].span_start == 3 and reports[1].span_end == 6
    assert reports[2].span_start == 6 and reports[2].span_end == 7


def test_calibration_ece_and_auroc_computation():
    """Verify ECE and AUROC computation on known synthetic binary ground truth."""
    # Perfect predictor
    scores = np.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])

    auroc = MetacognitiveSelfModel.compute_auroc(scores, labels)
    ece = MetacognitiveSelfModel.compute_ece(scores, labels, n_bins=5)

    assert auroc == 1.0
    assert ece < 0.15


def test_rich_table_logging():
    """Verify that report logging executes without raising an exception."""
    model = MetacognitiveSelfModel()
    reports = [
        SpanCompetenceReport(0, 2, [1, 2], 0.85, 0.05, 0.01, 0.1, EpistemicStatus.KNOWN, "ok"),
        SpanCompetenceReport(2, 4, [3, 4], 0.50, 0.30, 0.20, 0.4, EpistemicStatus.GUESSING, "extrapolating"),
        SpanCompetenceReport(4, 6, [5, 6], 0.20, 0.80, 0.60, 0.8, EpistemicStatus.UNKNOWN, "high barrier"),
    ]
    model.log_reports(reports)
