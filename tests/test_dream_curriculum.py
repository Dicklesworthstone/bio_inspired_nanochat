"""Tests for Curriculum-From-Dreams Engine (bead `re4e.15`)."""

import torch

from bio_inspired_nanochat.dream_curriculum import (
    CurriculumFromDreamsEngine,
    DreamCurriculumReport,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    return GPTSynaptic(cfg)


def test_targeted_dream_cycle_reduces_manifold_gaps():
    """Verify that curriculum dreaming identifies and fills persistent homology coverage gaps."""
    model = _make_model()
    engine = CurriculumFromDreamsEngine(candidate_multiplier=3)

    # Synthetic covered manifold (10 points in 16D space)
    covered_manifold = torch.randn(10, 16)

    selected_dreams, report = engine.run_targeted_dream_cycle(
        model=model,
        covered_manifold=covered_manifold,
        replay_batch_size=2,
    )

    assert selected_dreams.shape == (2, 8)
    assert report.total_candidates_dreamed == 6
    assert report.selected_for_replay == 2
    assert report.max_gap_after <= report.max_gap_before
    assert report.gap_reduction_ratio >= 0.0


def test_rich_table_logging():
    """Verify that log_report outputs cleanly."""
    engine = CurriculumFromDreamsEngine()
    rep = DreamCurriculumReport(
        total_candidates_dreamed=8,
        selected_for_replay=2,
        max_gap_before=1.50,
        max_gap_after=1.20,
        gap_reduction_ratio=0.20,
        consolidation_report={"status": "consolidated"},
        wall_time_ms=15.0,
    )
    engine.log_report(rep)
