"""Tests for Headline Kinetics Ablation (bead `yw9.6`)."""


from scripts.e2e.kinetics_ablation_eval import (
    KineticsAblationConfig,
    print_ablation_summary,
    run_kinetics_ablation,
)


def test_kinetics_ablation_eval_pipeline():
    """Verify that multi-seed kinetics ablation runs cleanly and confirms learned advantage."""
    cfg = KineticsAblationConfig(
        seeds=(1, 2, 3),
        steps=5,
        batch_size=4,
        sequence_len=16,
        vocab_size=32,
        n_embd=16,
    )
    summary = run_kinetics_ablation(cfg)

    assert summary.learned_res.mean_loss < summary.default_res.mean_loss
    assert summary.passed
    assert summary.p_learned_vs_default < 0.05


def test_rich_ablation_summary_printing():
    """Verify that summary printing formats and renders cleanly."""
    cfg = KineticsAblationConfig(
        seeds=(1, 2),
        steps=2,
        batch_size=4,
        sequence_len=16,
        vocab_size=32,
        n_embd=16,
    )
    summary = run_kinetics_ablation(cfg)
    print_ablation_summary(summary)
