"""Tests for Continual Learning & Catastrophic Forgetting Benchmark (bead `cel.4`)."""


from scripts.e2e.continual_learning_eval import (
    ContinualEvalConfig,
    print_continual_summary,
    run_continual_benchmark,
)


def test_continual_learning_benchmark_pipeline():
    """Verify that continual learning evaluation executes across tasks and confirms retention."""
    cfg = ContinualEvalConfig(
        seeds=(1, 2, 3),
        steps_per_task=3,
        batch_size=2,
        sequence_len=8,
        vocab_size=32,
        n_embd=16,
        sleep_steps=2,
    )
    summary = run_continual_benchmark(cfg)

    assert summary.bio_with_sleep_res.mean_acc_a > summary.vanilla_res.mean_acc_a
    assert summary.passed
    assert summary.p_sleep_vs_vanilla < 0.05


def test_rich_continual_summary_printing():
    """Verify that summary printing formats and outputs cleanly."""
    cfg = ContinualEvalConfig(
        seeds=(1, 2),
        steps_per_task=2,
        batch_size=2,
        sequence_len=8,
        vocab_size=32,
        n_embd=16,
        sleep_steps=1,
    )
    summary = run_continual_benchmark(cfg)
    print_continual_summary(summary)
