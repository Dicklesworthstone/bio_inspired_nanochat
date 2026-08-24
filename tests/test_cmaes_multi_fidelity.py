"""Tests for Multi-Fidelity, Multi-Seed & Composite Evaluator (bead `hea.1`)."""

import numpy as np

from bio_inspired_nanochat.multi_fidelity_evaluator import (
    FidelityLevel,
    MultiFidelityConfig,
    MultiFidelityEvaluator,
)


def mock_eval_fn(x: np.ndarray, seed: int, steps: int) -> tuple[float, float]:
    """Mock evaluation: base quadratic loss + seed noise + step bonus."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.05)
    loss = float(np.sum(x**2) + noise + 10.0 / steps)
    latency_ms = float(steps * 0.1)
    return loss, latency_ms


def test_multi_seed_evaluation_variance_reduction():
    """Verify that multi-seed evaluation aggregates across seeds properly."""
    evaluator = MultiFidelityEvaluator(mock_eval_fn, MultiFidelityConfig(seeds_high=3))
    x = np.array([0.5, -0.5])

    res = evaluator.evaluate_candidate(x, FidelityLevel.HIGH_FIDELITY, candidate_idx=1, base_seed=42)

    assert len(res.seeds_evaluated) == 3
    assert res.fidelity == FidelityLevel.HIGH_FIDELITY
    assert res.held_out_loss_std >= 0.0
    assert res.composite_score > res.held_out_loss_mean


def test_successive_halving_promotion():
    """Verify that population is filtered across successive halving stages."""
    evaluator = MultiFidelityEvaluator(mock_eval_fn)
    # 8 candidates: lower norm is better
    population = [np.array([i * 0.2, i * 0.2]) for i in range(8)]

    survivors = evaluator.evaluate_population_successive_halving(population, reduction_factor=2)

    # 8 -> 4 -> 2 survivors
    assert len(survivors) == 2
    assert survivors[0][1].fidelity == FidelityLevel.HIGH_FIDELITY
    # Best survivor should be index 0
    assert survivors[0][1].candidate_idx == 0


def test_composite_score_incorporates_latency_and_regularization():
    """Verify that composite score penalizes higher parameter norm and latency."""
    cfg = MultiFidelityConfig(alpha_latency=0.1, beta_regularization=0.1)
    evaluator = MultiFidelityEvaluator(mock_eval_fn, cfg)

    x_small = np.array([0.1, 0.1])
    x_large = np.array([5.0, 5.0])

    res_small = evaluator.evaluate_candidate(x_small, FidelityLevel.SCREENING)
    res_large = evaluator.evaluate_candidate(x_large, FidelityLevel.SCREENING)

    assert res_large.regularization_penalty > res_small.regularization_penalty
    assert res_large.composite_score > res_small.composite_score


def test_rich_table_rendering():
    """Verify that log_results formats and displays table cleanly."""
    evaluator = MultiFidelityEvaluator(mock_eval_fn)
    population = [np.array([0.1, 0.1]), np.array([0.5, 0.5])]
    survivors = evaluator.evaluate_population_successive_halving(population, reduction_factor=2)
    evaluator.log_results(survivors)
