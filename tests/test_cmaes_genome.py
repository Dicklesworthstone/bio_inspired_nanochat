"""Tests for NES / CMA-ES Xi-Genome Optimizer (bead `hea.5`)."""

import torch

from bio_inspired_nanochat.cmaes_genome_optimizer import (
    EvolutionConfig,
    NESGenomeOptimizer,
)
from scripts.e2e.cmaes_genome_eval import GenomeEvalConfig, run_cmaes_genome_evaluation


def test_nes_optimizer_fitness_improvement():
    """Verify that NES optimizer strictly increases fitness on a quadratic target."""
    target = torch.ones(8) * 0.5

    def fit_fn(x: torch.Tensor) -> float:
        return float(-torch.sum((x - target) ** 2).item())

    cfg = EvolutionConfig(population_size=8, generations=10, learning_rate_mu=0.2, seed=42)
    optimizer = NESGenomeOptimizer(genome_dim=8, cfg=cfg)

    res = optimizer.optimize(fit_fn)

    assert res.best_fitness > res.initial_fitness
    assert len(res.history) == 10


def test_nes_antithetic_sampling():
    """Verify that antithetic sampling produces zero-mean symmetric perturbation pairs."""
    cfg = EvolutionConfig(population_size=8, antithetic=True)
    optimizer = NESGenomeOptimizer(genome_dim=4, cfg=cfg)

    _, eps = optimizer.sample_population()
    assert eps.shape == (8, 4)
    assert torch.allclose(eps[:4] + eps[4:], torch.zeros(4, 4), atol=1e-6)


def test_e2e_cmaes_genome_eval():
    """Verify that the head-to-head evaluation pipeline runs cleanly and returns valid result."""
    cfg = GenomeEvalConfig(seeds=(1, 2), generations=5, population_size=6, sgd_steps=5)
    res = run_cmaes_genome_evaluation(cfg)

    assert res.nes_mean_fitness > res.random_mean_fitness
    assert res.passed
