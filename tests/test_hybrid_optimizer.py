"""Tests for Hybrid Bilevel Optimizer (bead `hea.4`)."""

import random

import torch
from torch import nn

from bio_inspired_nanochat.hybrid_optimizer import (
    BilevelResult,
    DiscreteConfig,
    HybridBilevelOptimizer,
)


class MockModel(nn.Module):
    def __init__(self, cfg: DiscreteConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.tensor([2.0, -1.0]))
        # Config bonus: normal_reparam and rank 16 give favorable inductive bias
        self.bias_bonus = 0.5 if cfg.stochastic_mode == "normal_reparam" else 1.0
        self.bias_bonus += 0.2 if cfg.rank_eligibility == 16 else 0.5

    def loss(self) -> float:
        # Distance to [0, 0] + discrete inductive bias penalty
        return float((self.weight**2).sum().item()) + self.bias_bonus


def test_discrete_config_mutation():
    """Verify that mutation stays within valid discrete sets."""
    rng = random.Random(42)
    cfg = DiscreteConfig()
    mutated = cfg.mutate(rng)

    assert mutated.stochastic_mode in ["normal_reparam", "bernoulli", "gumbel"]
    assert mutated.rank_eligibility in [4, 8, 16]
    assert mutated.attn_topk in [16, 32, 64]


def test_bilevel_optimization_progress():
    """Verify that inner SGD + outer evolution reduces validation loss."""
    def model_factory(cfg: DiscreteConfig) -> nn.Module:
        return MockModel(cfg)

    def train_fn(model: nn.Module, steps: int) -> float:
        assert isinstance(model, MockModel)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            loss_val = (model.weight**2).sum()
            loss_val.backward()
            opt.step()
        return model.loss()

    def eval_fn(model: nn.Module) -> float:
        assert isinstance(model, MockModel)
        return model.loss()

    optimizer = HybridBilevelOptimizer(
        model_factory=model_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        population_size=4,
        generations=3,
        inner_steps=5,
        seed=42,
    )

    result = optimizer.optimize()

    assert result.best_val_loss < result.initial_val_loss
    assert result.generations_run == 3
    assert len(result.history) == 3


def test_rich_table_logging():
    """Verify that log_results formats and outputs cleanly."""
    res = BilevelResult(
        best_discrete=DiscreteConfig(),
        best_val_loss=0.75,
        initial_val_loss=1.50,
        generations_run=2,
        population_size=4,
        inner_steps=5,
        history=[
            {"generation": 1, "best_loss": 1.10, "mean_loss": 1.30, "best_cfg": {"stochastic_mode": "normal_reparam", "rank_eligibility": 8, "attn_topk": 32}, "wall_time_ms": 10.0},
            {"generation": 2, "best_loss": 0.75, "mean_loss": 0.90, "best_cfg": {"stochastic_mode": "normal_reparam", "rank_eligibility": 16, "attn_topk": 32}, "wall_time_ms": 10.0},
        ],
        wall_time_ms=25.0,
    )
    optimizer = HybridBilevelOptimizer(lambda c: MockModel(c), lambda m, s: 0.0, lambda m: 0.0)
    optimizer.log_results(res)
