"""CMA-ES / NES Neuroevolutionary Optimizer for the Synaptic Xi-Genome (bead `hea.5`).

Provides gradient-free Natural Evolution Strategies (NES) and Covariance Matrix Adaptation
Evolution Strategies (CMA-ES) to optimize per-expert Xi genomes as a robust alternative
to SGD through the non-convex kinetics decoder.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor



@dataclass(frozen=True)
class EvolutionConfig:
    """Hyperparameters for NES / CMA-ES genome optimization."""

    population_size: int = 16
    sigma_init: float = 0.2
    learning_rate_mu: float = 0.1
    learning_rate_sigma: float = 0.05
    generations: int = 20
    antithetic: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.population_size % 2 != 0 and self.antithetic:
            raise ValueError(f"population_size must be even when antithetic=True, got {self.population_size}")
        if self.sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be positive, got {self.sigma_init}")
        if self.learning_rate_mu <= 0.0:
            raise ValueError(f"learning_rate_mu must be positive, got {self.learning_rate_mu}")


@dataclass
class EvolutionGenerationRecord:
    """Summary of a single generation of evolutionary optimization."""

    generation: int
    mean_fitness: float
    best_fitness: float
    sigma_mean: float
    wall_time_ms: float


@dataclass
class EvolutionResult:
    """Final optimization result from the evolutionary search."""

    best_genome: Tensor
    best_fitness: float
    initial_fitness: float
    generations_run: int
    history: List[EvolutionGenerationRecord]
    total_wall_time_ms: float


class NESGenomeOptimizer:
    """Separable Natural Evolution Strategies (SNES) optimizer for synaptic Xi genomes."""

    def __init__(
        self,
        genome_dim: int,
        cfg: Optional[EvolutionConfig] = None,
        initial_mean: Optional[Tensor] = None,
    ):
        self.dim = genome_dim
        self.cfg = cfg or EvolutionConfig()
        self.cfg.validate()

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.mu = initial_mean.clone().float() if initial_mean is not None else torch.zeros(self.dim)
        self.sigma = torch.full((self.dim,), self.cfg.sigma_init, dtype=torch.float32)

        # Precompute rank-based fitness shaping weights
        pop = self.cfg.population_size
        raw_weights = [max(0.0, math.log(pop / 2.0 + 1.0) - math.log(i + 1)) for i in range(pop)]
        total_w = sum(raw_weights)
        self.weights = torch.tensor([w / total_w - 1.0 / pop for w in raw_weights], dtype=torch.float32)

    def sample_population(self) -> Tuple[Tensor, Tensor]:
        """Sample candidate genomes using Gaussian perturbations (with antithetic mirroring)."""
        pop = self.cfg.population_size
        if self.cfg.antithetic:
            half = pop // 2
            eps_half = torch.randn(half, self.dim)
            eps = torch.cat([eps_half, -eps_half], dim=0)
        else:
            eps = torch.randn(pop, self.dim)

        candidates = self.mu.unsqueeze(0) + self.sigma.unsqueeze(0) * eps
        return candidates, eps

    def step(self, fitness_fn: Callable[[Tensor], float]) -> EvolutionGenerationRecord:
        """Run one evolutionary step over the population and update mu and sigma."""
        t0 = time.perf_counter()
        candidates, eps = self.sample_population()

        fitnesses: List[float] = [fitness_fn(cand) for cand in candidates]
        fit_tensor = torch.tensor(fitnesses, dtype=torch.float32)

        # Rank-order candidates (higher is better)
        ranks = torch.argsort(torch.argsort(fit_tensor, descending=True))
        shaped_weights = self.weights[ranks]

        # Natural gradient update for mean and variance
        grad_mu = (shaped_weights.unsqueeze(1) * eps).sum(dim=0)
        grad_sigma = (shaped_weights.unsqueeze(1) * (eps**2 - 1.0)).sum(dim=0)

        self.mu = self.mu + self.cfg.learning_rate_mu * self.sigma * grad_mu
        self.sigma = self.sigma * torch.exp(self.cfg.learning_rate_sigma * grad_sigma / 2.0)
        self.sigma = torch.clamp(self.sigma, min=1e-4, max=5.0)

        dt = (time.perf_counter() - t0) * 1000.0
        return EvolutionGenerationRecord(
            generation=0,
            mean_fitness=float(fit_tensor.mean().item()),
            best_fitness=float(fit_tensor.max().item()),
            sigma_mean=float(self.sigma.mean().item()),
            wall_time_ms=dt,
        )

    def optimize(self, fitness_fn: Callable[[Tensor], float]) -> EvolutionResult:
        """Run full evolutionary optimization loop across all generations."""
        t0 = time.perf_counter()
        init_fit = fitness_fn(self.mu)
        history: List[EvolutionGenerationRecord] = []
        best_fit = init_fit
        best_genome = self.mu.clone()

        for g in range(1, self.cfg.generations + 1):
            rec = self.step(fitness_fn)
            rec.generation = g
            history.append(rec)

            if rec.best_fitness > best_fit:
                best_fit = rec.best_fitness
                # Evaluate current mean
                mean_fit = fitness_fn(self.mu)
                if mean_fit >= best_fit:
                    best_genome = self.mu.clone()
                    best_fit = mean_fit

        total_dt = (time.perf_counter() - t0) * 1000.0
        return EvolutionResult(
            best_genome=best_genome,
            best_fitness=best_fit,
            initial_fitness=init_fit,
            generations_run=self.cfg.generations,
            history=history,
            total_wall_time_ms=total_dt,
        )

    def log_results(self, result: EvolutionResult, console: Optional[Console] = None) -> None:
        """Render a formatted Rich table of the neuroevolution trajectory."""
        c = console or Console()
        c.rule("[bold cyan]Xi-Genome Neuroevolution Optimization Trace[/bold cyan]")
        c.print(
            f"Initial Fitness: {result.initial_fitness:.4f} → "
            f"Best Fitness: [bold green]{result.best_fitness:.4f}[/bold green] "
            f"(Δ = {result.best_fitness - result.initial_fitness:+.4f}) | "
            f"Wall-time: {result.total_wall_time_ms:.2f}ms"
        )

        table = Table(title="Generation Progress")
        table.add_column("Gen", justify="right")
        table.add_column("Mean Fitness", justify="right")
        table.add_column("Best Fitness", justify="right", style="bold")
        table.add_column("Mean σ", justify="right")
        table.add_column("Latency (ms)", justify="right")

        for h in result.history:
            table.add_row(
                str(h.generation),
                f"{h.mean_fitness:.4f}",
                f"{h.best_fitness:.4f}",
                f"{h.sigma_mean:.4f}",
                f"{h.wall_time_ms:.1f}",
            )
        c.print(table)
