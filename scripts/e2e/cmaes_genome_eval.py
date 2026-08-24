"""Evaluation comparing NES / CMA-ES evolved Xi vs SGD-learned Xi on synthetic benchmarks (bead `hea.5`).

Runs multi-seed head-to-head comparison between:
1. NES Neuroevolution-optimized Xi genome
2. SGD-learned Xi genome
3. Random baseline Xi genome
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.cmaes_genome_optimizer import (
    EvolutionConfig,
    NESGenomeOptimizer,
)
from bio_inspired_nanochat.eval_stats import paired_comparison


@dataclass(frozen=True)
class GenomeEvalConfig:
    seeds: Tuple[int, ...] = (10, 20, 30, 40, 50)
    genome_dim: int = 8
    num_experts: int = 4
    generations: int = 15
    population_size: int = 12
    sgd_steps: int = 30
    sgd_lr: float = 0.05


@dataclass
class GenomeBenchmarkResult:
    nes_mean_fitness: float
    sgd_mean_fitness: float
    random_mean_fitness: float
    delta_nes_vs_sgd: float
    p_value: float
    passed: bool


def mock_fitness_function(xi: torch.Tensor, target_optimum: torch.Tensor) -> float:
    """Evaluates task fitness: negative distance to optimum with non-convex biophysical penalty."""
    dist = float(torch.norm(xi - target_optimum).item())
    # Add multimodal non-convex ripple
    ripple = float(torch.sin(3.0 * xi).sum().item()) * 0.1
    fitness = 1.0 / (1.0 + dist) + ripple
    return float(fitness)


def run_cmaes_genome_evaluation(config: GenomeEvalConfig) -> GenomeBenchmarkResult:
    """Run head-to-head evaluation between NES evolution and SGD genome optimization."""
    nes_fitnesses: Dict[int, float] = {}
    sgd_fitnesses: Dict[int, float] = {}
    random_fitnesses: Dict[int, float] = {}

    for seed in config.seeds:
        torch.manual_seed(seed)
        target = torch.randn(config.num_experts, config.genome_dim)

        def fit_fn(cand: torch.Tensor) -> float:
            c = cand.view(config.num_experts, config.genome_dim)
            return mock_fitness_function(c, target)

        # 1. Random baseline
        rand_xi = torch.randn(config.num_experts, config.genome_dim)
        random_fitnesses[seed] = fit_fn(rand_xi)

        # 2. NES Evolution
        evo_cfg = EvolutionConfig(
            population_size=config.population_size,
            generations=config.generations,
            seed=seed,
        )
        optimizer = NESGenomeOptimizer(
            genome_dim=config.num_experts * config.genome_dim,
            cfg=evo_cfg,
        )
        nes_res = optimizer.optimize(fit_fn)
        nes_fitnesses[seed] = nes_res.best_fitness

        # 3. SGD optimization
        sgd_param = nn.Parameter(torch.zeros(config.num_experts * config.genome_dim))
        sgd_opt = torch.optim.Adam([sgd_param], lr=config.sgd_lr)

        for _ in range(config.sgd_steps):
            sgd_opt.zero_grad()
            # Surrogate loss: quadratic distance to target + noise
            loss = ((sgd_param - target.view(-1)) ** 2).sum()
            loss.backward()
            sgd_opt.step()

        sgd_fitnesses[seed] = fit_fn(sgd_param.detach())

    # Paired comparison: NES vs SGD
    comp = paired_comparison(nes_fitnesses, sgd_fitnesses, lower_is_better=False)

    nes_mean = float(np.mean(list(nes_fitnesses.values())))
    sgd_mean = float(np.mean(list(sgd_fitnesses.values())))
    rand_mean = float(np.mean(list(random_fitnesses.values())))
    delta = comp.mean_delta if comp else (nes_mean - sgd_mean)
    p_val = comp.t_p_value if comp else 1.0

    passed = nes_mean >= rand_mean and nes_mean >= 0.8 * sgd_mean

    return GenomeBenchmarkResult(
        nes_mean_fitness=nes_mean,
        sgd_mean_fitness=sgd_mean,
        random_mean_fitness=rand_mean,
        delta_nes_vs_sgd=delta,
        p_value=p_val,
        passed=passed,
    )


def print_genome_benchmark(res: GenomeBenchmarkResult, console: Optional[Console] = None) -> None:
    """Print Rich table summarizing genome evolution benchmark."""
    c = console or Console()
    c.rule("[bold cyan]Xi-Genome Optimization Benchmark (NES Evolution vs SGD)[/bold cyan]")

    table = Table(title="Optimization Performance Comparison")
    table.add_column("Optimizer", style="bold")
    table.add_column("Mean Fitness", justify="right")
    table.add_column("Δ vs SGD", justify="right")
    table.add_column("Paired p-value", justify="right")

    table.add_row("Random Baseline", f"{res.random_mean_fitness:.4f}", f"{res.random_mean_fitness - res.sgd_mean_fitness:+.4f}", "—")
    table.add_row("SGD-Learned Genome", f"{res.sgd_mean_fitness:.4f}", "—", "—")
    table.add_row("NES-Evolved Genome", f"{res.nes_mean_fitness:.4f}", f"{res.delta_nes_vs_sgd:+.4f}", f"{res.p_value:.4f}")
    c.print(table)

    status_str = "[green]PASSED (Competitive Neuroevolutionary Optimizer)[/green]" if res.passed else "[red]FAILED[/red]"
    c.print(f"[bold]Verdict:[/bold] {status_str}\n")


def main() -> None:
    cfg = GenomeEvalConfig()
    res = run_cmaes_genome_evaluation(cfg)
    console = Console()
    print_genome_benchmark(res, console)


if __name__ == "__main__":
    main()
