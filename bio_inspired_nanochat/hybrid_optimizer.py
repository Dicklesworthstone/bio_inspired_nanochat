"""Hybrid Bilevel Optimizer: SGD for Differentiable, Evolution for Discrete (bead `hea.4`).

Implements a principled division of labor:
- Inner Loop: First-order SGD/Adam gradient descent on continuous weights and differentiable Xi kinetics.
- Outer Loop: Derivative-free evolutionary search over discrete/combinatorial structural hyperparameters.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from torch import nn


@dataclass(frozen=True)
class DiscreteConfig:
    """Categorical and integer structural hyperparameters searched by outer evolution."""

    stochastic_mode: str = "normal_reparam"  # "normal_reparam", "bernoulli", "gumbel"
    rank_eligibility: int = 8                # 4, 8, 16
    attn_topk: int = 32                      # 16, 32, 64

    def mutate(self, rng: random.Random) -> DiscreteConfig:
        """Apply random discrete point mutations."""
        modes = ["normal_reparam", "bernoulli", "gumbel"]
        ranks = [4, 8, 16]
        topks = [16, 32, 64]

        new_mode = rng.choice(modes) if rng.random() < 0.3 else self.stochastic_mode
        new_rank = rng.choice(ranks) if rng.random() < 0.3 else self.rank_eligibility
        new_topk = rng.choice(topks) if rng.random() < 0.3 else self.attn_topk

        return DiscreteConfig(
            stochastic_mode=new_mode,
            rank_eligibility=new_rank,
            attn_topk=new_topk,
        )


@dataclass
class BilevelResult:
    """Optimization summary for the hybrid bilevel search."""

    best_discrete: DiscreteConfig
    best_val_loss: float
    initial_val_loss: float
    generations_run: int
    population_size: int
    inner_steps: int
    history: list[dict[str, Any]]
    wall_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_discrete": asdict(self.best_discrete),
            "best_val_loss": float(self.best_val_loss),
            "initial_val_loss": float(self.initial_val_loss),
            "generations_run": self.generations_run,
            "population_size": self.population_size,
            "inner_steps": self.inner_steps,
            "history": self.history,
            "wall_time_ms": float(self.wall_time_ms),
        }


class HybridBilevelOptimizer:
    """Bilevel search engine coordinating outer evolutionary search with inner SGD training."""

    def __init__(
        self,
        model_factory: Callable[[DiscreteConfig], nn.Module],
        train_fn: Callable[[nn.Module, int], float],
        eval_fn: Callable[[nn.Module], float],
        population_size: int = 6,
        generations: int = 4,
        inner_steps: int = 10,
        seed: int = 42,
    ):
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.pop_size = population_size
        self.generations = generations
        self.inner_steps = inner_steps
        self.seed = seed
        self.rng = random.Random(seed)

    def optimize(self) -> BilevelResult:
        """Execute the bilevel optimization search loop."""
        t0 = time.perf_counter()

        # Initialize population with default and mutations
        default_cfg = DiscreteConfig()
        population: list[DiscreteConfig] = [default_cfg]
        for _ in range(self.pop_size - 1):
            population.append(default_cfg.mutate(self.rng))

        # Evaluate initial baseline before training
        init_model = self.model_factory(default_cfg)
        init_loss = self.eval_fn(init_model)

        best_loss = init_loss
        best_cfg = default_cfg
        history: list[dict[str, Any]] = []

        for gen in range(1, self.generations + 1):
            t_gen_0 = time.perf_counter()
            gen_evals: list[tuple[DiscreteConfig, float]] = []

            for cand in population:
                model = self.model_factory(cand)
                # Inner loop: first-order SGD optimization
                self.train_fn(model, self.inner_steps)
                # Validation evaluation
                val_loss = self.eval_fn(model)
                gen_evals.append((cand, val_loss))

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_cfg = cand

            gen_evals.sort(key=lambda item: item[1])
            mean_loss = float(np.mean([item[1] for item in gen_evals]))
            dt_gen = (time.perf_counter() - t_gen_0) * 1000.0

            history.append({
                "generation": gen,
                "best_loss": float(gen_evals[0][1]),
                "mean_loss": mean_loss,
                "best_cfg": asdict(gen_evals[0][0]),
                "wall_time_ms": dt_gen,
            })

            # Selection and reproduction for next generation (keep top-half + mutate)
            survivors = [item[0] for item in gen_evals[: max(1, self.pop_size // 2)]]
            new_pop = list(survivors)
            while len(new_pop) < self.pop_size:
                parent = self.rng.choice(survivors)
                new_pop.append(parent.mutate(self.rng))
            population = new_pop

        total_dt = (time.perf_counter() - t0) * 1000.0
        return BilevelResult(
            best_discrete=best_cfg,
            best_val_loss=best_loss,
            initial_val_loss=init_loss,
            generations_run=self.generations,
            population_size=self.pop_size,
            inner_steps=self.inner_steps,
            history=history,
            wall_time_ms=total_dt,
        )

    def log_results(self, result: BilevelResult, console: Console | None = None) -> None:
        """Render a formatted Rich table of bilevel optimization progress."""
        c = console or Console()
        c.rule("[bold cyan]Hybrid Bilevel Optimization Summary (SGD + Evolution)[/bold cyan]")
        c.print(
            f"Initial Loss: {result.initial_val_loss:.4f} → "
            f"Best Loss: [bold green]{result.best_val_loss:.4f}[/bold green] "
            f"(Δ = {result.best_val_loss - result.initial_val_loss:+.4f}) | "
            f"Wall-time: {result.wall_time_ms:.2f}ms"
        )
        c.print(f"Optimal Discrete Configuration: [bold]{result.best_discrete}[/bold]")

        table = Table(title="Bilevel Generation History")
        table.add_column("Gen", justify="right")
        table.add_column("Best Val Loss", justify="right", style="bold green")
        table.add_column("Mean Val Loss", justify="right", style="yellow")
        table.add_column("Best Mode")
        table.add_column("Rank / Top-k", justify="right")
        table.add_column("Latency (ms)", justify="right")

        for h in result.history:
            cfg = h["best_cfg"]
            table.add_row(
                str(h["generation"]),
                f"{h['best_loss']:.4f}",
                f"{h['mean_loss']:.4f}",
                cfg["stochastic_mode"],
                f"r={cfg['rank_eligibility']} / k={cfg['attn_topk']}",
                f"{h['wall_time_ms']:.1f}",
            )
        c.print(table)
