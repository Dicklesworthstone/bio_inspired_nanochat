"""Multi-Fidelity, Multi-Seed & Composite Evaluation Engine (bead `hea.1`).

Provides ASHA/Hyperband-style multi-fidelity evaluation, multi-seed aggregation for
variance reduction, and composite objective scoring (quality + throughput + regularization)
for evolutionary and hyperparameter optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table



class FidelityLevel(int, Enum):
    """Evaluation fidelity stages for successive halving / ASHA scheduling."""

    SCREENING = 1
    REFINEMENT = 2
    HIGH_FIDELITY = 3


@dataclass(frozen=True)
class MultiFidelityConfig:
    """Hyperparameters and cost weights for multi-fidelity composite evaluation."""

    r_min: int = 15
    r_med: int = 40
    r_max: int = 100
    seeds_screening: int = 1
    seeds_refinement: int = 2
    seeds_high: int = 3
    alpha_latency: float = 0.01
    beta_regularization: float = 0.001
    base_latency_ms: float = 10.0

    def validate(self) -> None:
        if not (1 <= self.r_min <= self.r_med <= self.r_max):
            raise ValueError(f"Invalid steps progression: {self.r_min} <= {self.r_med} <= {self.r_max}")
        if self.seeds_screening < 1 or self.seeds_refinement < 1 or self.seeds_high < 1:
            raise ValueError("Seed counts must be >= 1")
        if self.alpha_latency < 0.0 or self.beta_regularization < 0.0:
            raise ValueError("Penalty weights must be non-negative")


@dataclass
class CompositeEvaluationResult:
    """Detailed score breakdown for a multi-fidelity evaluation."""

    candidate_idx: int
    fidelity: FidelityLevel
    composite_score: float
    held_out_loss_mean: float
    held_out_loss_std: float
    latency_mean_ms: float
    regularization_penalty: float
    seeds_evaluated: List[int]
    wall_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_idx": self.candidate_idx,
            "fidelity": self.fidelity.name,
            "composite_score": float(self.composite_score),
            "held_out_loss_mean": float(self.held_out_loss_mean),
            "held_out_loss_std": float(self.held_out_loss_std),
            "latency_mean_ms": float(self.latency_mean_ms),
            "regularization_penalty": float(self.regularization_penalty),
            "seeds_evaluated": self.seeds_evaluated,
            "wall_time_ms": float(self.wall_time_ms),
        }


class MultiFidelityEvaluator:
    """Executes multi-fidelity, multi-seed evaluation with composite objective computation."""

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray, int, int], Tuple[float, float]],
        cfg: Optional[MultiFidelityConfig] = None,
    ):
        """
        eval_fn: Callable(x, seed, steps) -> (held_out_loss, latency_ms)
        """
        self.eval_fn = eval_fn
        self.cfg = cfg or MultiFidelityConfig()
        self.cfg.validate()

    def evaluate_candidate(
        self,
        x: np.ndarray,
        fidelity: FidelityLevel,
        candidate_idx: int = 0,
        base_seed: int = 42,
    ) -> CompositeEvaluationResult:
        """Evaluate a single candidate at a specified fidelity level across multiple seeds."""
        t0 = time.perf_counter()

        if fidelity == FidelityLevel.SCREENING:
            steps = self.cfg.r_min
            n_seeds = self.cfg.seeds_screening
        elif fidelity == FidelityLevel.REFINEMENT:
            steps = self.cfg.r_med
            n_seeds = self.cfg.seeds_refinement
        else:
            steps = self.cfg.r_max
            n_seeds = self.cfg.seeds_high

        seeds = [base_seed + i * 1000 for i in range(n_seeds)]
        losses: List[float] = []
        latencies: List[float] = []

        for s in seeds:
            loss_val, lat_val = self.eval_fn(x, s, steps)
            losses.append(loss_val)
            latencies.append(lat_val)

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses)) if len(losses) > 1 else 0.0
        mean_lat = float(np.mean(latencies))

        # Composite score computation: Loss + Latency Penalty + Parameter Regularization
        reg_penalty = float(np.sum(x**2)) * self.cfg.beta_regularization
        lat_penalty = (mean_lat / max(1e-3, self.cfg.base_latency_ms)) * self.cfg.alpha_latency
        composite = mean_loss + lat_penalty + reg_penalty

        dt = (time.perf_counter() - t0) * 1000.0
        return CompositeEvaluationResult(
            candidate_idx=candidate_idx,
            fidelity=fidelity,
            composite_score=composite,
            held_out_loss_mean=mean_loss,
            held_out_loss_std=std_loss,
            latency_mean_ms=mean_lat,
            regularization_penalty=reg_penalty,
            seeds_evaluated=seeds,
            wall_time_ms=dt,
        )

    def evaluate_population_successive_halving(
        self,
        population: List[np.ndarray],
        base_seed: int = 42,
        reduction_factor: int = 2,
    ) -> List[Tuple[np.ndarray, CompositeEvaluationResult]]:
        """Run Successive Halving across SCREENING -> REFINEMENT -> HIGH_FIDELITY stages."""
        current_pop = [(idx, cand) for idx, cand in enumerate(population)]

        # Stage 1: Screening
        results_s1 = [
            (cand, self.evaluate_candidate(cand, FidelityLevel.SCREENING, idx, base_seed))
            for idx, cand in current_pop
        ]
        results_s1.sort(key=lambda item: item[1].composite_score)
        n_surv_1 = max(1, len(results_s1) // reduction_factor)
        current_pop = [(item[1].candidate_idx, item[0]) for item in results_s1[:n_surv_1]]

        # Stage 2: Refinement
        results_s2 = [
            (cand, self.evaluate_candidate(cand, FidelityLevel.REFINEMENT, idx, base_seed))
            for idx, cand in current_pop
        ]
        results_s2.sort(key=lambda item: item[1].composite_score)
        n_surv_2 = max(1, len(results_s2) // reduction_factor)
        current_pop = [(item[1].candidate_idx, item[0]) for item in results_s2[:n_surv_2]]

        # Stage 3: High Fidelity
        results_s3 = [
            (cand, self.evaluate_candidate(cand, FidelityLevel.HIGH_FIDELITY, idx, base_seed))
            for idx, cand in current_pop
        ]
        results_s3.sort(key=lambda item: item[1].composite_score)

        return results_s3

    def log_results(
        self,
        results: List[Tuple[np.ndarray, CompositeEvaluationResult]],
        console: Optional[Console] = None,
    ) -> None:
        """Render a formatted Rich table of surviving multi-fidelity candidates."""
        c = console or Console()
        c.rule("[bold cyan]Multi-Fidelity Composite Evaluation Summary[/bold cyan]")

        table = Table(title="Surviving Candidates (Rank-Ordered by Composite Score)")
        table.add_column("Rank", justify="right")
        table.add_column("Cand ID", justify="right", style="cyan")
        table.add_column("Fidelity", style="bold")
        table.add_column("Composite Score", justify="right", style="bold green")
        table.add_column("Held-Out Loss (Mean ± Std)", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Reg Penalty", justify="right")

        for rank, (_, res) in enumerate(results, start=1):
            table.add_row(
                str(rank),
                str(res.candidate_idx),
                res.fidelity.name,
                f"{res.composite_score:.4f}",
                f"{res.held_out_loss_mean:.4f} ± {res.held_out_loss_std:.4f}",
                f"{res.latency_mean_ms:.2f}",
                f"{res.regularization_penalty:.5f}",
            )
        c.print(table)
