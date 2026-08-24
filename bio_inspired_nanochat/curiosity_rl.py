"""Intrinsic-Motivation RL via Neuromodulated Curiosity (bead `r00r.12`).

Implements intrinsic curiosity rewards derived from prediction error, free-energy reduction (r00r.1),
and norepinephrine (NE) novelty signals to drive exploratory reinforcement learning on sparse-reward tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.neuromod import NeuromodConfig, NeuromodulatoryBus


@dataclass(frozen=True)
class CuriosityConfig:
    """Hyperparameters for intrinsic motivation and free-energy curiosity rewards."""

    curiosity_weight: float = 0.25
    free_energy_weight: float = 0.15
    novelty_weight: float = 0.10
    ema_decay: float = 0.95


@dataclass
class StepRewardBreakdown:
    """Decomposition of extrinsic, intrinsic, and composite reward signals."""

    step: int
    extrinsic_reward: float
    intrinsic_curiosity: float
    free_energy_penalty: float
    composite_reward: float
    dopamine_level: float
    norepinephrine_level: float


class CuriosityRewardEngine:
    """Computes free-energy and novelty-based intrinsic curiosity rewards for RL."""

    def __init__(self, cfg: Optional[CuriosityConfig] = None):
        self.cfg = cfg or CuriosityConfig()
        self.bus = NeuromodulatoryBus(NeuromodConfig(enabled=True))
        self.running_mean_surprise = 0.0

    def compute_intrinsic_reward(
        self,
        token_loss: float,
        hidden_states: Tensor,
    ) -> Tuple[float, float]:
        """Compute intrinsic curiosity reward and free energy penalty from trajectory signals."""
        # 1. Prediction error surprise
        surprise = float(token_loss)
        self.running_mean_surprise = (
            self.cfg.ema_decay * self.running_mean_surprise + (1.0 - self.cfg.ema_decay) * surprise
        )
        novelty_bonus = max(0.0, surprise - self.running_mean_surprise)

        # 2. Latent Lyapunov Free Energy: E(h) = 0.5 * ||h||^2 - logsumexp(h)
        with torch.no_grad():
            h_flat = hidden_states.detach().float().view(-1, hidden_states.shape[-1])
            fe = 0.5 * torch.norm(h_flat, dim=-1).mean().item()

        intrinsic_r = (
            self.cfg.novelty_weight * novelty_bonus + self.cfg.free_energy_weight * (1.0 / (1.0 + fe))
        )
        return float(intrinsic_r), float(fe)

    def step(
        self,
        step_idx: int,
        extrinsic_reward: float,
        token_loss: float,
        hidden_states: Tensor,
    ) -> StepRewardBreakdown:
        """Combine extrinsic and intrinsic rewards and update the neuromodulatory bus."""
        int_r, fe = self.compute_intrinsic_reward(token_loss, hidden_states)
        total_r = extrinsic_reward + self.cfg.curiosity_weight * int_r

        self.bus.update(reward=total_r, entropy=fe, loss=token_loss)
        levels = self.bus.levels()

        return StepRewardBreakdown(
            step=step_idx,
            extrinsic_reward=float(extrinsic_reward),
            intrinsic_curiosity=float(int_r),
            free_energy_penalty=float(fe),
            composite_reward=float(total_r),
            dopamine_level=float(levels["da"]),
            norepinephrine_level=float(levels["ne"]),
        )

    def log_trace(
        self,
        history: List[StepRewardBreakdown],
        console: Optional[Console] = None,
    ) -> None:
        """Render Rich table of curiosity reward progression."""
        c = console or Console()
        c.rule("[bold cyan]Neuromodulated Curiosity RL Step Progression[/bold cyan]")

        table = Table(title="Intrinsic Curiosity & Neuromodulator Trace")
        table.add_column("Step", justify="right")
        table.add_column("Extrinsic R", justify="right")
        table.add_column("Curiosity R", justify="right", style="cyan")
        table.add_column("Composite R", justify="right", style="bold green")
        table.add_column("DA Level", justify="right", style="yellow")
        table.add_column("NE Level", justify="right", style="magenta")

        for h in history:
            table.add_row(
                str(h.step),
                f"{h.extrinsic_reward:.3f}",
                f"{h.intrinsic_curiosity:.3f}",
                f"{h.composite_reward:.3f}",
                f"{h.dopamine_level:.3f}",
                f"{h.norepinephrine_level:.3f}",
            )
        c.print(table)
