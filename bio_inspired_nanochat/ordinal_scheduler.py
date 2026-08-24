"""Ordinal Learning Rate Scheduler (bead vap.5).

Implements the multi-stage deterministic restart ladder (patience -> anneal -> restart,
omega^2 A + omega B + C) transferred from model_guided_research for reproducible training.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch.optim.lr_scheduler import LRScheduler


class OrdinalLRScheduler(LRScheduler):
    """Ordinal multi-stage learning rate scheduler with deterministic restarts.

    Divides the total training horizon into discrete ordinal cycles.
    Each cycle follows:
      1. Warmup (linear increase to peak LR)
      2. Plateau / Patience (constant peak LR)
      3. Cosine Annealing (decay to min LR)
      4. Deterministic Restart with geometric decay factor
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        cycle_steps: int = 1000,
        warmup_fraction: float = 0.05,
        patience_fraction: float = 0.20,
        min_lr_ratio: float = 0.05,
        restart_decay: float = 0.90,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.cycle_steps = max(10, cycle_steps)
        self.warmup_fraction = float(warmup_fraction)
        self.patience_fraction = float(patience_fraction)
        self.min_lr_ratio = float(min_lr_ratio)
        self.restart_decay = float(restart_decay)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < 0:
            return self.base_lrs

        cycle_idx = step // self.cycle_steps
        step_in_cycle = step % self.cycle_steps
        cycle_scale = self.restart_decay**cycle_idx

        warmup_steps = int(self.cycle_steps * self.warmup_fraction)
        patience_steps = int(self.cycle_steps * self.patience_fraction)
        decay_steps = self.cycle_steps - warmup_steps - patience_steps

        if step_in_cycle < warmup_steps:
            # Linear warmup
            progress = step_in_cycle / max(1, warmup_steps)
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * progress
        elif step_in_cycle < (warmup_steps + patience_steps):
            # Patience plateau
            factor = 1.0
        else:
            # Cosine decay to min_lr
            decay_progress = (step_in_cycle - warmup_steps - patience_steps) / max(1, decay_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        effective_factor = factor * cycle_scale
        return [base_lr * effective_factor for base_lr in self.base_lrs]
