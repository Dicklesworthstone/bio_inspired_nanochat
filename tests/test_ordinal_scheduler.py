"""Unit tests for the Ordinal Learning Rate Scheduler (bead vap.5)."""

from __future__ import annotations

import torch
import torch.nn as nn

from bio_inspired_nanochat.ordinal_scheduler import OrdinalLRScheduler


def test_ordinal_scheduler_warmup_plateau_and_restart():
    """Ordinal scheduler steps through warmup, patience, cosine decay, and restart."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = OrdinalLRScheduler(
        optimizer,
        total_steps=500,
        cycle_steps=100,
        warmup_fraction=0.1,   # 10 steps
        patience_fraction=0.3, # 30 steps
        min_lr_ratio=0.1,
        restart_decay=0.8,
    )

    lrs = []
    for _ in range(250):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Step 0: start at min_lr
    assert lrs[0] < 1e-3

    # Step 10-39: peak plateau (1e-3)
    assert abs(lrs[15] - 1e-3) < 1e-6
    assert abs(lrs[35] - 1e-3) < 1e-6

    # Step 99: decayed near min_lr
    assert lrs[99] < 2e-4

    # Step 100: restart scaled by 0.8
    assert abs(lrs[115] - 8e-4) < 1e-6
