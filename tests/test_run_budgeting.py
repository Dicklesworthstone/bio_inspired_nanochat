"""Unit tests for run budgeting and compute accounting (bead 2a7)."""

from __future__ import annotations

import time
from pathlib import Path

from bio_inspired_nanochat.budgeting import (
    RunBudgetTracker,
    load_budget_entries,
    log_run_cost,
)


def test_run_budget_tracker_and_logging(tmp_path: Path):
    """Test start/stop tracking and loading entries."""
    log_file = tmp_path / "budget.jsonl"

    tracker = RunBudgetTracker(
        run_id="test_run_1",
        purpose="unit_test_run",
        num_gpus=2,
        hourly_rate_usd=1.0,
        objective_type="proxy",
        log_path=log_file,
    ).start()

    time.sleep(0.05)
    entry = tracker.stop(extra={"batch_size": 32})

    assert entry.run_id == "test_run_1"
    assert entry.num_gpus == 2
    assert entry.gpu_hours > 0
    assert entry.estimated_cost_usd > 0
    assert entry.objective_type == "proxy"

    # Test direct log_run_cost helper
    entry2 = log_run_cost(
        run_id="test_run_2",
        purpose="bench_run",
        duration_seconds=3600.0,
        num_gpus=4,
        hourly_rate_usd=0.5,
        objective_type="full",
        log_path=log_file,
    )
    assert entry2.gpu_hours == 4.0
    assert entry2.estimated_cost_usd == 2.0

    # Test load entries
    loaded = load_budget_entries(log_file)
    assert len(loaded) == 2
    assert loaded[0].run_id == "test_run_1"
    assert loaded[1].run_id == "test_run_2"
