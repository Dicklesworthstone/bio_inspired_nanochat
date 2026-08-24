"""Run Budgeting & Compute Cost Accounting (bead 2a7).

Logs and tracks GPU-hours, estimated dollar costs, objective type (proxy vs full),
and resource utilization across training, benchmarks, profiling, and HPO runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BUDGET_LOG_PATH = "runs/budget_accounting.jsonl"
DEFAULT_GPU_HOURLY_RATE = 0.75  # Approximate RTX 4090 cloud rate ($/hr)


@dataclass
class RunBudgetEntry:
    run_id: str
    purpose: str
    num_gpus: int
    duration_seconds: float
    gpu_hours: float
    hourly_rate_usd: float
    estimated_cost_usd: float
    objective_type: str  # "proxy" or "full"
    timestamp: float
    extra: dict[str, Any]


class RunBudgetTracker:
    """Tracks and logs runtime and estimated financial cost of experimental runs."""

    def __init__(
        self,
        run_id: str,
        purpose: str,
        num_gpus: int = 1,
        hourly_rate_usd: float = DEFAULT_GPU_HOURLY_RATE,
        objective_type: str = "full",
        log_path: str | Path = DEFAULT_BUDGET_LOG_PATH,
    ) -> None:
        self.run_id = run_id
        self.purpose = purpose
        self.num_gpus = max(1, num_gpus)
        self.hourly_rate_usd = float(hourly_rate_usd)
        self.objective_type = objective_type
        self.log_path = Path(log_path)
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> RunBudgetTracker:
        self._start_time = time.time()
        return self

    def stop(self, extra: dict[str, Any] | None = None) -> RunBudgetEntry:
        if self._start_time is None:
            self._start_time = time.time()
        self._end_time = time.time()
        duration = max(0.0, self._end_time - self._start_time)
        gpu_hours = (duration / 3600.0) * self.num_gpus
        est_cost = gpu_hours * self.hourly_rate_usd

        entry = RunBudgetEntry(
            run_id=self.run_id,
            purpose=self.purpose,
            num_gpus=self.num_gpus,
            duration_seconds=duration,
            gpu_hours=gpu_hours,
            hourly_rate_usd=self.hourly_rate_usd,
            estimated_cost_usd=est_cost,
            objective_type=self.objective_type,
            timestamp=self._end_time,
            extra=extra or {},
        )
        self._append_log(entry)
        return entry

    def _append_log(self, entry: RunBudgetEntry) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


def log_run_cost(
    run_id: str,
    purpose: str,
    duration_seconds: float,
    num_gpus: int = 1,
    hourly_rate_usd: float = DEFAULT_GPU_HOURLY_RATE,
    objective_type: str = "full",
    log_path: str | Path = DEFAULT_BUDGET_LOG_PATH,
    extra: dict[str, Any] | None = None,
) -> RunBudgetEntry:
    """Helper function to log a completed run's cost immediately."""
    gpu_hours = (duration_seconds / 3600.0) * max(1, num_gpus)
    est_cost = gpu_hours * hourly_rate_usd
    entry = RunBudgetEntry(
        run_id=run_id,
        purpose=purpose,
        num_gpus=max(1, num_gpus),
        duration_seconds=duration_seconds,
        gpu_hours=gpu_hours,
        hourly_rate_usd=hourly_rate_usd,
        estimated_cost_usd=est_cost,
        objective_type=objective_type,
        timestamp=time.time(),
        extra=extra or {},
    )
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
    return entry


def load_budget_entries(log_path: str | Path = DEFAULT_BUDGET_LOG_PATH) -> list[RunBudgetEntry]:
    """Load all logged budget entries from disk."""
    path = Path(log_path)
    if not path.exists():
        return []
    entries: list[RunBudgetEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid budget JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if isinstance(data, dict):
                entries.append(RunBudgetEntry(**data))
    return entries
