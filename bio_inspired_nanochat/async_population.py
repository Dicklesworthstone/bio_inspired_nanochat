"""Async Distributed Population Evaluation Orchestrator (bead hea.2).

Decouples evolutionary / CMA-ES controller orchestration from worker execution.
Supports arbitrary population size (pop > num_workers), work-queue dispatching,
asynchronous future resolution, worker failure detection, and automatic requeue.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class CandidateTask:
    candidate_id: int
    params: Dict[str, float]
    generation: int
    retries: int = 0
    max_retries: int = 2


@dataclass
class EvaluationResult:
    candidate_id: int
    fitness: float
    duration_seconds: float
    worker_id: int
    status: str  # "success", "failed", "timeout"
    error: Optional[str] = None


class AsyncPopulationEvaluator:
    """Orchestrates parallel evaluation of candidate populations across workers."""

    def __init__(
        self,
        eval_fn: Callable[[Dict[str, float]], float],
        num_workers: int = 4,
        task_timeout_seconds: float = 30.0,
    ) -> None:
        self.eval_fn = eval_fn
        self.num_workers = max(1, num_workers)
        self.task_timeout_seconds = task_timeout_seconds

    def _execute_task(self, task: CandidateTask, worker_id: int) -> EvaluationResult:
        start_time = time.time()
        try:
            fitness = self.eval_fn(task.params)
            duration = time.time() - start_time
            return EvaluationResult(
                candidate_id=task.candidate_id,
                fitness=float(fitness),
                duration_seconds=duration,
                worker_id=worker_id,
                status="success",
            )
        except Exception as exc:
            duration = time.time() - start_time
            return EvaluationResult(
                candidate_id=task.candidate_id,
                fitness=float("inf"),
                duration_seconds=duration,
                worker_id=worker_id,
                status="failed",
                error=str(exc),
            )

    def evaluate_population(
        self,
        population: List[Dict[str, float]],
        generation: int = 0,
    ) -> List[EvaluationResult]:
        """Evaluate a full population concurrently across workers with automatic retry."""
        tasks = [
            CandidateTask(candidate_id=i, params=p, generation=generation)
            for i, p in enumerate(population)
        ]
        results_by_id: Dict[int, EvaluationResult] = {}
        pending_tasks = list(tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while pending_tasks:
                current_batch = list(pending_tasks)
                pending_tasks.clear()

                future_to_task = {
                    executor.submit(self._execute_task, task, worker_id=i % self.num_workers): task
                    for i, task in enumerate(current_batch)
                }

                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=self.task_timeout_seconds)
                        if result.status == "success":
                            results_by_id[result.candidate_id] = result
                        else:
                            # Retry if under retry limit
                            if task.retries < task.max_retries:
                                task.retries += 1
                                pending_tasks.append(task)
                            else:
                                results_by_id[task.candidate_id] = result
                    except concurrent.futures.TimeoutError:
                        if task.retries < task.max_retries:
                            task.retries += 1
                            pending_tasks.append(task)
                        else:
                            results_by_id[task.candidate_id] = EvaluationResult(
                                candidate_id=task.candidate_id,
                                fitness=float("inf"),
                                duration_seconds=self.task_timeout_seconds,
                                worker_id=-1,
                                status="timeout",
                                error="Task timed out",
                            )

        # Return results in candidate_id order
        return [results_by_id[i] for i in range(len(population))]
