"""Unit tests for the async distributed population evaluation orchestrator (bead hea.2)."""

from __future__ import annotations


from bio_inspired_nanochat.async_population import AsyncPopulationEvaluator


def test_async_population_evaluates_larger_than_worker_count():
    """Population size (10) strictly exceeds worker pool (2) and evaluates cleanly."""
    def sphere_eval(params: dict[str, float]) -> float:
        # Simple quadratic fitness function
        return sum(v**2 for v in params.values())

    evaluator = AsyncPopulationEvaluator(eval_fn=sphere_eval, num_workers=2)
    pop = [{"x": float(i), "y": float(i * 2)} for i in range(10)]

    results = evaluator.evaluate_population(pop, generation=1)

    assert len(results) == 10
    for i, res in enumerate(results):
        assert res.candidate_id == i
        assert res.status == "success"
        expected = float(i)**2 + float(i * 2)**2
        assert abs(res.fitness - expected) < 1e-6


def test_async_population_handles_worker_failures_and_retries():
    """Failing worker tasks are automatically retried and recorded."""
    fail_counts = {"count": 0}

    def flaky_eval(params: dict[str, float]) -> float:
        fail_counts["count"] += 1
        if fail_counts["count"] == 1:
            raise RuntimeError("Transient worker failure")
        return params["val"] * 2.0

    evaluator = AsyncPopulationEvaluator(eval_fn=flaky_eval, num_workers=2)
    pop = [{"val": 5.0}]

    results = evaluator.evaluate_population(pop)
    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].fitness == 10.0
