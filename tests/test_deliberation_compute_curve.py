"""Falsification harness tests for the deliberation compute/quality curve (``r00r.1.4``)."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from bio_inspired_nanochat.eval_stats import PairedResult
from scripts.e2e.deliberation_compute_curve import (
    ExperimentConfig,
    classify_verdict,
    render_report,
    run_experiment,
)


def _paired(delta: float, low: float, high: float, p_value: float) -> PairedResult:
    return PairedResult(
        n_pairs=5,
        mean_delta=delta,
        delta_ci_low=low,
        delta_ci_high=high,
        t_stat=3.0,
        t_p_value=p_value,
        wilcoxon_p_value=p_value,
        cohen_dz=1.0,
        n_favorable=5,
    )


def _verdict(comparison: PairedResult, *, baseline_accuracy: float = 0.6):
    return classify_verdict(
        comparison,
        budget=16,
        alpha=0.05,
        baseline_accuracy=baseline_accuracy,
        chance_accuracy=0.1,
        min_skill_over_chance=0.1,
    )


@pytest.mark.unit
def test_verdict_is_predeclared_and_allows_honest_null():
    improved = _verdict(_paired(0.1, 0.02, 0.2, 0.01))
    null = _verdict(_paired(0.01, -0.1, 0.1, 0.8))
    worse = _verdict(_paired(-0.1, -0.2, -0.02, 0.01))
    inconclusive = _verdict(_paired(0.1, 0.02, 0.2, 0.01), baseline_accuracy=0.15)
    assert improved.outcome == "improved"
    assert null.outcome == "null" and null.honest_null_allowed
    assert worse.outcome == "worse"
    assert inconclusive.outcome == "inconclusive"


@pytest.mark.unit
def test_config_rejects_nonstatistical_or_unordered_sweeps():
    with pytest.raises(ValueError, match="at least two unique"):
        ExperimentConfig(seeds=(1,)).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        ExperimentConfig(budgets=(8, 1)).validate()
    with pytest.raises(ValueError, match="divisible"):
        ExperimentConfig(n_head=3, n_embd=16).validate()


@pytest.mark.e2e
def test_tiny_compute_quality_curve_is_complete_stats_backed_and_strict_json():
    config = ExperimentConfig(
        seeds=(3, 7),
        budgets=(1, 3),
        vocab_size=16,
        copy_length=2,
        train_batch_size=2,
        train_steps=1,
        eval_sequences=2,
        n_layer=1,
        n_head=2,
        n_embd=16,
        bootstrap_samples=100,
    )
    report = run_experiment(config)
    assert [point.max_iters for point in report.curve] == [0, 1, 3]
    baseline, *treatments = report.curve
    assert baseline.token_accuracy_vs_baseline is None
    assert baseline.mean_effort_per_token.mean == 0.0
    assert baseline.deliberation_coverage.mean == 0.0
    expected_generated = config.eval_sequences * config.copy_length
    for point in treatments:
        assert point.token_accuracy_vs_baseline is not None
        assert point.token_accuracy_vs_baseline.n_pairs == 2
        assert 0.0 <= point.token_accuracy.mean <= 1.0
        assert 0.0 <= point.exact_match.mean <= 1.0
        assert point.deliberation_coverage.mean == 1.0
        assert 0.0 < point.mean_effort_per_token.mean <= point.max_iters
        assert all(item.generated_tokens in {expected_generated} for item in point.per_seed)
        assert all(item.pondered_tokens in {item.generated_tokens} for item in point.per_seed)
    assert report.verdict.outcome in {"improved", "null", "worse", "inconclusive"}
    assert "not fed back into logits" in report.mechanism_scope
    assert "every generated token" in report.mechanism_scope
    json.dumps(report.to_dict(), allow_nan=False)

    console = Console(record=True, width=140)
    render_report(report, console)
    rendered = console.export_text()
    assert "compute/quality curve" in rendered
    assert "Verdict" in rendered
