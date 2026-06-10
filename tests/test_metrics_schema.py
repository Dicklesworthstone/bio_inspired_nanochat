"""
Unified metrics schema — bead hm4.2.

Locks: every metric is registered exactly once (no duplicates); validation rejects unknown names
and non-finite values; the canonical set covers each harness; strict vs non-strict behavior.

Run:  pytest tests/test_metrics_schema.py -v
"""

from __future__ import annotations

import math

import pytest

from bio_inspired_nanochat.metrics_schema import (
    Direction,
    MetricSpec,
    UnknownMetricError,
    all_metrics,
    get_metric,
    is_known,
    register_metric,
    validate_metrics,
)


@pytest.mark.unit
def test_registry_is_populated_and_well_formed():
    metrics = all_metrics()
    assert len(metrics) >= 20, "the canonical set should cover train/eval/tune/neuroscore"
    for name, spec in metrics.items():
        assert spec.name == name
        assert isinstance(spec, MetricSpec)
        assert spec.unit and spec.description
        assert isinstance(spec.direction, Direction)


@pytest.mark.unit
def test_each_harness_area_is_represented():
    names = set(all_metrics())
    assert {"train_loss", "val_bpb", "grad_norm"} <= names           # training
    assert {"eval_bpb", "eval_accuracy"} <= names                    # eval
    assert {"neuroscore_efficiency", "neuroscore_resilience"} <= names  # neuroscore
    assert {"tune_objective", "tune_generation"} <= names            # tuning


@pytest.mark.unit
def test_duplicate_registration_is_rejected():
    existing = next(iter(all_metrics()))
    with pytest.raises(ValueError, match="duplicate"):
        register_metric(MetricSpec(existing, "x", Direction.NEUTRAL, "dup"))


@pytest.mark.unit
def test_validate_accepts_known_metrics_and_coerces_to_float():
    out = validate_metrics({"train_loss": 4.5, "step": 10, "mfu": 0.3})
    assert out == {"train_loss": 4.5, "step": 10.0, "mfu": 0.3}
    assert all(isinstance(v, float) for v in out.values())


@pytest.mark.unit
def test_validate_rejects_unknown_metric_when_strict():
    with pytest.raises(UnknownMetricError, match="unknown metric"):
        validate_metrics({"train_loss": 4.5, "totally_made_up": 1.0})


@pytest.mark.unit
def test_validate_drops_unknown_metric_when_not_strict():
    out = validate_metrics({"train_loss": 4.5, "totally_made_up": 1.0}, strict=False)
    assert out == {"train_loss": 4.5}


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_validate_rejects_non_finite_values(bad):
    with pytest.raises(ValueError, match="not finite"):
        validate_metrics({"train_loss": bad})


@pytest.mark.unit
def test_validate_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="not numeric"):
        validate_metrics({"train_loss": "abc"})


@pytest.mark.unit
def test_helpers():
    assert is_known("val_bpb") and not is_known("nope")
    assert get_metric("val_bpb").direction == Direction.LOWER_BETTER
    assert get_metric("nope") is None
    assert math.isfinite(validate_metrics({"step": 3})["step"])
