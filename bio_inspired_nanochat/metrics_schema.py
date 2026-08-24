"""
Unified metrics schema — bead hm4.2.

ONE canonical registry of metric names / units / semantics, shared by base_train, eval,
tune_bio_params, neuroscore, and the results registry (hm4.1). Today each script defines metrics
ad-hoc, so cross-harness comparison and the committed results corpus are unreliable (a typo or a
renamed key silently breaks aggregation). This module:

- defines every metric once (`MetricSpec`: name, unit, optimization direction, description),
- rejects unknown or duplicate metric names and non-finite values (`validate_metrics`),
- is the single import all harnesses + the registry use.

Adding a metric = add one `_M(...)` line here (and only here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class Direction(str, Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    direction: Direction
    description: str


class UnknownMetricError(KeyError):
    """Raised when a metric name is not in the canonical schema."""


_METRICS: Dict[str, MetricSpec] = {}


def register_metric(spec: MetricSpec) -> MetricSpec:
    """Register a metric; raises on a duplicate name (the schema is the single source of truth)."""
    if spec.name in _METRICS:
        raise ValueError(f"duplicate metric name in schema: {spec.name!r}")
    _METRICS[spec.name] = spec
    return spec


def _M(name: str, unit: str, direction: Direction, description: str) -> MetricSpec:
    return register_metric(MetricSpec(name, unit, direction, description))


# --------------------------------------------------------------------------- #
# The canonical metric set (migrated from the ad-hoc per-script definitions).
# --------------------------------------------------------------------------- #
# -- Training (base_train) --
_M("step", "count", Direction.NEUTRAL, "optimizer step index")
_M("train_loss", "nats/token", Direction.LOWER_BETTER, "cross-entropy training loss")
_M("smooth_train_loss", "nats/token", Direction.LOWER_BETTER, "EMA-smoothed training loss")
_M("val_bpb", "bits/byte", Direction.LOWER_BETTER, "validation bits-per-byte")
_M("lr", "ratio", Direction.NEUTRAL, "learning-rate multiplier this step")
_M("grad_norm", "L2", Direction.LOWER_BETTER, "global gradient L2 norm")
_M("tok_per_sec", "tokens/s", Direction.HIGHER_BETTER, "training throughput")
_M("mfu", "fraction", Direction.HIGHER_BETTER, "model FLOPs utilization")
_M("total_training_time", "seconds", Direction.NEUTRAL, "cumulative wall-clock training time")
# -- Divergence guard (vg9.7) / bio state --
_M("loss_ema", "nats/token", Direction.LOWER_BETTER, "divergence-guard loss EMA")
_M("camkii_mean", "au", Direction.NEUTRAL, "mean CaMKII (consolidation write) level")
_M("bdnf_mean", "au", Direction.NEUTRAL, "mean BDNF (metaplasticity) level")
_M("calcium_norm", "L2", Direction.NEUTRAL, "presynaptic calcium state L2 norm")
_M("w_fast_norm", "L2", Direction.NEUTRAL, "fast-weight L2 norm")
_M("rrp_mean", "vesicles", Direction.NEUTRAL, "mean readily-releasable vesicle pool")
# -- Evaluation (base_eval / eval_matrix) --
_M("eval_bpb", "bits/byte", Direction.LOWER_BETTER, "evaluation bits-per-byte")
_M("eval_accuracy", "fraction", Direction.HIGHER_BETTER, "evaluation accuracy")
_M("niah_accuracy", "fraction", Direction.HIGHER_BETTER, "needle-in-haystack retrieval accuracy")
_M("id_ece", "fraction", Direction.LOWER_BETTER, "in-distribution expected calibration error")
_M("ood_auroc", "fraction", Direction.HIGHER_BETTER, "out-of-distribution detection AUROC")
_M(
    "forgetting_rate",
    "fraction",
    Direction.LOWER_BETTER,
    "mean peak-to-final accuracy loss across previously learned continual tasks",
)
_M(
    "moe_gini",
    "coefficient",
    Direction.NEUTRAL,
    "mean per-layer Gini coefficient of observed MoE routing counts",
)
_M(
    "dead_expert_frac",
    "fraction",
    Direction.LOWER_BETTER,
    "fraction of MoE experts below the predeclared routing-share floor",
)
_M(
    "live_ft_max_crooks_residual",
    "log-probability ratio",
    Direction.LOWER_BETTER,
    "maximum absolute detailed fluctuation-theorem residual on live release counts",
)
_M(
    "live_ft_integral_residual",
    "absolute error",
    Direction.LOWER_BETTER,
    "absolute residual of the live integral fluctuation theorem",
)
_M(
    "live_tur_relative_variance",
    "ratio",
    Direction.NEUTRAL,
    "exact relative current variance for the live paired-binomial protocol",
)
_M(
    "live_tur_entropy_bound",
    "ratio",
    Direction.NEUTRAL,
    "classic continuous-time TUR entropy lower bound evaluated on the live protocol",
)
_M(
    "live_tur_slack",
    "ratio",
    Direction.NEUTRAL,
    "live relative current variance minus the classic continuous-time TUR bound",
)
_M(
    "live_tur_bound_ratio",
    "ratio",
    Direction.NEUTRAL,
    "live relative current variance divided by the classic continuous-time TUR bound",
)
_M(
    "integrator_endpoint_loss",
    "mean_squared_error",
    Direction.LOWER_BETTER,
    "fixed-horizon endpoint error against an independent continuous-flow reference",
)
_M(
    "integrator_divergence_rate",
    "fraction",
    Direction.LOWER_BETTER,
    "fraction of predeclared stress-step evaluations classified as divergent",
)
_M(
    "tropical_exactness_rate",
    "fraction",
    Direction.HIGHER_BETTER,
    "fraction of the preregistered temperature sweep whose soft argmax matches the active vertex",
)
_M(
    "tropical_readout_l1_error",
    "L1",
    Direction.LOWER_BETTER,
    "low-temperature soft-readout L1 distance from the exact tropical hard readout",
)
_M(
    "soft_readout_l1_error",
    "L1",
    Direction.LOWER_BETTER,
    "ordinary tau=1 soft-readout L1 distance from the exact hard readout",
)
_M(
    "tropical_certified_radius",
    "L2 input distance",
    Direction.HIGHER_BETTER,
    "safety-adjusted affine selection radius under the preregistered L2 threat model",
)
_M(
    "empirical_adversarial_radius",
    "L2 input distance",
    Direction.HIGHER_BETTER,
    "nearest decision flip found by the independent black-box angular adversarial search",
)
_M(
    "tropical_radius_tightness",
    "ratio",
    Direction.HIGHER_BETTER,
    "certified selection radius divided by the empirical adversarial flip radius",
)
_M(
    "tropical_attribution_l1_error",
    "L1",
    Direction.LOWER_BETTER,
    "active-vertex attribution L1 distance from the exact hard-selection attribution",
)
_M(
    "attention_rollout_attribution_l1_error",
    "L1",
    Direction.LOWER_BETTER,
    "ordinary one-layer attention-rollout L1 distance from exact hard-selection attribution",
)
# -- NeuroScore (neuroscore.py) --
_M("neuroscore_efficiency", "ratio", Direction.HIGHER_BETTER, "expert efficiency (contribution/energy)")
_M("neuroscore_specialization", "ratio", Direction.HIGHER_BETTER, "expert input-distribution specialization")
_M("neuroscore_resilience", "ratio", Direction.HIGHER_BETTER, "expert resilience to perturbation")
# -- Tuning (tune_bio_params, CMA-ES) --
_M("tune_objective", "score", Direction.LOWER_BETTER, "CMA-ES objective (val_bpb proxy)")
_M("tune_generation", "count", Direction.NEUTRAL, "CMA-ES generation index")


def get_metric(name: str) -> Optional[MetricSpec]:
    return _METRICS.get(name)


def all_metrics() -> Dict[str, MetricSpec]:
    """A copy of the full canonical registry."""
    return dict(_METRICS)


def is_known(name: str) -> bool:
    return name in _METRICS


def validate_metrics(metrics: Mapping[str, Any], *, strict: bool = True) -> Dict[str, float]:
    """Validate a metrics dict against the canonical schema.

    - Unknown metric names raise UnknownMetricError when strict (default); when not strict they
      are dropped (the caller logs a warning).
    - Values must be finite and coercible to float, else ValueError.

    Returns the validated metrics coerced to float.
    """
    out: Dict[str, float] = {}
    for name, val in metrics.items():
        if name not in _METRICS:
            if strict:
                raise UnknownMetricError(
                    f"unknown metric {name!r}: not in the canonical schema "
                    f"(add it to metrics_schema.py)"
                )
            continue
        try:
            f = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {name!r} value {val!r} is not numeric") from exc
        if not math.isfinite(f):
            raise ValueError(f"metric {name!r} is not finite ({f})")
        out[name] = f
    return out
