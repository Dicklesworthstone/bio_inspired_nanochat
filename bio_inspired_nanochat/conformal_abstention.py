"""Top-1 selective classification with a finite-sample calibration risk bound.

The classifier and threshold grid must be fixed before an independent, iid labelled
calibration sample. At confidence 1-delta over that sample, every admitted policy's
population error conditional on answering is at most alpha under the same distribution.
This statement concerns a population rate. Individual answers can be wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConformalDecision:
    prediction: int | None
    abstained: bool
    non_conformity_score: float
    calibrated_threshold: float
    target_alpha: float
    calibration_confidence: float
    answered_risk_upper_bound: float | None


# Fixed before seeing labels or predictions. Testing every distinct observed score would
# instead create a data-dependent family with a different simultaneous-coverage obligation.
_THRESHOLDS = tuple(i / 100 for i in range(101))


def _upper_risk(errors: int, answered: int, log_inverse_tail: float) -> float:
    """Invert the binomial Chernoff lower-tail bound; return the conservative endpoint."""
    if errors == answered:
        return 1.0
    if errors == 0:
        return -math.expm1(-log_inverse_tail / answered)
    p = errors / answered
    low, high = p, 1.0
    for _ in range(50):
        mid = (low + high) / 2
        if mid == high or mid == low:
            break
        divergence = p * math.log(p / mid) + (1 - p) * (math.log1p(-p) - math.log1p(-mid))
        if answered * divergence >= log_inverse_tail:
            high = mid
        else:
            low = mid
    return high


def _predictions(probs: Tensor) -> tuple[Tensor, Tensor]:
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 2 or not probs.is_floating_point():
        raise ValueError("expected a nonempty floating probability matrix with at least two classes")
    values = probs.detach().to(device="cpu", dtype=torch.float64)
    if (not values.isfinite().all() or (values < 0).any() or (values > 1).any()
        or not torch.allclose(values.sum(1), torch.ones_like(values[:, 0]), atol=1e-5, rtol=0)):
        raise ValueError("probability rows must be finite, nonnegative and sum to one")
    confidence, predicted = values.max(dim=1)
    return 1.0 - confidence, predicted


def _label_errors(probs: Tensor, labels: Tensor, predicted: Tensor) -> Tensor:
    if labels.shape != predicted.shape or labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise ValueError("one integer class label is required per prediction")
    labels = labels.detach().cpu()
    if (labels < 0).any() or (labels >= probs.shape[1]).any():
        raise ValueError("label outside the class range")
    return predicted != labels


class ConformalAbstentionEngine:
    """Calibrate an acceptance policy on held-out labels using simultaneous binomial bounds."""

    def __init__(self, target_alpha: float = 0.05, *, delta: float = 0.05) -> None:
        if not (0 < target_alpha < 1 and 0 < delta < 1):
            raise ValueError("target_alpha and delta must be in (0, 1)")
        self.target_alpha = target_alpha
        self.delta = delta
        self.calibrated_threshold: float | None = None
        self.answered_risk_upper_bound: float | None = None
        self.calibration_size = 0
        self._classes = 0

    def calibrate(self, predicted_probs: Tensor, ground_truth_labels: Tensor) -> float:
        scores, predicted = _predictions(predicted_probs)
        errors = _label_errors(predicted_probs, ground_truth_labels, predicted)
        order = scores.argsort()
        ordered = scores[order]
        cum_errors = errors[order].long().cumsum(dim=0)
        counts = torch.searchsorted(ordered, ordered.new_tensor(_THRESHOLDS), right=True)
        # Conditional on each fixed policy's acceptance count, accepted errors are binomial.
        # A union bound over the grid makes selection of the widest passing policy valid.
        tail = math.log(len(_THRESHOLDS)) - math.log(self.delta)
        threshold, bound = -1.0, None
        for candidate, count in reversed(list(zip(_THRESHOLDS, counts.tolist(), strict=True))):
            if count:
                upper = _upper_risk(int(cum_errors[count - 1]), count, tail)
                if upper <= self.target_alpha:
                    threshold, bound = candidate, upper
                    break
        self.calibrated_threshold = threshold
        self.answered_risk_upper_bound = bound
        self.calibration_size, self._classes = predicted_probs.shape
        return threshold

    def _threshold_for(self, probs: Tensor) -> float:
        if self.calibrated_threshold is None:
            raise RuntimeError("calibrate before predicting")
        if probs.shape[-1:] != (self._classes,):
            raise ValueError("prediction class count differs from calibration")
        return self.calibrated_threshold

    def predict(self, prob_distribution: Tensor) -> ConformalDecision:
        probs = prob_distribution.reshape(1, -1) if prob_distribution.ndim == 1 else prob_distribution
        threshold = self._threshold_for(probs)
        scores, predicted = _predictions(probs)
        if len(scores) != 1:
            raise ValueError("predict expects one distribution")
        score = float(scores[0])
        abstained = score > threshold
        return ConformalDecision(None if abstained else int(predicted[0]), abstained, score, threshold,
                                 self.target_alpha, 1 - self.delta, self.answered_risk_upper_bound)

    def evaluate_answered_error(self, test_probs: Tensor, test_labels: Tensor) -> tuple[float, float]:
        """Return empirical conditional error (NaN for zero answers) and abstention fraction."""
        threshold = self._threshold_for(test_probs)
        scores, predicted = _predictions(test_probs)
        errors = _label_errors(test_probs, test_labels, predicted)
        kept = scores <= threshold
        answered = int(kept.sum())
        rate = int(errors[kept].sum()) / answered if answered else math.nan
        return rate, 1 - answered / len(scores)
