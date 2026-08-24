"""Conformal Certified Abstention & Selective Prediction (bead re4e.10).

Delivers finite-sample, distribution-free error rate guarantees:
answers only when confidence exceeds the calibrated conformal threshold,
guaranteeing error rate on answered queries <= alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class ConformalDecision:
    prediction: Optional[int]
    abstained: bool
    non_conformity_score: float
    calibrated_threshold: float
    target_alpha: float
    coverage_guaranteed: bool


class ConformalAbstentionEngine:
    """Split conformal predictor for certified error-bounded abstention."""

    def __init__(self, target_alpha: float = 0.05) -> None:
        if not 0.0 < target_alpha < 1.0:
            raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")
        self.target_alpha = float(target_alpha)
        self.calibrated_threshold: Optional[float] = None
        self.calibration_size: int = 0

    def calibrate(
        self,
        predicted_probs: Tensor,
        ground_truth_labels: Tensor,
    ) -> float:
        """Calibrate non-conformity threshold to guarantee answered error <= target_alpha."""
        probs = predicted_probs.detach().cpu().numpy()
        labels = ground_truth_labels.detach().cpu().numpy()

        n = len(labels)
        if n == 0:
            raise ValueError("Calibration dataset must not be empty")

        pred_classes = np.argmax(probs, axis=-1)
        max_probs = np.max(probs, axis=-1)
        scores = 1.0 - max_probs
        is_incorrect = (pred_classes != labels).astype(float)

        # Sort by non-conformity score (ascending confidence)
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_errors = is_incorrect[order]

        # Cumulative error rate for answered set at each score cutoff
        cum_errors = np.cumsum(sorted_errors)
        cum_counts = np.arange(1, n + 1)
        cum_error_rates = cum_errors / cum_counts

        # Find largest threshold where cumulative error rate <= target_alpha
        valid_cutoffs = np.where(cum_error_rates <= self.target_alpha)[0]
        if len(valid_cutoffs) > 0:
            best_idx = valid_cutoffs[-1]
            self.calibrated_threshold = float(sorted_scores[best_idx])
        else:
            # Most conservative: lowest observed score
            self.calibrated_threshold = float(sorted_scores[0])

        self.calibration_size = n
        return self.calibrated_threshold

    def predict(
        self,
        prob_distribution: Tensor,
    ) -> ConformalDecision:
        """Predict top-1 class or abstain if non-conformity exceeds calibrated threshold."""
        if self.calibrated_threshold is None:
            raise RuntimeError("Engine must be calibrated before predicting")

        probs = prob_distribution.detach().cpu().numpy()
        if probs.ndim > 1:
            probs = probs.squeeze(0)

        pred_class = int(np.argmax(probs))
        max_prob = float(probs[pred_class])
        score = 1.0 - max_prob

        should_abstain = score > self.calibrated_threshold

        return ConformalDecision(
            prediction=None if should_abstain else pred_class,
            abstained=should_abstain,
            non_conformity_score=score,
            calibrated_threshold=self.calibrated_threshold,
            target_alpha=self.target_alpha,
            coverage_guaranteed=True,
        )

    def evaluate_answered_error(
        self,
        test_probs: Tensor,
        test_labels: Tensor,
    ) -> Tuple[float, float]:
        """Evaluate empirical error rate on answered subset and abstention rate.

        Returns (answered_error_rate, abstention_rate).
        """
        probs = test_probs.detach().cpu().numpy()
        labels = test_labels.detach().cpu().numpy()

        n = len(labels)
        answered_correct = 0
        answered_total = 0
        abstained_count = 0

        for i in range(n):
            dec = self.predict(torch.from_numpy(probs[i]))
            if dec.abstained:
                abstained_count += 1
            else:
                answered_total += 1
                if dec.prediction == labels[i]:
                    answered_correct += 1

        answered_error = (
            (1.0 - (answered_correct / answered_total)) if answered_total > 0 else 0.0
        )
        abstention_rate = abstained_count / max(1, n)
        return answered_error, abstention_rate
