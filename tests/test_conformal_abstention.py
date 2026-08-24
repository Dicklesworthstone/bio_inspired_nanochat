"""Unit and empirical coverage tests for Conformal Abstention Engine (bead re4e.10)."""

from __future__ import annotations

import numpy as np
import torch

from bio_inspired_nanochat.conformal_abstention import ConformalAbstentionEngine


def test_conformal_abstention_guarantees_target_error():
    """Conformal abstention ensures answered subset error <= target alpha."""
    rng = np.random.default_rng(42)
    n_calib = 1000
    n_test = 1000
    n_classes = 5
    target_alpha = 0.10  # 10% maximum allowable error

    # Generate synthetic softmax probabilities with 20% noisy low-confidence samples
    def generate_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = rng.integers(0, n_classes, size=n)
        logits = rng.normal(size=(n, n_classes))
        # Boost true label logits for easy samples
        for i in range(n):
            if rng.random() > 0.2:  # 80% confident
                logits[i, labels[i]] += 4.0
        probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1)
        return probs, torch.from_numpy(labels).long()

    calib_probs, calib_labels = generate_data(n_calib)
    test_probs, test_labels = generate_data(n_test)

    engine = ConformalAbstentionEngine(target_alpha=target_alpha)
    threshold = engine.calibrate(calib_probs, calib_labels)
    assert threshold > 0.0

    answered_error, abstention_rate = engine.evaluate_answered_error(test_probs, test_labels)

    # Assert that answered error respects the target alpha (with finite-sample margin)
    assert answered_error <= target_alpha + 0.03, (
        f"Answered error {answered_error:.2%} exceeded target alpha {target_alpha:.2%}"
    )
    assert abstention_rate > 0.0, "Engine should selectively abstain on uncertain points"


def test_conformal_abstention_predict_decision_types():
    """ConformalAbstentionEngine returns valid ConformalDecision object."""
    engine = ConformalAbstentionEngine(target_alpha=0.05)
    probs = torch.tensor([[0.9, 0.1], [0.55, 0.45]])
    labels = torch.tensor([0, 0])

    engine.calibrate(probs, labels)

    # Confident input
    dec_high = engine.predict(torch.tensor([0.95, 0.05]))
    assert not dec_high.abstained
    assert dec_high.prediction == 0

    # Uncertain input
    dec_low = engine.predict(torch.tensor([0.51, 0.49]))
    assert dec_low.abstained
    assert dec_low.prediction is None
