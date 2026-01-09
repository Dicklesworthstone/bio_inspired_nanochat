from __future__ import annotations

import math

from scripts.tune_bio_params import _lce_predict_from_points


def test_lce_predict_from_points_recovers_powerlaw() -> None:
    a = 2.0
    b = 10.0
    exponent = 0.5

    points = []
    for step in range(1, 101):
        loss = a + b * (step ** (-exponent))
        points.append((step, loss))

    pred = _lce_predict_from_points(points[-50:], target_step=400, exponent=exponent)
    assert pred is not None

    expected = a + b * (400 ** (-exponent))
    assert math.isfinite(pred)
    assert abs(pred - expected) < 1e-6


def test_lce_predict_from_points_rejects_increasing_curve() -> None:
    points = [(step, 1.0 + 0.1 * step) for step in range(1, 20)]
    pred = _lce_predict_from_points(points, target_step=40, exponent=0.5)
    assert pred is None


def test_lce_predict_from_points_requires_valid_exponent() -> None:
    points = [(1, 1.0), (2, 0.9), (3, 0.85), (4, 0.83)]
    assert _lce_predict_from_points(points, target_step=10, exponent=0.0) is None
    assert _lce_predict_from_points(points, target_step=10, exponent=-0.5) is None


def test_lce_predict_from_points_requires_enough_points() -> None:
    points = [(1, 1.0), (2, 0.9), (3, 0.85)]
    assert _lce_predict_from_points(points, target_step=10, exponent=0.5) is None

