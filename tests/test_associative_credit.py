import itertools

import numpy as np
import pytest

from scripts.e2e.associative_credit import MIXES, choose, components, curves


def test_counterfactual_matches_exhaustive_permutations_and_all_arms_normalize():
    # Two retained blocks have different marginals; globally shuffling is a different null.
    x = np.array([1, 1, 2, 1, 2, 2, 1])
    past_y = [0, 0, 1, 2, 2, 0]
    h = np.ones((7, 2)) / np.sqrt(2)
    distributions = {}
    for target, base in enumerate([0.2, 0.3, 0.5]):
        y = np.array([*past_y, target])
        parts = components(x, y, np.full(7, base), h, block=3)
        vals = []
        for a, b in itertools.product(itertools.permutations(past_y[:3]), itertools.permutations(past_y[3:])):
            hits = sum(c == 1 and t == target for c, t in zip(x[:6], a + b, strict=True))
            vals.append((hits + 4 * base) / 7)
        assert parts["exchangeable"][-1] == pytest.approx(np.mean(vals))
        for arm, probs in parts.items():
            distributions.setdefault(arm, []).append(probs[-1])
    assert all(sum(p) == pytest.approx(1) for p in distributions.values())
    assert distributions["binding"] != distributions["exchangeable"]


def test_future_writes_cannot_change_scored_prefix_and_zero_mixture_is_the_lm():
    x = np.arange(96) % 4
    y = np.arange(96) % 3
    p = np.full(96, 0.2)
    h = np.ones((96, 1))
    old = components(x, y, p, h)
    changed = y.copy()
    changed[61:] = 7
    new = components(x, changed, p, h)
    for arm in old:
        np.testing.assert_array_equal(old[arm][:61], new[arm][:61])
        np.testing.assert_array_equal(old[arm][:32], p[:32])
    assert all(b[0] == pytest.approx(-96 * np.log2(0.2)) for b in curves(p, old).values())


def test_calibration_ties_and_impossible_pure_cache():
    data = {"cosine_5": [9.0] * len(MIXES), "cosine_1": [9.0] * (len(MIXES)-1) + [None]}
    assert choose([{"bits": data}], ["cosine_5", "cosine_1"]) == {
        "index": 0, "mixture": 0, "temperature": 1, "arm": "cosine_1"}
    row = curves(np.array([0.5, 0.5]), {"cache": np.array([1.0, 0.0])})
    assert row["cache"][-1] is None
