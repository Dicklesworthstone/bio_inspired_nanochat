"""Independent draws and live fast-weight writes for the two reduced-model bounds."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from bio_inspired_nanochat.stochastic_thermo import ReleaseRates, empirical_tur, simulate_currents
from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig


@pytest.mark.parametrize("a", [2, 4, 8, 16])
def test_jump_current_precision(a, record_property):
    b, duration = 1, 8
    draws = simulate_currents(ReleaseRates(a, b), duration, 65536, seed=17001 + a)
    entropy = (a - b) * duration * math.log(a / b)
    observed = empirical_tur(draws, entropy)
    exact = (a + b) / (duration * (a - b) ** 2)
    batches = [empirical_tur(batch, entropy).relative_variance for batch in np.split(draws, 32)]
    se = float(np.std(batches, ddof=1) / math.sqrt(len(batches)))
    assert abs(observed.relative_variance - exact) < 6 * se
    assert observed.relative_variance + 6 * se >= 2 / entropy
    record_property("curve", json.dumps(dict(forward=a, reverse=b, duration=duration,
                    draws=draws.size, variance_ratio=observed.relative_variance, se=se,
                    exact=exact, entropy_floor=2 / entropy)))


def test_discrete_release_cannot_borrow_the_poisson_floor():
    # At these per-tick probabilities Bernoulli event counts are far from rare.
    rng = np.random.default_rng(30211)
    current = rng.binomial(1, 0.9, 65536) - rng.binomial(1, 0.1, 65536)
    affinity = math.log((0.9 * 0.9) / (0.1 * 0.1))
    assert not empirical_tur(current, 0.8 * affinity).satisfied


@pytest.mark.parametrize("decay", [0.75, 0.90])
@torch.no_grad()
def test_live_fast_weight_reconstruction(decay, record_property):
    # Each output coordinate is an independent scalar trial. This is the raw
    # leaky molecular update, with no normalized writes, gates or feedback.
    count = 65536
    cfg = SynapticConfig(post_fast_lr=1.0, post_fast_decay=decay, fast_weight_normalized=False)
    layer = PostsynapticHebb(1, count, cfg)
    generator = torch.Generator().manual_seed(8021)
    signal = torch.randn(count, generator=generator)
    layer.fast.copy_(signal)
    unit = torch.ones(1, 1)
    curve = []
    for lag in range(1, 7):
        distractor = 0.2 * torch.randn(1, count, generator=generator)
        noise = 0.3 * torch.randn(1, count, generator=generator)
        layer.hebb_fast(unit, distractor + noise)
        if lag not in {1, 3, 6}:
            continue
        power = decay ** (2 * lag)
        geometric_sum = (1 - power) / (1 - decay ** 2)
        noise_var = 0.09 * geometric_sum
        total_var = 0.13 * geometric_sum
        # For these Gaussian innovations the MMSE decoder is linear. Its
        # coefficient is fixed analytically before seeing any observations.
        coefficient = decay ** lag / (power + total_var)
        residual_sq = (signal.double() - coefficient * layer.fast.double()).square()
        measured = float(residual_sq.mean())
        se = float(residual_sq.std()) / math.sqrt(count)
        floor = noise_var / (power + noise_var)
        exact = total_var / (power + total_var)
        assert abs(measured - exact) < 6 * se
        assert measured + 6 * se >= floor
        # Independently verify the channel gain, so a stale or ignored write
        # cannot pass just by making reconstruction arbitrarily bad.
        gain = float((signal.double() * layer.fast.double()).mean() / signal.double().square().mean())
        assert abs(gain - decay ** lag) < 6 * math.sqrt(total_var / count)
        curve.append(dict(lag=lag, decay=decay, draws=count, measured_mse=measured,
                          se=se, exact_mmse=exact, noise_only_floor=floor, measured_gain=gain))
    record_property("curve", json.dumps(curve))
