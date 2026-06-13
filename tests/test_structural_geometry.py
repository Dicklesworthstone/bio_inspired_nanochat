"""Numerical corroboration of the free-prob / TDA / OT structural-plasticity note (Thrust C, `0642.5.1`).

Checks the falsifiable results of `docs/theory/structural_geometry.md` against the reference
implementation (`bio_inspired_nanochat/structural_geometry.py`):

  - §1 FREE PROBABILITY — a noisy expert split moves singular values by ≤ ‖δN‖ (Weyl), so the child
    condition number is bounded by the spectral-conditioning certificate (`0642.5.1.1`);
  - §2 PERSISTENT HOMOLOGY — the H0 coverage signal (largest MST gap) flags genuine topological holes
    and is bottleneck-stable under perturbation (`0642.5.1.2`);
  - §3 OPTIMAL TRANSPORT — the Wasserstein barycenter is the min-cost, spread-preserving merge, beating
    naive value averaging (`0642.5.1.3`).

Run:  pytest tests/test_structural_geometry.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat import structural_geometry as sg

pytestmark = pytest.mark.unit


def _well_conditioned(n: int, s_min: float, s_max: float, rng: np.random.Generator) -> np.ndarray:
    """A matrix with prescribed extreme singular values (the rest interpolated) — known κ."""
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.linspace(s_max, s_min, n)
    return u @ np.diag(s) @ v.T


# --------------------------------------------------------------------------- #
# §1. Free-probability spectral conditioning
# --------------------------------------------------------------------------- #
def test_condition_number_matches_construction():
    rng = np.random.default_rng(0)
    w = _well_conditioned(16, s_min=1.0, s_max=4.0, rng=rng)
    assert sg.condition_number(w) == pytest.approx(4.0, rel=1e-6)


def test_spectral_certificate_bounds_child_kappa_weyl():
    rng = np.random.default_rng(1)
    w = _well_conditioned(16, s_min=1.0, s_max=4.0, rng=rng)
    noise = 0.2
    cert = sg.spectral_conditioning_certificate(w, noise)
    assert cert.well_conditioned and np.isfinite(cert.kappa_bound)
    assert cert.kappa_bound == pytest.approx((4.0 + noise) / (1.0 - noise), rel=1e-6)
    # The Weyl bound must actually hold for real split children.
    for _ in range(10):
        c1, c2 = sg.function_preserving_split(w, noise, rng)
        assert sg.condition_number(c1) <= cert.kappa_bound + 1e-6
        assert sg.condition_number(c2) <= cert.kappa_bound + 1e-6


def test_split_is_output_preserving_on_average():
    rng = np.random.default_rng(2)
    w = _well_conditioned(12, 1.0, 3.0, rng=rng)
    c1, c2 = sg.function_preserving_split(w, 0.3, rng)
    assert np.allclose(0.5 * (c1 + c2), w), "the antisymmetric split must average back to the parent"


def test_max_noise_for_kappa_achieves_the_target():
    rng = np.random.default_rng(3)
    w = _well_conditioned(16, s_min=1.0, s_max=3.0, rng=rng)
    x = sg.max_noise_for_kappa(w, kappa_target=5.0)
    assert x == pytest.approx((5.0 * 1.0 - 3.0) / (5.0 + 1.0), rel=1e-6)
    assert sg.spectral_conditioning_certificate(w, x).kappa_bound == pytest.approx(5.0, rel=1e-6)
    with pytest.raises(ValueError):
        sg.max_noise_for_kappa(w, 1.0)


def test_certificate_is_void_when_noise_exceeds_smallest_singular_value():
    rng = np.random.default_rng(4)
    w = _well_conditioned(8, s_min=1.0, s_max=2.0, rng=rng)
    cert = sg.spectral_conditioning_certificate(w, noise_norm=1.5)  # > σ_min ⟹ child may be singular
    assert not cert.well_conditioned and not np.isfinite(cert.kappa_bound)


# --------------------------------------------------------------------------- #
# §2. Persistent-homology coverage signal
# --------------------------------------------------------------------------- #
def test_mst_edges_count_and_sorted():
    rng = np.random.default_rng(5)
    pts = rng.standard_normal((30, 3))
    e = sg.mst_edge_lengths(pts)
    assert e.size == 29 and np.all(np.diff(e) >= 0)


def test_coverage_signal_flags_a_real_hole_not_a_uniform_cloud():
    rng = np.random.default_rng(6)
    two_clusters = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal([8, 0], 0.3, (60, 2))])
    uniform = rng.uniform(0, 8, (120, 2))
    sig_hole = sg.coverage_signal(two_clusters)
    sig_flat = sg.coverage_signal(uniform)
    assert sig_hole.significant and sig_hole.persistence_ratio > 10.0
    assert not sig_flat.significant
    assert sig_hole.max_gap > sig_flat.max_gap


def test_coverage_signal_is_bottleneck_stable():
    rng = np.random.default_rng(7)
    pts = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal([8, 0], 0.3, (50, 2))])
    eps = 0.02
    perturbed = pts + rng.normal(0, eps, pts.shape)
    base = sg.coverage_signal(pts).max_gap
    pert = sg.coverage_signal(perturbed).max_gap
    assert abs(pert - base) <= 6 * eps, "the H0 diagram must move by ~the perturbation (bottleneck stability)"


# --------------------------------------------------------------------------- #
# §3. Optimal-transport merge
# --------------------------------------------------------------------------- #
def test_wasserstein_1d_basic_properties():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 4000)
    assert sg.wasserstein_1d(a, a) == pytest.approx(0.0, abs=1e-9)
    assert sg.wasserstein_1d(a, a + 3.0) == pytest.approx(3.0, abs=0.05)  # pure shift ⟹ W2 = shift


def test_gaussian_barycenter_averages_mean_and_std():
    rng = np.random.default_rng(9)
    g1, g2 = rng.normal(0, 1, 6000), rng.normal(5, 3, 6000)
    bary = sg.wasserstein_barycenter_1d(g1, g2, t=0.5)
    assert bary.mean() == pytest.approx(2.5, abs=0.1)   # W2 barycenter mean = average mean
    assert bary.std() == pytest.approx(2.0, abs=0.1)    # ... and std = average std (geodesic midpoint)


def test_ot_merge_preserves_spread_and_is_min_cost():
    rng = np.random.default_rng(10)
    a, b = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)  # same law, random order
    cert = sg.ot_merge_certificate(a, b)
    assert cert.ot_preserves_spread, "the OT barycenter must keep the marginal spread"
    assert cert.barycenter_std > cert.naive_std + 0.1, "naive averaging must collapse the variance"
    assert cert.transport_cost < cert.naive_cost, "the barycenter must be the lower-cost merge"
