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

import io
import json

import numpy as np
import pytest
from rich.console import Console

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
    noise_norm = 0.3
    c1, c2 = sg.function_preserving_split(w, noise_norm, rng)
    assert np.allclose(0.5 * (c1 + c2), w), "the antisymmetric split must average back to the parent"
    assert not np.allclose(c1, c2), "the split must actually perturb (antisymmetric, non-degenerate)"
    # The achieved noise spectral norm must equal the request — the contract the Weyl certificate (and
    # max_noise_for_kappa) rely on; a wrong rescaling would silently break the κ bound.
    delta_n = 0.5 * (c1 - c2)
    assert float(np.linalg.svd(delta_n, compute_uv=False)[0]) == pytest.approx(noise_norm, rel=1e-6)


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


def test_coverage_signal_zero_median_reports_inf_not_a_huge_number():
    # A tight cluster of duplicate points plus one far outlier ⟹ median MST edge = 0; the ratio must be
    # reported as ∞ (a genuine isolated hole), not a meaningless ~1e14 from an epsilon floor.
    pts = np.vstack([np.zeros((10, 2)), np.array([[100.0, 0.0]])])
    sig = sg.coverage_signal(pts)
    assert sig.typical_gap == 0.0 and sig.persistence_ratio == float("inf") and sig.significant
    # All-identical points (every MST edge 0, no hole) ⟹ ratio 0, not ∞.
    flat = sg.coverage_signal(np.zeros((8, 2)))
    assert flat.max_gap == 0.0 and flat.persistence_ratio == 0.0 and not flat.significant


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
    assert cert.comparator_available
    assert cert.transport_optimal, "the quantile transport plan must beat the naive merge"
    assert cert.barycenter_std > cert.naive_std + 0.1, "naive averaging must collapse the variance"
    assert cert.transport_cost < cert.naive_cost, "the barycenter must be the lower-cost merge"


# --------------------------------------------------------------------------- #
# 0642.5.2.1 — bounded runtime certificates + Rich/JSONL logging
# --------------------------------------------------------------------------- #
def test_runtime_monitor_emits_all_certificates_and_caps_homology_work():
    rng = np.random.default_rng(11)
    w = _well_conditioned(12, s_min=1.0, s_max=3.0, rng=rng)
    # More rows/dimensions than the monitor budget, with a real inter-cluster H0 gap.
    points = np.vstack([
        rng.normal(0.0, 0.2, (100, 12)),
        rng.normal(np.r_[8.0, np.zeros(11)], 0.2, (100, 12)),
    ])
    a, b = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    monitor = sg.StructuralGeometryMonitor(sg.StructuralGeometryMonitorConfig(
        max_points=32, max_dim=4, max_persistence_features=3,
    ))

    rec = monitor.record(
        step=7,
        parent_weight=w,
        split_noise_norm=0.2,
        routing_points=points,
        merge_a=a,
        merge_b=b,
    )

    assert (rec.routing_points_input, rec.routing_dim_input) == (200, 12)
    assert (rec.routing_points_used, rec.routing_dim_used) == (32, 4)
    assert rec.routing_was_capped
    assert rec.homology_dimension == 0  # H0-only is the explicit homology-dimension cap
    assert rec.split_well_conditioned and rec.kappa_bound is not None
    assert rec.persistence_significant and 1 <= len(rec.top_persistence_features) <= 3
    assert list(rec.top_persistence_features) == sorted(rec.top_persistence_features, reverse=True)
    assert rec.merge_transport_optimal and rec.merge_cost_saving >= -1e-12
    monitor.assert_certificates()

    payload = json.loads(monitor.to_jsonl()[0])
    assert payload["step"] == 7
    assert payload["routing_points_used"] == 32
    assert payload["top_persistence_features"]
    assert payload["merge_transport_cost"] <= payload["merge_naive_cost"] + 1e-12

    output = io.StringIO()
    monitor.render(Console(file=output, force_terminal=False, width=100))
    rendered = output.getvalue()
    assert "Structural geometry certificates" in rendered
    assert "homology_dimension" in rendered and "routing_point_cap" in rendered
    assert "max_persistence" in rendered and "mean_merge_cost" in rendered


def test_runtime_monitor_fails_closed_for_uncertified_split_and_logs_standard_json():
    rng = np.random.default_rng(12)
    w = _well_conditioned(8, s_min=1.0, s_max=2.0, rng=rng)
    points = np.vstack([np.zeros((10, 2)), np.array([[10.0, 0.0]])])
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    monitor = sg.StructuralGeometryMonitor()

    rec = monitor.record(
        step=0,
        parent_weight=w,
        split_noise_norm=1.5,
        routing_points=points,
        merge_a=a,
        merge_b=b,
    )

    assert not rec.split_well_conditioned and rec.kappa_bound is None
    assert rec.persistence_ratio is None  # unbounded ratio is serialized as JSON null, not Infinity
    with pytest.raises(AssertionError, match="split conditioning failed"):
        monitor.assert_certificates()
    line = monitor.to_jsonl()[0]
    assert "Infinity" not in line and json.loads(line)["kappa_bound"] is None


def test_runtime_monitor_fails_closed_without_records_or_a_naive_merge_comparator():
    monitor = sg.StructuralGeometryMonitor()
    with pytest.raises(AssertionError, match="no structural geometry records"):
        monitor.assert_certificates()

    rec = monitor.record(
        step=0,
        parent_weight=np.eye(2),
        split_noise_norm=0.1,
        routing_points=np.array([[0.0], [1.0]]),
        merge_a=np.array([0.0, 1.0]),
        merge_b=np.array([0.0, 1.0, 2.0]),
    )
    assert not rec.merge_comparator_available and not rec.merge_transport_optimal
    with pytest.raises(AssertionError, match="OT merge certificate failed"):
        monitor.assert_certificates()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persistence_ratio_threshold": 0.0},
        {"persistence_ratio_threshold": np.nan},
        {"max_points": 1},
        {"max_dim": 0},
        {"max_persistence_features": 0},
    ],
)
def test_runtime_monitor_rejects_invalid_cost_bounds(kwargs):
    with pytest.raises(ValueError):
        sg.StructuralGeometryMonitorConfig(**kwargs)


def test_runtime_monitor_rejects_malformed_or_nonfinite_routing_points():
    monitor = sg.StructuralGeometryMonitor()
    w = np.eye(2)
    merge = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="2D"):
        monitor.record(
            step=0,
            parent_weight=w,
            split_noise_norm=0.1,
            routing_points=np.arange(4.0),
            merge_a=merge,
            merge_b=merge,
        )
    with pytest.raises(ValueError, match="finite"):
        monitor.record(
            step=0,
            parent_weight=w,
            split_noise_norm=0.1,
            routing_points=np.array([[0.0], [np.nan]]),
            merge_a=merge,
            merge_b=merge,
        )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("parent", "parent_weight.*finite"),
        ("noise", "split_noise_norm.*finite"),
        ("merge_empty", "merge samples.*non-empty"),
        ("merge_nonfinite", "merge samples.*finite"),
    ],
)
def test_runtime_monitor_rejects_inputs_that_cannot_form_standard_json(case, error):
    parent = np.array([[np.nan]]) if case == "parent" else np.eye(2)
    noise = np.inf if case == "noise" else 0.1
    merge_a = np.array([]) if case == "merge_empty" else np.array([0.0, 1.0])
    merge_b = np.array([np.nan]) if case == "merge_nonfinite" else np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=error):
        sg.StructuralGeometryMonitor().record(
            step=0,
            parent_weight=parent,
            split_noise_norm=noise,
            routing_points=np.array([[0.0], [1.0]]),
            merge_a=merge_a,
            merge_b=merge_b,
        )
