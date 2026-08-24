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
from typing import Any

import numpy as np
import pytest
import torch
from rich.console import Console

from bio_inspired_nanochat import structural_geometry as sg
from bio_inspired_nanochat.ablation_registry import MECHANISMS
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)

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


def test_ot_certificate_uses_every_rank_for_equal_empirical_measures():
    a = np.zeros(2048)
    b = np.zeros(2048)
    b[1] = 10.0

    cert = sg.ot_merge_certificate(a, b)

    expected_transport = float(np.mean((np.sort(a) - np.sort(b)) ** 2)) / 4.0
    assert cert.transport_cost == pytest.approx(expected_transport, rel=1e-12)
    assert cert.comparator_available and cert.transport_optimal


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


# --------------------------------------------------------------------------- #
# 0642.5.2.2 — geometry-driven lifecycle + deterministic UTA fallback
# --------------------------------------------------------------------------- #
def _geometry_moe(seed: int = 21, *, topological: bool = True) -> SynapticMoE:
    torch.manual_seed(seed)
    cfg = SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=True,
        router_contrastive_push=0.0,
        router_contrastive_lr=0.0,
        topological_nas=topological,
    )
    moe = SynapticMoE(
        n_embd=4,
        num_experts=4,
        top_k=2,
        hidden_mult=1,
        cfg=cfg,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        for index, expert in enumerate(moe.experts):
            scale = 1.0 + index
            expert.fc1.w_slow.copy_(torch.diag(torch.linspace(scale, scale + 0.3, 4)))
            expert.fc2.w_slow.copy_(torch.diag(torch.linspace(scale, scale + 0.3, 4)))
        # Experts 0 and 1 have identical empirical weight laws, making them the
        # unique zero-cost OT pair while retaining a well-conditioned split source.
        moe.experts[1].fc1.w_slow.copy_(moe.experts[0].fc1.w_slow)
        moe.experts[1].fc2.w_slow.copy_(moe.experts[0].fc2.w_slow)
        moe.experts[1].fc1.bias.copy_(moe.experts[0].fc1.bias)
        moe.experts[1].fc2.bias.copy_(moe.experts[0].fc2.bias)
        moe.router.weight[1].copy_(moe.router.weight[0])
        moe.Xi[1].copy_(moe.Xi[0])
        moe.router_embeddings[1].copy_(moe.router_embeddings[0])
    return moe


def _topological_cfg(**overrides: Any) -> SplitMergeConfig:
    values: dict[str, Any] = dict(
        enabled=True,
        warmup_steps=0,
        min_step_interval=0,
        merges_per_call=1,
        splits_per_call=1,
        resets_per_call=0,
        ddp_broadcast=False,
        function_preserving=True,
        fp_divergence_noise=0.05,
        topological_kappa_target=10.0,
        topological_merge_cost_ratio_max=0.01,
        topological_persistence_ratio_threshold=2.0,
    )
    values.update(overrides)
    return SplitMergeConfig(**values)


def _prime_routing(moe: SynapticMoE, *, with_gap: bool) -> None:
    dtype = moe.router.weight.dtype
    device = moe.router.weight.device
    if with_gap:
        points = torch.cat(
            [
                torch.zeros(4, 4, dtype=dtype, device=device),
                torch.full((4, 4), 8.0, dtype=dtype, device=device),
            ],
            dim=0,
        )
    else:
        points = torch.zeros(8, 4, dtype=dtype, device=device)
        points[:, 0] = torch.arange(8, dtype=dtype, device=device)
    with torch.no_grad():
        moe(points.unsqueeze(0), update_mem=False)


def test_topological_split_prediction_bounds_measured_child_spectrum(tmp_path):
    moe = _geometry_moe()
    _prime_routing(moe, with_gap=True)
    originals = [expert.fc1.w_slow.detach().clone() for expert in moe.experts]
    logger = RunLogger(tmp_path, name="topological_split", console=False)
    controller = SplitMergeController(
        moe,
        _topological_cfg(homeostasis_guards=True),
        event_logger=logger,
    )
    redundant_before = controller._expert_weight_samples(moe, 0).copy()

    controller.step(global_step=7)

    decision = controller.topological_decisions[-1]
    assert decision.mode == "topological" and decision.action == "merge_split"
    assert decision.split_source is not None and decision.split_destination is not None
    assert decision.merge_pair is not None
    assert decision.split_destination in decision.merge_pair
    assert set(controller.homeo._ramps[0]) == {decision.split_destination}
    assert controller.homeo._ramps[0][decision.split_destination][1] == decision.split_source
    assert decision.kappa_bound is not None and decision.split_noise_norm is not None
    parent = moe.experts[decision.split_source].fc1.w_slow.detach()
    child = moe.experts[decision.split_destination].fc1.w_slow.detach()
    assert torch.allclose(0.5 * (parent + child), originals[decision.split_source])
    measured_noise = float(torch.linalg.matrix_norm(0.5 * (child - parent), ord=2))
    assert measured_noise <= decision.split_noise_norm + 1e-6
    assert sg.condition_number(parent.numpy()) <= decision.kappa_bound + 1e-6
    assert sg.condition_number(child.numpy()) <= decision.kappa_bound + 1e-6
    assert np.allclose(controller._expert_weight_samples(moe, 0), redundant_before)
    events = [event for event in logger.read_events() if event["event"] == "topological_nas"]
    assert events[-1]["decision"]["reason"] == "persistent_uncovered_h0_gap"
    assert events[-1]["certificates"]["persistence_significant"]
    logger.close()


def test_topological_ot_signal_selects_lowest_cost_merge_pair():
    moe = _geometry_moe()
    with torch.no_grad():
        moe.experts[1].fc1.w_slow.add_(0.01 * torch.eye(4))
    _prime_routing(moe, with_gap=False)
    controller = SplitMergeController(moe, _topological_cfg())
    a = controller._expert_weight_samples(moe, 0)
    b = controller._expert_weight_samples(moe, 1)
    expected_barycenter = 0.5 * (np.sort(a) + np.sort(b))

    controller.step(global_step=8)

    decision = controller.topological_decisions[-1]
    assert decision.mode == "topological" and decision.action == "merge"
    assert decision.merge_pair == (0, 1)
    assert decision.merge_cost_ratio is not None
    assert decision.merge_cost_ratio <= controller.cfg.topological_merge_cost_ratio_max
    assert decision.persistence_ratio == pytest.approx(1.0, abs=1e-6)
    actual_barycenter = np.sort(controller._expert_weight_samples(moe, 0))
    assert np.allclose(actual_barycenter, expected_barycenter, atol=1e-7)
    assert np.allclose(
        controller._expert_weight_samples(moe, 1),
        controller._expert_weight_samples(moe, 0),
    )


def test_topological_ot_rejects_marginally_equal_but_functionally_permuted_experts():
    moe = _geometry_moe()
    with torch.no_grad():
        moe.experts[1].fc1.w_slow.copy_(torch.flip(moe.experts[0].fc1.w_slow, (0, 1)))
        moe.experts[1].fc2.w_slow.copy_(torch.flip(moe.experts[0].fc2.w_slow, (0, 1)))
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_functional_distance_max=0.05),
    )
    a = controller._expert_weight_samples(moe, 0)
    b = controller._expert_weight_samples(moe, 1)

    assert sg.ot_merge_certificate(a, b).transport_cost == pytest.approx(0.0, abs=1e-12)
    candidate_pairs = {(candidate[2], candidate[3]) for candidate in controller._ot_merge_candidates(moe)}
    assert (0, 1) not in candidate_pairs


def test_topological_ot_guard_includes_live_postsynaptic_state():
    torch.manual_seed(29)
    cfg = SynapticConfig(
        enable_hebbian=True,
        enable_metabolism=True,
        topological_nas=True,
    )
    moe = SynapticMoE(4, 3, 2, 1, cfg, 0.0).eval()
    with torch.no_grad():
        moe.experts[1].load_state_dict(moe.experts[0].state_dict())
        moe.router.weight[1].copy_(moe.router.weight[0])
        moe.Xi[1].copy_(moe.Xi[0])
        moe.router_embeddings[1].copy_(moe.router_embeddings[0])
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_functional_distance_max=0.05),
    )
    assert (0, 1) in {
        (candidate[2], candidate[3]) for candidate in controller._ot_merge_candidates(moe)
    }

    with torch.no_grad():
        assert moe.experts[1].fc1.post is not None
        moe.experts[1].fc1.post.fast.add_(1.0)

    assert (0, 1) not in {
        (candidate[2], candidate[3]) for candidate in controller._ot_merge_candidates(moe)
    }


@pytest.mark.parametrize("state_name", ["bdnf_hebb_accum", "_last_hebb_delta_mag"])
def test_topological_exact_guard_includes_plasticity_accumulators(state_name):
    torch.manual_seed(30)
    cfg = SynapticConfig(
        enable_hebbian=True,
        enable_metabolism=True,
        topological_nas=True,
    )
    moe = SynapticMoE(4, 3, 2, 1, cfg, 0.0).eval()
    with torch.no_grad():
        moe.experts[1].load_state_dict(moe.experts[0].state_dict())
        moe.router.weight[1].copy_(moe.router.weight[0])
        moe.Xi[1].copy_(moe.Xi[0])
        moe.router_embeddings[1].copy_(moe.router_embeddings[0])
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_functional_distance_max=0.05),
    )
    assert controller._exact_ot_merge_candidate(moe, 0, 1) is not None

    with torch.no_grad():
        post = moe.experts[1].fc1.post
        assert post is not None
        getattr(post, state_name).add_(1.0)

    assert controller._exact_ot_merge_candidate(moe, 0, 1) is None


def test_topological_persistence_signal_can_birth_under_budget():
    moe = _geometry_moe().to(dtype=torch.float64)
    _prime_routing(moe, with_gap=True)
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-3)
    for parameter in moe.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    untouched = moe.experts[2].fc1.w_slow
    assert untouched in optimizer.state
    controller = SplitMergeController(
        moe,
        _topological_cfg(
            variable_expert_count=True,
            min_experts=2,
            max_experts=5,
            growth_budget_pct=0.5,
        ),
    )

    controller.step(global_step=9, optimizer=optimizer)

    decision = controller.topological_decisions[-1]
    assert decision.action == "birth" and decision.split_destination == 4
    assert moe.num_experts == 5 and len(moe.experts) == 5
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in moe.parameters()
    }
    assert untouched in optimizer.state
    assert all(
        parameter.device == moe.router.weight.device
        and parameter.dtype == moe.router.weight.dtype
        for parameter in moe.experts[-1].parameters()
    )


def test_topological_birth_does_not_require_an_ot_merge_candidate():
    moe = _geometry_moe()
    with torch.no_grad():
        moe.router_logit_bias[1] = 1.0
    _prime_routing(moe, with_gap=True)
    controller = SplitMergeController(
        moe,
        _topological_cfg(
            variable_expert_count=True,
            min_experts=2,
            max_experts=5,
            growth_budget_pct=0.5,
            topological_functional_distance_max=0.0,
        ),
    )
    assert controller._ot_merge_candidates(moe) == []

    controller.step(global_step=10)

    decision = controller.topological_decisions[-1]
    assert decision.mode == "topological" and decision.action == "birth"
    assert decision.merge_pair is None and moe.num_experts == 5


def test_topological_merge_split_respects_ot_cost_ceiling():
    moe = _geometry_moe()
    with torch.no_grad():
        moe.experts[1].fc1.w_slow.add_(0.01 * torch.eye(4))
    _prime_routing(moe, with_gap=True)
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_merge_cost_ratio_max=0.0),
    )

    controller.step(global_step=11)

    decision = controller.topological_decisions[-1]
    assert decision.mode == "uta_fallback"
    assert decision.reason == "ot_pair_above_merge_split_cost_ceiling"


def test_topological_persistence_requires_uncovered_capacity():
    moe = _geometry_moe()
    _prime_routing(moe, with_gap=True)
    controller = SplitMergeController(
        moe,
        _topological_cfg(
            variable_expert_count=True,
            max_experts=5,
            growth_budget_pct=0.5,
            topological_coverage_distance_threshold=2.0,
        ),
    )

    controller.step(global_step=12)

    decision = controller.topological_decisions[-1]
    assert decision.action != "birth"
    assert decision.coverage_distance is not None and decision.coverage_distance < 2.0


def test_topological_monitor_samples_are_bounded_per_tensor():
    moe = _geometry_moe()
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_max_samples_per_tensor=3),
    )

    assert controller._expert_weight_samples(moe, 0).size == 6
    assert all(
        component.size <= 3
        for component in controller._expert_function_components(moe, 0).values()
    )


def test_topological_spectral_work_is_bounded_and_not_repeated(monkeypatch):
    moe = _geometry_moe()
    _prime_routing(moe, with_gap=True)
    controller = SplitMergeController(
        moe,
        _topological_cfg(topological_max_spectral_candidates=1),
    )
    calls = 0
    real_svdvals = torch.linalg.svdvals

    def counted_svdvals(matrix):
        nonlocal calls
        calls += 1
        return real_svdvals(matrix)

    monkeypatch.setattr(torch.linalg, "svdvals", counted_svdvals)

    decision, record = controller._plan_topological_lifecycle(
        moe, step=13, layer_index=0
    )

    assert calls == 1
    assert decision.kappa_bound is not None
    assert record is not None and record.kappa_bound == decision.kappa_bound


def test_topological_exact_merge_rejects_unsampled_functional_difference():
    torch.manual_seed(31)
    cfg = SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=True,
        topological_nas=True,
    )
    moe = SynapticMoE(16, 3, 2, 2, cfg, 0.0).eval()
    with torch.no_grad():
        moe.experts[1].load_state_dict(moe.experts[0].state_dict())
        moe.router.weight[1].copy_(moe.router.weight[0])
        moe.Xi[1].copy_(moe.Xi[0])
        moe.router_embeddings[1].copy_(moe.router_embeddings[0])
        # With a two-point linspace sample, index 1 is deliberately invisible
        # to the shortlist but must be caught by exact verification.
        moe.experts[1].fc1.w_slow.reshape(-1)[1].add_(100.0)
    routing_input = torch.zeros(1, 8, 16)
    routing_input[0, :, 0] = torch.arange(8)
    with torch.no_grad():
        moe(routing_input, update_mem=False)
    controller = SplitMergeController(
        moe,
        _topological_cfg(
            topological_max_samples_per_tensor=2,
            topological_max_exact_merge_candidates=1,
            topological_functional_distance_max=0.01,
        ),
    )

    shortlist = controller._ot_merge_candidates(moe)
    assert shortlist[0][2:4] == (0, 1)
    assert controller._exact_ot_merge_candidate(moe, 0, 1) is None

    decision, _ = controller._plan_topological_lifecycle(
        moe, step=14, layer_index=0
    )
    assert decision.merge_pair != (0, 1)


def test_splitmerge_controller_state_roundtrips_and_validates_growth_budget():
    moe = _geometry_moe()
    controller = SplitMergeController(moe, _topological_cfg(min_step_interval=7))
    controller._last_step = 123
    state = controller.state_dict()

    restored = SplitMergeController(moe, _topological_cfg(min_step_interval=7))
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    with pytest.raises(ValueError, match="inconsistent"):
        restored.load_state_dict({**state, "net_added_experts": 1})
    with pytest.raises(ValueError, match="must be an integer"):
        restored.load_state_dict({**state, "last_step": True})


def test_topological_missing_evidence_falls_back_to_uta_deterministically():
    first = _geometry_moe()
    second = _geometry_moe(seed=22, topological=False)
    second.load_state_dict(first.state_dict())
    for moe in (first, second):
        with torch.no_grad():
            moe.fatigue.copy_(torch.tensor([1.0, 0.1, 0.2, 0.3]))
            moe.energy.fill_(1.0)
    cfg = _topological_cfg(merges_per_call=0, fp_divergence_noise=0.05)

    controllers = [SplitMergeController(moe, cfg) for moe in (first, second)]
    rng_state = torch.random.get_rng_state()
    for controller in controllers:
        torch.random.set_rng_state(rng_state)
        controller.step(global_step=10)

    decision = controllers[0].topological_decisions[-1]
    assert decision.mode == "uta_fallback" and decision.action == "uta"
    assert decision.reason == "missing_routing_points"
    assert controllers[1].topological_decisions == []
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_topological_nas_toggle_is_default_off_and_registered():
    assert not SynapticConfig().topological_nas
    flag = next(mechanism for mechanism in MECHANISMS if mechanism.field == "topological_nas")
    assert not flag.default and not flag.off_value and not flag.default_on
    assert SplitMergeController(_geometry_moe(), SplitMergeConfig()).topological_nas
    assert not SplitMergeController(
        _geometry_moe(topological=False), SplitMergeConfig()
    ).topological_nas
    with pytest.raises(ValueError, match="consistently"):
        SplitMergeController(
            torch.nn.ModuleList(
                [_geometry_moe(topological=True), _geometry_moe(topological=False)]
            ),
            SplitMergeConfig(),
        )
    with pytest.raises(ValueError, match="function_preserving"):
        SplitMergeController(_geometry_moe(), SplitMergeConfig(function_preserving=False))
    with pytest.raises(ValueError, match="functional_distance"):
        SplitMergeConfig(topological_functional_distance_max=-1.0)
    with pytest.raises(ValueError, match="coverage_distance"):
        SplitMergeConfig(topological_coverage_distance_threshold=2.1)
    with pytest.raises(ValueError, match="spectral_candidates"):
        SplitMergeConfig(topological_max_spectral_candidates=0)
    with pytest.raises(ValueError, match="exact_merge_candidates"):
        SplitMergeConfig(topological_max_exact_merge_candidates=0)
