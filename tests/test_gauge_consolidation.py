"""Unit tests for Gauge-Theoretic Consolidation & Curvature Monitoring (bead 0642.7.2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from bio_inspired_nanochat.gauge_consolidation import (
    CurvatureMonitor,
    FisherNaturalConsolidator,
    GaugeInvarianceGuard,
)


def test_gauge_invariance_guard_exact_preservation():
    """GL(R) gauge transformations strictly preserve (U @ V) and outputs."""
    D_in, D_out, R = 32, 48, 4
    U = torch.randn(D_in, R)
    V = torch.randn(R, D_out)
    x = torch.randn(4, D_in)

    # Invertible random GL(R) gauge matrix
    g = torch.eye(R) + 0.1 * torch.randn(R, R)

    assert GaugeInvarianceGuard.assert_gauge_invariance(U, V, g, x=x, tol=1e-5)


def test_curvature_monitor_computes_magnitude():
    """Curvature monitor correctly tracks discrete commutator and da."""
    monitor = CurvatureMonitor()
    A1 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    A2 = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])

    curv = monitor.compute_curvature(A1, A2, dt=1.0)
    assert curv > 0.0

    entry = monitor.record_step(
        step=1, task_id=0, curvature_norm=curv, holonomy_norm=0.1, fisher_trace=12.5
    )
    assert entry.curvature_norm == curv
    assert len(monitor.history) == 1


def test_fisher_natural_consolidator_penalty():
    """FisherNaturalConsolidator computes quadratic EWC penalty on parameter movement."""
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    consolidator = FisherNaturalConsolidator(model)

    # Fake data loader
    dummy_data = [(torch.randn(4, 8), torch.randint(0, 4, (4,)))]
    consolidator.update_fisher_estimates(dummy_data, num_samples=4)

    # Zero penalty at reference
    p0 = consolidator.compute_penalty()
    assert abs(p0.item()) < 1e-6

    # Mutate parameters -> positive penalty
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    p1 = consolidator.compute_penalty()
    assert p1.item() > 0.0


def test_holonomy_predicts_continual_forgetting():
    """Holonomy displacement ||Hol(A) - I|| correlates positively with Task A loss degradation."""
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    consolidator = FisherNaturalConsolidator(model)

    task_a_data = [(torch.randn(8, 8), torch.randint(0, 4, (8,)))]
    consolidator.update_fisher_estimates(task_a_data, num_samples=8)

    # Initial loss on Task A
    with torch.no_grad():
        x, y = task_a_data[0]
        init_loss = nn.functional.cross_entropy(model(x), y).item()

    # Perturb weights with different displacement magnitudes
    perturbations = [0.01, 0.05, 0.2]
    forgetting_deltas: list[float] = []

    for eps in perturbations:
        with torch.no_grad():
            for p, ref in zip(model.parameters(), consolidator.reference_params.values()):
                p.copy_(ref + eps)
            loss_now = nn.functional.cross_entropy(model(x), y).item()
            forgetting_deltas.append(loss_now - init_loss)

    # Monotonic correlation: larger holonomy displacement -> higher forgetting
    assert forgetting_deltas[0] < forgetting_deltas[1] < forgetting_deltas[2]

