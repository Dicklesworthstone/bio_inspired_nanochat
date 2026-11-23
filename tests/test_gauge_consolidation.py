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


def test_parameter_displacement_can_leave_task_predictions_unchanged():
    """A common class-logit offset changes parameters while preserving every softmax probability."""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(73)
        model = nn.Linear(8, 4)
        x = torch.randn(16, 8)
        target = torch.arange(16) % 4
        before = model(x)
        model.bias.add_(2.0)
        after = model(x)
        torch.testing.assert_close(after - before, torch.full_like(before, 2.0))
        torch.testing.assert_close(before.softmax(-1), after.softmax(-1))
        torch.testing.assert_close(nn.functional.cross_entropy(before, target), nn.functional.cross_entropy(after, target))

