"""Unit and stress tests for Homeostatic Safety Guard (bead re4e.13)."""

from __future__ import annotations

import torch

from bio_inspired_nanochat.homeostatic_safety import HomeostaticSafetyGuard


def test_homeostatic_safety_guard_accepts_safe_state():
    """Safe synaptic state requires no intervention and maintains positive barrier margin."""
    guard = HomeostaticSafetyGuard(max_fast_weight_norm=5.0, max_energy_budget=2.0)

    w_fast = torch.randn(8, 8) * 0.1  # norm << 5.0
    energy = torch.ones(8) * 0.5      # energy << 2.0

    w_safe, e_safe, report = guard.enforce_safety(w_fast=w_fast, energy=energy)

    assert report.is_safe
    assert not report.intervention_applied
    assert report.min_margin > 0.0
    assert w_safe is not None and torch.equal(w_safe, w_fast)


def test_homeostatic_safety_guard_prevents_runaway_potentiation():
    """Safety guard strictly bounds fast weight norm when runaway potentiation occurs."""
    max_norm = 4.0
    margin = 0.1
    guard = HomeostaticSafetyGuard(max_fast_weight_norm=max_norm, gamma_margin=margin)

    # Explosive runaway fast weights: norm = 100.0
    w_explosive = torch.ones(10, 10) * 10.0

    w_safe, _, report = guard.enforce_safety(w_fast=w_explosive)

    assert report.intervention_applied
    assert w_safe is not None

    safe_norm = float(torch.linalg.norm(w_safe).item())
    assert safe_norm <= max_norm - margin + 1e-5, (
        f"Safe norm {safe_norm:.4f} exceeded upper limit {max_norm - margin:.4f}"
    )


def test_homeostatic_safety_guard_clamps_energy_budget():
    """Safety guard clamps excessive energy dissipation to safe upper bound."""
    max_energy = 1.5
    guard = HomeostaticSafetyGuard(max_energy_budget=max_energy, gamma_margin=0.05)

    excessive_energy = torch.tensor([1.2, 2.5, 3.0])
    _, e_safe, report = guard.enforce_safety(energy=excessive_energy)

    assert report.intervention_applied
    assert e_safe is not None
    assert (e_safe <= max_energy).all()
