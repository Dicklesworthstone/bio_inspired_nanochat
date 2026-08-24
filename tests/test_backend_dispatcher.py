"""Unit and parity tests for Unified Backend Dispatcher (bead jyb.6)."""

from __future__ import annotations

import torch

from bio_inspired_nanochat.kernels.dispatcher import (
    Backend,
    dispatch_accumulate_router_stats,
    dispatch_update_metabolism,
    get_available_backends,
    select_backend,
)


def test_select_backend_auto_and_override():
    """select_backend selects appropriate backend by device and honors overrides."""
    available = get_available_backends()
    assert Backend.PYTORCH in available

    # Override PyTorch always works
    b_pt = select_backend("cpu", override=Backend.PYTORCH)
    assert b_pt == Backend.PYTORCH

    # Auto selection on CPU
    b_auto_cpu = select_backend("cpu")
    if Backend.RUST in available:
        assert b_auto_cpu == Backend.RUST
    else:
        assert b_auto_cpu == Backend.PYTORCH


def test_dispatcher_accumulate_router_stats_parity():
    """Dispatching router stats produces identical results across backends."""
    B, T, K, num_experts = 2, 64, 2, 8
    indices = torch.randint(0, num_experts, (B, T, K))
    gates = torch.rand(B, T, K)

    # Reference eager PyTorch
    counts_pt, sums_pt = dispatch_accumulate_router_stats(
        indices, gates, num_experts, backend=Backend.PYTORCH
    )

    # Auto dispatch (Rust or PyTorch on CPU)
    counts_auto, sums_auto = dispatch_accumulate_router_stats(
        indices, gates, num_experts, backend=Backend.AUTO
    )

    torch.testing.assert_close(counts_pt, counts_auto, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sums_pt, sums_auto, rtol=1e-4, atol=1e-4)


def test_dispatcher_update_metabolism_parity():
    """Dispatching metabolism updates produces identical results across backends."""
    num_experts = 8
    fatigue = torch.rand(num_experts)
    energy = torch.rand(num_experts)
    alpha_f = torch.rand(num_experts) * 0.1
    alpha_e = torch.rand(num_experts) * 0.1
    util = torch.rand(num_experts)

    # Reference eager PyTorch
    f_pt, e_pt = dispatch_update_metabolism(
        fatigue, energy, alpha_f, alpha_e, util, backend=Backend.PYTORCH
    )

    # Auto dispatch (Rust or PyTorch on CPU)
    f_auto, e_auto = dispatch_update_metabolism(
        fatigue, energy, alpha_f, alpha_e, util, backend=Backend.AUTO
    )

    torch.testing.assert_close(f_pt, f_auto, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(e_pt, e_auto, rtol=1e-5, atol=1e-5)
