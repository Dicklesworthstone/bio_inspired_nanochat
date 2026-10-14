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


def test_synaptic_moe_live_dispatches_rust_genetics_and_preserves_gradients(monkeypatch):
    """SynapticMoE(native_genetics=True) dispatches through kernels/dispatcher.py on CPU and matches eager forward + backward."""
    import copy

    import bio_inspired_nanochat.kernels.dispatcher as disp
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

    torch.manual_seed(42)
    cfg_eager = SynapticConfig(
        enable_metabolism=True,
        enable_hebbian=False,
        native_genetics=False,
        router_contrastive_lr=0.0,
    )
    cfg_native = SynapticConfig(
        enable_metabolism=True,
        enable_hebbian=False,
        native_genetics=True,
        router_contrastive_lr=0.0,
    )

    moe_eager = SynapticMoE(n_embd=16, num_experts=4, top_k=2, hidden_mult=2, cfg=cfg_eager)
    moe_native = copy.deepcopy(moe_eager)
    object.__setattr__(moe_native, "cfg", cfg_native)

    calls = {"router_stats": 0, "metabolism": 0}
    orig_stats = disp.dispatch_accumulate_router_stats
    orig_metab = disp.dispatch_update_metabolism

    def wrapped_stats(*args, **kwargs):
        calls["router_stats"] += 1
        return orig_stats(*args, **kwargs)

    def wrapped_metab(*args, **kwargs):
        calls["metabolism"] += 1
        return orig_metab(*args, **kwargs)

    monkeypatch.setattr(disp, "dispatch_accumulate_router_stats", wrapped_stats)
    monkeypatch.setattr(disp, "dispatch_update_metabolism", wrapped_metab)

    x_eager = torch.randn(2, 8, 16, requires_grad=True)
    x_native = x_eager.detach().clone().requires_grad_(True)

    out_eager, aux_eager = moe_eager(x_eager, update_mem=True)
    out_native, aux_native = moe_native(x_native, update_mem=True)

    assert calls["router_stats"] == 1
    assert calls["metabolism"] == 1

    torch.testing.assert_close(out_native, out_eager, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(aux_native, aux_eager, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(moe_native.fatigue, moe_eager.fatigue, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(moe_native.energy, moe_eager.energy, rtol=1e-5, atol=1e-5)

    (out_eager.sum() + 0.1 * aux_eager).backward()
    (out_native.sum() + 0.1 * aux_native).backward()

    assert x_native.grad is not None and x_eager.grad is not None
    torch.testing.assert_close(x_native.grad, x_eager.grad, rtol=1e-5, atol=1e-5)
    assert moe_native.router.weight.grad is not None and moe_eager.router.weight.grad is not None
    torch.testing.assert_close(
        moe_native.router.weight.grad, moe_eager.router.weight.grad, rtol=1e-5, atol=1e-5
    )


def test_accumulate_metrics_fused_matches_moe_last_ctx():
    """accumulate_metrics_fused aggregates routing counts and gate sums across SynapticMoE.last_ctx dicts."""
    from bio_inspired_nanochat.metrics_fused import accumulate_metrics_fused
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

    torch.manual_seed(7)
    cfg = SynapticConfig(enable_metabolism=True, enable_hebbian=False, native_genetics=True)
    moe = SynapticMoE(n_embd=16, num_experts=4, top_k=2, hidden_mult=2, cfg=cfg)
    x = torch.randn(2, 6, 16)
    _ = moe(x, update_mem=True)

    counts, probs = accumulate_metrics_fused([moe.last_ctx], num_experts=4)
    assert counts.shape == (4,)
    assert probs.shape == (4,)
    assert torch.isclose(counts.sum(), torch.tensor(float(2 * 6 * 2)))
    assert torch.isclose(probs.sum(), torch.tensor(float(2 * 6)), atol=1e-4)

    # Verify bfloat16 autocast compatibility on CPU (both dispatcher Rust path and SynapticMoE.forward)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16, aux_bf16 = moe(x, update_mem=True)
    assert out_bf16.shape == x.shape
    assert aux_bf16.requires_grad
