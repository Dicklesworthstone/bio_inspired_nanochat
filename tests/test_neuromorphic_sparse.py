"""Unit and benchmark tests for Neuromorphic Sparse Synapse (bead r00r.13)."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F

from bio_inspired_nanochat.neuromorphic_sparse import EventDrivenSparseSynapse


def test_event_driven_sparse_synapse_skips_quiescent_channels():
    """EventDrivenSparseSynapse achieves high sparsity and respects theoretical error bound."""
    in_dim, out_dim = 32, 64
    synapse = EventDrivenSparseSynapse(
        in_features=in_dim,
        out_features=out_dim,
        event_threshold=0.1,
    )

    # Input with 80% quiescent zeros/low values
    x = torch.zeros(4, 16, in_dim)
    # Activate only 20% of entries
    generator = torch.Generator().manual_seed(927)
    mask = torch.rand(4, 16, in_dim, generator=generator) < 0.2
    x[mask] = torch.randn(int(mask.sum().item()), generator=generator)

    y, stats = synapse(x)
    assert y.shape == (4, 16, out_dim)
    assert stats.sparsity_ratio > 0.75, f"Expected >75% sparsity, got {stats.sparsity_ratio:.2%}"
    assert stats.projection_backend == "sparse_coo"
    assert stats.executed_projection_macs == int((x.abs() > 0.1).sum()) * out_dim
    assert stats.dense_projection_macs == x.numel() * out_dim
    assert 1 - stats.executed_projection_macs / stats.dense_projection_macs > 0.75
    assert stats.max_error_bound > 0.0
    assert (y - F.linear(x, synapse.weight)).abs().max().item() <= stats.max_error_bound


def test_event_driven_sparse_exact_at_zero_threshold():
    """At event_threshold=0, event-driven synapse exactly matches dense linear."""
    in_dim, out_dim = 16, 32
    synapse = EventDrivenSparseSynapse(
        in_features=in_dim,
        out_features=out_dim,
        event_threshold=0.0,
    )

    x = torch.randn(2, 8, in_dim)
    y, stats = synapse(x)
    y_dense = torch.nn.functional.linear(x, synapse.weight, synapse.bias)

    torch.testing.assert_close(y, y_dense, rtol=1e-5, atol=1e-5)
    assert stats.sparsity_ratio == 0.0
    assert stats.projection_backend == "dense"
    assert stats.max_error_bound == 0.0


def test_quiet_external_calcium_cannot_drop_a_large_input():
    # November 22 audit: the old module produced 0, claimed error <= 0.1
    # and claimed 100% FLOP savings, after doing dense GEMM with the input masked.
    synapse = EventDrivenSparseSynapse(1, 1, event_threshold=0.1)
    with torch.no_grad():
        synapse.weight.fill_(1.0)
    x = torch.tensor([100.0])
    y, stats = synapse(x, torch.zeros_like(x))
    assert y.item() == 100.0
    assert stats.max_error_bound == 0.0
    assert stats.executed_projection_macs == stats.dense_projection_macs == 1


def test_bound_uses_row_norm_and_respects_broadcast_calcium_and_threshold_equality():
    synapse = EventDrivenSparseSynapse(4, 1, event_threshold=0.125, bias=True).double()
    with torch.no_grad():
        synapse.weight.fill_(1.0)
        synapse.bias.fill_(2.0)
    x = torch.full((2, 4), 0.125, dtype=torch.double)
    activity = torch.tensor([[0.125], [1.0]])  # broadcasts across four features
    y, stats = synapse(x, activity)
    assert stats.projection_backend == "dense_masked"
    assert stats.active_elements == 4 and stats.total_elements == 8
    assert stats.executed_projection_macs == stats.dense_projection_macs == 8
    torch.testing.assert_close(y, torch.tensor([[2.0], [2.5]], dtype=torch.double))
    actual = (F.linear(x, synapse.weight, synapse.bias) - y).abs().max().item()
    assert actual == stats.max_error_bound == 0.5  # old column-norm formula yielded 0.125


@pytest.mark.parametrize("active_per_row", [0, 1, 2, 3, 8])
def test_random_projection_and_gradients_against_dense_control(active_per_row):
    rng = torch.Generator().manual_seed(1977)
    synapse = EventDrivenSparseSynapse(8, 5, event_threshold=0.25, bias=True).double()
    assert synapse.bias is not None
    x = torch.rand((5, 3, 8), generator=rng, dtype=torch.double) * 0.25
    x[..., :active_per_row] += 1
    x.requires_grad_()
    active_mask = x.detach().abs() > 0.25
    retained = x.masked_fill(~active_mask, 0)
    reference = F.linear(retained, synapse.weight, synapse.bias)
    dense = F.linear(x, synapse.weight, synapse.bias)
    y, stats = synapse(x)
    torch.testing.assert_close(y, reference, rtol=1e-12, atol=1e-12)
    assert (dense - y).abs().max().item() <= stats.max_error_bound + 1e-12
    upstream = torch.randn(y.shape, generator=rng, dtype=torch.double)
    operands = (x, synapse.weight, synapse.bias)
    for got, expected in zip(torch.autograd.grad(y, operands, upstream),
                             torch.autograd.grad(reference, operands, upstream), strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    expected_backend = "sparse_coo" if active_per_row <= 2 else "dense_masked" if active_per_row < 8 else "dense"
    assert stats.projection_backend == expected_backend


def test_sparse_path_invokes_sparse_matrix_multiply():
    synapse = EventDrivenSparseSynapse(32, 8)
    x = torch.zeros(8, 32)
    x[:, 5] = 1
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as recorded:
        y, stats = synapse(x)
    operators = {op.key for op in recorded.key_averages()}
    assert "aten::_sparse_mm" in operators
    assert "aten::linear" not in operators
    assert stats.executed_projection_macs == 8 * 8
    torch.testing.assert_close(y, F.linear(x, synapse.weight))


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_invalid_event_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        EventDrivenSparseSynapse(4, 2, threshold)


@pytest.mark.slow
def test_cpu_dense_and_sparse_latency_at_bounded_error():
    """Time complete inference calls with the incumbent present in the same benchmark."""
    from torch.utils.benchmark import Timer

    rng = torch.Generator().manual_seed(1432)
    synapse = EventDrivenSparseSynapse(512, 512, event_threshold=0.01).eval()
    with torch.inference_mode():
        synapse.weight.uniform_(-0.04, 0.04, generator=rng)
        for fraction in (0.01, 0.20, 1.0):
            x = torch.rand((256, 512), generator=rng) * 0.005
            keep = torch.rand(x.shape, generator=rng) < fraction
            x[keep] += 1.0
            y, stats = synapse(x)
            dense = F.linear(x, synapse.weight)
            err = (y - dense).abs().max().item()
            assert err <= stats.max_error_bound + 1e-5
            args = {"F": F, "x": x, "module": synapse}
            dense_time = Timer("F.linear(x, module.weight)", globals=args, num_threads=1).blocked_autorange(min_run_time=0.35)
            sparse_time = Timer("module(x)", globals=args, num_threads=1).blocked_autorange(min_run_time=0.35)
            logging.getLogger(__name__).info(
                "CPU FP32, 1 thread, 256x512->512, backend=%s active=%d/%d "
                "dense_us=%.2f module_us=%.2f speedup=%.3f worst_abs_error=%.7g bound=%.7g",
                stats.projection_backend, stats.active_elements, stats.total_elements,
                dense_time.median * 1e6, sparse_time.median * 1e6, dense_time.median / sparse_time.median,
                err, stats.max_error_bound,
            )
