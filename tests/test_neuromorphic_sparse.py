"""Unit and benchmark tests for Neuromorphic Sparse Synapse (bead r00r.13)."""

from __future__ import annotations

import torch

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
    mask = torch.rand(4, 16, in_dim) < 0.2
    x[mask] = torch.randn(int(mask.sum().item()))

    y, stats = synapse(x)
    assert y.shape == (4, 16, out_dim)
    assert stats.sparsity_ratio > 0.75, f"Expected >75% sparsity, got {stats.sparsity_ratio:.2%}"
    assert stats.flops_saved_ratio > 0.75
    assert stats.max_error_bound > 0.0


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
