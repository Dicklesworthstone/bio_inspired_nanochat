"""Unit & benchmark tests for cross-pollinated Reversible & Simplicial modules (beads 6cb, 3zd)."""

from __future__ import annotations

import torch

from bio_inspired_nanochat.xpoll import ReversibleBlock, SimplicialAttention


def test_reversible_block_exact_reconstruction():
    """ReversibleBlock achieves exact bitwise/numerical reconstruction in inverse()."""
    block = ReversibleBlock(d_model=64)
    block.eval()

    x = torch.randn(4, 16, 64)
    y = block(x)
    x_reconstructed = block.inverse(y)

    diff = (x - x_reconstructed).abs().max().item()
    assert diff < 1e-5, f"Reconstruction error too high: {diff}"


def test_reversible_block_gradcheck_and_train():
    """ReversibleBlock forward/backward runs cleanly during training."""
    block = ReversibleBlock(d_model=32)
    block.train()

    x = torch.randn(2, 8, 32, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_simplicial_attention_shapes_and_mixing():
    """SimplicialAttention mixes 1-hop and 2-hop attention within convex hull."""
    attn = SimplicialAttention(d_model=64, n_heads=4, initial_lambda=0.2)
    x = torch.randn(2, 12, 64, requires_grad=True)

    y = attn(x)
    assert y.shape == (2, 12, 64)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert attn.simplex_logit.grad is not None
    assert torch.isfinite(attn.simplex_logit.grad).all()
