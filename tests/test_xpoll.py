"""Unit & benchmark tests for cross-pollinated Reversible & Simplicial modules (beads 6cb, 3zd)."""

from __future__ import annotations

import copy

import pytest
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


@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("freeze_f", [False, True])
def test_reversible_block_matches_eager_optimizer_step(input_grad, freeze_f):
    torch.manual_seed(612)
    block = ReversibleBlock(d_model=8).double()
    block.f.requires_grad_(not freeze_f)
    reference = copy.deepcopy(block)
    x = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=input_grad)
    x_ref = x.detach().clone().requires_grad_(input_grad)
    target = torch.randn_like(x)
    before = {name: p.detach().clone() for name, p in block.named_parameters()}

    x1, x2 = x_ref.chunk(2, dim=-1)
    y1 = x1 + reference.f(x2)
    expected = torch.cat((y1, x2 + reference.g(y1)), dim=-1)
    actual = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual - target).square().mean().backward()
    (expected - target).square().mean().backward()
    if input_grad:
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
    ref_params = dict(reference.named_parameters())
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().max() > 0, name
            torch.testing.assert_close(p.grad, ref_params[name].grad, rtol=0, atol=0)
        else:
            assert p.grad is None
    torch.optim.SGD(block.parameters(), lr=0.1).step()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    for name, p in block.named_parameters():
        torch.testing.assert_close(p, ref_params[name], rtol=0, atol=0)
        assert torch.equal(p, before[name]) == (not p.requires_grad), name


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
