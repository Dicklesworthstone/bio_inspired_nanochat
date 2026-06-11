"""Router lateral-inhibition (contrastive) update — bead fkkc.

The README advertises "router embeddings + contrastive update" / "Lateral Inhibition:
forces experts to specialize" (router_contrastive_lr / router_contrastive_push). The
params were read by an update block in SynapticMoE.forward, but its co-occurrence matrix
was diagonal-only, so the pull term was identically zero and the net update did NOT
reduce pairwise similarity — i.e. it was wired but ineffective.

fkkc replaces it with a pure similarity-weighted repulsion that provably spreads the
router embeddings apart. These tests lock the claimed effect and the disable switch.

Run:  pytest tests/test_router_contrastive.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

pytestmark = pytest.mark.unit


def _moe(push: float, lr: float, num_experts: int = 6) -> SynapticMoE:
    cfg = SynapticConfig(router_contrastive_push=push, router_contrastive_lr=lr)
    return SynapticMoE(n_embd=8, num_experts=num_experts, top_k=2, hidden_mult=1, cfg=cfg)


def _collapse_embeddings(moe: SynapticMoE, jitter: float = 0.05) -> None:
    """Force all router embeddings near a single direction (worst case for diversity)."""
    with torch.no_grad():
        d = moe.cfg.router_embed_dim
        base = torch.randn(d)
        base = base / base.norm()
        emb = base.unsqueeze(0).repeat(moe.num_experts, 1) + jitter * torch.randn(
            moe.num_experts, d
        )
        moe.router_embeddings.copy_(emb / emb.norm(dim=-1, keepdim=True))


def _mean_offdiag_cos(emb: torch.Tensor) -> float:
    e = emb / emb.norm(dim=-1, keepdim=True)
    s = e @ e.T
    n = e.shape[0]
    return float((s.sum() - n) / (n * n - n))


def test_lateral_inhibition_spreads_embeddings_apart():
    torch.manual_seed(0)
    moe = _moe(push=0.1, lr=1e-2)  # larger lr so the effect is visible in few steps
    _collapse_embeddings(moe)
    before = _mean_offdiag_cos(moe.router_embeddings)
    x = torch.randn(4, 10, 8)
    for _ in range(50):
        moe(x)
    after = _mean_offdiag_cos(moe.router_embeddings)
    assert after < before - 1e-3, (
        f"lateral inhibition must reduce mean pairwise similarity "
        f"(before={before:.4f}, after={after:.4f}) — this is the claimed specialization"
    )


def test_disabled_when_push_is_zero():
    torch.manual_seed(1)
    moe = _moe(push=0.0, lr=1e-2)
    _collapse_embeddings(moe)
    before = moe.router_embeddings.clone()
    x = torch.randn(4, 10, 8)
    for _ in range(10):
        moe(x)
    assert torch.allclose(before, moe.router_embeddings, atol=1e-6), "push=0 must disable"


def test_disabled_when_lr_is_zero():
    torch.manual_seed(2)
    moe = _moe(push=0.1, lr=0.0)
    _collapse_embeddings(moe)
    before = moe.router_embeddings.clone()
    x = torch.randn(4, 10, 8)
    for _ in range(10):
        moe(x)
    assert torch.allclose(before, moe.router_embeddings, atol=1e-6), "lr=0 must disable"


def test_embeddings_stay_unit_norm_and_finite():
    torch.manual_seed(3)
    moe = _moe(push=0.5, lr=5e-2)  # aggressive settings
    _collapse_embeddings(moe)
    x = torch.randn(4, 10, 8)
    for _ in range(30):
        moe(x)
    norms = moe.router_embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "must stay unit-norm"
    assert torch.isfinite(moe.router_embeddings).all(), "must stay finite"


def test_update_is_live_in_forward():
    # A single forward must change the embeddings when enabled and they are not yet
    # maximally separated — proving the mechanism is wired into the live path.
    torch.manual_seed(4)
    moe = _moe(push=0.1, lr=1e-2)
    _collapse_embeddings(moe)
    before = moe.router_embeddings.clone()
    moe(torch.randn(4, 10, 8))
    assert not torch.allclose(before, moe.router_embeddings), "update must run in forward"
