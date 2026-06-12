"""Weight-tying tests (bead hwxb.2.9).

Verify the Phase-0 embedding-tying decision: wte/lm_head share ONE matrix, the param
budget drops by exactly V·d, the shared weight is optimized exactly once (no AdamW
double-update), it trains as a single weight, and the tie survives a
load_state_dict(assign=True) round-trip via tie_weights(). Fast, CPU-only.
"""
from __future__ import annotations

import pytest

from bio_inspired_nanochat.torch_imports import torch
from _bio_testkit import TINY, make_tiny_synaptic, make_tiny_vanilla

_V, _D = TINY["vocab_size"], TINY["n_embd"]


def _build(synaptic: bool, *, tie: bool, train: bool = False):
    mk = make_tiny_synaptic if synaptic else make_tiny_vanilla
    m = mk(tie_embeddings=tie, train=train)
    m.init_weights()  # tie_weights() runs at the end of init_weights (matches base_train flow)
    m.train(train)
    return m


def _forward_loss(m, x, y, synaptic: bool):
    if synaptic:
        _, loss = m(x, y, None, train_mode=True)
    else:
        loss = m(x, y)
    return loss


@pytest.mark.unit
@pytest.mark.parametrize("synaptic", [False, True], ids=["vanilla", "synaptic"])
def test_tying_shares_weight_and_cuts_params(synaptic):
    untied = _build(synaptic, tie=False)
    tied = _build(synaptic, tie=True)
    # The two weights are the SAME tensor object only when tied.
    assert tied.lm_head.weight is tied.wte.weight
    assert untied.lm_head.weight is not untied.wte.weight
    # parameters() dedupes shared tensors, so the count drops by exactly one V×d matrix.
    n_untied = sum(p.numel() for p in untied.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_untied - n_tied == _V * _D


@pytest.mark.unit
@pytest.mark.parametrize("synaptic", [False, True], ids=["vanilla", "synaptic"])
def test_tied_weight_optimized_exactly_once(synaptic):
    m = _build(synaptic, tie=True)
    opts = m.setup_optimizers()
    seen = [id(p) for opt in opts for g in opt.param_groups for p in g["params"]]
    # No parameter (especially the shared weight) is double-counted across the optimizers.
    assert len(seen) == len(set(seen)), "a parameter was placed in two optimizer groups"
    # The shared weight is present exactly once (not dropped, not duplicated).
    assert seen.count(id(m.wte.weight)) == 1
    # Every model parameter is covered exactly once.
    assert set(seen) == {id(p) for p in m.parameters()}


@pytest.mark.unit
@pytest.mark.parametrize("synaptic", [False, True], ids=["vanilla", "synaptic"])
def test_tied_weight_trains_as_one(synaptic):
    torch.manual_seed(0)
    m = _build(synaptic, tie=True, train=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, _V, (2, 16))
    y = torch.randint(0, _V, (2, 16))
    before = m.wte.weight.detach().clone()
    loss = _forward_loss(m, x, y, synaptic)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # The shared weight receives ONE grad (sum of the embedding + output paths).
    assert m.wte.weight.grad is m.lm_head.weight.grad
    opt.step()
    # After the step the tie still holds and the weight actually moved.
    assert m.wte.weight is m.lm_head.weight
    assert not torch.equal(m.wte.weight, before)


@pytest.mark.unit
@pytest.mark.parametrize("synaptic", [False, True], ids=["vanilla", "synaptic"])
def test_tie_survives_state_dict_roundtrip_with_assign(synaptic):
    """load_state_dict(assign=True) breaks the tie (replaces param objects); tie_weights() restores it.

    This is exactly the path base_train + checkpoint_manager.build_model take on resume.
    """
    src = _build(synaptic, tie=True)
    sd = src.state_dict()
    # Fresh model whose tie has NOT been established yet (init_weights not called).
    mk = make_tiny_synaptic if synaptic else make_tiny_vanilla
    dst = mk(tie_embeddings=True)
    dst.load_state_dict(sd, strict=True, assign=True)
    assert dst.wte.weight is not dst.lm_head.weight, "assign=True should replace params (break tie)"
    dst.tie_weights()
    assert dst.wte.weight is dst.lm_head.weight, "tie_weights() must re-establish the share"
    # And the restored shared weight equals the source's.
    assert torch.equal(dst.wte.weight, src.wte.weight)
