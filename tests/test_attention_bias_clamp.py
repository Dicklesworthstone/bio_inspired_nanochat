"""
Bounded log-release attention bias (bead vg9.5).

SynapticCausalSelfAttention injects a biological bias ``lambda_loge * log(ε+release)``
into the attention logits. The normalized release can spike, so without a clamp a
single edge's bias can dominate the softmax and destabilize attention. vg9.5 adds a
configurable symmetric clamp (``SynapticConfig.loge_bias_clamp``; 0 disables). These
tests pin: the default value, the formula bound, model finiteness under an extreme
(injected) release, and that the clamp is actually wired into attention.

Run:  pytest tests/test_attention_bias_clamp.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn

from _bio_testkit import assert_finite, random_tokens, set_seed

# attn_topk < Tk so the bias lands on only SOME edges (otherwise a uniform bias is a
# softmax shift and would be invisible). n_layer=1 keeps it fast.
SMALL = dict(sequence_len=32, vocab_size=97, n_layer=1, n_head=4, n_kv_head=4, n_embd=64)


def _build(clamp: float, **syn) -> GPTSynaptic:
    set_seed(0)
    syn_cfg = SynapticConfig(loge_bias_clamp=clamp, attn_topk=4, **syn)
    # ty can't match **dict unpacking to the dataclass fields here.
    cfg = GPTSynapticConfig(syn_cfg=syn_cfg, **SMALL)  # ty: ignore[invalid-argument-type]
    return GPTSynaptic(cfg).eval()


def _huge_release(self, state, drive, idx, train, valid=None, **_kwargs):
    """Drop-in for SynapticPresyn.release_canonical (the live attention entry point) that
    returns an enormous release everywhere,
    which would make the RAW log-bias ~log(1e6)=13.8 — well past a clamp of 0.5/10."""
    return torch.full_like(drive, 1e6)


@pytest.mark.unit
def test_default_clamp_value():
    assert SynapticConfig().loge_bias_clamp == 10.0


@pytest.mark.unit
def test_clamp_bounds_the_log_bias_formula():
    lam, eps, c = 1.0, 1e-6, 10.0
    e = torch.tensor([0.0, 1e-3, 1.0, 1e6, 1e30])
    raw = lam * torch.log(eps + e)
    bias = raw.clamp(-c, c)
    assert bias.abs().max().item() <= c
    assert raw.max().item() > c, "without clamp, extreme release blows past the bound"
    assert raw.min().item() < -c + 1.0 or eps > 0  # e=0 -> log(eps) ~ -13.8 (clamped to -c)


@pytest.mark.unit
def test_clamp_zero_leaves_formula_unbounded():
    e = torch.tensor([1e30])
    raw = 1.0 * torch.log(1e-6 + e)
    assert raw.item() > 10.0  # clamp=0 path would keep this enormous bias


@pytest.mark.unit
def test_model_forward_finite_under_extreme_release(monkeypatch):
    monkeypatch.setattr(SynapticPresyn, "release_canonical", _huge_release)
    m = _build(clamp=10.0)
    logits, _ = m(random_tokens(2, 16))
    assert_finite(logits, "logits under extreme injected release")
    # Composes with the softcap (vg9.1): logits also bounded by 15.
    assert logits.abs().max().item() <= 15.0 + 1e-4


@pytest.mark.unit
def test_clamp_is_wired_into_attention(monkeypatch):
    # Same weights/input/(injected) release, different clamp -> the bias on the top-k
    # edges differs (0.5 vs 10) while non-top-k edges stay 0, so attention differs.
    monkeypatch.setattr(SynapticPresyn, "release_canonical", _huge_release)
    x = random_tokens(2, 16)
    tight, _ = _build(clamp=0.5)(x)
    loose, _ = _build(clamp=10.0)(x)
    assert torch.isfinite(tight).all() and torch.isfinite(loose).all()
    assert not torch.allclose(tight, loose, atol=1e-3), "loge_bias_clamp must affect attention"
