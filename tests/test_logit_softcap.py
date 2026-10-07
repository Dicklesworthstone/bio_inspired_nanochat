"""
Language-head parity for GPTSynaptic: the logit softcap (bead vg9.1) and the final norm.

The vanilla GPT head bounds logits via ``softcap*tanh(logits/softcap)`` (softcap=15);
GPTSynaptic previously did not, leaving logits unbounded — a stability regression made
worse by the synaptic attention's unbounded ``log(ε+release)`` bias. These tests pin
the parity behavior: logits are bounded on both the inference and the loss paths, and
the cap is cleanly toggleable (``logit_softcap=0`` disables it for ablation).

The vanilla head also reads ``norm(x)``, the RMS-normalized final residual stream. GPTSynaptic
fed the raw pre-norm stream to its head until 2026-10-07; at 2L/128d that cost +0.13 val bpb
and made the mechanisms-off scaffold lose to vanilla (results/scaffold_diagnosis_2026-10-07.json).
The final-norm tests pin the normalization, its toggle, and that checkpoints saved before the
field existed rebuild without it.

Run:  pytest tests/test_logit_softcap.py -v
"""

from __future__ import annotations

import pytest
import torch

import bio_inspired_nanochat.checkpoint_manager as cm
from bio_inspired_nanochat.checkpoint_manager import (
    checkpoint_model_config,
    save_checkpoint,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig

from _bio_testkit import assert_finite, make_tiny_synaptic, random_tokens

SOFTCAP = 15.0


def _blow_up_head(model, factor: float = 1000.0):
    """Scale the lm_head so RAW (pre-cap) logits are enormous — this is what makes
    the softcap observable (otherwise small models produce already-small logits)."""
    with torch.no_grad():
        model.lm_head.weight.mul_(factor)


@pytest.mark.unit
def test_default_config_has_softcap_parity():
    cfg = GPTSynapticConfig(n_layer=1, n_embd=32, n_head=2, n_kv_head=2, vocab_size=64, sequence_len=16)
    assert cfg.logit_softcap == SOFTCAP, "default must match the vanilla GPT softcap (15)"


@pytest.mark.unit
def test_inference_logits_bounded_by_softcap():
    m = make_tiny_synaptic(seed=0)  # default softcap=15
    _blow_up_head(m)
    logits, _ = m(random_tokens(2, 16))
    amax = logits.abs().max().item()
    assert amax <= SOFTCAP + 1e-4, f"logits must be bounded by softcap, got |max|={amax}"
    assert amax > SOFTCAP - 1.0, "with a blown-up head the cap should be near-saturated"
    assert_finite(logits, "softcapped logits")


@pytest.mark.unit
def test_loss_path_logits_bounded_and_loss_finite():
    m = make_tiny_synaptic(seed=0, train=True)
    _blow_up_head(m)
    x, y = random_tokens(2, 16), random_tokens(2, 16)
    logits, loss = m(x, targets=y)
    assert logits.abs().max().item() <= SOFTCAP + 1e-4
    assert torch.isfinite(loss), "loss must be finite with softcap applied"


@pytest.mark.unit
def test_softcap_zero_disables_capping():
    m = make_tiny_synaptic(seed=0, logit_softcap=0.0)
    _blow_up_head(m)
    logits, _ = m(random_tokens(2, 16))
    assert logits.abs().max().item() > SOFTCAP, "softcap=0 must leave logits uncapped"


@pytest.mark.unit
def test_softcap_formula_is_bounded_monotone_and_near_identity_at_zero():
    # The mathematical property the model relies on: softcap*tanh(z/softcap) is a
    # smooth, strictly-monotone squash, bounded in (-softcap, softcap), and ~identity
    # for small z (so it doesn't distort already-reasonable logits).
    z = torch.linspace(-1000.0, 1000.0, 401)
    capped = SOFTCAP * torch.tanh(z / SOFTCAP)
    # Asymptotes to ±softcap (reaches it exactly in float32 at large |z|).
    assert capped.abs().max().item() <= SOFTCAP                     # bounded
    assert torch.all(capped.diff() >= 0)                           # monotone (flat at saturation)
    assert capped.diff()[len(z) // 2] > 0                           # strictly increasing through 0
    small = torch.linspace(-0.5, 0.5, 11)
    assert torch.allclose(SOFTCAP * torch.tanh(small / SOFTCAP), small, atol=2e-3)  # ~identity near 0


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1).sqrt()


@pytest.mark.unit
def test_head_reads_the_rms_normalized_final_stream_by_default():
    m = make_tiny_synaptic(seed=0)
    assert m.config.final_norm is True
    x = random_tokens(2, 16)
    hidden = m.get_hidden_states(x)
    torch.testing.assert_close(_rms(hidden), torch.ones(hidden.shape[:-1]), rtol=1e-4, atol=1e-4)
    logits, _ = m(x)
    torch.testing.assert_close(m.hidden_to_logits(hidden), logits)


@pytest.mark.unit
def test_final_norm_off_hands_the_raw_stream_to_the_head():
    m = make_tiny_synaptic(seed=0, final_norm=False)
    hidden = m.get_hidden_states(random_tokens(2, 16))
    # The N(0, 1) embedding plus the residual branches: nowhere near unit RMS.
    assert (_rms(hidden) - 1.0).abs().max().item() > 0.05


def _round_trip(tmp_path, monkeypatch, model, *, drop_final_norm: bool):
    cfg = model.config
    architecture = checkpoint_model_config(
        model,
        {k: getattr(cfg, k) for k in ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd")},
    )
    if drop_final_norm:  # what a checkpoint saved before the field existed looks like
        architecture.pop("final_norm")
    save_checkpoint(
        str(tmp_path), 1, model.state_dict(), None,
        {"model_config": architecture, "synapses": True, "synaptic_config": synaptic_config_to_meta(cfg.syn_cfg)},
    )

    class _Tokenizer:
        @staticmethod
        def get_vocab_size():
            return cfg.vocab_size

    monkeypatch.setattr(cm, "get_tokenizer", lambda: _Tokenizer())
    loaded, _, _ = cm.build_model(str(tmp_path), 1, torch.device("cpu"), "eval")
    return loaded


@pytest.mark.unit
@pytest.mark.parametrize("final_norm", [True, False])
def test_checkpoint_round_trips_final_norm(tmp_path, monkeypatch, final_norm):
    model = make_tiny_synaptic(seed=0, final_norm=final_norm)
    model.init_weights()
    loaded = _round_trip(tmp_path, monkeypatch, model, drop_final_norm=False)
    assert loaded.config.final_norm is final_norm
    x = random_tokens(1, 16)
    torch.testing.assert_close(loaded(x)[0], model(x)[0])


@pytest.mark.unit
def test_checkpoint_without_the_field_rebuilds_the_unnormalized_head(tmp_path, monkeypatch):
    legacy = make_tiny_synaptic(seed=0, final_norm=False)
    legacy.init_weights()
    loaded = _round_trip(tmp_path, monkeypatch, legacy, drop_final_norm=True)
    assert loaded.config.final_norm is False
    x = random_tokens(1, 16)
    torch.testing.assert_close(loaded(x)[0], legacy(x)[0])
