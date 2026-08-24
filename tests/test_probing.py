"""Unit tests for In-Silico Neuroscience probing, lesion, and stimulation API (beads odq.1, odq.2, odq.3).

Run:
    pytest tests/test_probing.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.probing import (
    PatchClampProbe,
    compute_causal_effect,
    lesion_head,
    lesion_mechanism,
    optogenetic_clamp,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear


def _make_model() -> GPTSynaptic:
    syn_cfg = SynapticConfig(
        enable_hebbian=True,
        bistable_latch=True,
        fast_weight_normalized=True,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    return GPTSynaptic(gpt_cfg).eval()


def test_patch_clamp_probe_records_and_cleans_up():
    """PatchClampProbe collects per-layer snapshots and removes hooks upon exit."""
    model = _make_model()
    x = torch.randint(0, 32, (2, 8))

    with PatchClampProbe(model) as probe:
        with torch.no_grad():
            model.reset_sequence_state(reset_fast_weights=True)
            model(x, train_mode=False)
        traces = probe.get_trace()
        assert len(traces) > 0
        assert any(t["camkii"] is not None for t in traces)

    # After exit, hooks are removed
    assert len(probe.handles) == 0


def test_lesion_head_modifies_output_and_restores():
    """lesion_head knocks out the target head and restores original weights/hooks on exit."""
    model = _make_model()
    x = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        model.reset_sequence_state(reset_fast_weights=True)
        out_orig = model(x, train_mode=False)[0]

    with lesion_head(model, layer_idx=0, head_idx=0):
        with torch.no_grad():
            model.reset_sequence_state(reset_fast_weights=True)
            out_lesioned = model(x, train_mode=False)[0]

    with torch.no_grad():
        model.reset_sequence_state(reset_fast_weights=True)
        out_after = model(x, train_mode=False)[0]

    assert not torch.allclose(out_orig, out_lesioned, atol=1e-4)
    assert torch.allclose(out_orig, out_after, atol=1e-6)


def test_lesion_mechanism_toggles_config():
    """lesion_mechanism temporarily switches off biological pathways and restores them."""
    model = _make_model()

    with lesion_mechanism(model, "hebbian"):
        for lin in model.modules():
            if isinstance(lin, SynapticLinear):
                assert lin.cfg.enable_hebbian is False

    # Restored
    for lin in model.modules():
        if isinstance(lin, SynapticLinear):
            assert lin.cfg.enable_hebbian is True


def test_optogenetic_clamp_pins_values_and_restores():
    """optogenetic_clamp pins synaptic state and cleans up on exit."""
    model = _make_model()

    with optogenetic_clamp(model, target="camkii", value=3.0, layer_idx=0):
        for name, m in model.named_modules():
            if name.startswith("h.0") and hasattr(m, "post") and hasattr(m.post, "camkii"):
                assert m.post.camkii.mean().item() == pytest.approx(3.0)

    with optogenetic_clamp(model, target="dopamine", value=2.5):
        for m in model.modules():
            if hasattr(m, "_nm_da_gain"):
                assert m._nm_da_gain == pytest.approx(2.5)


def test_compute_causal_effect():
    """compute_causal_effect produces valid MSE, KL divergence, and flip rates."""
    logits_a = torch.randn(2, 4, 10)
    logits_b = logits_a + 0.5 * torch.randn(2, 4, 10)

    effect = compute_causal_effect(logits_a, logits_b)
    assert effect["logit_mse"] > 0
    assert effect["kl_divergence"] >= 0
    assert 0.0 <= effect["prediction_flip_rate"] <= 1.0
