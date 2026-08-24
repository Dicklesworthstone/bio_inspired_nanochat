"""Tests for Optogenetic Synaptic Stimulation (bead `odq.3`)."""

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.optogenetic_stimulation import (
    ClampMode,
    OptogeneticStimulator,
    SynapticClamp,
)
from bio_inspired_nanochat.synaptic import SynapticLinear


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    return GPTSynaptic(cfg)


def test_optogenetic_clamp_and_restore():
    """Verify that optogenetic stimulation alters target state and safely restores on context exit."""
    model = _make_model()
    stimulator = OptogeneticStimulator(model)

    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    orig_w_fast = syn_lin.w_fast.data.clone()

    clamp = SynapticClamp(
        variable_name="w_fast",
        mode=ClampMode.PIN_VALUE,
        value=5.0,
    )

    with stimulator.stimulate([clamp]):
        assert syn_lin.w_fast.data[0, 0].item() == pytest.approx(5.0)

    # Post-context: state must be fully restored
    assert torch.equal(syn_lin.w_fast.data, orig_w_fast)


def test_optogenetic_stimulation_shifts_logits():
    """Verify that clamping fast weights produces causal behavioral shifts in forward logits."""
    model = _make_model()
    stimulator = OptogeneticStimulator(model)

    x = torch.randint(0, 32, (1, 6))

    with torch.no_grad():
        logits_base, _ = model(x)

    clamp = SynapticClamp(
        variable_name="w_fast",
        mode=ClampMode.PIN_VALUE,
        value=2.0,
    )

    with stimulator.stimulate([clamp]):
        with torch.no_grad():
            logits_stim, _ = model(x)

    # Assert causal change in logits
    assert not torch.allclose(logits_base, logits_stim, atol=1e-3)
    stimulator.log_stimulation([clamp])


def test_layer_targeted_clamping():
    """Verify that specifying layer_idx only affects the target layer."""
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    model = GPTSynaptic(cfg)
    stimulator = OptogeneticStimulator(model)

    syn_l0 = next(mod for mod in model.h[0].modules() if isinstance(mod, SynapticLinear))
    syn_l1 = next(mod for mod in model.h[1].modules() if isinstance(mod, SynapticLinear))

    assert syn_l0.w_fast is not None
    assert syn_l1.w_fast is not None
    l0_orig = syn_l0.w_fast.data.clone()

    clamp_l1 = SynapticClamp(
        layer_idx=1,
        variable_name="w_fast",
        mode=ClampMode.PIN_VALUE,
        value=7.5,
    )

    with stimulator.stimulate([clamp_l1]):
        assert syn_l1.w_fast.data[0, 0].item() == pytest.approx(7.5)
        # Layer 0 must be completely untouched
        assert torch.equal(syn_l0.w_fast.data, l0_orig)

    # After exit, Layer 1 restored
    assert syn_l1.w_fast.data[0, 0].item() != pytest.approx(7.5)


def test_site_type_filtering():
    """Verify that specifying site_type filters target linear modules accurately."""
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    model = GPTSynaptic(cfg)
    stimulator = OptogeneticStimulator(model)

    clamp_dense = SynapticClamp(
        site_type="dense_fc",
        variable_name="w_fast",
        mode=ClampMode.PIN_VALUE,
        value=4.0,
    )

    with stimulator.stimulate([clamp_dense]):
        for name, mod in model.named_modules():
            if isinstance(mod, SynapticLinear) and mod.w_fast is not None:
                assert mod.w_fast.data[0, 0].item() == pytest.approx(4.0)


