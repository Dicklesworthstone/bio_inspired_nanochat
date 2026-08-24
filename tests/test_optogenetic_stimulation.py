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
