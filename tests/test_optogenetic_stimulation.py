"""Tests for Optogenetic Synaptic Stimulation (bead `odq.3`)."""

import math

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


def test_overlapping_clamps_restore_original_state():
    """Multiple interventions on one variable unwind to the pre-context value."""
    model = _make_model()
    stimulator = OptogeneticStimulator(model)
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    original = syn_lin.w_fast.detach().clone()

    clamps = [
        SynapticClamp(variable_name="w_fast", mode=ClampMode.PIN_VALUE, value=1.0),
        SynapticClamp(variable_name="w_fast", mode=ClampMode.PIN_VALUE, value=2.0),
    ]
    with stimulator.stimulate(clamps):
        assert syn_lin.w_fast[0, 0].item() == pytest.approx(2.0)

    assert torch.equal(syn_lin.w_fast, original)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layer_idx": -1}, "layer_idx"),
        ({"layer_idx": 1.5}, "layer_idx"),
        ({"site_type": "typo"}, "site_type"),
        ({"variable_name": "typo"}, "variable_name"),
        ({"mode": "pin_value"}, "ClampMode"),
        ({"value": math.nan}, "finite"),
        ({"value": math.inf}, "finite"),
    ],
)
def test_invalid_clamp_specifications_fail_closed(kwargs, match):
    """Malformed experiments must not silently broaden or skip causal interventions."""
    with pytest.raises(ValueError, match=match):
        SynapticClamp(**kwargs)


def test_unmatched_clamp_rolls_back_other_interventions():
    """A partially invalid experiment fails atomically instead of leaving a clamp applied."""
    model = _make_model()
    stimulator = OptogeneticStimulator(model)
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    original = syn_lin.w_fast.detach().clone()
    clamps = [
        SynapticClamp(variable_name="w_fast", value=2.0),
        SynapticClamp(layer_idx=99, variable_name="w_fast", value=3.0),
    ]

    with pytest.raises(ValueError, match="matched no synaptic sites"):
        stimulator.apply_clamps(clamps)

    assert torch.equal(syn_lin.w_fast, original)


def test_pin_value_remains_clamped_during_model_execution():
    """A PIN intervention is a maintained clamp, not a one-time assignment."""
    model = _make_model()
    stimulator = OptogeneticStimulator(model)
    synaptic_layers = [
        mod
        for mod in model.modules()
        if isinstance(mod, SynapticLinear) and mod.w_fast is not None
    ]
    originals = [mod.w_fast.detach().clone() for mod in synaptic_layers]
    clamp = SynapticClamp(variable_name="w_fast", value=2.0)

    with stimulator.stimulate([clamp]), torch.no_grad():
        model(torch.randint(0, 32, (1, 6)))
        assert all(torch.all(mod.w_fast == 2.0) for mod in synaptic_layers)

    assert all(
        torch.equal(mod.w_fast, original)
        for mod, original in zip(synaptic_layers, originals)
    )


def test_maintained_pin_rejects_ambiguous_mixed_mode_overlap():
    model = _make_model()
    stimulator = OptogeneticStimulator(model)
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    original = syn_lin.w_fast.detach().clone()
    clamps = [
        SynapticClamp(variable_name="w_fast", mode=ClampMode.PIN_VALUE, value=1.0),
        SynapticClamp(variable_name="w_fast", mode=ClampMode.ADD_DELTA, value=1.0),
    ]

    with pytest.raises(ValueError, match="cannot overlap"):
        with stimulator.stimulate(clamps):
            pass

    assert torch.equal(syn_lin.w_fast, original)


def test_rrp_clamp_engages_on_live_release_path():
    """Regression (wave-two review): the calcium/RRP clamps used forward hooks
    on SynapticPresyn, but the live attention path calls release_canonical as a
    plain bound method — hooks never fired, so clamps were silent no-ops and
    causal experiments reported false nulls. The instance-level wrapper must be
    installed while active, removed on exit, and must actually suppress RRP
    relative to an unclamped run (refill from the reserve ring means the pinned
    value shows up as suppression, not exact zeros, after release consumes it)."""

    from bio_inspired_nanochat.probing import optogenetic_clamp
    from bio_inspired_nanochat.synaptic import SynapticPresyn

    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    prompt = torch.randint(0, 32, (1, 4))
    presyn = next(m for m in model.modules() if isinstance(m, SynapticPresyn))

    def _mean_rrp() -> float:
        states = model._last_presyn_state or []
        vals = [float(s["RRP"].mean()) for s in states if s and "RRP" in s]
        return sum(vals) / len(vals)

    with torch.no_grad():
        model(prompt)  # prime telemetry/state once for a comparable baseline
        baseline_rrp = _mean_rrp()

        with optogenetic_clamp(model, target="rrp", value=0.0):
            # Wiring proof: the instance attribute shadows the class method.
            assert "release_canonical" in presyn.__dict__
            model(prompt)
            clamped_rrp = _mean_rrp()

        assert "release_canonical" not in presyn.__dict__, "wrapper must be removed on exit"
        model(prompt)
    restored_rrp = _mean_rrp()

    assert clamped_rrp < baseline_rrp, (
        f"clamped RRP ({clamped_rrp:.3f}) must sit below the unclamped level "
        f"({baseline_rrp:.3f}): pinning to zero before every release drains the pool"
    )
    assert restored_rrp > 0.0
