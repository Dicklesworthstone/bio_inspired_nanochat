"""
Global neuromodulatory bus — DA / ACh / NE (bead hy8.1).

Acceptance: neuromodulator signals are COMPUTED from model signals, BROADCAST to the synapses,
GATE real quantities (plasticity / exploration / global gain), exposed via TELEMETRY, and an
ABLATION shows they change learning/exploration. These tests lock each clause.

  • DA (dopamine ≈ reward-prediction error)  -> plasticity gain on the online Hebbian write.
  • ACh (acetylcholine ≈ uncertainty/entropy) -> stochastic vesicle-release fraction (exploration).
  • NE (norepinephrine ≈ arousal/novelty)     -> global output gain (+ optional reset).

Run:  pytest tests/test_neuromod.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear, SynapticPresyn

from _bio_testkit import make_tiny_synaptic, set_seed

IN, OUT, B = 16, 16, 4


def _make_lin(**over) -> SynapticLinear:
    set_seed(0)
    cfg = SynapticConfig(enable_hebbian=True, enable_metabolism=True, fast_weight_max_norm=0.0, **over)
    return SynapticLinear(IN, OUT, cfg)


# --------------------------------------------------------------------------- #
# 1. SIGNALS ARE COMPUTED with the right sign
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dopamine_tracks_reward_prediction_error():
    # loss improvement => positive DA => plasticity gain > 1
    up = NeuromodulatoryBus()
    up.update(loss=2.0)  # seed baseline
    up.update(loss=1.0)  # improved
    assert up.levels()["da"] > 0 and up.gains()["plasticity"] > 1.0

    # loss worsening => negative DA => plasticity gain < 1 (don't consolidate bad steps)
    down = NeuromodulatoryBus()
    down.update(loss=1.0)
    down.update(loss=3.0)
    assert down.levels()["da"] < 0 and down.gains()["plasticity"] < 1.0


@pytest.mark.unit
def test_dopamine_from_explicit_reward():
    bus = NeuromodulatoryBus()
    bus.update(reward=0.0)
    bus.update(reward=1.0)  # positive reward-prediction error
    assert bus.levels()["da"] > 0


@pytest.mark.unit
def test_acetylcholine_rises_with_uncertainty():
    bus = NeuromodulatoryBus()
    bus.update(entropy=1.0)
    bus.update(entropy=3.0)  # rising uncertainty
    assert bus.levels()["ach"] > 0 and bus.gains()["explore"] > 1.0


@pytest.mark.unit
def test_norepinephrine_rises_with_novelty():
    bus = NeuromodulatoryBus()
    bus.update(loss=1.0)
    bus.update(loss=5.0)  # large surprise
    assert bus.levels()["ne"] > 0 and bus.gains()["global"] > 1.0


@pytest.mark.unit
def test_disabled_bus_is_inert():
    bus = NeuromodulatoryBus(NeuromodConfig(enabled=False))
    bus.update(loss=2.0)
    bus.update(loss=1.0)
    assert bus.levels() == {"da": 0.0, "ach": 0.0, "ne": 0.0}
    assert bus.gains() == {"plasticity": 1.0, "explore": 1.0, "attend": 1.0, "global": 1.0}


# --------------------------------------------------------------------------- #
# 2. BROADCAST sets gains on every synaptic layer
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_broadcast_sets_gains_on_synaptic_layers():
    model = make_tiny_synaptic(seed=0)
    bus = NeuromodulatoryBus()
    bus.update(loss=2.0)
    bus.update(loss=1.0, entropy=2.0)
    n = bus.broadcast(model)

    lins = [m for m in model.modules() if isinstance(m, SynapticLinear)]
    pres = [m for m in model.modules() if isinstance(m, SynapticPresyn)]
    assert n == len(lins) + len(pres) and n > 0
    g = bus.gains()
    assert all(getattr(m, "_nm_da_gain", None) == g["plasticity"] for m in lins)
    assert all(getattr(m, "_nm_ne_gain", None) == g["global"] for m in lins)
    assert all(getattr(m, "_nm_ach_gain", None) == g["explore"] for m in pres)


# --------------------------------------------------------------------------- #
# 3. GATING REAL QUANTITIES (+ ablation)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dopamine_gain_scales_the_hebbian_write():
    # One inference pass from a zero fast-weight start => ||w_fast|| ∝ post_fast_lr * DA * delta.
    # Same seed/input, so a 3× DA gain must give ~3× the write magnitude (DA gates learning).
    x = torch.randn(B, IN)
    ca, en = torch.ones(B), torch.ones(B)

    def write_norm(da_gain: float) -> float:
        lin = _make_lin().eval()
        lin.reset_sequence_state(reset_fast_weights=True)
        lin._nm_da_gain = da_gain
        with torch.no_grad():
            lin(x, ca, en)
        return lin.w_fast.norm().item()

    base = write_norm(1.0)
    boosted = write_norm(3.0)
    assert base > 0, "precondition: there is a nonzero write to scale"
    assert boosted == pytest.approx(3.0 * base, rel=1e-4), "DA gain must scale the Hebbian write"


@pytest.mark.unit
def test_norepinephrine_gain_scales_the_output():
    # Two fresh identical layers (same seed) so plasticity carryover can't confound the
    # comparison: the only difference is the NE output gain.
    x = torch.randn(B, IN)
    ca, en = torch.ones(B), torch.ones(B)
    lin1 = _make_lin().eval()
    lin2 = _make_lin().eval()
    lin2._nm_ne_gain = 2.0
    with torch.no_grad():
        y1 = lin1(x, ca, en)
        y2 = lin2(x, ca, en)
    assert torch.allclose(y2, 2.0 * y1, atol=1e-5), "NE gain must scale the synaptic output"


@pytest.mark.unit
def test_acetylcholine_gain_gates_stochastic_exploration():
    # ACh=0 -> stochastic release fraction 0 -> deterministic across train-mode forwards.
    # Default (no broadcast, frac=0.12) -> stochastic -> forwards differ. So ACh gates exploration.
    V = 97
    seq = torch.randint(0, V, (1, 16))

    def two_forwards(ach_gain):
        def _one_fwd(s):
            m = make_tiny_synaptic(seed=0, train=True)
            if ach_gain is not None:
                for mod in m.modules():
                    if isinstance(mod, SynapticPresyn):
                        mod._nm_ach_gain = ach_gain
            with torch.no_grad():
                set_seed(s)
                out, _ = m(seq)
                return out

        a = _one_fwd(123)
        b = _one_fwd(456)
        return (a - b).abs().max().item()

    explore_off = two_forwards(0.0)   # ACh=0 -> no stochastic release
    explore_default = two_forwards(None)  # default frac -> stochastic
    assert explore_off == pytest.approx(0.0, abs=1e-6), "ACh=0 must disable stochastic exploration"
    assert explore_default > explore_off, "default ACh must permit stochastic exploration"


# --------------------------------------------------------------------------- #
# 4. DEFAULT-NEUTRAL: no broadcast (or neutral gains) == un-modulated behavior
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_neutral_broadcast_does_not_change_eval_output():
    # Two fresh identical models; one gets a neutral broadcast. Same reset state + single
    # forward, so any difference would be the broadcast (it must be a no-op at neutral gains).
    seq = torch.randint(0, 97, (1, 16))
    m_plain = make_tiny_synaptic(seed=0).eval()
    m_bcast = make_tiny_synaptic(seed=0).eval()
    NeuromodulatoryBus().broadcast(m_bcast)  # no signals -> levels 0 -> gains all 1.0
    with torch.no_grad():
        # train_mode=False -> no stochastic release, so the forward is deterministic and the
        # only thing that could differ between the two models is the (neutral) broadcast.
        a, _ = m_plain(seq, train_mode=False)
        b, _ = m_bcast(seq, train_mode=False)
    assert torch.allclose(a, b, atol=1e-6), "neutral neuromodulation must be a no-op"


# --------------------------------------------------------------------------- #
# 5. TELEMETRY
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_telemetry_exposes_levels_and_gains():
    bus = NeuromodulatoryBus()
    bus.update(loss=2.0, entropy=1.0)
    tel = bus.telemetry()
    for k in ("nm/da", "nm/ach", "nm/ne", "nm/gain_plasticity", "nm/gain_explore", "nm/gain_global"):
        assert k in tel and isinstance(tel[k], float)


@pytest.mark.unit
def test_checkpoint_state_restores_levels_and_broadcast_gains():
    original = NeuromodulatoryBus()
    original.update(loss=2.0, entropy=1.0)
    original.update(loss=0.5, entropy=3.0)

    restored = NeuromodulatoryBus()
    restored.load_state_dict(original.state_dict(), strict=True)
    model = make_tiny_synaptic(seed=0)
    restored.broadcast(model)

    assert restored.levels() == original.levels()
    assert restored.gains() == original.gains()
    linear = next(module for module in model.modules() if isinstance(module, SynapticLinear))
    assert linear._nm_da_gain == restored.gains()["plasticity"]
    assert linear._nm_ne_gain == restored.gains()["global"]


# --------------------------------------------------------------------------- #
# 6. ABLATION: the bus measurably changes learning
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ablation_dopamine_changes_consolidation():
    x = torch.randn(B, IN)
    ca, en = torch.ones(B), torch.ones(B)

    def slow_drift(da_gain):
        lin = _make_lin().eval()
        lin.reset_sequence_state(reset_fast_weights=True)
        w0 = lin.w_slow.detach().clone()
        lin._nm_da_gain = da_gain
        with torch.no_grad():
            for _ in range(3):
                lin(x, ca, en)
        return (lin.w_slow - w0).abs().sum().item()

    high = slow_drift(2.0)
    low = slow_drift(0.0)  # DA=0 gain -> no consolidation
    assert low == pytest.approx(0.0, abs=1e-9), "zero dopamine gain must freeze consolidation"
    assert high > 0.0, "positive dopamine gain must drive consolidation"


# --------------------------------------------------------------------------- #
# 7. NE novelty-triggered reset (arousal)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_norepinephrine_novelty_triggers_reset():
    model = make_tiny_synaptic(seed=0)
    # adapt some fast-weight state first
    for m in model.modules():
        if isinstance(m, SynapticLinear) and m.cfg.enable_hebbian:
            m.cfg.fast_weight_normalized = True
    V = model.config.vocab_size
    seq = torch.randint(0, V, (1, 16))
    with torch.no_grad():
        for _ in range(3):
            model(seq)
    lins = [m for m in model.modules() if isinstance(m, SynapticLinear) and m.w_fast is not None]
    assert max(lin.w_fast.norm().item() for lin in lins) > 0, "precondition: adapted state exists"

    bus = NeuromodulatoryBus(NeuromodConfig(novelty_reset_thresh=0.01))
    bus.update(loss=1.0)
    bus.update(loss=10.0)  # big novelty -> NE above threshold
    bus.broadcast(model)
    assert all(lin.w_fast.norm().item() == 0.0 for lin in lins), "NE novelty must trigger a reset"


# --------------------------------------------------------------------------- #
# 8. END-TO-END: the exact base_train hook (forward -> entropy -> update -> broadcast)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_training_loop_hook_computes_broadcasts_and_logs():
    model = make_tiny_synaptic(seed=0, train=True)
    V = model.config.vocab_size
    x = torch.randint(0, V, (2, 16))
    bus = NeuromodulatoryBus()
    # Mimic two training steps of the base_train loop hook.
    for _ in range(2):
        logits, loss = model(x, x, train_mode=True)
        ent = bus.entropy_from_logits(logits)
        bus.update(loss=float(loss.detach()), entropy=ent)
        n = bus.broadcast(model)
    assert n == len([m for m in model.modules() if isinstance(m, (SynapticLinear, SynapticPresyn))])
    # gains were applied to the synapses and telemetry is finite
    lin = next(m for m in model.modules() if isinstance(m, SynapticLinear))
    assert hasattr(lin, "_nm_da_gain") and lin._nm_da_gain == bus.gains()["plasticity"]
    assert all(v == v for v in bus.telemetry().values()), "telemetry must be finite"
