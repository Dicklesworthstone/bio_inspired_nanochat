"""Tests for Living-Model Synaptic Debugger (bead `re4e.14`)."""

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic_debugger import (
    BioBreakpoint,
    SynapticDebugger,
)


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


def test_debugger_hits_breakpoint_and_inspects_frame():
    """Verify that SynapticDebugger halts generation when a conditional breakpoint triggers."""
    model = _make_model()
    debugger = SynapticDebugger(model)

    # Breakpoint that triggers on step >= 2
    bp = BioBreakpoint(
        name="Step Limit Guard",
        condition_fn=lambda step, tok, telem: step >= 2,
    )
    debugger.add_breakpoint(bp)

    prompt = torch.randint(0, 32, (1, 3))
    tokens, hit_frame = debugger.run_until_breakpoint(prompt, max_tokens=6)

    assert hit_frame is not None
    assert hit_frame.step == 2
    assert hit_frame.hit_breakpoint == "Step Limit Guard"
    assert debugger.is_paused
    assert bp.hit_count == 1
    assert "telemetry_snapshot" in dir(hit_frame)

    debugger.log_debugger_frame(hit_frame)


def test_debugger_edit_state_and_resume():
    """Verify that editing synaptic parameters at a breakpoint alters continuation upon resume."""
    model = _make_model()
    debugger = SynapticDebugger(model)

    debugger.add_breakpoint(
        BioBreakpoint(
            name="Early Pause",
            condition_fn=lambda step, tok, telem: step == 1,
        )
    )

    prompt = torch.randint(0, 32, (1, 3))
    tokens_paused, frame = debugger.run_until_breakpoint(prompt, max_tokens=4)

    assert debugger.is_paused
    assert frame is not None

    # Hot-patch synaptic fast weights while paused
    debugger.edit_synaptic_state(var_name="w_fast", value=3.0)

    # Resume generation
    tokens_resumed = debugger.resume_generation(max_additional_tokens=3)

    assert tokens_resumed.shape[1] == tokens_paused.shape[1] + 3
    assert not debugger.is_paused


def test_debugger_step_over_execution():
    """Verify single-step stepping through generation with step_over."""
    model = _make_model()
    debugger = SynapticDebugger(model)

    prompt = torch.randint(0, 32, (1, 2))
    debugger.run_until_breakpoint(prompt, max_tokens=0)

    # Step over twice
    frame1 = debugger.step_over()
    assert frame1 is not None
    assert frame1.step == 0
    assert debugger.current_tokens is not None
    assert debugger.current_tokens.shape[1] == 3

    frame2 = debugger.step_over()
    assert frame2 is not None
    assert frame2.step == 1
    assert debugger.current_tokens.shape[1] == 4


def test_energy_history_records_real_telemetry_not_constant():
    """uta-review regression: the energy trajectory used to read nonexistent
    telemetry keys ('global'/'energy') and silently recorded a constant 1.0."""
    from bio_inspired_nanochat.synaptic_debugger import _mean_bio_energy

    model = _make_model()
    debugger = SynapticDebugger(model)
    prompt = torch.randint(0, 32, (1, 3))
    debugger.run_until_breakpoint(prompt, max_tokens=2)

    assert len(debugger._energy_history) == 2
    telem = model.bio_telemetry()
    expected = _mean_bio_energy(telem)
    assert expected is not None
    assert debugger._energy_history[-1] == pytest.approx(expected)


def test_step_over_decodes_incrementally_through_kv_cache():
    """Regression: step_over re-forwarded the whole prefix each call, re-applying
    online plasticity to every prefix token (O(k^2) contamination). It must feed
    only the new token through the live KV cache instead."""
    model = _make_model()
    debugger = SynapticDebugger(model)
    prompt = torch.randint(0, 32, (1, 3))
    debugger.run_until_breakpoint(prompt, max_tokens=1)

    cache = debugger._kv_cache
    assert cache is not None
    pos_after_first = cache.get_pos()
    assert pos_after_first == 4  # 3 prompt tokens + 1 generated

    frame = debugger.step_over()
    assert frame is not None
    assert cache.get_pos() == pos_after_first + 1
    assert debugger.current_tokens.shape[1] == pos_after_first + 1
