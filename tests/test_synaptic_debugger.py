"""Tests for Living-Model Synaptic Debugger (bead `re4e.14`)."""

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
