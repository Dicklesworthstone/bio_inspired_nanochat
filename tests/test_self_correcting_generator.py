"""Tests for the Self-Correcting Generation Loop (beads `re4e.1`, `re4e.1.3`)."""

import pytest
import torch
import torch.nn as nn

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.self_correcting_generator import (
    CorrectionOutcome,
    SelfCorrectingGenerator,
    SelfCorrectionConfig,
    SelfCorrectionEvent,
    SelfCorrectingTrajectory,
)


def test_rejects_models_without_real_hidden_states():
    """The detector must never run on fabricated random representations."""
    with pytest.raises(TypeError, match="get_hidden_states"):
        SelfCorrectingGenerator(nn.Linear(8, 8))


def test_self_correction_passthrough_when_disabled():
    """Verify that disabled generator returns PASSTHROUGH immediately."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    generator = SelfCorrectingGenerator(model, SelfCorrectionConfig(enabled=False))
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.PASSTHROUGH
    assert traj.attempts_used == 0
    assert not traj.is_abstention


def test_verified_consistent_when_no_obstruction():
    """Verify that clean sequences return VERIFIED_CONSISTENT."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    # Very high threshold -> never flags obstruction
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(obstruction_threshold=1.0),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.VERIFIED_CONSISTENT
    assert not traj.is_abstention


def test_certified_abstain_on_exhaustion():
    """Verify that persistent inconsistency terminates in CERTIFIED_ABSTAIN."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    # Impossibly low threshold -> always flags obstruction
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(
            obstruction_threshold=0.0001,
            max_repair_attempts=2,
            abstain_on_exhaustion=True,
            abstain_token_id=99,
        ),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.CERTIFIED_ABSTAIN
    assert traj.is_abstention
    assert traj.final_tokens[-1] == 99


def test_rich_table_lineage_logging():
    """Verify that logging history functions without exceptions."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    generator = SelfCorrectingGenerator(model)

    event = SelfCorrectionEvent(
        attempt_idx=1,
        span_start=3,
        span_end=5,
        corrupted_tokens=[10, 11],
        repaired_tokens=[12, 13],
        initial_obstruction=0.65,
        repaired_obstruction=0.20,
        repaired_successfully=True,
        wall_time_ms=12.5,
    )
    traj = SelfCorrectingTrajectory(
        final_tokens=[1, 2, 3, 12, 13, 14],
        outcome=CorrectionOutcome.REPAIRED,
        attempts_used=1,
        events=[event],
        total_wall_time_ms=25.0,
        is_abstention=False,
    )
    generator.log_trajectory(traj)
