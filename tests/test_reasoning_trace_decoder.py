"""Tests for Faithful Reasoning-Trace Decoder (bead `re4e.9`)."""

import pytest

from bio_inspired_nanochat.reasoning_trace_decoder import (
    ReasoningTraceDecoder,
    StepOperation,
)


def test_decode_trajectory_steps_and_operations():
    """Verify that deliberation trajectory is decoded into structured reasoning steps."""
    decoder = ReasoningTraceDecoder()

    traj = [1.50, 0.90, 0.30, 0.20]
    trace = decoder.decode_energy_trajectory(traj)

    assert len(trace.steps) == 3
    assert trace.steps[0].operation == StepOperation.INITIAL_HYPOTHESIS
    assert trace.steps[1].operation == StepOperation.INCONSISTENCY_RESOLVED
    assert trace.steps[2].energy_after == 0.20
    assert trace.total_energy_reduction == pytest.approx(1.30, abs=1e-3)
    assert trace.is_causally_faithful


def test_rich_table_logging():
    """Verify that formatting and rich rendering execute cleanly."""
    decoder = ReasoningTraceDecoder()

    traj = [2.00, 1.20, 0.50]
    trace = decoder.decode_energy_trajectory(traj)
    decoder.log_trace(trace)
