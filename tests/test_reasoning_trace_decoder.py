"""Tests for the scalar energy-trajectory reporter (bead `re4e.9`)."""

import math

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
    assert trace.steps[0].operation == StepOperation.ENERGY_RELAXATION
    assert trace.steps[1].operation == StepOperation.ENERGY_RELAXATION
    assert trace.steps[2].energy_after == 0.20
    assert all(step.state_norm_delta is None for step in trace.steps)
    assert "energy decrease of 0.600" in trace.steps[0].explanation
    assert trace.total_energy_reduction == pytest.approx(1.30, abs=1e-3)
    assert not trace.is_causally_faithful
    assert all(not step.top_token_concepts for step in trace.steps)


def test_rich_table_logging():
    """Verify that formatting and rich rendering execute cleanly."""
    decoder = ReasoningTraceDecoder()

    traj = [2.00, 1.20, 0.50]
    trace = decoder.decode_energy_trajectory(traj)
    decoder.log_trace(trace)


def test_decode_short_trajectory():
    """Verify single-step or empty trajectories produce valid traces."""
    decoder = ReasoningTraceDecoder()

    trace_single = decoder.decode_energy_trajectory([1.0])
    assert len(trace_single.steps) == 0
    assert trace_single.total_energy_reduction == 0.0
    assert not trace_single.is_causally_faithful

    trace_empty = decoder.decode_energy_trajectory([])
    assert len(trace_empty.steps) == 0
    assert not trace_empty.is_causally_faithful
    assert trace_empty.summary_narrative == "No deliberation trajectory was observed."


def test_decode_rejects_nonfinite_energy_and_reports_increase():
    """Invalid measurements fail closed and rising energy is never called dissipation."""
    decoder = ReasoningTraceDecoder()

    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must all be finite"):
            decoder.decode_energy_trajectory([1.0, invalid])

    trace = decoder.decode_energy_trajectory([1.0, 1.5, 2.0])
    assert trace.steps[0].operation == StepOperation.ENERGY_INCREASE
    assert trace.steps[-1].operation == StepOperation.ENERGY_INCREASE
    assert "increased from 1.000 to 2.000" in trace.summary_narrative
    assert "--" not in trace.summary_narrative

    one_transition = decoder.decode_energy_trajectory([1.0, 2.0])
    assert one_transition.steps[0].operation == StepOperation.ENERGY_INCREASE


def test_single_flat_transition_is_reported_as_unchanged_not_convergence():
    trace = ReasoningTraceDecoder().decode_energy_trajectory([1.0, 1.0])

    assert trace.steps[0].operation == StepOperation.ENERGY_UNCHANGED
    assert "within reporting tolerance" in trace.steps[0].explanation
    assert "reduced energy" not in trace.steps[0].explanation


def test_small_positive_and_negative_changes_use_the_same_tolerance():
    decoder = ReasoningTraceDecoder()

    positive = decoder.decode_energy_trajectory([1.0, 1.00001])
    negative = decoder.decode_energy_trajectory([1.0, 0.99999])

    assert positive.steps[0].operation == StepOperation.ENERGY_UNCHANGED
    assert negative.steps[0].operation == StepOperation.ENERGY_UNCHANGED
