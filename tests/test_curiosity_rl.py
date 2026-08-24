"""Tests for Intrinsic-Motivation Curiosity RL (bead `r00r.12`)."""

import torch

from bio_inspired_nanochat.curiosity_rl import (
    CuriosityConfig,
    CuriosityRewardEngine,
    StepRewardBreakdown,
)


def test_curiosity_reward_novelty_bonus():
    """Verify that unexpected loss spikes generate intrinsic novelty curiosity rewards."""
    engine = CuriosityRewardEngine()
    h = torch.randn(2, 4, 16)

    # Initial baseline
    r_base, _ = engine.compute_intrinsic_reward(token_loss=1.0, hidden_states=h)

    # Spike in surprise
    r_spike, _ = engine.compute_intrinsic_reward(token_loss=4.0, hidden_states=h)

    assert r_spike > r_base


def test_curiosity_step_progression():
    """Verify that step breakdown combines extrinsic and intrinsic rewards and updates neuromodulators."""
    engine = CuriosityRewardEngine(CuriosityConfig(curiosity_weight=0.5))
    h = torch.randn(1, 8, 16)

    breakdown = engine.step(
        step_idx=1,
        extrinsic_reward=0.0,
        token_loss=2.0,
        hidden_states=h,
    )

    assert breakdown.extrinsic_reward == 0.0
    assert breakdown.intrinsic_curiosity > 0.0
    assert breakdown.composite_reward > breakdown.extrinsic_reward
    assert breakdown.dopamine_level >= 0.0
    assert breakdown.norepinephrine_level >= 0.0


def test_rich_trace_logging():
    """Verify that log_trace formats and prints table cleanly."""
    engine = CuriosityRewardEngine()
    history = [
        StepRewardBreakdown(step=1, extrinsic_reward=0.0, intrinsic_curiosity=0.4, free_energy_penalty=0.5, composite_reward=0.1, dopamine_level=0.5, norepinephrine_level=0.8),
        StepRewardBreakdown(step=2, extrinsic_reward=1.0, intrinsic_curiosity=0.2, free_energy_penalty=0.4, composite_reward=1.05, dopamine_level=1.2, norepinephrine_level=0.3),
    ]
    engine.log_trace(history)
