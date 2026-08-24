"""Tests for Prioritized Replay Buffer and Sleep Consolidation (bead `cel.2`)."""

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.sleep_consolidation import (
    PrioritizedReplayBuffer,
    SleepConsolidationController,
)
from bio_inspired_nanochat.synaptic import SynapticLinear


def test_replay_buffer_capacity_and_sampling():
    """Verify prioritized buffer inserts, respects capacity, and samples proportionally."""
    buf = PrioritizedReplayBuffer(capacity=4, alpha=0.6)

    for i in range(6):
        buf.push(torch.tensor([i, i + 1]), surprise_score=float(i + 1))

    assert len(buf) == 4
    # Lowest surprises (1.0, 2.0) should have been evicted; highest (3,4,5,6) remain
    min_surprise = min(item.surprise_score for item in buf.items)
    assert min_surprise >= 3.0

    samples, indices = buf.sample(batch_size=2)
    assert len(samples) == 2
    assert len(indices) == 2


def test_sleep_consolidation_transfers_fast_to_slow():
    """Verify that offline sleep phase transfers W_fast into W_slow and resets W_fast."""
    model_cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    model = GPTSynaptic(model_cfg)

    # Locate a SynapticLinear layer and inject fast weight
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    w_slow_orig = syn_lin.w_slow.data.clone()
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.5)

    buf = PrioritizedReplayBuffer(capacity=8)
    for _ in range(4):
        buf.push(torch.randint(0, 32, (8,)), surprise_score=2.5)

    controller = SleepConsolidationController(consolidation_lr=0.2)
    report = controller.run_sleep_phase(model, buf, sleep_steps=1, batch_size=2)

    assert report["status"] == "consolidated"
    assert report["total_transferred_norm"] > 0.0

    # W_fast must be cleared to zero
    assert syn_lin.w_fast.norm().item() == pytest.approx(0.0, abs=1e-6)

    # W_slow must have absorbed the transferred norm
    assert not torch.equal(syn_lin.w_slow.data, w_slow_orig)


def test_shy_homeostatic_downscaling():
    """Verify that SHY homeostatic downscaling caps the Frobenius norm of slow weights."""
    model_cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    model = GPTSynaptic(model_cfg)

    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    syn_lin.w_slow.data.fill_(10.0)  # Large norm

    buf = PrioritizedReplayBuffer(capacity=4)
    buf.push(torch.randint(0, 32, (8,)), surprise_score=1.0)

    controller = SleepConsolidationController(downscale_target_norm=2.0)
    controller.run_sleep_phase(model, buf, sleep_steps=1, batch_size=1)

    assert syn_lin.w_slow.data.norm().item() <= 2.001
