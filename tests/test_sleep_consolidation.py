"""Tests for Prioritized Replay Buffer and Sleep Consolidation (bead `cel.2`)."""

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.sleep_consolidation import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    SleepConsolidationController,
    consolidate_sleep_replay,
    homeostatic_downscale,
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

    controller = SleepConsolidationController(consolidation_lr=0.2, latch_threshold=0.0)
    report = controller.run_sleep_phase(model, buf, sleep_steps=1, batch_size=2)

    assert report["status"] == "consolidated"
    assert report["total_transferred_norm"] > 0.0

    # W_fast is a trainable parameter as well as an adaptive path, so sleep must not erase it.
    assert syn_lin.w_fast.norm().item() > 0.0

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


def test_sleep_phase_restores_training_modes_on_success_and_error():
    """Offline replay must not silently change the caller's training configuration."""
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
    model.train()
    model.h[0].mlp.eval()
    expected_modes = [module.training for module in model.modules()]

    controller = SleepConsolidationController()
    valid_buffer = PrioritizedReplayBuffer(capacity=1)
    valid_buffer.push(torch.randint(0, 32, (8,)), surprise_score=1.0)
    controller.run_sleep_phase(model, valid_buffer, sleep_steps=1, batch_size=1)
    assert [module.training for module in model.modules()] == expected_modes

    invalid_buffer = PrioritizedReplayBuffer(capacity=1)
    invalid_buffer.push(torch.full((8,), 32), surprise_score=1.0)
    with pytest.raises(ValueError, match="inside the vocabulary"):
        controller.run_sleep_phase(model, invalid_buffer, sleep_steps=1, batch_size=1)
    assert [module.training for module in model.modules()] == expected_modes


def test_dream_generation_infers_the_model_device():
    """Scheduler callers that omit a device must not create CPU inputs for another device."""

    class MetaDreamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.empty((), device="meta"))
            self.config = SimpleNamespace(vocab_size=32, sequence_len=8)

        def forward(self, tokens, train_mode=True):
            assert not train_mode
            assert tokens.device.type == "meta"
            logits = torch.empty((*tokens.shape, 32), device=tokens.device)
            return logits, None

    dreams = SleepConsolidationController().generate_dreams(
        cast(Any, MetaDreamModel()), num_dreams=2, seq_len=3
    )
    assert dreams.device.type == "meta"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"consolidation_lr": -0.1}, "consolidation_lr"),
        ({"consolidation_lr": math.nan}, "consolidation_lr"),
        ({"downscale_target_norm": -1.0}, "downscale_target_norm"),
        ({"latch_threshold": math.inf}, "latch_threshold"),
    ],
)
def test_sleep_controller_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SleepConsolidationController(**kwargs)


def test_sleep_execution_rejects_invalid_loop_and_dream_parameters():
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    controller = SleepConsolidationController()

    with pytest.raises(ValueError, match="sleep_steps"):
        controller.run_sleep_phase(model, sleep_steps=-1)
    with pytest.raises(ValueError, match="batch_size"):
        controller.run_sleep_phase(model, batch_size=0)
    with pytest.raises(ValueError, match="num_dreams"):
        controller.generate_dreams(model, num_dreams=0)
    with pytest.raises(ValueError, match="seq_len"):
        controller.generate_dreams(model, seq_len=0)
    with pytest.raises(ValueError, match="model sequence length"):
        controller.generate_dreams(model, seq_len=9)
    with pytest.raises(ValueError, match="temperature"):
        controller.generate_dreams(model, seq_len=8, temperature=0.0)


def test_sleep_phase_crops_mixed_length_replay_without_fabricating_padding():
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.push(torch.randint(0, 32, (5,)), surprise_score=1.0)
    replay.push(torch.randint(0, 32, (8,)), surprise_score=1.0)

    report = SleepConsolidationController().run_sleep_phase(
        model, replay, sleep_steps=1, batch_size=2
    )

    assert report["status"] == "consolidated"
    assert report["steps_run"] == 1

    long_replay = PrioritizedReplayBuffer(capacity=2)
    long_replay.push(torch.randint(0, 32, (10,)), surprise_score=1.0)
    long_replay.push(torch.randint(0, 32, (12,)), surprise_score=1.0)
    long_report = SleepConsolidationController().run_sleep_phase(
        model, long_replay, sleep_steps=1, batch_size=2
    )
    assert long_report["status"] == "consolidated"


def test_prioritized_replay_buffer_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="capacity"):
        PrioritizedReplayBuffer(capacity=0)
    with pytest.raises(ValueError, match="alpha"):
        PrioritizedReplayBuffer(alpha=math.nan)

    replay = PrioritizedReplayBuffer()
    with pytest.raises(ValueError, match="non-empty"):
        replay.push(torch.empty(0, dtype=torch.long), surprise_score=1.0)
    with pytest.raises(ValueError, match="integer token IDs"):
        replay.push(torch.ones(2), surprise_score=1.0)
    for invalid_surprise in (-0.1, math.inf):
        with pytest.raises(ValueError, match="surprise_score"):
            replay.push(torch.ones(2, dtype=torch.long), surprise_score=invalid_surprise)
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(0)


def test_prioritized_replay_buffer_stores_token_snapshots():
    replay = PrioritizedReplayBuffer()
    tokens = torch.tensor([1, 2, 3])
    replay.push(tokens, surprise_score=1.0)

    tokens.fill_(9)

    assert replay.items[0].tokens.tolist() == [1, 2, 3]


def test_prioritized_sampling_is_stable_for_extreme_finite_weights():
    replay = PrioritizedReplayBuffer(capacity=2, alpha=1e308)
    replay.push(torch.tensor([1]), surprise_score=1e-300)
    replay.push(torch.tensor([2]), surprise_score=1e300)

    items, indices = replay.sample(1)

    assert len(items) == 1
    assert len(indices) == 1


def test_legacy_replay_buffer_validates_inputs_and_stores_snapshots():
    with pytest.raises(ValueError, match="max_capacity"):
        ReplayBuffer(max_capacity=0)
    with pytest.raises(ValueError, match="alpha"):
        ReplayBuffer(alpha=math.nan)

    replay = ReplayBuffer(max_capacity=2)
    inputs = torch.tensor([1, 2])
    targets = torch.tensor([2, 3])
    replay.add(inputs, targets, loss=1.0)
    inputs.fill_(9)
    targets.fill_(9)
    assert replay.buffer[0].inputs.tolist() == [1, 2]
    assert replay.buffer[0].targets.tolist() == [2, 3]

    with pytest.raises(ValueError, match="loss"):
        replay.add(inputs, targets, loss=math.inf)
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(0)

    stable = ReplayBuffer(max_capacity=2, alpha=1e308)
    stable.add(torch.tensor([1]), torch.tensor([1]), loss=1e-300)
    stable.add(torch.tensor([2]), torch.tensor([2]), loss=1e300)
    assert len(stable.sample(1)) == 1


def test_legacy_sleep_helpers_reject_invalid_downscaling_and_pass_counts():
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    with pytest.raises(ValueError, match="max_slow_norm"):
        homeostatic_downscale(model, max_slow_norm=-1.0)
    with pytest.raises(ValueError, match="decay_factor"):
        homeostatic_downscale(model, decay_factor=-0.1)
    with pytest.raises(ValueError, match="consolidation_passes"):
        consolidate_sleep_replay(model, [], consolidation_passes=0)
    with pytest.raises(ValueError, match="downscale_decay"):
        consolidate_sleep_replay(model, [], downscale_decay=1.1)


def test_sleep_uses_scaled_native_slow_delta_without_clearing_fast(monkeypatch):
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    assert syn_lin.post is not None
    syn_lin.w_fast.data.fill_(0.5)
    slow_before = syn_lin.w_slow.detach().clone()
    fast_before = syn_lin.w_fast.detach().clone()

    def controlled_forward(_tokens, train_mode=True):
        assert train_mode
        with torch.no_grad():
            syn_lin.w_slow.add_(2.0)
            syn_lin.post.camkii.fill_(1.0)
        return torch.empty(0), None

    monkeypatch.setattr(model, "forward", controlled_forward)
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.push(torch.ones(4, dtype=torch.long), surprise_score=1.0)

    report = SleepConsolidationController(
        consolidation_lr=0.25, latch_threshold=0.5
    ).run_sleep_phase(model, replay, sleep_steps=1, batch_size=1)

    assert torch.allclose(syn_lin.w_slow, slow_before + 0.5)
    assert torch.equal(syn_lin.w_fast, fast_before)
    assert report["total_transferred_norm"] > 0.0


def test_sleep_flushes_pending_wake_write_once_and_clears_traces():
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    model.train()
    tokens = torch.randint(0, 32, (1, 8))
    logits, _ = model(tokens)
    logits.float().sum().backward()
    assert any(
        mod._plasticity_pending
        for mod in model.modules()
        if isinstance(mod, SynapticLinear)
    )

    replay = PrioritizedReplayBuffer(capacity=1)
    replay.push(tokens[0], surprise_score=1.0)
    SleepConsolidationController(latch_threshold=0.0).run_sleep_phase(
        model, replay, sleep_steps=1, batch_size=1
    )

    for mod in model.modules():
        if isinstance(mod, SynapticLinear):
            assert not mod._plasticity_pending
            assert mod._last_gate_scale is None
            if mod.u_buf is not None:
                assert torch.count_nonzero(mod.u_buf) == 0
            if mod.v_buf is not None:
                assert torch.count_nonzero(mod.v_buf) == 0


def test_sleep_failure_rolls_back_partial_synaptic_mutation(monkeypatch):
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
        )
    )
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    assert syn_lin.post is not None
    state_before = {
        "w_slow": syn_lin.w_slow.detach().clone(),
        "w_fast": syn_lin.w_fast.detach().clone(),
        "camkii": syn_lin.post.camkii.detach().clone(),
    }

    def partially_mutate_then_fail(_tokens, train_mode=True):
        assert train_mode
        with torch.no_grad():
            syn_lin.w_slow.add_(1.0)
            syn_lin.w_fast.add_(1.0)
            syn_lin.post.camkii.fill_(1.0)
        raise RuntimeError("replay failed")

    monkeypatch.setattr(model, "forward", partially_mutate_then_fail)
    replay = PrioritizedReplayBuffer(capacity=1)
    replay.push(torch.ones(4, dtype=torch.long), surprise_score=1.0)

    with pytest.raises(RuntimeError, match="replay failed"):
        SleepConsolidationController().run_sleep_phase(
            model, replay, sleep_steps=1, batch_size=1
        )

    assert torch.equal(syn_lin.w_slow, state_before["w_slow"])
    assert torch.equal(syn_lin.w_fast, state_before["w_fast"])
    assert torch.equal(syn_lin.post.camkii, state_before["camkii"])
