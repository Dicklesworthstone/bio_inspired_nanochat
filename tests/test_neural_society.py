"""Tests for Multi-Agent Neural Society & Shared Synaptic-Memory Bus (bead `re4e.12`)."""

import math
from typing import Any, cast

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.neural_society import (
    NeuralSociety,
    SharedSynapticMemoryBus,
)
from bio_inspired_nanochat.synaptic import SynapticLinear


def _make_agent() -> GPTSynaptic:
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


def test_shared_synaptic_memory_bus_publish_and_aggregate():
    """Verify publishing associations to memory bus and aggregating topic matrices."""
    bus = SharedSynapticMemoryBus(max_norm=3.0)

    k = torch.randn(16)
    v = torch.randn(16)

    bus.publish("agent_alpha", "math_reasoning", k, v, confidence=1.0)
    assert "math_reasoning" in bus.topics
    assert len(bus.topics["math_reasoning"]) == 1

    agg = bus.aggregate_topic("math_reasoning", in_dim=16, out_dim=16)
    assert agg.shape == (16, 16)
    assert agg.norm().item() <= 3.001


def test_neural_society_collaborative_solving():
    """Verify that multiple specialized agents solve a collective task via the shared memory bus."""
    society = NeuralSociety()

    agent1 = _make_agent()
    agent2 = _make_agent()

    society.register_agent("agent_retrieval", agent1, "Context Extractor")
    society.register_agent("agent_reasoner", agent2, "Deductive Engine")

    task_prompt = torch.randint(0, 32, (1, 6))
    result = society.solve_collaborative_task(task_prompt, topic="shared_workspace")

    assert result["status"] == "solved_collaborative"
    assert result["participating_agents"] == 2
    assert result["output_logits_norm"] > 0.0

    society.log_society_status()


def test_memory_bus_rejects_negative_layer_index():
    """Negative indices must not silently mutate the final synaptic layer."""
    model = _make_agent()
    bus = SharedSynapticMemoryBus()
    bus.publish("agent", "topic", torch.ones(16), torch.ones(64))

    synaptic_weights = [
        mod.w_fast.detach().clone()
        for mod in model.modules()
        if hasattr(mod, "w_fast") and mod.w_fast is not None
    ]
    assert not bus.sync_to_agent(model, "topic", layer_idx=-1)
    weights_after = [
        mod.w_fast
        for mod in model.modules()
        if hasattr(mod, "w_fast") and mod.w_fast is not None
    ]
    assert all(torch.equal(before, after) for before, after in zip(synaptic_weights, weights_after))


def test_memory_bus_rejects_nonfinite_payloads_and_invalid_configuration():
    """One malformed agent message must not poison shared fast weights with NaNs."""
    with pytest.raises(ValueError, match="max_norm"):
        SharedSynapticMemoryBus(max_norm=math.nan)
    with pytest.raises(ValueError, match="max_messages_per_topic"):
        SharedSynapticMemoryBus(max_messages_per_topic=0)

    bus = SharedSynapticMemoryBus()
    with pytest.raises(ValueError, match="confidence"):
        bus.publish("agent", "topic", torch.ones(2), torch.ones(2), confidence=math.nan)
    with pytest.raises(ValueError, match="finite values"):
        bus.publish("agent", "topic", torch.tensor([1.0, math.inf]), torch.ones(2))
    with pytest.raises(ValueError, match="real floating"):
        bus.publish("agent", "topic", torch.ones(2, dtype=torch.long), torch.ones(2))
    with pytest.raises(ValueError, match="non-empty"):
        bus.publish("agent", "topic", torch.empty(0), torch.ones(2))
    with pytest.raises(ValueError, match="sender_id"):
        bus.publish("", "topic", torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="topic"):
        bus.publish("agent", " ", torch.ones(2), torch.ones(2))
    assert bus.topics == {}
    with pytest.raises(ValueError, match="in_dim"):
        bus.aggregate_topic("topic", in_dim=0, out_dim=2)


def test_sync_missing_topic_does_not_erase_existing_fast_weights():
    model = _make_agent()
    bus = SharedSynapticMemoryBus()
    syn_lin = next(mod for mod in model.modules() if hasattr(mod, "w_fast") and mod.w_fast is not None)
    syn_lin.w_fast.data.fill_(0.25)
    original = syn_lin.w_fast.detach().clone()

    assert not bus.sync_to_agent(model, "misspelled-topic")
    assert torch.equal(syn_lin.w_fast, original)


def test_memory_bus_bounds_topic_history():
    bus = SharedSynapticMemoryBus(max_messages_per_topic=2)
    for value in (1.0, 2.0, 3.0):
        bus.publish("agent", "topic", torch.tensor([value]), torch.ones(1))

    assert [message.key_vector.item() for message in bus.topics["topic"]] == [2.0, 3.0]


def test_memory_bus_stores_payload_snapshots_and_accepts_noncontiguous_vectors():
    bus = SharedSynapticMemoryBus()
    source = torch.arange(8.0).reshape(2, 4).transpose(0, 1)
    expected = source.reshape(-1).clone()

    bus.publish("agent", "topic", source, source, confidence=1.0)
    source.fill_(99.0)

    assert torch.equal(bus.topics["topic"][0].key_vector, expected)
    assert torch.equal(bus.topics["topic"][0].value_vector, expected)


def test_collaboration_mounts_bus_without_forward_adaptation_and_restores_modes():
    society = NeuralSociety()
    retrieval = _make_agent()
    reasoner = _make_agent()
    reasoner.train()
    reasoner.h[0].mlp.eval()
    modes_before = [module.training for module in reasoner.modules()]
    society.register_agent("retrieval", retrieval, "retrieve")
    society.register_agent("reasoner", reasoner, "reason")

    society.solve_collaborative_task(torch.ones((1, 3), dtype=torch.long), topic="shared")

    first_synaptic = next(
        module
        for module in reasoner.modules()
        if hasattr(module, "w_fast") and module.w_fast is not None
    )
    expected = society.bus.aggregate_topic(
        "shared", *first_synaptic.w_fast.shape
    ).to(first_synaptic.w_fast.device)
    assert torch.equal(first_synaptic.w_fast, expected)
    assert [module.training for module in reasoner.modules()] == modes_before


def test_society_rejects_invalid_identity_and_prompt_boundaries():
    society = NeuralSociety()
    with pytest.raises(ValueError, match="agent_id"):
        society.register_agent("", _make_agent(), "role")
    with pytest.raises(ValueError, match="role"):
        society.register_agent("agent", _make_agent(), "")

    society.register_agent("agent", _make_agent(), "role")
    with pytest.raises(ValueError, match="must be a tensor"):
        society.solve_collaborative_task(cast(Any, [[1, 2]]))
    with pytest.raises(ValueError, match="rank-2"):
        society.solve_collaborative_task(torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match="integer token IDs"):
        society.solve_collaborative_task(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="outside an agent vocabulary"):
        society.solve_collaborative_task(torch.tensor([[32]], dtype=torch.long))
    with pytest.raises(ValueError, match="context window"):
        society.solve_collaborative_task(torch.ones((1, 9), dtype=torch.long))
    assert society.bus.topics == {}


def test_society_reports_only_agents_that_execute_and_rejects_failed_bus_mount(monkeypatch):
    society = NeuralSociety()
    society.register_agent("retrieval", _make_agent(), "retrieve")
    society.register_agent("reasoner", _make_agent(), "reason")
    society.register_agent("observer", _make_agent(), "observe")
    prompt = torch.ones((1, 2), dtype=torch.long)

    result = society.solve_collaborative_task(prompt)
    assert result["participating_agents"] == 2

    monkeypatch.setattr(society.bus, "sync_to_agent", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="no compatible synaptic"):
        society.solve_collaborative_task(prompt, topic="failed-mount")
    assert "failed-mount" not in society.bus.topics


def test_collaboration_rolls_back_bus_and_fast_weights_when_reasoning_fails(monkeypatch):
    society = NeuralSociety()
    retrieval = _make_agent()
    reasoner = _make_agent()
    society.register_agent("retrieval", retrieval, "retrieve")
    society.register_agent("reasoner", reasoner, "reason")
    fast = next(
        module.w_fast
        for module in reasoner.modules()
        if isinstance(module, SynapticLinear) and module.w_fast is not None
    )
    fast_before = fast.detach().clone()

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("reasoner failed")

    monkeypatch.setattr(reasoner, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="reasoner failed"):
        society.solve_collaborative_task(
            torch.ones((1, 2), dtype=torch.long), topic="transaction"
        )

    assert "transaction" not in society.bus.topics
    assert torch.equal(fast, fast_before)
