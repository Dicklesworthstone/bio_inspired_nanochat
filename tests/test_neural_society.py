"""Tests for Multi-Agent Neural Society & Shared Synaptic-Memory Bus (bead `re4e.12`)."""

import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.neural_society import (
    NeuralSociety,
    SharedSynapticMemoryBus,
)


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
