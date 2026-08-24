"""Multi-Agent Neural Society & Shared Synaptic-Memory Bus (bead `re4e.12`).

Coordinates a collective of specialized bio-inspired transformer agents communicating
through a shared, tensor-level synaptic memory bus:
1. `SharedSynapticMemoryBus`: Topic-partitioned fast-weight memory blackboard.
2. `NeuralSociety`: Orchestrates multi-agent division of labor, enabling agents to directly read/write
   shared fast synaptic weights to solve complex collective tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


@dataclass
class BusMessage:
    """An associative synaptic memory payload published by a specialized agent."""

    sender_id: str
    topic: str
    key_vector: Tensor
    value_vector: Tensor
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class SharedSynapticMemoryBus:
    """Topic-partitioned synaptic fast-weight communication and memory blackboard."""

    def __init__(self, max_norm: float = 4.0):
        self.max_norm = max_norm
        self.topics: Dict[str, List[BusMessage]] = {}

    def publish(
        self,
        sender_id: str,
        topic: str,
        key_vector: Tensor,
        value_vector: Tensor,
        confidence: float = 1.0,
    ) -> None:
        """Publish an associative vector binding onto a memory topic."""
        k = key_vector.detach().cpu().view(-1)
        v = value_vector.detach().cpu().view(-1)
        msg = BusMessage(
            sender_id=sender_id,
            topic=topic,
            key_vector=k,
            value_vector=v,
            confidence=float(confidence),
        )
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(msg)

    def clear_topic(self, topic: str) -> None:
        """Clear all messages from a topic."""
        self.topics.pop(topic, None)

    def list_topics(self) -> List[str]:
        """List all active topic names."""
        return list(self.topics.keys())

    def aggregate_topic(self, topic: str, in_dim: int, out_dim: int) -> Tensor:
        """Aggregate and blend all published associative bindings for a given topic."""
        delta = torch.zeros(in_dim, out_dim)
        if topic not in self.topics:
            return delta

        for msg in self.topics[topic]:
            # Robust dimension adaptation with zero padding / truncation
            k = msg.key_vector
            v = msg.value_vector

            k_pad = torch.zeros(in_dim)
            k_len = min(in_dim, k.shape[0])
            if k_len > 0:
                k_pad[:k_len] = k[:k_len]

            v_pad = torch.zeros(out_dim)
            v_len = min(out_dim, v.shape[0])
            if v_len > 0:
                v_pad[:v_len] = v[:v_len]

            delta.add_(msg.confidence * torch.outer(k_pad, v_pad))

        # Norm bounding
        curr_norm = float(delta.norm().item())
        if curr_norm > self.max_norm:
            delta.mul_(self.max_norm / max(1e-6, curr_norm))

        return delta

    def sync_to_agent(
        self,
        model: GPTSynaptic,
        topic: str,
        layer_idx: int = 0,
    ) -> bool:
        """Mount aggregated topic memories into target agent's synaptic fast weights."""
        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
        if layer_idx >= len(syn_layers):
            return False

        mod = syn_layers[layer_idx]
        if mod.w_fast is None:
            return False

        in_d, out_d = mod.w_fast.shape
        delta = self.aggregate_topic(topic, in_d, out_d).to(mod.w_fast.device)
        mod.w_fast.data.copy_(delta)
        return True


class NeuralSociety:
    """Society of specialized synaptic agents collaborating through the memory bus."""

    def __init__(self, bus: Optional[SharedSynapticMemoryBus] = None):
        self.bus = bus or SharedSynapticMemoryBus()
        self.agents: Dict[str, GPTSynaptic] = {}
        self.agent_roles: Dict[str, str] = {}

    def register_agent(self, agent_id: str, model: GPTSynaptic, role: str) -> None:
        """Add a specialized agent to the neural society."""
        self.agents[agent_id] = model
        self.agent_roles[agent_id] = role

    def solve_collaborative_task(
        self,
        task_prompt: Tensor,
        topic: str = "collective_scratchpad",
    ) -> Dict[str, Any]:
        """Execute collaborative reasoning pipeline across the agent society."""
        t0 = time.perf_counter()
        agent_names = list(self.agents.keys())
        if len(agent_names) == 0:
            return {"status": "no_agents", "steps": 0}

        # Step 1: Agent 1 (Perception/Retrieval) inspects prompt and publishes memory binding
        a1_id = agent_names[0]
        a1_model = self.agents[a1_id]
        with torch.no_grad():
            wte_out = a1_model.wte(task_prompt)
            # Robust 1D vector aggregation across batch and sequence dimensions
            if wte_out.ndim == 3:
                emb = wte_out.mean(dim=(0, 1))
            elif wte_out.ndim == 2:
                emb = wte_out.mean(dim=0)
            else:
                emb = wte_out.view(-1)

            self.bus.publish(
                sender_id=a1_id,
                topic=topic,
                key_vector=emb,
                value_vector=emb,
                confidence=1.0,
            )

        # Step 2: Agent 2 (Reasoner) mounts collective memory and executes forward deduction
        a2_id = agent_names[min(1, len(agent_names) - 1)]
        a2_model = self.agents[a2_id]
        self.bus.sync_to_agent(a2_model, topic=topic, layer_idx=0)

        with torch.no_grad():
            logits, _ = a2_model(task_prompt)

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "solved_collaborative",
            "topic": topic,
            "participating_agents": len(self.agents),
            "output_logits_norm": float(logits.norm().item()),
            "wall_time_ms": dt,
        }

    def log_society_status(self, console: Optional[Console] = None) -> None:
        """Render Rich table of neural society agents and active memory topics."""
        c = console or Console()
        c.rule("[bold cyan]Neural Society Collective Roster & Memory Bus[/bold cyan]")

        table = Table(title="Specialized Agents")
        table.add_column("Agent ID", style="bold")
        table.add_column("Specialization Role", style="cyan")
        table.add_column("Memory Topics Active", justify="right")

        for aid, role in self.agent_roles.items():
            table.add_row(aid, role, str(len(self.bus.topics)))
        c.print(table)
