"""Multi-Agent Neural Society & Shared Synaptic-Memory Bus (bead `re4e.12`).

Coordinates a collective of specialized bio-inspired transformer agents communicating
through a shared, tensor-level synaptic memory bus:
1. `SharedSynapticMemoryBus`: Topic-partitioned fast-weight memory blackboard.
2. `NeuralSociety`: Orchestrates multi-agent division of labor, enabling agents to directly read/write
   shared fast synaptic weights to solve complex collective tasks.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


@contextmanager
def _temporary_eval(model: torch.nn.Module) -> Iterator[None]:
    """Run inference without permanently changing mixed per-module training modes."""
    training_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in training_modes:
            module.training = was_training


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

    def __init__(self, max_norm: float = 4.0, max_messages_per_topic: int = 128):
        if not math.isfinite(max_norm) or max_norm <= 0.0:
            raise ValueError("max_norm must be finite and positive")
        if (
            isinstance(max_messages_per_topic, bool)
            or not isinstance(max_messages_per_topic, int)
            or max_messages_per_topic <= 0
        ):
            raise ValueError("max_messages_per_topic must be a positive integer")
        self.max_norm = max_norm
        self.max_messages_per_topic = max_messages_per_topic
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
        if not isinstance(sender_id, str) or not sender_id.strip():
            raise ValueError("sender_id must be non-empty")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be non-empty")
        if not isinstance(key_vector, Tensor) or not isinstance(value_vector, Tensor):
            raise ValueError("key_vector and value_vector must be tensors")
        if not key_vector.is_floating_point() or not value_vector.is_floating_point():
            raise ValueError("key_vector and value_vector must be real floating tensors")
        # Messages are immutable historical snapshots. Flatten non-contiguous inputs safely,
        # and clone after moving to CPU so later caller mutations cannot rewrite the bus.
        k = key_vector.detach().cpu().reshape(-1).clone()
        v = value_vector.detach().cpu().reshape(-1).clone()
        confidence_value = float(confidence)
        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if k.numel() == 0 or v.numel() == 0:
            raise ValueError("key_vector and value_vector must be non-empty")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError("key_vector and value_vector must contain only finite values")
        msg = BusMessage(
            sender_id=sender_id,
            topic=topic,
            key_vector=k,
            value_vector=v,
            confidence=confidence_value,
        )
        if topic not in self.topics:
            self.topics[topic] = []
        topic_messages = self.topics[topic]
        topic_messages.append(msg)
        if len(topic_messages) > self.max_messages_per_topic:
            del topic_messages[: -self.max_messages_per_topic]

    def clear_topic(self, topic: str) -> None:
        """Clear all messages from a topic."""
        self.topics.pop(topic, None)

    def list_topics(self) -> List[str]:
        """List all active topic names."""
        return list(self.topics.keys())

    def aggregate_topic(self, topic: str, in_dim: int, out_dim: int) -> Tensor:
        """Aggregate and blend all published associative bindings for a given topic."""
        for name, value in (("in_dim", in_dim), ("out_dim", out_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
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
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be non-empty")
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise ValueError("layer_idx must be an integer")
        syn_layers = [m for m in model.modules() if isinstance(m, SynapticLinear)]
        if topic not in self.topics or not 0 <= layer_idx < len(syn_layers):
            return False

        mod = syn_layers[layer_idx]
        if mod.w_fast is None:
            return False

        in_d, out_d = mod.w_fast.shape
        delta = self.aggregate_topic(topic, in_d, out_d).to(
            device=mod.w_fast.device, dtype=mod.w_fast.dtype
        )
        # A version-tracked copy fails loudly if a caller violates the required safe
        # boundary by mounting between a forward and its backward pass.
        with torch.no_grad():
            mod.w_fast.copy_(delta)
        return True


class NeuralSociety:
    """Society of specialized synaptic agents collaborating through the memory bus."""

    def __init__(self, bus: Optional[SharedSynapticMemoryBus] = None):
        self.bus = bus or SharedSynapticMemoryBus()
        self.agents: Dict[str, GPTSynaptic] = {}
        self.agent_roles: Dict[str, str] = {}

    def register_agent(self, agent_id: str, model: GPTSynaptic, role: str) -> None:
        """Add a specialized agent to the neural society."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be non-empty")
        self.agents[agent_id] = model
        self.agent_roles[agent_id] = role

    def solve_collaborative_task(
        self,
        task_prompt: Tensor,
        topic: str = "collective_scratchpad",
    ) -> Dict[str, Any]:
        """Execute collaborative reasoning pipeline across the agent society."""
        t0 = time.perf_counter()
        if not isinstance(task_prompt, Tensor):
            raise ValueError("task_prompt must be a tensor")
        if task_prompt.ndim != 2 or task_prompt.numel() == 0:
            raise ValueError("task_prompt must be a non-empty rank-2 tensor")
        if task_prompt.dtype not in {torch.int32, torch.int64}:
            raise ValueError("task_prompt must contain integer token IDs")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be non-empty")
        agent_names = list(self.agents.keys())
        if len(agent_names) == 0:
            return {"status": "no_agents", "steps": 0}

        # Step 1: Agent 1 (Perception/Retrieval) inspects prompt and publishes memory binding
        a1_id = agent_names[0]
        a1_model = self.agents[a1_id]
        a2_id = agent_names[min(1, len(agent_names) - 1)]
        a2_model = self.agents[a2_id]
        min_token = task_prompt.min().item()
        max_token = task_prompt.max().item()
        if (
            min_token < 0
            or max_token >= a1_model.config.vocab_size
            or max_token >= a2_model.config.vocab_size
        ):
            raise ValueError("task_prompt contains token IDs outside an agent vocabulary")
        prompt_length = task_prompt.shape[1]
        if (
            prompt_length > a1_model.config.sequence_len
            or prompt_length > a2_model.config.sequence_len
        ):
            raise ValueError("task_prompt exceeds an executing agent's context window")

        reasoner_layers = [
            module for module in a2_model.modules() if isinstance(module, SynapticLinear)
        ]
        if not reasoner_layers or reasoner_layers[0].w_fast is None:
            raise RuntimeError("reasoning agent has no compatible synaptic fast-weight target")
        reasoner_fast = reasoner_layers[0].w_fast
        fast_before = reasoner_fast.detach().clone()
        topic_existed = topic in self.bus.topics
        topic_before = list(self.bus.topics.get(topic, []))

        a1_device = next(a1_model.parameters()).device
        try:
            with torch.no_grad():
                wte_out = a1_model.wte(task_prompt.to(a1_device))
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

            # Step 2: Agent 2 mounts collective memory and executes a non-adaptive forward.
            a2_device = next(a2_model.parameters()).device
            if not self.bus.sync_to_agent(a2_model, topic=topic, layer_idx=0):
                raise RuntimeError("reasoning agent has no compatible synaptic fast-weight target")

            with _temporary_eval(a2_model), torch.no_grad():
                logits, _ = a2_model(task_prompt.to(a2_device), train_mode=False)
        except Exception:
            with torch.no_grad():
                reasoner_fast.copy_(fast_before)
            if topic_existed:
                self.bus.topics[topic] = topic_before
            else:
                self.bus.topics.pop(topic, None)
            raise

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "solved_collaborative",
            "topic": topic,
            "participating_agents": len({a1_id, a2_id}),
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
