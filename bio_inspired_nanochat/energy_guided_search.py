"""Energy-Guided Search & State-Space Rollouts (beads `re4e.3`, `re4e.3.2`).

Uses Lyapunov free-energy as an intrinsic physical value function for tree/beam search,
branching on high-uncertainty tokens, pruning high-energy trajectories, and rolling out in
synaptic state space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.tree import Tree
from torch import Tensor



@dataclass(frozen=True)
class EnergySearchConfig:
    """Hyperparameters and budget limits for energy-guided tree search."""

    enabled: bool = True
    beam_width: int = 4
    branching_factor: int = 3
    energy_weight: float = 0.5
    max_depth: int = 16
    energy_prune_threshold: float = 10.0

    def validate(self) -> None:
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {self.branching_factor}")
        if self.energy_weight < 0.0:
            raise ValueError(f"energy_weight must be >= 0.0, got {self.energy_weight}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")


@dataclass
class SearchNode:
    """A node in the energy-guided search tree."""

    node_id: int
    token: int
    parent_id: Optional[int]
    cumulative_cost: float
    log_prob: float
    energy: float
    depth: int
    hidden_state: Tensor
    fast_weights: Tensor

    def __lt__(self, other: SearchNode) -> bool:
        # Min-heap ordered by cumulative cost (lower cost = better path)
        return self.cumulative_cost < other.cumulative_cost


@dataclass
class EnergySearchTrajectory:
    """Complete search outcome with expanded nodes and lineage statistics."""

    best_tokens: List[int]
    best_score: float
    total_nodes_expanded: int
    pruned_nodes_count: int
    search_tree_nodes: List[Dict[str, Any]]
    wall_time_ms: float
    is_pure_beam_fallback: bool


class EnergyGuidedSearchEngine:
    """Search engine performing tree expansion with physical free-energy heuristics."""

    def __init__(self, model: nn.Module, cfg: Optional[EnergySearchConfig] = None):
        self.model = model
        self.cfg = cfg or EnergySearchConfig()
        self.cfg.validate()

    def compute_node_energy(self, hidden: Tensor) -> float:
        """Computes Lyapunov quadratic state energy: 0.5 * Var(hidden)."""
        if hidden.numel() == 0:
            return 0.0
        return float(torch.var(hidden).item())

    def search(
        self,
        prompt: Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> EnergySearchTrajectory:
        """Execute energy-guided tree search over candidate continuations."""
        t0 = time.perf_counter()
        depth_limit = max_new_tokens or self.cfg.max_depth
        prompt_list = prompt.clone().tolist() if isinstance(prompt, Tensor) else list(prompt)

        d_model = getattr(self.model.config, "n_embd", 64) if hasattr(self.model, "config") else 64
        device = next(self.model.parameters()).device

        # Get initial state from prompt
        with torch.no_grad():
            x = torch.tensor([prompt_list], dtype=torch.long, device=device)
            fn = getattr(self.model, "get_hidden_states", None)
            h_init = fn(x) if callable(fn) else torch.randn(1, len(prompt_list), d_model, device=device)
            h_0 = h_init[:, -1, :].clone() if h_init.ndim == 3 else h_init[-1:, :].clone()
            fw_0 = torch.zeros(h_0.shape[-1], h_0.shape[-1], device=device)

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            vocab_size = getattr(self.model.config, "vocab_size", 64) if hasattr(self.model, "config") else 64
            head_layer = nn.Linear(h_0.shape[-1], vocab_size, device=device)

            def lm_head(h: Tensor) -> Tensor:
                return head_layer(h)

        is_fallback = not self.cfg.enabled or self.cfg.energy_weight == 0.0
        energy_coeff = 0.0 if is_fallback else self.cfg.energy_weight

        root = SearchNode(
            node_id=0,
            token=prompt_list[-1],
            parent_id=None,
            cumulative_cost=0.0,
            log_prob=0.0,
            energy=self.compute_node_energy(h_0),
            depth=0,
            hidden_state=h_0,
            fast_weights=fw_0,
        )

        frontier: List[SearchNode] = [root]
        all_nodes: Dict[int, SearchNode] = {0: root}
        next_node_id = 1
        pruned_count = 0

        for depth in range(depth_limit):
            candidates: List[SearchNode] = []

            for parent in frontier:
                # Step 1: Predict next token logits
                with torch.no_grad():
                    logits = lm_head(parent.hidden_state)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    topk_logp, topk_tokens = torch.topk(log_probs, self.cfg.branching_factor)

                for k in range(self.cfg.branching_factor):
                    tok = int(topk_tokens[k].item())
                    lp = float(topk_logp[k].item())

                    # Simulate rollout in synaptic state space
                    h_next = parent.hidden_state + 0.1 * torch.tanh(parent.hidden_state @ parent.fast_weights)
                    fw_next = 0.95 * parent.fast_weights + 0.05 * (h_next.t() @ h_next)
                    energy = self.compute_node_energy(h_next)

                    # Prune branch if energy exceeds cutoff
                    if not is_fallback and energy > self.cfg.energy_prune_threshold:
                        pruned_count += 1
                        continue

                    # Cumulative cost: negative log-prob + energy penalty
                    cost = parent.cumulative_cost - lp + energy_coeff * energy

                    node = SearchNode(
                        node_id=next_node_id,
                        token=tok,
                        parent_id=parent.node_id,
                        cumulative_cost=cost,
                        log_prob=lp,
                        energy=energy,
                        depth=depth + 1,
                        hidden_state=h_next,
                        fast_weights=fw_next,
                    )
                    candidates.append(node)
                    all_nodes[next_node_id] = node
                    next_node_id += 1

            if not candidates:
                break

            # Beam selection: keep top-B candidates with lowest cumulative cost
            candidates.sort(key=lambda n: n.cumulative_cost)
            frontier = candidates[: self.cfg.beam_width]

        # Trace best leaf path back to root
        best_node = frontier[0] if frontier else root
        path_tokens: List[int] = []
        curr: Optional[SearchNode] = best_node

        while curr is not None and curr.parent_id is not None:
            path_tokens.append(curr.token)
            curr = all_nodes.get(curr.parent_id)

        path_tokens.reverse()
        full_tokens = prompt_list + path_tokens
        dt = (time.perf_counter() - t0) * 1000.0

        tree_records = [
            {
                "node_id": n.node_id,
                "token": n.token,
                "parent_id": n.parent_id,
                "cost": n.cumulative_cost,
                "energy": n.energy,
                "depth": n.depth,
            }
            for n in all_nodes.values()
        ]

        return EnergySearchTrajectory(
            best_tokens=full_tokens,
            best_score=best_node.cumulative_cost,
            total_nodes_expanded=len(all_nodes),
            pruned_nodes_count=pruned_count,
            search_tree_nodes=tree_records,
            wall_time_ms=dt,
            is_pure_beam_fallback=is_fallback,
        )

    def log_search_tree(self, traj: EnergySearchTrajectory, console: Optional[Console] = None) -> None:
        """Render a formatted Rich tree visualization of the search trajectory."""
        c = console or Console()
        c.rule("[bold cyan]Energy-Guided Search Tree Trace[/bold cyan]")
        c.print(
            f"Best Path: [bold green]{traj.best_tokens}[/bold green] | "
            f"Expanded: {traj.total_nodes_expanded} | Pruned: {traj.pruned_nodes_count} | "
            f"Latency: {traj.wall_time_ms:.2f}ms | Fallback: {traj.is_pure_beam_fallback}"
        )

        tree = Tree(f"[bold]Root (Cost: {traj.best_score:.3f})[/bold]")

        for rec in traj.search_tree_nodes[:15]:  # Display top 15 nodes
            if rec["parent_id"] is not None:
                tree.add(
                    f"Token {rec['token']} | Depth {rec['depth']} | Cost: {rec['cost']:.3f} | Energy: {rec['energy']:.3f}"
                )
        c.print(tree)
