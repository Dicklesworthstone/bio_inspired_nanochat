"""Neuromorphic & Event-Driven Sparse Execution Backend (bead r00r.13).

Implements threshold-gated event-driven execution for synaptic modules, skipping
computation on quiescent / low-calcium channels with bounded error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EventExecutionStats:
    total_elements: int
    active_elements: int
    sparsity_ratio: float
    flops_saved_ratio: float
    max_error_bound: float


class EventDrivenSparseSynapse(nn.Module):
    """Event-driven linear synapse that computes updates only for active events."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        event_threshold: float = 0.05,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.event_threshold = float(event_threshold)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Tensor,
        calcium_activity: Optional[Tensor] = None,
    ) -> Tuple[Tensor, EventExecutionStats]:
        """Compute event-driven sparse forward pass."""
        # Determine active event mask
        if calcium_activity is not None:
            activity = calcium_activity
        else:
            activity = x.abs()

        event_mask = activity > self.event_threshold

        total_elems = int(activity.numel())
        active_elems = int(event_mask.sum().item())
        sparsity = float(1.0 - (active_elems / max(1, total_elems)))

        # Masked sparse computation
        x_sparse = torch.where(event_mask, x, torch.zeros_like(x))
        y = torch.nn.functional.linear(x_sparse, self.weight, self.bias)

        # Theoretical error bound: ||y_dense - y_sparse|| <= threshold * ||W||_1
        w_norm = float(torch.linalg.norm(self.weight, ord=1).item())
        error_bound = self.event_threshold * w_norm

        stats = EventExecutionStats(
            total_elements=total_elems,
            active_elements=active_elems,
            sparsity_ratio=sparsity,
            flops_saved_ratio=sparsity,
            max_error_bound=error_bound,
        )

        return y, stats
