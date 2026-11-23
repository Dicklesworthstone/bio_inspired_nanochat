"""Neuromorphic & Event-Driven Sparse Execution Backend (bead r00r.13).

Implements threshold-gated event-driven execution for synaptic modules, skipping
computation on quiescent / low-calcium channels with bounded error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class EventExecutionStats:
    """Projection arithmetic only. Masking, bound calculation and dispatch also take time."""

    total_elements: int
    active_elements: int
    sparsity_ratio: float
    projection_backend: str
    dense_projection_macs: int
    executed_projection_macs: int
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
        if min(in_features, out_features) <= 0:
            raise ValueError("feature dimensions must be positive")
        if not math.isfinite(event_threshold) or event_threshold < 0:
            raise ValueError("event_threshold must be finite and nonnegative")
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
        calcium_activity: Tensor | None = None,
    ) -> tuple[Tensor, EventExecutionStats]:
        """Skip small inputs, retaining any extra events requested by calcium activity.

        The bound is on the worst output-coordinate truncation error in exact
        arithmetic. Different kernels may also round their accumulated sums differently.
        """
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"input must end with {self.in_features} features")
        if not x.is_floating_point():
            raise TypeError("event-driven projection requires a real floating input")
        magnitude = x.detach().abs()
        if not torch.isfinite(magnitude).all():
            raise ValueError("event input must be finite")
        event_mask = magnitude > self.event_threshold
        if calcium_activity is not None:
            # Calcium may retain a subthreshold signal. It cannot suppress a large
            # signal and invalidate an input-amplitude truncation bound.
            activity = calcium_activity.detach().expand_as(x)
            if not torch.isfinite(activity).all():
                raise ValueError("calcium activity must be finite")
            event_mask = event_mask | (activity > self.event_threshold)

        total_elems = x.numel()
        active_elems = int(event_mask.sum().item())
        dense_macs = total_elems * self.out_features
        executed_macs = dense_macs
        error_bound = 0.0
        if active_elems == total_elems:
            backend = "dense"
            y = F.linear(x, self.weight, self.bias)
        else:
            omitted_max = float(magnitude.masked_fill(event_mask, 0).max().item())
            if omitted_max:
                # Each output's omitted sum is at most max|omitted x| times
                # sum_i |W[o, i]|. The induced infinity norm is a max ROW sum.
                weights = self.weight.detach()
                if weights.device.type == "mps":
                    weights = weights.cpu()  # MPS cannot accumulate the bound in float64.
                row_sums = weights.to(torch.float64).abs().sum(dim=1)
                error_bound = omitted_max * float(row_sums.max().item())
            sparse_supported = x.device.type in {"cpu", "cuda"} and x.dtype in {torch.float32, torch.float64}
            # COO index/dispatch overhead grows with density; dense fallback
            # retains the same approximation and charges all the projection MACs.
            if sparse_supported and active_elems * 4 <= total_elems:
                flat = x.reshape(-1, self.in_features)
                indices = event_mask.reshape_as(flat).nonzero().T.contiguous()
                values = flat[indices[0], indices[1]]
                events = torch.sparse_coo_tensor(
                    indices, values, size=flat.shape, is_coalesced=True,
                )
                y = torch.sparse.mm(events, self.weight.T).reshape(*x.shape[:-1], self.out_features)
                if self.bias is not None:
                    y = y + self.bias
                backend = "sparse_coo"
                executed_macs = active_elems * self.out_features
            else:
                backend = "dense_masked"
                y = F.linear(x.masked_fill(~event_mask, 0), self.weight, self.bias)

        return y, EventExecutionStats(
            total_elements=total_elems,
            active_elements=active_elems,
            sparsity_ratio=(total_elems - active_elems) / total_elems if total_elems else 0.0,
            projection_backend=backend,
            dense_projection_macs=dense_macs,
            executed_projection_macs=executed_macs,
            max_error_bound=error_bound,
        )
