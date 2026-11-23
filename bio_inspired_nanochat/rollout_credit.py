"""Masked, per-rollout Hebbian credit for the chat policy gradient."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


def pack_rollout_tokens(ids: Tensor, sampled: Tensor, *, include_last: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Build shifted policy targets and the action-token observation mask."""
    shifted = ids[:, 1:].masked_fill(~sampled[:, 1:], -1)
    if not include_last:
        return ids[:, :-1], shifted, sampled[:, :-1]
    targets = torch.full_like(ids, -1)
    targets[:, :-1] = shifted
    return ids, targets, sampled


def synaptic_token_logp(model: GPTSynaptic, inputs: Tensor, targets: Tensor, *, update_mem: bool) -> Tensor:
    """Return a log probability at every supervised position, including on large CUDA batches."""
    hidden, _ = model._forward_hidden(inputs, train_mode=True, structural_training=True, update_mem=update_mem)
    states, labels = hidden.flatten(0, 1), targets.reshape(-1)

    def project_loss(x: Tensor, y: Tensor) -> Tensor:
        logits = model.hidden_to_logits(x, for_loss=True)
        return F.cross_entropy(logits, y, reduction="none", ignore_index=-1)

    if hidden.is_cuda and labels.numel() > 2048:
        losses = torch.cat([
            checkpoint(project_loss, states[start:start + 2048], labels[start:start + 2048], use_reentrant=False)
            for start in range(0, labels.numel(), 2048)
        ])
    else:
        losses = project_loss(states, labels)
    return -losses.view_as(targets)


def permute_advantages(advantages: Tensor, seed: int) -> Tensor:
    """Shuffle a complete question group with an independent, reproducible RNG."""
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(advantages.numel(), generator=rng).to(advantages.device)
    return advantages[order]


def rollout_direction(x: Tensor, y: Tensor, mask: Tensor, pin: Tensor, pout: Tensor) -> Tensor:
    """Unit direction of the un-decayed co-activity product for one rollout."""
    # Mask both factors prior to contraction so forced tokens and padding have
    # no contribution, including if an excluded activation is non-finite.
    keep = mask.unsqueeze(-1)
    x = torch.where(keep, x, 0)
    y = torch.where(keep, y, 0)
    scale = .05 / mask.sum().clamp_min(1)
    u = (x.T @ (y @ pout)) * scale
    v = ((x @ pin).T @ y) * scale
    raw = u @ v
    norm = torch.linalg.vector_norm(raw)
    safe = torch.where(norm.isfinite(), norm.clamp_min(1e-12), torch.full_like(norm, math.nan))
    return torch.where(norm <= 1e-12, 0, raw / safe)


class RolloutCredit:
    """Collect detached directions from real dense MLP calls and commit after backward."""

    def __init__(self, model: GPTSynaptic, *, lr: float, rollouts_per_rank: int):
        cfg = model.config
        if cfg.use_moe or not cfg.syn_cfg.enable_hebbian or cfg.syn_cfg.enable_stdp:
            raise ValueError("Rollout credit requires dense co-activity Hebbian layers with STDP off")
        if not math.isfinite(lr) or lr <= 0 or rollouts_per_rank <= 0:
            raise ValueError("Rollout credit requires a positive finite rate and rollout count")
        self.lr = lr
        self.rollouts_per_rank = rollouts_per_rank
        self.layers = [layer for layer in model.modules() if isinstance(layer, SynapticLinear)]
        for layer in self.layers:
            if (layer.proj_in is None or layer.proj_out is None or layer._plasticity_pending
                    or getattr(layer, "_nm_ach_input_gain", 1.0) != 1.0
                    or getattr(layer, "_nm_ne_gain", 1.0) != 1.0):
                raise ValueError("Rollout credit requires projections, no deferred writes, and neutral input/output gains")
        if not self.layers:
            raise ValueError("No eligible dense layers found")
        self._deltas = [torch.zeros_like(layer.w_slow, dtype=torch.float32) for layer in self.layers]
        self._seen = 0
        self._credited = 0

    @contextmanager
    def observe(self, action_mask: Tensor, advantages: Tensor) -> Iterator[None]:
        """Attach to one policy forward. All input rows count toward the denominator."""
        if (action_mask.ndim != 2 or action_mask.dtype != torch.bool or advantages.ndim != 1
                or action_mask.size(0) != advantages.size(0)):
            raise ValueError("Expected an action mask and one advantage per rollout")
        selected = ((advantages != 0) & action_mask.any(dim=1)).nonzero().flatten().tolist()
        batch_deltas = {}

        @torch.no_grad()
        def capture(layer: SynapticLinear, args: tuple[Tensor, ...], output: Tensor) -> None:
            if layer in batch_deltas:
                raise RuntimeError("A rollout trace site was called twice")
            if args[0].shape[:2] != action_mask.shape or output.shape[:2] != action_mask.shape:
                raise RuntimeError("Rollout action mask does not match the observed activation positions")
            assert layer.proj_in is not None and layer.proj_out is not None
            # LayerNorm is the existing transformation at this site. The
            # returned post-activity equals its eligibility activity at neutral NE.
            x = args[0].detach()
            if layer.input_ln is not None:
                x = layer.input_ln(x)
            with torch.autocast(device_type=x.device.type, enabled=False):
                x, y = x.float(), output.float()
                pi, po = layer.proj_in.float(), layer.proj_out.float()
                delta = torch.zeros_like(layer.w_slow, dtype=torch.float32)
                for i in selected:
                    direction = rollout_direction(x[i], y[i], action_mask[i], pi, po)
                    delta.add_(advantages[i] * direction)
            batch_deltas[layer] = delta

        if selected:
            with ExitStack() as stack:
                for layer in self.layers:
                    stack.callback(layer.register_forward_hook(capture).remove)
                yield
            if len(batch_deltas) != len(self.layers):
                raise RuntimeError("A policy forward skipped rollout trace sites")
            for layer, delta in zip(self.layers, self._deltas, strict=True):
                delta.add_(batch_deltas[layer])
        else:
            yield
        self._seen += advantages.numel()
        self._credited += len(selected)

    @torch.no_grad()
    def commit(self, *, average_replicas: Callable[[Tensor], None] | None = None) -> dict[str, int | float]:
        """Apply one step's slow-matrix credit after the gradient optimizer."""
        if self._seen != self.rollouts_per_rank:
            raise RuntimeError(f"Expected {self.rollouts_per_rank} rollouts, saw {self._seen}")
        updates = [delta * (self.lr / self._seen) for delta in self._deltas]
        if average_replicas is not None:
            for update in updates:
                average_replicas(update)
        norms = [float(torch.linalg.vector_norm(update)) for update in updates]
        if not all(math.isfinite(norm) for norm in norms):
            raise FloatingPointError("Non-finite reduced rollout credit")
        for layer, update, norm, delta in zip(self.layers, updates, norms, self._deltas, strict=True):
            if norm > 0:
                layer.w_slow.add_(update)
            delta.zero_()
        result = {"local_rollouts": self._seen, "local_credited_rollouts": self._credited,
                  "max_update_norm": max(norms)}
        self._seen = self._credited = 0
        return result
