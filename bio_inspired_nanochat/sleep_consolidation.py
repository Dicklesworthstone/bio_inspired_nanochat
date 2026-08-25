"""Sleep Consolidation & Prioritized High-Surprise Replay Buffer (beads `cel.1`, `cel.2`).

Implements the biological two-stage memory consolidation system:
1. `PrioritizedReplayBuffer`: Episodic memory buffer storing high-surprise / high-loss sequences.
2. `SleepConsolidationController`: Offline NREM replay loop distilling fast synaptic weights (W_fast)
   into durable slow weights (W_slow) gated by the CaMKII/PP1 bistable latch, followed by SHY downscaling.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from torch import Tensor, nn

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


@contextmanager
def _temporary_eval(model: nn.Module) -> Iterator[None]:
    """Temporarily evaluate a model without flattening caller-owned per-module modes."""
    training_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in training_modes:
            module.training = was_training


@dataclass
class ReplayBufferItem:
    """An episodic memory experience stored in the replay buffer with inputs and targets."""

    inputs: Tensor
    targets: Tensor
    loss: float
    step: int = 0
    task_id: int = 0
    priority: float = 1.0


class ReplayBuffer:
    """Prioritized experience replay buffer for offline sleep consolidation."""

    def __init__(self, max_capacity: int = 64, alpha: float = 0.6, seed: int = 42):
        if isinstance(max_capacity, bool) or not isinstance(max_capacity, int) or max_capacity <= 0:
            raise ValueError("max_capacity must be a positive integer")
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.max_capacity = max_capacity
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.buffer: List[ReplayBufferItem] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        inputs: Tensor,
        targets: Tensor,
        loss: float = 1.0,
        step: int = 0,
        task_id: int = 0,
    ) -> None:
        """Insert sequence into the buffer; evicts lowest-loss item on capacity overflow."""
        if inputs.numel() == 0 or targets.numel() == 0:
            raise ValueError("inputs and targets must be non-empty tensors")
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("loss must be finite and non-negative")
        for name, value in (("step", step), ("task_id", task_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        priority = max(1e-4, float(loss))
        item = ReplayBufferItem(
            inputs=inputs.detach().cpu().clone(),
            targets=targets.detach().cpu().clone(),
            loss=float(loss),
            step=step,
            task_id=task_id,
            priority=priority,
        )
        if len(self.buffer) < self.max_capacity:
            self.buffer.append(item)
        else:
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i].loss)
            if item.loss > self.buffer[min_idx].loss:
                self.buffer[min_idx] = item

    def sample(self, batch_size: int) -> List[ReplayBufferItem]:
        """Sample batch proportionally to priority^alpha."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not self.buffer:
            return []
        n = len(self.buffer)
        k = min(batch_size, n)
        priorities = np.array([item.priority for item in self.buffer], dtype=np.float64)
        if self.alpha == 0.0:
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            # Subtract the maximum log-priority before exponentiation. Direct
            # ``priority ** alpha`` overflows for valid but large finite settings.
            log_priorities = np.log(priorities)
            centered = log_priorities - log_priorities.max()
            cutoff = np.log(np.finfo(np.float64).tiny) / self.alpha
            log_weights = np.full_like(centered, -np.inf)
            active = centered >= cutoff
            log_weights[active] = self.alpha * centered[active]
            weights = np.exp(log_weights)
            probs = weights / weights.sum()
        indices = self.rng.choice(n, size=k, replace=False, p=probs)
        return [self.buffer[i] for i in indices]


def get_synaptic_layers(model: nn.Module) -> List[SynapticLinear]:
    """Return all SynapticLinear layers found in the model."""
    return [m for m in model.modules() if isinstance(m, SynapticLinear)]


def homeostatic_downscale(
    model: nn.Module,
    max_slow_norm: Optional[float] = None,
    decay_factor: float = 0.95,
) -> Dict[str, float]:
    """Apply synaptic homeostasis downscaling (SHY protocol) to slow weights."""
    if max_slow_norm is not None and (
        not math.isfinite(max_slow_norm) or max_slow_norm < 0.0
    ):
        raise ValueError("max_slow_norm must be finite and non-negative")
    if not math.isfinite(decay_factor) or not 0.0 <= decay_factor <= 1.0:
        raise ValueError("decay_factor must be finite and in [0, 1]")
    syn_layers = get_synaptic_layers(model)
    sq_sum = 0.0
    for lin in syn_layers:
        if lin.w_slow is not None:
            sq_sum += float(lin.w_slow.detach().norm() ** 2)
        if lin.post is not None and lin.post.slow is not None:
            sq_sum += float(lin.post.slow.detach().norm() ** 2)
    curr_norm = math.sqrt(sq_sum)
    if not math.isfinite(curr_norm):
        raise ValueError("slow weights must contain only finite values")
    initial_norm = curr_norm

    scale = decay_factor
    if max_slow_norm is not None:
        if curr_norm > max_slow_norm and curr_norm > 1e-6:
            target = max_slow_norm * decay_factor
            scale = target / curr_norm
        else:
            scale = 1.0

    if scale < 1.0:
        for lin in syn_layers:
            if lin.w_slow is not None:
                lin.w_slow.data.mul_(scale)
            if lin.post is not None and lin.post.slow is not None:
                lin.post.slow.data.mul_(scale)
        post_norm = curr_norm * scale
    else:
        post_norm = curr_norm

    return {
        "initial_norm": initial_norm,
        "scaling_factor": scale,
        "post_norm": post_norm,
    }


def consolidate_sleep_replay(
    model: GPTSynaptic,
    replay_items: Sequence[ReplayBufferItem],
    device: str | torch.device | None = None,
    consolidation_passes: int = 1,
    downscale_decay: float = 0.95,
    max_slow_norm: Optional[float] = 15.0,
    reset_fast_after: bool = False,
) -> Dict[str, Any]:
    """Replay buffered episodes through the model's native adaptive-plasticity path.

    This must run after backward at a safe parameter-mutation boundary. Setting
    ``reset_fast_after=True`` explicitly clears a trainable fast-weight parameter and is
    intended only for experiments that deliberately probe slow-state retention.
    """
    if (
        isinstance(consolidation_passes, bool)
        or not isinstance(consolidation_passes, int)
        or consolidation_passes <= 0
    ):
        raise ValueError("consolidation_passes must be a positive integer")
    if not isinstance(reset_fast_after, bool):
        raise ValueError("reset_fast_after must be a boolean")
    if max_slow_norm is not None and (
        not math.isfinite(max_slow_norm) or max_slow_norm < 0.0
    ):
        raise ValueError("max_slow_norm must be finite and non-negative")
    if not math.isfinite(downscale_decay) or not 0.0 <= downscale_decay <= 1.0:
        raise ValueError("downscale_decay must be finite and in [0, 1]")
    if not replay_items:
        return {
            "replayed_items": 0,
            "passes": 0,
            "status": "empty",
            "homeostasis": {"scaling_factor": 1.0, "initial_norm": 0.0, "post_norm": 0.0},
            "scaling_factor": 1.0,
        }

    model_device = next(model.parameters()).device
    device_obj = torch.device(device) if device is not None else model_device
    if device_obj != model_device:
        raise ValueError(
            f"sleep replay device {device_obj} does not match model device {model_device}"
        )

    syn_layers = get_synaptic_layers(model)
    sq_sum = 0.0
    for lin in syn_layers:
        if lin.w_slow is not None:
            sq_sum += float(lin.w_slow.detach().norm() ** 2)
        if lin.post is not None and lin.post.slow is not None:
            sq_sum += float(lin.post.slow.detach().norm() ** 2)
    init_norm = math.sqrt(sq_sum)

    with torch.no_grad():
        for lin in syn_layers:
            if lin._plasticity_pending:
                lin._apply_hebb_weight_writes(lin._last_gate_scale)
                lin._plasticity_pending = False
        model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)

    try:
        with _temporary_eval(model), torch.no_grad():
            for _ in range(consolidation_passes):
                for item in replay_items:
                    inp = item.inputs.to(device_obj)
                    if (
                        inp.dtype not in {torch.int32, torch.int64}
                        or inp.numel() == 0
                        or inp.min().item() < 0
                        or inp.max().item() >= model.config.vocab_size
                    ):
                        raise ValueError(
                            "replay inputs must be non-empty integer IDs inside the vocabulary"
                        )
                    model(inp, train_mode=True)
    finally:
        model.reset_sequence_state(
            reset_fast_weights=reset_fast_after, reset_consolidation=False
        )

    downscale_stats = homeostatic_downscale(
        model, max_slow_norm=max_slow_norm, decay_factor=downscale_decay
    )

    return {
        "replayed_items": len(replay_items),
        "passes": consolidation_passes,
        "initial_slow_norm": init_norm,
        "post_slow_norm": downscale_stats["post_norm"],
        "scaling_factor": downscale_stats["scaling_factor"],
        "homeostasis": downscale_stats,
    }


@dataclass
class ReplayItem:
    """An episodic memory experience stored in the prioritized replay buffer."""

    tokens: Tensor
    surprise_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrioritizedReplayBuffer:
    """Bounded prioritized experience replay buffer ranked by prediction surprise."""

    def __init__(self, capacity: int = 128, alpha: float = 0.6):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        self.capacity = capacity
        self.alpha = alpha
        self.items: List[ReplayItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def push(
        self,
        tokens: Tensor,
        surprise_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert sequence(s) into the buffer; evicts the lowest-surprise item when full."""
        if tokens.ndim not in {1, 2} or tokens.numel() == 0:
            raise ValueError("tokens must be a non-empty rank-1 or rank-2 tensor")
        if tokens.dtype not in {torch.int32, torch.int64}:
            raise ValueError("tokens must contain integer token IDs")
        if not math.isfinite(surprise_score) or surprise_score < 0.0:
            raise ValueError("surprise_score must be finite and non-negative")
        # A replay item is a historical snapshot. ``Tensor.cpu()`` aliases storage when
        # the caller is already on CPU, so clone explicitly before retaining it.
        tok = tokens.detach().cpu().clone()
        if tok.ndim > 1:
            for i in range(tok.shape[0]):
                self._push_single(tok[i], surprise_score, metadata)
        else:
            self._push_single(tok, surprise_score, metadata)

    def _push_single(
        self,
        tok: Tensor,
        surprise_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        item = ReplayItem(
            tokens=tok,
            surprise_score=float(surprise_score),
            timestamp=time.time(),
            metadata=dict(metadata) if metadata is not None else {},
        )

        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            # Find item with minimal priority
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i].surprise_score)
            if item.surprise_score > self.items[min_idx].surprise_score:
                self.items[min_idx] = item

    def sample(self, batch_size: int) -> Tuple[List[ReplayItem], List[int]]:
        """Sample a batch of items with probability proportional to (surprise)^alpha."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if len(self.items) == 0:
            return [], []

        n = len(self.items)
        k = min(batch_size, n)

        scores = np.array([max(1e-4, item.surprise_score) for item in self.items], dtype=np.float64)
        if self.alpha == 0.0:
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            log_scores = np.log(scores)
            centered = log_scores - log_scores.max()
            cutoff = np.log(np.finfo(np.float64).tiny) / self.alpha
            log_weights = np.full_like(centered, -np.inf)
            active = centered >= cutoff
            log_weights[active] = self.alpha * centered[active]
            weights = np.exp(log_weights)
            probs = weights / weights.sum()

        chosen_indices = np.random.choice(n, size=k, replace=False, p=probs).tolist()
        sampled_items = [self.items[idx] for idx in chosen_indices]
        return sampled_items, chosen_indices


class SleepConsolidationController:
    """Executes offline replay, native synaptic consolidation, and homeostatic downscaling."""

    def __init__(
        self,
        consolidation_lr: float = 0.1,
        downscale_target_norm: Optional[float] = None,
        latch_threshold: float = 0.4,
    ):
        if not math.isfinite(consolidation_lr) or consolidation_lr < 0.0:
            raise ValueError("consolidation_lr must be finite and non-negative")
        if downscale_target_norm is not None and (
            not math.isfinite(downscale_target_norm) or downscale_target_norm < 0.0
        ):
            raise ValueError("downscale_target_norm must be finite and non-negative")
        if not math.isfinite(latch_threshold) or not 0.0 <= latch_threshold <= 1.0:
            raise ValueError("latch_threshold must be finite and in [0, 1]")
        self.consolidation_lr = consolidation_lr
        self.downscale_target_norm = downscale_target_norm
        self.latch_threshold = latch_threshold

    @torch.no_grad()
    def generate_dreams(
        self,
        model: GPTSynaptic,
        num_dreams: int = 4,
        seq_len: int = 16,
        temperature: float = 0.8,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """Autoregressively generate synthetic replay sequences from the current model state."""
        if isinstance(num_dreams, bool) or not isinstance(num_dreams, int) or num_dreams <= 0:
            raise ValueError("num_dreams must be a positive integer")
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if seq_len > model.config.sequence_len:
            raise ValueError("seq_len cannot exceed the model sequence length")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        with _temporary_eval(model):
            vocab_size = model.config.vocab_size
            device_obj = (
                torch.device(device)
                if device is not None
                else next(model.parameters()).device
            )
            # Start from random prompt seeds
            dreams = torch.randint(0, vocab_size, (num_dreams, 1), device=device_obj)

            for _ in range(seq_len - 1):
                logits, _ = model(dreams, train_mode=False)
                next_logits = logits[:, -1, :] / max(1e-3, temperature)
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                dreams = torch.cat([dreams, next_tok], dim=1)

        return dreams

    def run_sleep_phase(
        self,
        model: GPTSynaptic,
        replay_buffer: Optional[PrioritizedReplayBuffer] = None,
        sleep_steps: int = 4,
        batch_size: int = 4,
        use_dream_replay: bool = False,
        device: str | torch.device | None = None,
    ) -> Dict[str, Any]:
        """Run native replay consolidation at a post-backward/optimizer safe boundary.

        The final deferred wake update is flushed once before replay. Replay then uses the
        model's native adaptive write as the sole source of slow-state change; this controller
        scales that measured change and never copies or clears the trainable ``w_fast`` value.
        """
        if isinstance(sleep_steps, bool) or not isinstance(sleep_steps, int) or sleep_steps < 0:
            raise ValueError("sleep_steps must be a non-negative integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not use_dream_replay and (replay_buffer is None or len(replay_buffer) == 0):
            return {"status": "skipped_empty_buffer", "steps_run": 0}
        if sleep_steps == 0:
            return {"status": "skipped_zero_steps", "steps_run": 0}

        t0 = time.perf_counter()
        total_transferred_norm = 0.0
        consolidated_layer_ids: set[int] = set()
        model_device = next(model.parameters()).device
        device_obj = (
            torch.device(device)
            if device is not None
            else model_device
        )
        if device_obj != model_device:
            raise ValueError(
                f"sleep replay device {device_obj} does not match model device {model_device}"
            )

        syn_layers = get_synaptic_layers(model)

        # The training path defers its local write until the next safe forward. Sleep is a
        # parameter-mutating post-step hook, so land that owed wake write exactly once, then
        # clear its traces before replay so they cannot be incorporated a second time.
        with torch.no_grad():
            for mod in syn_layers:
                if mod._plasticity_pending:
                    mod._apply_hebb_weight_writes(mod._last_gate_scale)
                    mod._plasticity_pending = False
            model.reset_sequence_state(
                reset_fast_weights=False, reset_consolidation=False
            )

        # Snapshot only state that native adaptive forwards can mutate. This is substantially
        # smaller than cloning the entire model and lets a failed/non-finite phase roll back.
        rollback_tensors: List[Tuple[Tensor, Tensor]] = []
        seen_tensors: set[int] = set()

        def remember(tensor: Optional[Tensor]) -> None:
            if tensor is not None and id(tensor) not in seen_tensors:
                seen_tensors.add(id(tensor))
                rollback_tensors.append((tensor, tensor.detach().clone()))

        for mod in syn_layers:
            remember(mod.w_slow)
            remember(mod.w_fast)
            for buffer in mod.buffers(recurse=True):
                remember(buffer)
            if mod.post is not None:
                remember(mod.post.fast)
                remember(mod.post.slow)
        for module in model.modules():
            ema_e = getattr(module, "ema_e", None)
            if isinstance(ema_e, Tensor):
                remember(ema_e)

        bookkeeping = [
            (mod, mod._plasticity_pending, mod._last_gate_scale)
            for mod in syn_layers
        ]

        def restore_phase_state() -> None:
            with torch.no_grad():
                for target, original in rollback_tensors:
                    target.copy_(original)
                for mod, pending, gate_scale in bookkeeping:
                    mod._plasticity_pending = pending
                    mod._last_gate_scale = gate_scale

        try:
            with _temporary_eval(model):
                for _step in range(sleep_steps):
                    if use_dream_replay:
                        batch_tokens = self.generate_dreams(
                            model,
                            num_dreams=batch_size,
                            seq_len=model.config.sequence_len,
                            device=device_obj,
                        )
                    else:
                        if replay_buffer is None:
                            break
                        items, _ = replay_buffer.sample(batch_size)
                        if not items:
                            break

                        # Replay items can come from sessions with different context lengths.
                        # Crop to the shortest sampled sequence so stacking is deterministic and
                        # never pads with fabricated token IDs.
                        sequences = [
                            item.tokens if item.tokens.ndim == 1 else item.tokens.squeeze(0)
                            for item in items
                        ]
                        replay_len = min(
                            model.config.sequence_len,
                            *(sequence.shape[0] for sequence in sequences),
                        )
                        batch_tokens = torch.stack(
                            [sequence[:replay_len] for sequence in sequences]
                        ).to(device_obj)

                    if (
                        batch_tokens.dtype not in {torch.int32, torch.int64}
                        or batch_tokens.numel() == 0
                        or batch_tokens.min().item() < 0
                        or batch_tokens.max().item() >= model.config.vocab_size
                    ):
                        raise ValueError(
                            "replay tokens must be non-empty integer IDs inside the vocabulary"
                        )

                    slow_before = [
                        (
                            mod,
                            mod.w_slow.detach().clone(),
                            (
                                mod.post.slow.detach().clone()
                                if mod.post is not None and mod.post.slow is not None
                                else None
                            ),
                        )
                        for mod in syn_layers
                    ]

                    with torch.no_grad():
                        # Native adaptive replay updates eligibility, fast state, and gated slow
                        # state exactly once. We scale the observed slow delta below rather than
                        # applying a second, unrelated whole-fast-weight copy.
                        model(batch_tokens, train_mode=True)

                        for layer_idx, (mod, w_slow_before, post_slow_before) in enumerate(
                            slow_before
                        ):
                            slow_targets = [(mod.w_slow, w_slow_before)]
                            if (
                                mod.post is not None
                                and mod.post.slow is not None
                                and post_slow_before is not None
                            ):
                                slow_targets.append((mod.post.slow, post_slow_before))

                            latch_open = (
                                mod.post is not None
                                and float(mod.post.camkii.detach().mean().item())
                                >= self.latch_threshold
                            )
                            layer_delta_sq = 0.0
                            for target, before in slow_targets:
                                native_delta = target.detach() - before
                                if not torch.isfinite(native_delta).all():
                                    raise FloatingPointError(
                                        "sleep replay produced a non-finite slow-state delta"
                                    )
                                if latch_open:
                                    retained_delta = self.consolidation_lr * native_delta
                                    target.copy_(before + retained_delta)
                                    layer_delta_sq += float(retained_delta.norm().item()) ** 2
                                else:
                                    target.copy_(before)

                            if layer_delta_sq > 0.0:
                                total_transferred_norm += math.sqrt(layer_delta_sq)
                                consolidated_layer_ids.add(layer_idx)

                            if self.downscale_target_norm is not None:
                                curr_norm = float(mod.w_slow.detach().norm().item())
                                if not math.isfinite(curr_norm):
                                    raise FloatingPointError(
                                        "sleep replay produced non-finite slow weights"
                                    )
                                if curr_norm > self.downscale_target_norm:
                                    scale = self.downscale_target_norm / max(1e-6, curr_norm)
                                    mod.w_slow.mul_(scale)

                model.reset_sequence_state(
                    reset_fast_weights=False, reset_consolidation=False
                )
        except Exception:
            restore_phase_state()
            raise

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "consolidated",
            "steps_run": sleep_steps,
            "layers_consolidated": len(consolidated_layer_ids),
            "total_transferred_norm": total_transferred_norm,
            "wall_time_ms": dt,
        }

    def log_consolidation(
        self,
        report: Dict[str, Any],
        console: Optional[Console] = None,
    ) -> None:
        """Render a formatted Rich panel of sleep consolidation metrics."""
        c = console or Console()
        c.rule("[bold cyan]Offline Sleep Consolidation Report (SHY Protocol)[/bold cyan]")
        c.print(
            f"Status: [bold green]{report['status']}[/bold green] | "
            f"Steps Replayed: {report['steps_run']} | "
            f"Transferred Norm: {report.get('total_transferred_norm', 0.0):.4f} | "
            f"Latency: {report.get('wall_time_ms', 0.0):.2f}ms"
        )


@dataclass(frozen=True)
class WakeSleepConfig:
    """Configuration for interleaved wake-sleep training and inference scheduler."""

    enabled: bool = True
    sleep_every_n_steps: int = 20
    sleep_duration_steps: int = 3
    surprise_threshold: float = 1.0
    consolidate_on_session_end: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if (
            isinstance(self.sleep_every_n_steps, bool)
            or not isinstance(self.sleep_every_n_steps, int)
            or self.sleep_every_n_steps <= 0
        ):
            raise ValueError("sleep_every_n_steps must be a positive integer")
        if (
            isinstance(self.sleep_duration_steps, bool)
            or not isinstance(self.sleep_duration_steps, int)
            or self.sleep_duration_steps < 0
        ):
            raise ValueError("sleep_duration_steps must be a non-negative integer")
        if not math.isfinite(self.surprise_threshold) or self.surprise_threshold < 0.0:
            raise ValueError("surprise_threshold must be finite and non-negative")
        if not isinstance(self.consolidate_on_session_end, bool):
            raise ValueError("consolidate_on_session_end must be a boolean")


class WakeSleepScheduler:
    """Interleaves wake (online learning) and sleep (offline replay & consolidation) phases."""

    def __init__(
        self,
        model: GPTSynaptic,
        buffer: Optional[PrioritizedReplayBuffer] = None,
        controller: Optional[SleepConsolidationController] = None,
        cfg: Optional[WakeSleepConfig] = None,
    ):
        self.model = model
        self.buffer = buffer if buffer is not None else PrioritizedReplayBuffer()
        self.controller = (
            controller if controller is not None else SleepConsolidationController()
        )
        self.cfg = cfg if cfg is not None else WakeSleepConfig()
        self.total_sleep_phases = 0
        self.last_consolidation_report: Optional[Dict[str, Any]] = None

    def step_training(
        self,
        step_idx: int,
        tokens: Tensor,
        step_loss: float,
    ) -> Optional[Dict[str, Any]]:
        """Post-backward/post-optimizer hook that records surprise and may trigger sleep."""
        if not self.cfg.enabled:
            return None
        if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
            raise ValueError("step_idx must be a non-negative integer")
        if not math.isfinite(step_loss) or step_loss < 0.0:
            raise ValueError("step_loss must be finite and non-negative")

        if step_loss >= self.cfg.surprise_threshold:
            self.buffer.push(tokens.detach().cpu(), surprise_score=step_loss)

        if step_idx > 0 and step_idx % self.cfg.sleep_every_n_steps == 0:
            report = self.controller.run_sleep_phase(
                self.model,
                self.buffer,
                sleep_steps=self.cfg.sleep_duration_steps,
            )
            self.total_sleep_phases += 1
            self.last_consolidation_report = report
            return report
        return None

    def on_session_end(self, session_tokens: Optional[Tensor] = None) -> Optional[Dict[str, Any]]:
        """Hook called at the conclusion of an interactive chat or inference session."""
        if not self.cfg.enabled or not self.cfg.consolidate_on_session_end:
            return None

        if session_tokens is not None:
            self.buffer.push(session_tokens.detach().cpu(), surprise_score=2.0)

        report = self.controller.run_sleep_phase(
            self.model,
            self.buffer,
            sleep_steps=self.cfg.sleep_duration_steps,
        )
        self.total_sleep_phases += 1
        self.last_consolidation_report = report
        return report
