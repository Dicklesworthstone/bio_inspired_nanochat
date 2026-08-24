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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from torch import Tensor, nn

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


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
        priority = max(1e-4, float(loss))
        item = ReplayBufferItem(
            inputs=inputs.detach().cpu(),
            targets=targets.detach().cpu(),
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
        if not self.buffer:
            return []
        n = len(self.buffer)
        k = min(batch_size, n)
        priorities = np.array([item.priority for item in self.buffer], dtype=np.float64)
        probs = priorities**self.alpha
        probs /= probs.sum()
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
    syn_layers = get_synaptic_layers(model)
    sq_sum = 0.0
    for lin in syn_layers:
        if lin.w_slow is not None:
            sq_sum += float(lin.w_slow.detach().norm() ** 2)
        if lin.post is not None and lin.post.slow is not None:
            sq_sum += float(lin.post.slow.detach().norm() ** 2)
    curr_norm = math.sqrt(sq_sum)
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
    device: str = "cpu",
    consolidation_passes: int = 1,
    downscale_decay: float = 0.95,
    max_slow_norm: Optional[float] = 15.0,
    reset_fast_after: bool = True,
) -> Dict[str, Any]:
    """Replay buffered high-surprise episodes to distill fast weights into slow weights."""
    if not replay_items:
        return {
            "replayed_items": 0,
            "passes": 0,
            "status": "empty",
            "homeostasis": {"scaling_factor": 1.0, "initial_norm": 0.0, "post_norm": 0.0},
            "scaling_factor": 1.0,
        }

    syn_layers = get_synaptic_layers(model)
    sq_sum = sum(float(lin.w_slow.detach().norm() ** 2) for lin in syn_layers if lin.w_slow is not None)
    init_norm = math.sqrt(sq_sum)

    for _ in range(consolidation_passes):
        for item in replay_items:
            inp = item.inputs.to(device)
            tgt = item.targets.to(device)
            # Forward pass through model to activate eligibility and fast weights
            with torch.no_grad():
                model(inp, targets=tgt)

            # Consolidate fast weights into slow weights
            for lin in syn_layers:
                lr = 0.05
                if lin.post is not None and hasattr(lin.post, "cfg"):
                    lr = lin.post.cfg.post_slow_lr
                if lin.w_fast is not None and lin.w_fast.norm() > 1e-6:
                    lin.w_slow.data.add_(lr * lin.w_fast.detach())
                if lin.post is not None and lin.post.fast is not None and lin.post.slow is not None:
                    lin.post.slow.data.add_(lr * lin.post.fast.detach())

    if reset_fast_after:
        model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=False)

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
        tok = tokens.detach().cpu()
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
            metadata=metadata or {},
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
        if len(self.items) == 0:
            return [], []

        n = len(self.items)
        k = min(batch_size, n)

        scores = np.array([max(1e-4, item.surprise_score) for item in self.items], dtype=np.float64)
        probs = scores**self.alpha
        probs = probs / probs.sum()

        chosen_indices = np.random.choice(n, size=k, replace=False, p=probs).tolist()
        sampled_items = [self.items[idx] for idx in chosen_indices]
        return sampled_items, chosen_indices


class SleepConsolidationController:
    """Executes offline NREM memory replay, fast->slow weight consolidation, and SHY downscaling."""

    def __init__(
        self,
        consolidation_lr: float = 0.1,
        downscale_target_norm: Optional[float] = None,
        latch_threshold: float = 0.4,
    ):
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
        device: str = "cpu",
    ) -> Tensor:
        """Autoregressively generate synthetic dream sequences from slow weights for privacy-safe replay."""
        model.eval()
        vocab_size = model.config.vocab_size
        # Start from random prompt seeds
        dreams = torch.randint(0, vocab_size, (num_dreams, 1), device=device)

        for _ in range(seq_len - 1):
            logits, _ = model(dreams)
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
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Execute the offline sleep consolidation loop across stored or dreamed memories."""
        if not use_dream_replay and (replay_buffer is None or len(replay_buffer) == 0):
            return {"status": "skipped_empty_buffer", "steps_run": 0}

        t0 = time.perf_counter()
        total_transferred_norm = 0.0
        n_layers_consolidated = 0

        model.eval()

        for step in range(sleep_steps):
            if use_dream_replay:
                batch_tokens = self.generate_dreams(
                    model,
                    num_dreams=batch_size,
                    seq_len=model.config.sequence_len,
                    device=device,
                )
            else:
                if replay_buffer is None:
                    break
                items, _ = replay_buffer.sample(batch_size)
                if not items:
                    break

                # Stack replayed 1D tokens into a 2D batch tensor (B, T)
                batch_tokens = torch.stack([
                    item.tokens if item.tokens.ndim == 1 else item.tokens.squeeze(0)
                    for item in items
                ]).to(device)

            with torch.no_grad():
                # Forward replay pass to activate local latches and eligibility traces
                model(batch_tokens)

                # Consolidate W_fast -> W_slow in all SynapticLinear layers
                for mod in model.modules():
                    if isinstance(mod, SynapticLinear):
                        if mod.w_fast is not None and mod.w_fast.norm() > 1e-6:
                            # Consolidation transfer: delta_w = lr * w_fast
                            delta_w = self.consolidation_lr * mod.w_fast.detach()
                            mod.w_slow.data.add_(delta_w)
                            total_transferred_norm += float(delta_w.norm().item())
                            n_layers_consolidated += 1

                            # Reset fast weights following consolidation
                            mod.w_fast.data.zero_()

                        # Optional SHY homeostatic downscaling
                        if self.downscale_target_norm is not None:
                            curr_norm = float(mod.w_slow.data.norm().item())
                            if curr_norm > self.downscale_target_norm:
                                scale = self.downscale_target_norm / max(1e-6, curr_norm)
                                mod.w_slow.data.mul_(scale)

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "consolidated",
            "steps_run": sleep_steps,
            "layers_consolidated": n_layers_consolidated,
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
        self.buffer = buffer or PrioritizedReplayBuffer()
        self.controller = controller or SleepConsolidationController()
        self.cfg = cfg or WakeSleepConfig()
        self.total_sleep_phases = 0
        self.last_consolidation_report: Optional[Dict[str, Any]] = None

    def step_training(
        self,
        step_idx: int,
        tokens: Tensor,
        step_loss: float,
    ) -> Optional[Dict[str, Any]]:
        """Hook called at every training step; collects high-surprise sequences and triggers sleep."""
        if not self.cfg.enabled:
            return None

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
