"""Sleep Consolidation & Prioritized High-Surprise Replay Buffer (beads `cel.1`, `cel.2`).

Implements the biological two-stage memory consolidation system:
1. `PrioritizedReplayBuffer`: Episodic memory buffer storing high-surprise / high-loss sequences.
2. `SleepConsolidationController`: Offline NREM replay loop distilling fast synaptic weights (W_fast)
   into durable slow weights (W_slow) gated by the CaMKII/PP1 bistable latch, followed by SHY downscaling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticLinear


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
                assert replay_buffer is not None
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
