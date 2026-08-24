"""Energy-Based Constrained & Controllable Decoding (bead re4e.8).

Shapes the generation sampling distribution via Boltzmann energy functions:
p(x) ~ exp(-(E_base + sum lambda_k C_k) / T) for lexical, format, and style constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EnergyConstraint:
    name: str
    weight: float
    penalty_fn: Callable[[Tensor, Tensor], Tensor]  # (history_tokens, next_token_logits) -> token_penalties


@dataclass
class EnergyDecompositionRecord:
    step: int
    selected_token: int
    base_energy: float
    constraint_energies: Dict[str, float]
    total_energy: float


class EnergyConstrainedDecoder:
    """Decodes tokens by shaping candidate logits with additive constraint energy terms."""

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        constraints: Optional[List[EnergyConstraint]] = None,
    ) -> None:
        self.model = model
        self.temperature = max(1e-4, float(temperature))
        self.constraints: List[EnergyConstraint] = constraints or []

    def add_constraint(self, constraint: EnergyConstraint) -> None:
        self.constraints.append(constraint)

    def add_forbidden_tokens_constraint(
        self,
        forbidden_token_ids: Set[int],
        penalty_weight: float = 100.0,
    ) -> None:
        """Add constraint penalizing generation of forbidden tokens."""
        def penalty(history: Tensor, logits: Tensor) -> Tensor:
            penalties = torch.zeros_like(logits)
            for tok_id in forbidden_token_ids:
                if 0 <= tok_id < logits.size(-1):
                    penalties[..., tok_id] = 1.0
            return penalties

        self.add_constraint(
            EnergyConstraint(
                name="forbidden_tokens",
                weight=penalty_weight,
                penalty_fn=penalty,
            )
        )

    def add_repetition_penalty_constraint(
        self,
        penalty_weight: float = 2.0,
    ) -> None:
        """Add constraint penalizing repetition of recently generated tokens."""
        def penalty(history: Tensor, logits: Tensor) -> Tensor:
            penalties = torch.zeros_like(logits)
            unique_tokens, counts = torch.unique(history, return_counts=True)
            for tok, cnt in zip(unique_tokens, counts):
                tok_idx = int(tok.item())
                if 0 <= tok_idx < logits.size(-1):
                    penalties[..., tok_idx] = float(cnt.item())
            return penalties

        self.add_constraint(
            EnergyConstraint(
                name="repetition_penalty",
                weight=penalty_weight,
                penalty_fn=penalty,
            )
        )

    def decode_step(
        self,
        history_tokens: Tensor,
        step: int = 0,
    ) -> Tuple[int, EnergyDecompositionRecord]:
        """Compute constrained energy step and sample next token."""
        with torch.no_grad():
            out = self.model(history_tokens)
            logits = out.logits if hasattr(out, "logits") else out
            next_logits = logits[:, -1, :]  # (1, vocab_size)

        # Base energy: E_base = -logits
        base_energy = -next_logits
        total_energy = base_energy.clone()

        for c in self.constraints:
            c_energy = c.penalty_fn(history_tokens, next_logits)
            total_energy = total_energy + c.weight * c_energy

        # Boltzmann sampling: p ~ exp(-E / T) = exp((logits - sum lambda C) / T)
        shaped_logits = -total_energy
        probs = F.softmax(shaped_logits / self.temperature, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())

        record = EnergyDecompositionRecord(
            step=step,
            selected_token=next_token,
            base_energy=float(base_energy[0, next_token].item()),
            constraint_energies={
                c.name: float((c.weight * c.penalty_fn(history_tokens, next_logits)[0, next_token]).item())
                for c in self.constraints
            },
            total_energy=float(total_energy[0, next_token].item()),
        )

        return next_token, record

    def generate(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int = 10,
    ) -> Tuple[Tensor, List[EnergyDecompositionRecord]]:
        """Generate constrained token sequence."""
        current = prompt_tokens.clone()
        records = []
        for s in range(max_new_tokens):
            next_token, rec = self.decode_step(current, step=s)
            records.append(rec)
            current = torch.cat([current, torch.tensor([[next_token]], device=current.device)], dim=-1)
        return current, records
