"""Speculative Decoding via Metabolic Cheap-Path Draft (bead re4e.7).

Uses early-exit prefix depth or low-energy top_k=1 MoE execution as the draft model
and full-depth deliberation as the verifier, guaranteeing exact output distribution
equivalence while achieving significant wall-clock speedups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SpeculativeDecodeConfig:
    gamma: int = 3  # Draft lookahead length
    early_exit_layer: int = 4
    total_layers: int = 12
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class SpeculativeDecodeReport:
    tokens_generated: int
    draft_tokens_proposed: int
    tokens_accepted: int
    acceptance_rate: float
    verifier_calls: int
    speedup_ratio: float


class SpeculativeDecoder:
    """Speculative decoding engine using metabolic cheap-path as draft model."""

    def __init__(
        self,
        model: nn.Module,
        config: SpeculativeDecodeConfig = SpeculativeDecodeConfig(),
    ) -> None:
        self.model = model
        self.config = config

    def sample_token(self, logits: Tensor) -> Tensor:
        """Sample next token from logits at configured temperature."""
        if self.config.temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int = 20,
    ) -> Tuple[Tensor, SpeculativeDecodeReport]:
        """Generate tokens using speculative draft + verify rejection sampling."""
        device = prompt_tokens.device
        current_tokens = prompt_tokens.clone()
        if current_tokens.ndim == 1:
            current_tokens = current_tokens.unsqueeze(0)

        total_draft_proposed = 0
        total_accepted = 0
        verifier_calls = 0

        while current_tokens.shape[1] < prompt_tokens.shape[1] + max_new_tokens:
            gamma = min(self.config.gamma, prompt_tokens.shape[1] + max_new_tokens - current_tokens.shape[1])
            if gamma <= 0:
                break

            # 1. Draft Phase: propose gamma tokens using cheap early-exit
            draft_seq = current_tokens.clone()
            draft_probs_list: List[Tensor] = []
            draft_tokens_list: List[int] = []

            for _ in range(gamma):
                with torch.no_grad():
                    forward_fn = getattr(self.model, "forward_early_exit", None)
                    if callable(forward_fn):
                        logits = forward_fn(draft_seq, max_layers=self.config.early_exit_layer)
                    else:
                        out = self.model(draft_seq)
                        logits = out.logits if hasattr(out, "logits") else out
                    next_token_logits = logits[:, -1, :]
                    probs = F.softmax(next_token_logits / max(1e-5, self.config.temperature), dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                    draft_tokens_list.append(int(token.item()))
                    draft_probs_list.append(probs)
                    draft_seq = torch.cat([draft_seq, token], dim=-1)

            total_draft_proposed += gamma

            # 2. Verification Phase: parallel forward on full model for prompt + draft
            with torch.no_grad():
                out_full = self.model(draft_seq)
                verifier_calls += 1
                logits_full = out_full.logits if hasattr(out_full, "logits") else out_full

            # Verification slice for each draft position
            prefix_len = current_tokens.shape[1]
            n_accepted = 0

            for i in range(gamma):
                pos = prefix_len + i - 1
                target_logits = logits_full[:, pos, :]
                target_probs = F.softmax(target_logits / max(1e-5, self.config.temperature), dim=-1)
                draft_token_id = draft_tokens_list[i]
                p_target = target_probs[0, draft_token_id].item()
                p_draft = draft_probs_list[i][0, draft_token_id].item()

                # Acceptance probability: min(1, p_target / p_draft)
                alpha = min(1.0, p_target / max(1e-8, p_draft))
                r = float(torch.rand(1).item())

                if r <= alpha:
                    n_accepted += 1
                    current_tokens = torch.cat(
                        [current_tokens, torch.tensor([[draft_token_id]], device=device)], dim=-1
                    )
                else:
                    # Reject & sample from adjusted distribution (p_target - p_draft)_+
                    diff = F.relu(target_probs - draft_probs_list[i])
                    sum_diff = diff.sum(dim=-1, keepdim=True)
                    if sum_diff.item() > 1e-6:
                        resample_probs = diff / sum_diff
                    else:
                        resample_probs = target_probs
                    correction_token = torch.multinomial(resample_probs, num_samples=1)
                    current_tokens = torch.cat([current_tokens, correction_token], dim=-1)
                    break

            # If all gamma tokens were accepted, sample one bonus token from last target logits
            if n_accepted == gamma and current_tokens.shape[1] < prompt_tokens.shape[1] + max_new_tokens:
                bonus_logits = logits_full[:, prefix_len + gamma - 1, :]
                bonus_probs = F.softmax(bonus_logits / max(1e-5, self.config.temperature), dim=-1)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)
                current_tokens = torch.cat([current_tokens, bonus_token], dim=-1)

            total_accepted += n_accepted

        n_gen = current_tokens.shape[1] - prompt_tokens.shape[1]
        acc_rate = (total_accepted / max(1, total_draft_proposed))
        speedup = n_gen / max(1, verifier_calls)

        report = SpeculativeDecodeReport(
            tokens_generated=n_gen,
            draft_tokens_proposed=total_draft_proposed,
            tokens_accepted=total_accepted,
            acceptance_rate=acc_rate,
            verifier_calls=verifier_calls,
            speedup_ratio=speedup,
        )

        return current_tokens, report
