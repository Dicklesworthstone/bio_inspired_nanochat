"""Unit and constraint satisfaction tests for EnergyConstrainedDecoder (bead re4e.8)."""

from __future__ import annotations

import torch
import torch.nn as nn

from bio_inspired_nanochat.constrained_energy_decoder import EnergyConstrainedDecoder


class SimpleVocabModel(nn.Module):
    """Simple model outputting fixed logits across a small vocabulary."""

    def __init__(self, vocab_size: int = 10) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Uniform initial logits
        return torch.ones(x.shape[0], x.shape[1], self.vocab_size)


def test_forbidden_tokens_constraint_never_samples_forbidden():
    """EnergyConstrainedDecoder strictly avoids forbidden tokens under heavy penalty."""
    model = SimpleVocabModel(vocab_size=10)
    decoder = EnergyConstrainedDecoder(model, temperature=1.0)

    # Forbid tokens {3, 5, 7}
    forbidden = {3, 5, 7}
    decoder.add_forbidden_tokens_constraint(forbidden, penalty_weight=100.0)

    prompt = torch.tensor([[0, 1]])
    generated, records = decoder.generate(prompt, max_new_tokens=30)

    gen_tokens = generated[0, 2:].tolist()
    for tok in gen_tokens:
        assert tok not in forbidden, f"Forbidden token {tok} was generated!"

    assert len(records) == 30
    for rec in records:
        assert rec.selected_token not in forbidden


def test_repetition_penalty_encourages_diversity():
    """Repetition penalty lowers duplicate token frequency."""
    model = SimpleVocabModel(vocab_size=6)
    decoder = EnergyConstrainedDecoder(model, temperature=0.5)
    decoder.add_repetition_penalty_constraint(penalty_weight=5.0)

    prompt = torch.tensor([[0]])
    generated, _ = decoder.generate(prompt, max_new_tokens=20)

    gen_tokens = generated[0, 1:].tolist()
    unique_count = len(set(gen_tokens))
    # Across 20 steps with 6 vocab tokens, high penalty should use all 6 unique tokens
    assert unique_count >= 5, f"Expected high diversity, got only {unique_count} unique tokens"
