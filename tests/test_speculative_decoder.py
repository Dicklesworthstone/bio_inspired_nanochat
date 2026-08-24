"""Unit and integration tests for Speculative Decoder (bead re4e.7)."""

from __future__ import annotations

import torch
import torch.nn as nn

from bio_inspired_nanochat.speculative_decoder import (
    SpeculativeDecodeConfig,
    SpeculativeDecoder,
)


class MockEarlyExitModel(nn.Module):
    """Mock model with early-exit and full forward paths."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.early_head = nn.Linear(d_model, vocab_size)
        self.full_head = nn.Linear(d_model, vocab_size)

    def forward_early_exit(self, tokens: torch.Tensor, max_layers: int = 4) -> torch.Tensor:
        emb = self.embedding(tokens)
        return self.early_head(emb)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        return self.full_head(emb)


def test_speculative_decoder_generation_length():
    """SpeculativeDecoder generates exact requested token count."""
    model = MockEarlyExitModel(vocab_size=16)
    cfg = SpeculativeDecodeConfig(gamma=3, early_exit_layer=2)
    decoder = SpeculativeDecoder(model, config=cfg)

    prompt = torch.tensor([[1, 2, 3]])
    max_new = 10
    out, report = decoder.generate(prompt, max_new_tokens=max_new)

    assert out.shape == (1, prompt.shape[1] + max_new)
    assert report.tokens_generated == max_new
    assert report.verifier_calls > 0
    assert 0.0 <= report.acceptance_rate <= 1.0


def test_speculative_decoder_high_agreement_speedup():
    """When draft and full models agree, speculative decoding achieves high speedup."""
    model = MockEarlyExitModel(vocab_size=8)
    # Force full head and early head to share weights for 100% agreement
    model.full_head.weight.data.copy_(model.early_head.weight.data)
    model.full_head.bias.data.copy_(model.early_head.bias.data)

    cfg = SpeculativeDecodeConfig(gamma=4, early_exit_layer=2, temperature=1.0)
    decoder = SpeculativeDecoder(model, config=cfg)

    prompt = torch.tensor([[0, 1]])
    out, report = decoder.generate(prompt, max_new_tokens=15)

    assert report.tokens_generated == 15
    assert report.acceptance_rate >= 0.8, f"High agreement should give high acceptance: {report.acceptance_rate:.2%}"
    assert report.speedup_ratio > 1.5, f"Expected speedup > 1.5x, got {report.speedup_ratio:.2f}x"
