"""Cross-Pollination Prototypes: Reversible Blocks & Simplicial Attention (beads 6cb, 3zd).

Imports key mathematical architectures from model_guided_research into the
bio_inspired_nanochat transformer ecosystem with full autograd, exact inversions,
and performance benchmarking.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


# -----------------------------------------------------------------------------
# 1. Reversible / Measure-Preserving Coupling Block (bead 6cb, vap.3)
# -----------------------------------------------------------------------------


class ReversibleBlock(nn.Module):
    """Invertible additive coupling with one saved input per training block."""

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        half_dim = d_model // 2
        h_dim = hidden_dim or half_dim * 2

        self.f = nn.Sequential(
            nn.LayerNorm(half_dim),
            nn.Linear(half_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, half_dim),
        )
        self.g = nn.Sequential(
            nn.LayerNorm(half_dim),
            nn.Linear(half_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, half_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training and torch.is_grad_enabled():
            # Non-reentrant replay also differentiates weights when the input is frozen.
            return checkpoint(self._couple, x, use_reentrant=False)
        return self._couple(x)

    def _couple(self, x: Tensor) -> Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        """Recover the input from the output up to floating-point roundoff."""
        y1, y2 = torch.chunk(y, 2, dim=-1)
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=-1)


# -----------------------------------------------------------------------------
# 2. Simplicial Higher-Order Attention (bead 3zd, vap.4)
# -----------------------------------------------------------------------------


class SimplicialAttention(nn.Module):
    """Simplicial 2-hop attention mixing: y = (1 - λ) A v + λ A (A v)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        initial_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable simplex mixing logit (sigmoid maps to [0, 1])
        init_logit = math.log(initial_lambda / (1.0 - initial_lambda + 1e-6))
        self.simplex_logit = nn.Parameter(torch.full((n_heads, 1, 1), init_logit))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Pairwise dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)  # A: (B, H, T, T)

        # 1-hop standard output
        y1 = attn_weights @ v  # (B, H, T, D)

        # 2-hop simplicial diffusion: A @ (A @ v)
        y2 = attn_weights @ y1  # (B, H, T, D)

        # Simplicial convex mixing
        lam = torch.sigmoid(self.simplex_logit)  # (H, 1, 1)
        y = (1.0 - lam) * y1 + lam * y2

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
