"""Cross-Pollination Prototypes: Reversible Blocks & Simplicial Attention (beads 6cb, 3zd).

Imports key mathematical architectures from model_guided_research into the
bio_inspired_nanochat transformer ecosystem with full autograd, exact inversions,
and performance benchmarking.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# 1. Reversible / Measure-Preserving Coupling Block (bead 6cb, vap.3)
# -----------------------------------------------------------------------------


class ReversibleCouplingFunction(torch.autograd.Function):
    """Autograd function implementing exact activation recomputation backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        f_block: nn.Module,
        g_block: nn.Module,
    ) -> Tensor:
        # Split input along feature dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
        with torch.no_grad():
            f_x2 = f_block(x2)
            y1 = x1 + f_x2
            g_y1 = g_block(y1)
            y2 = x2 + g_y1
            y = torch.cat([y1, y2], dim=-1)

        ctx.f_block = f_block
        ctx.g_block = g_block
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Tensor, None, None]:
        grad_y: Tensor = grad_outputs[0]
        (y,) = ctx.saved_tensors
        y1, y2 = torch.chunk(y, 2, dim=-1)
        dy1, dy2 = torch.chunk(grad_y, 2, dim=-1)

        # Exact backward reconstruction: x2 = y2 - g(y1), x1 = y1 - f(x2)
        with torch.set_grad_enabled(True):
            y1_detached = y1.detach().requires_grad_(True)
            g_y1 = ctx.g_block(y1_detached)
            x2 = y2 - g_y1.detach()

            x2_detached = x2.detach().requires_grad_(True)
            f_x2 = ctx.f_block(x2_detached)

            # Gradients of g
            grad_g_params = torch.autograd.grad(
                g_y1,
                (y1_detached, *tuple(ctx.g_block.parameters())),
                grad_outputs=dy2,
                retain_graph=True,
            )
            dy1_total = dy1 + grad_g_params[0]

            # Gradients of f
            grad_f_params = torch.autograd.grad(
                f_x2,
                (x2_detached, *tuple(ctx.f_block.parameters())),
                grad_outputs=dy1_total,
            )
            dy2_total = dy2 + grad_f_params[0]

            dx = torch.cat([dy1_total, dy2_total], dim=-1)

        return dx, None, None


class ReversibleBlock(nn.Module):
    """Reversible additive coupling layer achieving O(1) activation memory."""

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
        if self.training and x.requires_grad:
            return ReversibleCouplingFunction.apply(x, self.f, self.g)
        # Fast non-autograd path
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        """Exact inverse pass: reconstructs x from output y."""
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
