"""Mathematical Geometric Research (MGR) cross-pollination modules for Bio-Inspired Nanochat.

Implements the top-3 geometric inductive biases and optimization transfers from MGR
(beads vap.3, vap.4, vap.5, eqyk.21):
  1. ``SimplicialCausalSelfAttention``: Multi-hop graph/simplicial diffusion on attention
     manifolds ($Y = \\alpha (A V) + \\beta A (A V)$), capturing higher-order relational dependencies.
  2. ``ReversibleBlock`` & ``ReversibleAdditiveCoupling``: Invertible additive coupling on channel-split
     representations ($y_1 = x_1 + F(x_2), y_2 = x_2 + G(y_1)$) guaranteeing volume preservation
     ($|\\det J| \\equiv 1$) and exact analytical inversion ($x_2 = y_2 - G(y_1), x_1 = y_1 - F(x_2)$).
  3. ``OrdinalLRScheduler``: Transfinite ordinal learning rate scheduler indexed by
     $\\rho = \\omega^2 A + \\omega B + C$ with well-founded patience, geometric annealing, and
     deterministic restarts with optimizer state clearing.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_inspired_nanochat.gpt import GPTConfig, apply_rotary_emb, norm


class SimplicialCausalSelfAttention(nn.Module):
    """Simplicial (Higher-Order) Causal Self-Attention.

    Augments standard pairwise attention with 2-hop simplicial diffusion along the
    attention complex 1-skeleton:
        y_1 = A @ v                (1-hop pairwise attention)
        y_2 = A @ (A @ v) = A @ y_1 (2-hop simplicial path diffusion)
        y   = mix_1 * y_1 + mix_2 * y_2
    """

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Learnable mixing parameters for 1-hop (edges) and 2-hop (simplices/triangles)
        self.mix_1 = nn.Parameter(torch.tensor(1.0))
        self.mix_2 = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: Any = None,
    ) -> torch.Tensor:
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)

        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # 1-hop and 2-hop diffusion
        y1 = attn_weights @ v
        y2 = attn_weights @ y1
        y = self.mix_1 * y1 + self.mix_2 * y2

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class ReversibleAdditiveCoupling(nn.Module):
    """Additive coupling reversible layer.

    Splits the input stream x in half along the embedding dimension into (x1, x2):
        Forward:
            y1 = x1 + F(x2)
            y2 = x2 + G(y1)
        Inverse:
            x2 = y2 - G(y1)
            x1 = y1 - F(x2)
    """

    def __init__(self, f_block: nn.Module, g_block: nn.Module) -> None:
        super().__init__()
        self.f_block = f_block
        self.g_block = g_block

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        f_out = self.f_block(x2, **kwargs) if kwargs else self.f_block(x2)
        y1 = x1 + f_out
        g_out = self.g_block(y1, **kwargs) if kwargs else self.g_block(y1)
        y2 = x2 + g_out
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Exact analytical reconstruction of x from y."""
        y1, y2 = torch.chunk(y, 2, dim=-1)
        g_out = self.g_block(y1, **kwargs) if kwargs else self.g_block(y1)
        x2 = y2 - g_out
        f_out = self.f_block(x2, **kwargs) if kwargs else self.f_block(x2)
        x1 = y1 - f_out
        return torch.cat([x1, x2], dim=-1)


class ReversibleBlock(nn.Module):
    """Reversible transformer block wrapping attention and MLP in additive coupling."""

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert config.n_embd % 2 == 0, "n_embd must be even for reversible split"
        half_dim = config.n_embd // 2

        # Sub-configs for half-dimension streams
        half_config = GPTConfig(
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=max(1, config.n_head // 2),
            n_kv_head=max(1, config.n_kv_head // 2),
            n_embd=half_dim,
            attention_type="standard",
        )

        from bio_inspired_nanochat.gpt import CausalSelfAttention, MLP

        self.f_attn = CausalSelfAttention(half_config, layer_idx)
        self.g_mlp = MLP(half_config)
        self.coupling = ReversibleAdditiveCoupling(
            f_block=nn.Sequential(self.f_attn),
            g_block=nn.Sequential(self.g_mlp),
        )

    def _ensure_cos_sin(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos_sin is not None:
            return cos_sin
        t = x.shape[1]
        head_dim = self.f_attn.head_dim
        d_half = head_dim // 2
        theta = 10000.0 ** (-torch.arange(0, d_half, dtype=torch.float32, device=x.device) / d_half)
        t_idx = torch.arange(t, dtype=torch.float32, device=x.device)
        freqs = torch.outer(t_idx, theta)
        cos = torch.cos(freqs).view(1, t, 1, d_half).to(dtype=torch.bfloat16)
        sin = torch.sin(freqs).view(1, t, 1, d_half).to(dtype=torch.bfloat16)
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: Any = None,
    ) -> torch.Tensor:
        cos_sin_use = self._ensure_cos_sin(x, cos_sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        f_out = self.f_attn(norm(x2), cos_sin_use, kv_cache)
        y1 = x1 + f_out
        g_out = self.g_mlp(norm(y1))
        y2 = x2 + g_out
        return torch.cat([y1, y2], dim=-1)

    def inverse(
        self,
        y: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: Any = None,
    ) -> torch.Tensor:
        cos_sin_use = self._ensure_cos_sin(y, cos_sin)
        y1, y2 = torch.chunk(y, 2, dim=-1)
        g_out = self.g_mlp(norm(y1))
        x2 = y2 - g_out
        f_out = self.f_attn(norm(x2), cos_sin_use, kv_cache)
        x1 = y1 - f_out
        return torch.cat([x1, x2], dim=-1)


class OrdinalLRScheduler:
    """Transfinite Ordinal Learning Rate Scheduler.

    Scheduler state indexed by rank rho = omega^2 * A + omega * B + C:
      - A: Restart budget (highest order transfinite limit)
      - B: Anneal levels (curriculum)
      - C: Patience (finite step counter)

    Transitions:
      - Step: Update EMA loss. If improved: maintain (A, B, C); else: C <- C - 1.
      - Limit (C <= 0):
        - Anneal (B > 0): B <- B - 1, lr <- max(min_lr, lr * gamma), C <- P_init.
        - Restart (B == 0, A > 0): A <- A - 1, B <- B_init, lr <- eta_init, C <- P_init,
          and clear optimizer state moments.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        a_init: int = 2,
        b_init: int = 3,
        p_init: int = 20,
        eta_init: float = 1e-3,
        gamma: float = 0.5,
        min_lr: float = 1e-6,
        alpha: float = 0.1,
    ) -> None:
        if a_init < 0 or b_init < 0 or p_init < 1:
            raise ValueError("a_init and b_init must be >= 0, and p_init must be >= 1")
        if eta_init <= 0 or gamma <= 0 or min_lr <= 0:
            raise ValueError("eta_init, gamma, and min_lr must be positive")

        self.optimizer = optimizer
        self.a_init = a_init
        self.b_init = b_init
        self.p_init = p_init
        self.a = a_init
        self.b = b_init
        self.c = p_init
        self.eta_init = eta_init
        self.gamma = gamma
        self.min_lr = min_lr
        self.alpha = alpha

        self.best_loss = float("inf")
        self.ema_loss: float | None = None
        self.step_count = 0
        self.restart_events = 0
        self.anneal_events = 0

        # Set initial LR
        for group in self.optimizer.param_groups:
            group["lr"] = self.eta_init

    def step(self, loss: float | torch.Tensor) -> dict[str, Any]:
        """Execute one ordinal schedule update step with current batch loss."""
        if isinstance(loss, torch.Tensor):
            loss_val = float(loss.detach().item())
        else:
            loss_val = float(loss)

        self.step_count += 1
        transition_type = "step"

        if self.ema_loss is None:
            self.ema_loss = loss_val
        else:
            self.ema_loss = (1.0 - self.alpha) * self.ema_loss + self.alpha * loss_val

        # Check for loss improvement
        if self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            # Reset patience on genuine progress
            self.c = self.p_init
            transition_type = "progress"
        else:
            self.c -= 1

        # Check limit ordinal conditions
        if self.c <= 0:
            if self.b > 0:
                # Anneal (omega-term drop)
                self.b -= 1
                self.c = self.p_init
                self.anneal_events += 1
                transition_type = "anneal"
                for group in self.optimizer.param_groups:
                    group["lr"] = max(self.min_lr, group["lr"] * self.gamma)
                self.best_loss = float("inf")

            elif self.a > 0:
                # Restart (omega^2-term drop)
                self.a -= 1
                self.b = self.b_init
                self.c = self.p_init
                self.restart_events += 1
                transition_type = "restart"
                for group in self.optimizer.param_groups:
                    group["lr"] = self.eta_init
                self.optimizer.state.clear()
                self.best_loss = float("inf")
            else:
                transition_type = "plateau"

        current_lr = self.get_last_lr()[0]
        return {
            "step": self.step_count,
            "transition": transition_type,
            "ordinal_rank": f"ω²·{self.a} + ω·{self.b} + {self.c}",
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "lr": current_lr,
            "ema_loss": self.ema_loss,
            "best_loss": self.best_loss,
        }

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "a_init": self.a_init,
            "b_init": self.b_init,
            "p_init": self.p_init,
            "eta_init": self.eta_init,
            "gamma": self.gamma,
            "min_lr": self.min_lr,
            "alpha": self.alpha,
            "best_loss": self.best_loss,
            "ema_loss": self.ema_loss,
            "step_count": self.step_count,
            "restart_events": self.restart_events,
            "anneal_events": self.anneal_events,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.a = state["a"]
        self.b = state["b"]
        self.c = state["c"]
        self.a_init = state.get("a_init", self.a_init)
        self.b_init = state.get("b_init", self.b_init)
        self.p_init = state.get("p_init", self.p_init)
        self.eta_init = state.get("eta_init", self.eta_init)
        self.gamma = state.get("gamma", self.gamma)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.alpha = state.get("alpha", self.alpha)
        self.best_loss = state["best_loss"]
        self.ema_loss = state["ema_loss"]
        self.step_count = state.get("step_count", 0)
        self.restart_events = state.get("restart_events", 0)
        self.anneal_events = state.get("anneal_events", 0)
