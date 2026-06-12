"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_inspired_nanochat.common import ca_init_weight_, get_dist_info
from bio_inspired_nanochat.muon import Muon, DistMuon
from bio_inspired_nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Attention mechanism switch (experimental)
    attention_type: str = "standard"  # "standard" | "ultrametric"
    # Ultrametric (p-adic / LCP-kernel) attention hyperparams
    ultrametric_k: int = 8
    ultrametric_p: int = 2
    ultrametric_alpha: float = 2.0
    ultrametric_lcp_beta: float = 32.0
    ultrametric_query_chunk_size: int = 64
    # Weight initialization
    init_type: str = "baseline"  # "baseline" | "ca_rule30" | "ca_rule116"
    init_seed: int = 42
    # Tie the token embedding and the output (lm_head) weights into ONE shared matrix
    # (hwxb.2.9). For a small model the untied embeddings (2·vocab·d) dominate the param
    # budget; tying halves that so the budget goes into depth/width. Default off preserves
    # the legacy untied behaviour + the separate embedding/unembedding LRs.
    tie_embeddings: bool = False


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


def _repeat_kv_heads(x: torch.Tensor, *, n_head: int) -> torch.Tensor:
    """Repeat KV heads to match query heads (manual GQA handling)."""
    if x.size(1) == n_head:
        return x
    n_kv = x.size(1)
    if n_kv <= 0 or n_head % n_kv != 0:
        raise ValueError(f"Invalid GQA: n_head={n_head}, n_kv_head={n_kv}")
    return x.repeat_interleave(n_head // n_kv, dim=1)


def _autoregressive_keep_mask(
    *,
    Tq: int,
    Tk: int,
    kv_cache,
    device: torch.device,
) -> torch.Tensor:
    """Return bool keep-mask shaped (Tq, Tk) for autoregressive attention."""
    if kv_cache is None or Tq == Tk:
        return torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))
    if Tq == 1:
        return torch.ones((Tq, Tk), dtype=torch.bool, device=device)
    prefix_len = Tk - Tq
    keep = torch.zeros((Tq, Tk), dtype=torch.bool, device=device)
    if prefix_len > 0:
        keep[:, :prefix_len] = True
    keep[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=device))
    return keep


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class UltrametricCausalSelfAttention(nn.Module):
    """Ultrametric (p-adic / LCP-kernel) causal attention (dense, chunked).

    This is a prototype port of the MGR idea. It is still O(T^2) compute, but
    query-chunked to avoid allocating an (Tq×Tk×K) tensor.
    """

    _ultra_p_minus_1: torch.Tensor
    _ultra_lcp_beta: torch.Tensor
    _ultra_log_alpha: torch.Tensor
    _ultra_query_chunk_size: int

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        K = int(config.ultrametric_k)
        p = int(config.ultrametric_p)
        alpha = float(config.ultrametric_alpha)
        lcp_beta = float(config.ultrametric_lcp_beta)
        query_chunk_size = int(config.ultrametric_query_chunk_size)

        if K <= 0:
            raise ValueError("ultrametric_k must be > 0")
        if p < 2:
            raise ValueError("ultrametric_p must be >= 2")
        if alpha <= 1.0:
            raise ValueError("ultrametric_alpha must be > 1")
        if query_chunk_size <= 0:
            raise ValueError("ultrametric_query_chunk_size must be > 0")

        self.register_buffer("_ultra_p_minus_1", torch.tensor(float(p - 1), dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_ultra_lcp_beta",
            torch.tensor(lcp_beta, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_ultra_log_alpha",
            torch.tensor(math.log(alpha), dtype=torch.float32),
            persistent=False,
        )
        object.__setattr__(self, "_ultra_query_chunk_size", query_chunk_size)

        # Learned per-head projections into K "digits" used for an LCP-kernel.
        self.to_digits_q = nn.Linear(self.head_dim, K, bias=False)
        self.to_digits_k = nn.Linear(self.head_dim, K, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, _C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,D)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        # Expand KV heads for manual GQA.
        if self.n_kv_head != self.n_head:
            k = _repeat_kv_heads(k, n_head=self.n_head)
            v = _repeat_kv_heads(v, n_head=self.n_head)

        Tq = q.size(2)
        Tk = k.size(2)
        keep = _autoregressive_keep_mask(Tq=Tq, Tk=Tk, kv_cache=kv_cache, device=q.device)

        # Soft digits in [0, p-1] (continuous relaxation).
        q_digits = torch.sigmoid(self.to_digits_q(q.to(dtype=torch.float32))) * self._ultra_p_minus_1
        k_digits = torch.sigmoid(self.to_digits_k(k.to(dtype=torch.float32))) * self._ultra_p_minus_1

        out = torch.empty((B, self.n_head, Tq, self.head_dim), dtype=v.dtype, device=v.device)

        query_chunk_size = self._ultra_query_chunk_size
        K = q_digits.size(-1)

        for qs in range(0, Tq, query_chunk_size):
            qe = min(Tq, qs + query_chunk_size)
            qd = q_digits[:, :, qs:qe, :]  # (B,H,qchunk,K)

            prefix = torch.ones((B, self.n_head, qe - qs, Tk), dtype=torch.float32, device=q.device)
            lcp = torch.zeros_like(prefix)
            for d in range(K):
                diff = qd[..., d].unsqueeze(-1) - k_digits[..., d].unsqueeze(-2)
                match = torch.exp(-self._ultra_lcp_beta * diff.square())
                prefix.mul_(match)
                lcp.add_(prefix)

            weights = torch.exp(lcp * self._ultra_log_alpha)
            keep_chunk = keep[qs:qe].unsqueeze(0).unsqueeze(0)  # (1,1,qchunk,Tk)
            weights = weights.masked_fill(~keep_chunk, 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn = (weights / denom).to(dtype=v.dtype)

            out[:, :, qs:qe, :] = attn @ v

        y = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        y = self.c_proj(y)
        return y


def _build_attention(config: GPTConfig, layer_idx: int) -> nn.Module:
    attn_type = str(getattr(config, "attention_type", "standard"))
    if attn_type == "standard":
        return CausalSelfAttention(config, layer_idx)
    if attn_type == "ultrametric":
        return UltrametricCausalSelfAttention(config, layer_idx)
    raise ValueError(f"Unknown attention_type={attn_type!r}")


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = _build_attention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @property
    def wte(self) -> nn.Embedding:
        return cast(nn.Embedding, self.transformer["wte"])

    @property
    def blocks(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.transformer["h"])

    def init_weights(self):
        init_type = str(getattr(self.config, "init_type", "baseline"))
        if init_type.startswith("ca_") or init_type.startswith("ca"):
            self._init_weights_ca()
        else:
            self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            torch.nn.init.zeros_(cast(torch.Tensor, block.mlp.c_proj.weight))
            torch.nn.init.zeros_(cast(torch.Tensor, block.attn.c_proj.weight))
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
        # Tie LAST (after the zeroing + bf16 cast above) so lm_head shares the FINAL wte
        # tensor — otherwise the `zeros_(lm_head.weight)` above would zero the embedding,
        # or lm_head would keep a stale fp32 copy after the cast. See tie_weights().
        self.tie_weights()

    def tie_weights(self) -> None:
        """Share lm_head.weight with the token embedding when tie_embeddings is set.

        Idempotent. Must be called after init AND after any load_state_dict(assign=True)
        (which replaces the param objects and thus breaks an earlier tie) — callers in
        scripts/base_train.py and checkpoint_manager.build_model re-tie on resume.
        """
        if getattr(self.config, "tie_embeddings", False):
            self.lm_head.weight = self.wte.weight

    def _init_weights_ca(self) -> None:
        init_type = str(getattr(self.config, "init_type", "ca_rule30"))
        init_seed = int(getattr(self.config, "init_seed", 42))
        rule = 116 if "116" in init_type else 30

        # Avoid pathological init-time overhead for very large "out" dims (e.g. lm_head).
        max_ca_fan_out = 8192

        for module_name, module in self.named_modules():
            if module is self:
                continue
            if isinstance(module, nn.Linear):
                fan_out = int(module.weight.size(0))
                if fan_out > max_ca_fan_out:
                    self._init_weights(module)
                else:
                    ca_init_weight_(
                        module.weight,
                        rule=rule,
                        seed=init_seed,
                        salt=f"{module_name}.weight",
                        layout="out_in",
                    )
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.wte.weight.numel()
        n_layers = self.config.n_layer
        n_heads = self.config.n_head
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        num_flops_per_token = (
            6 * (nparams - nparams_embedding) + 12 * n_layers * n_heads * head_dim * seq_len
        )
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # When tie_embeddings is on, lm_head.weight IS wte.weight (one shared tensor). It must
        # be optimized EXACTLY ONCE, else AdamW double-updates it. We keep it in the lm_head
        # (unembedding) group — the smaller, more conservative LR — because the shared weight
        # also serves the dense output-projection role where the large embedding LR (0.2) would
        # risk logit instability. So drop it from the embedding group (which becomes empty).
        if getattr(self.config, "tie_embeddings", False):
            embedding_params = [p for p in embedding_params if p is not self.lm_head.weight]
        # parameters() already dedupes shared tensors, so the assert holds in both cases.
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        ]
        if embedding_params:  # empty when tied (the shared weight sits in the lm_head group)
            adam_groups.append(dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale))
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        use_fused = (not ddp) and any(p.is_cuda for p in (embedding_params + lm_head_params))
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=use_fused)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
