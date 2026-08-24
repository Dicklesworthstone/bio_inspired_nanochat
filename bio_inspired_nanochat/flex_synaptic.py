"""
Fused Synaptic Attention using PyTorch FlexAttention.
Requires PyTorch >= 2.5.0

This module replaces the memory-heavy (B, H, T, T) materialization of biological biases
with an on-the-fly "score mod" that fuses directly into the FlashAttention kernel.

Key Benefits:
1. O(N) memory usage (vs O(N^2)).
2. FlashAttention speeds.
3. Automatic backward pass differentiation.
"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

class SynapticFlexAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We need to ensure state tensors are registered buffers or similar?
        # No, they are passed per forward pass.

    def precompute_bio_factors(self, state, cfg):
        """Canonical faithful O(N) bio factors for the O(N^2) flex attention map (s3w9).

        Uses the SAME formulation as the live standard path's release_canonical: Hill-function
        calcium sensing (Syt1 fast + Syt7 slow) + Doc2 facilitation, a complexin/SNARE fuse gate,
        and an energy-gated AMPA amplitude (replacing the legacy sigmoid mix + raw AMP state). The
        per-(q,k) bilinear term sigmoid(score) is applied in score_mod, so
        release = key_factor * sigmoid(score) with key_factor = fuse_base * RRP.

        NOTE (flex O(N) approximation): the Hill here uses the per-KEY STATE calcium (already
        advanced by release_canonical), NOT a per-edge drive-augmented calcium. The standard path
        adds the per-edge influx alpha_ca*softplus(drive) before the Hill, which is not
        O(N)-precomputable. This is the one intended difference from the standard path.

        Returns:
            key_factor: (B, H, T) - fuse_base * RRP, the per-key release readiness.
            qamp: (B, H, T) - energy-gated quantal amplitude.
        """
        c = state["C"]      # calcium
        rrp = state["RRP"]  # readily-releasable vesicle pool
        cl = state["CL"]    # complexin clamp
        pr = state["PR"]    # priming / SNARE
        e = state["E"]      # metabolic energy

        # Hill Syt1/Syt7 calcium sensing + preserved Doc2 (matches _faithful_release_prob).
        fast = c / (c + cfg.syt_fast_kd)
        slow = c / (c + cfg.syt_slow_kd)
        syt = 0.7 * fast + 0.3 * slow + cfg.doc2_gain * torch.sigmoid(4.0 * (c - 0.12))
        # Complexin/SNARE fuse gate (additive-in-sigmoid, faithful form).
        fuse_base = torch.sigmoid(3.0 * syt + 2.0 * pr - 2.0 * (cl + cfg.complexin_bias))

        key_factor = fuse_base * rrp
        qamp = torch.sigmoid(cfg.q_beta * (e - 0.5)) * cfg.qmax  # energy-gated AMPA amplitude
        return key_factor, qamp

    def forward(self, q, k, v, presyn_state, block_mask=None, *, query_offset: int = 0):
        """
        q, k, v: (B, H, T, D)
        presyn_state: Dict of (B, H, T)
        """
        B, H, _Tq, _D = q.shape
        
        # 1. Pre-compute biological factors O(N)
        key_factor, qamp = self.precompute_bio_factors(presyn_state, self.config)
        
        # 2. Define the score modifier
        # We need to wrap this in a closure to capture tensors
        # Capture constants to avoid lookup inside
        barrier_strength = self.config.barrier_strength
        epsilon = self.config.epsilon
        lambda_loge = self.config.lambda_loge  # s3w9: match the standard path's bias scaling

        def score_mod(score, b, h, q_idx, kv_idx):
            # Note: flex_attention applies `scale` before calling score_mod. With the default
            # scale (1/sqrt(E)), `score` is already scaled.
            scaled_score = score

            # Bio modulation (canonical): release = fuse_base*RRP * sigmoid(score), biased into
            # the logits as lambda_loge*log(release*qamp + eps) -- the same form as the standard
            # path (release_canonical -> lambda_loge*log(eps+e)). Flex does not EMA-normalize e,
            # which is the second intended difference from the standard path.
            kf_val = key_factor[b, h, kv_idx]
            qa_val = qamp[b, h, kv_idx]
            release = kf_val * torch.sigmoid(scaled_score)
            bio_bias = lambda_loge * torch.log(release * qa_val + epsilon)

            # Septin-like distance barrier.
            absolute_q_idx = q_idx + query_offset
            causal_extent = absolute_q_idx + 1
            dist = abs(absolute_q_idx - kv_idx) / causal_extent
            barrier = barrier_strength * dist

            return scaled_score + bio_bias - barrier

        # 3. Run FlexAttention
        # block_mask can be used for causal masking
        out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        
        return out
