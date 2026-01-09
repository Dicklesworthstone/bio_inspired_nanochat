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
        """
        Computes the O(N) biological factors needed for the O(N^2) attention map.
        Returns:
            key_factor: (B, H, T) - The 'readiness' of the key to release.
            qamp: (B, H, T) - The quantal amplitude of the key.
        """
        # Extract state
        c = state["C"]  # Calcium
        rrp = state["RRP"]  # Vesicle pool
        cl = state["CL"]  # Complexin clamp
        sn = state["PR"]  # Priming (SNARE)
        amp = state["AMP"]  # AMPA/Quantal size
        
        # 1. Mix Prob (Synaptotagmin & Complexin)
        # p1 = sigmoid(syt1_slope * (c - 0.55))
        p1 = torch.sigmoid(cfg.syt1_slope * (c - 0.55))
        # p7 = sigmoid(syt7_slope * (c - 0.25))
        p7 = torch.sigmoid(cfg.syt7_slope * (c - 0.25))
        # p = p1*0.8 + p7*0.2 + doc2_gain * sigmoid(4*(c-0.12))
        p = p1 * 0.8 + p7 * 0.2 + cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        # Complexin clamp gate: higher clamp reduces release probability.
        cpx_gate = torch.sigmoid(
            8.0 * (c - cfg.cpx_thresh) - 2.0 * (cl + cfg.complexin_bias)
        )

        mix_prob = p * cpx_gate * sn
        mix_prob = torch.clamp(mix_prob, 0, 0.999)
        
        # 2. Key Factor
        # release = mix_prob * rrp
        key_factor = mix_prob * rrp
        
        # 3. QAmp
        # Corresponds to 'amp' state
        qamp = amp
        
        return key_factor, qamp

    def forward(self, q, k, v, presyn_state, block_mask=None):
        """
        q, k, v: (B, H, T, D)
        presyn_state: Dict of (B, H, T)
        """
        B, H, T, _D = q.shape
        
        # 1. Pre-compute biological factors O(N)
        key_factor, qamp = self.precompute_bio_factors(presyn_state, self.config)
        
        # 2. Define the score modifier
        # We need to wrap this in a closure to capture tensors
        # Capture constants to avoid lookup inside
        barrier_strength = self.config.barrier_strength
        epsilon = self.config.epsilon
        
        def score_mod(score, b, h, q_idx, kv_idx):
            # Note: flex_attention applies `scale` before calling score_mod. With the default
            # scale (1/sqrt(E)), `score` is already scaled.
            scaled_score = score
            
            # Bio modulation
            # kf[b, h, kv_idx]
            kf_val = key_factor[b, h, kv_idx]
            qa_val = qamp[b, h, kv_idx]
            
            # release = kf * sigmoid(scaled_score)
            release = kf_val * torch.sigmoid(scaled_score)
            
            # log_term = log(release * qa + eps)
            bio_bias = torch.log(release * qa_val + epsilon)
            
            # Barrier
            # Ensure T is treated as tensor for division if symbolic
            dist = abs(q_idx - kv_idx) / max(1, T)
            barrier = barrier_strength * dist
            
            return scaled_score + bio_bias - barrier

        # 3. Run FlexAttention
        # block_mask can be used for causal masking
        out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        
        return out
