"""Unit and sequence ablation tests for STDP Plasticity (bead sax.3)."""

from __future__ import annotations

import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear


def test_stdp_toggle_and_temporal_order_sensitivity():
    """STDP produces asymmetric weight updates dependent on temporal sequence order."""
    # 1. Standard rate-Hebbian (time-symmetric)
    cfg_rate = SynapticConfig(
        enable_hebbian=True,
        enable_stdp=False,
        post_fast_lr=0.01,
    )
    layer_rate = SynapticLinear(8, 8, cfg=cfg_rate)

    # 2. STDP (time-asymmetric)
    cfg_stdp = SynapticConfig(
        enable_hebbian=True,
        enable_stdp=True,
        stdp_a_plus=0.02,
        stdp_a_minus=0.01,
        post_fast_lr=0.01,
    )
    layer_stdp = SynapticLinear(8, 8, cfg=cfg_stdp)

    # Sequence of 4 time steps
    x_forward = torch.randn(4, 8)
    ca = torch.ones(4, 8)
    en = torch.ones(4, 8)

    layer_rate.reset_sequence_state()
    layer_stdp.reset_sequence_state()

    _ = layer_rate(x_forward, ca, en)
    _ = layer_stdp(x_forward, ca, en)

    # Forward updates differ between rate-Hebbian and STDP
    assert not torch.allclose(layer_rate.u_buf, layer_stdp.u_buf)

    # 3. Test temporal order asymmetry for STDP: reversing sequence inverts pre/post timing
    x_reversed = torch.flip(x_forward, dims=[0])
    layer_stdp_rev = SynapticLinear(8, 8, cfg=cfg_stdp)
    layer_stdp_rev.reset_sequence_state()
    _ = layer_stdp_rev(x_reversed, ca, en)

    # Reversing sequence order should produce distinct STDP trace updates
    assert not torch.allclose(layer_stdp.u_buf, layer_stdp_rev.u_buf)
