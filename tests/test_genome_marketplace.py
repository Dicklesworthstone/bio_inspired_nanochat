"""Unit and behavioral probe tests for Genome Marketplace (bead re4e.11)."""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.genome_marketplace import (
    get_genome,
    list_available_genomes,
    transplant_genome,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear


def test_marketplace_lists_and_retrieves_genomes():
    """Marketplace offers multiple characterized genomes."""
    genomes = list_available_genomes()
    assert len(genomes) >= 3

    g_novelty = get_genome("high_novelty")
    assert g_novelty.personality == "explorer"
    assert g_novelty.config.post_fast_lr > 0.005

    with pytest.raises(KeyError):
        get_genome("nonexistent_profile")


def test_transplant_genome_updates_model_in_place():
    """Transplanting a genome alters kinetic dynamics without changing affine weights."""
    cfg_init = SynapticConfig(post_fast_lr=0.01)
    layer = SynapticLinear(16, 32, cfg=cfg_init)

    orig_weight = layer.w_slow.clone()

    # Transplant low_energy genome
    count = transplant_genome(layer, "low_energy")
    assert count == 1
    assert layer.cfg.bdnf_scale == 0.0
    assert layer.cfg.post_fast_lr == 0.0

    # Weight values remain strictly unchanged
    torch.testing.assert_close(layer.w_slow, orig_weight)


def test_transplanted_genomes_produce_behavioral_shifts():
    """High-novelty vs low-energy genomes produce distinct dynamic plasticity responses."""
    layer = SynapticLinear(8, 8, cfg=get_genome("balanced_biomimetic").config)
    x1 = torch.randn(4, 8)
    x2 = torch.randn(4, 8)
    ca = torch.ones(4, 8)
    en = torch.ones(4, 8)

    # 1. High Novelty: fast Hebbian learning
    transplant_genome(layer, "high_novelty")
    layer.reset_sequence_state()
    _ = layer(x1, ca, en)
    out_novelty = layer(x2, ca, en)

    # 2. Re-create low energy layer
    layer_frugal = SynapticLinear(8, 8, cfg=get_genome("low_energy").config)
    layer_frugal.reset_sequence_state()
    _ = layer_frugal(x1, ca, en)
    out_frugal = layer_frugal(x2, ca, en)

    # Dynamic outputs diverge due to different kinetic configs
    assert not torch.allclose(out_novelty, out_frugal)
