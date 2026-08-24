"""Tests for Model-Provided bio_telemetry() API (bead `hm4.6`)."""

import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.telemetry import collect_bio_telemetry


def test_bio_telemetry_schema_and_keys():
    """Verify that bio_telemetry() produces schema-valid dictionary for dense synaptic model."""
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg)

    # Perform a dummy forward pass to populate state
    x = torch.randint(0, 32, (2, 8))
    model(x)

    telemetry = model.bio_telemetry()

    assert telemetry["schema"] == "bio-telemetry/1"
    assert telemetry["num_layers"] == 2
    assert len(telemetry["layers"]) == 2

    l0 = telemetry["layers"][0]
    assert l0["index"] == 0
    assert "mlp" in l0
    assert l0["mlp"]["type"] == "dense"
    assert "fc" in l0["mlp"]
    fc_site = l0["mlp"]["fc"]
    assert "camkii" in fc_site
    assert "pp1" in fc_site
    assert "bdnf" in fc_site


def test_bio_telemetry_moe():
    """Verify that bio_telemetry() properly captures MoE expert energy, fatigue, and site metrics."""
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=True,
        num_experts=4,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg)

    x = torch.randint(0, 32, (2, 8))
    model(x)

    telemetry = collect_bio_telemetry(model, include_routing=True)

    assert telemetry["num_layers"] == 1
    l0 = telemetry["layers"][0]
    mlp = l0["mlp"]
    assert mlp["type"] == "moe"
    assert mlp["num_experts"] == 4
    assert len(mlp["energy"]) == 4
    assert len(mlp["fatigue"]) == 4
    assert len(mlp["experts"]) == 4
