"""Tests for staged and full-space CMA-ES search spaces (bead `hea.3`)."""

import numpy as np
import pytest

from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.tune_bio_params import (
    FULL_LIVE_PARAM_SPECS,
    PARAM_SPACES,
    _validate_param_specs,
    decode_params,
    encode_params,
    evaluate_candidate_detailed,
)


def test_param_spaces_validation():
    """Verify all staged parameter spaces are structurally valid with no duplicates."""
    for name, specs in PARAM_SPACES.items():
        _validate_param_specs(specs)
        # Ensure all specs correspond to real fields in SynapticConfig
        cfg = SynapticConfig()
        for spec in specs:
            assert hasattr(cfg, spec.name), f"{name} contains non-existent field {spec.name}"


def test_full_space_encode_decode_roundtrip():
    """Verify round-trip encoding and decoding across full 21D parameter space."""
    cfg = SynapticConfig()
    specs = FULL_LIVE_PARAM_SPECS
    vec = encode_params(cfg, specs)

    assert len(vec) == len(specs)
    decoded = decode_params(vec, specs)

    for spec in specs:
        orig = float(getattr(cfg, spec.name))
        assert decoded[spec.name] == pytest.approx(orig, rel=1e-3, abs=1e-4)


def test_full_space_evaluation_smoke():
    """Verify that a candidate from the full 21D space evaluates without error."""
    specs = FULL_LIVE_PARAM_SPECS
    vec = encode_params(SynapticConfig(), specs)

    tiny_model = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )

    res = evaluate_candidate_detailed(
        vec,
        specs=specs,
        seed=1,
        steps=2,
        batch_size=4,
        device="cpu",
        lr=1e-3,
        weight_decay=0.0,
        timeout_seconds=None,
        max_retries=0,
        model_config=tiny_model,
        held_out_batches=2,
        reset_state=True,
        raise_on_error=True,
    )

    assert res.objective is not None and np.isfinite(res.objective)
