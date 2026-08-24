"""Unit tests for the centralized determinism framework (beads aiq, hm4.4)."""

from __future__ import annotations

import torch

from bio_inspired_nanochat.determinism import (
    configure_determinism,
    determinism_provenance_dict,
)


def test_configure_determinism_sets_reproducible_rng():
    """Configuring determinism with same seed produces identical random tensors."""
    configure_determinism(seed=1234, deterministic=True)
    t1 = torch.randn(10, 10)

    configure_determinism(seed=1234, deterministic=True)
    t2 = torch.randn(10, 10)

    assert torch.equal(t1, t2), "Same seed must produce identical random tensors"

    configure_determinism(seed=5678, deterministic=True)
    t3 = torch.randn(10, 10)
    assert not torch.equal(t1, t3), "Different seeds must produce different tensors"


def test_determinism_state_and_provenance():
    """Determinism state is serializable and captures current environment flags."""
    state = configure_determinism(seed=999, deterministic=True)
    assert state.seed == 999
    assert state.deterministic_algorithms is True

    prov = determinism_provenance_dict(seed=999)
    assert prov["seed"] == 999
    assert "deterministic_algorithms" in prov
    assert "cublas_workspace_config" in prov
