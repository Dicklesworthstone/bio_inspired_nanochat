"""
Shared pytest fixtures & hooks for bio_inspired_nanochat (bead eqyk.1).

This is the spine of the project's test discipline. Every later "unit tests + e2e
+ detailed logging" bead reuses these fixtures so tests are deterministic, fast,
and portable (CPU dev boxes, GPU CI). The reusable *functions* live in
``tests/_bio_testkit.py``; this file exposes them as fixtures and wires global
hooks (per-test seeding, GPU auto-skip).

Conventions (see TESTING.md):
- Mark fast deterministic tests ``@pytest.mark.unit``; whole-flow tests ``e2e``;
  long ones ``slow``; CUDA-only ones ``gpu`` (auto-skipped on CPU hosts).
- Use the ``seed``/``rng`` fixtures (or call ``set_seed``) — never rely on global
  RNG state leaking between tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# tests/ is on sys.path (pytest prepends the conftest dir), so this resolves.
from _bio_testkit import (
    make_tiny_synaptic,
    make_tiny_vanilla,
    set_seed,
)

DEFAULT_SEED = 1234


# --------------------------------------------------------------------------- #
# Global hooks
# --------------------------------------------------------------------------- #
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.gpu`` tests when CUDA is unavailable.

    Keeps the same suite green on CPU dev boxes and GPU CI without per-test guards.
    """
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available (gpu-marked test)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _per_test_determinism():
    """Seed RNGs before EVERY test so tests can't leak randomness into each other.

    Autouse + function-scoped: a fresh, identical RNG state per test. Tests needing
    a specific seed should request the ``seed`` fixture or call ``set_seed`` again.
    """
    set_seed(DEFAULT_SEED)
    yield


# --------------------------------------------------------------------------- #
# Determinism / device fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def seed() -> int:
    """The canonical test seed (also applied by the autouse determinism fixture)."""
    return DEFAULT_SEED


@pytest.fixture
def rng() -> torch.Generator:
    """A freshly-seeded ``torch.Generator`` for ops that accept ``generator=``."""
    return set_seed(DEFAULT_SEED)


@pytest.fixture
def device() -> torch.device:
    """CPU by default; CUDA when present. Most unit tests should stay on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """CUDA device, skipping the test if unavailable (for ``gpu``-marked tests)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """A throwaway run directory (checkpoints/logs/artifacts) isolated per test."""
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Tiny model fixtures (CPU, fast)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_synaptic_model():
    """A tiny eval-mode ``GPTSynaptic`` on CPU (sub-second forward)."""
    return make_tiny_synaptic(seed=DEFAULT_SEED)


@pytest.fixture
def tiny_synaptic_model_factory():
    """Factory so a test can build several models / patch config fields.

    Usage: ``m = tiny_synaptic_model_factory(n_layer=1, train=True)``.
    """
    return make_tiny_synaptic


@pytest.fixture
def tiny_vanilla_model():
    """A tiny vanilla ``GPT`` on CPU (baseline for bio-vs-vanilla comparisons)."""
    return make_tiny_vanilla(seed=DEFAULT_SEED)
