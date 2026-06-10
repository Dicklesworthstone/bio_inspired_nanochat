"""
Smoke tests for the test framework itself + a sample per bio module (bead eqyk.1).

Two jobs:
1. Prove the eqyk.1 framework works — determinism, tensor diagnostics, golden I/O,
   and the tiny-model fixtures all behave. If this file is green, later beads can
   trust the harness.
2. Provide a minimal, real "sample unit test for each bio module" so coverage
   touches every core module and import regressions are caught immediately.

Run:  pytest tests/test_framework_smoke.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

import _bio_testkit as tk


# ======================================================================= #
# 1. Framework self-tests
# ======================================================================= #
@pytest.mark.unit
def test_set_seed_is_deterministic():
    tk.set_seed(0)
    a = torch.rand(5)
    tk.set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b), "set_seed must make torch RNG reproducible"


@pytest.mark.unit
def test_rng_fixture_is_reproducible(rng):
    x = torch.rand(4, generator=rng)
    rng2 = tk.set_seed(1234)  # same seed the fixture uses
    y = torch.rand(4, generator=rng2)
    assert torch.equal(x, y)


@pytest.mark.unit
def test_tensor_stats_never_raises_on_pathological_input():
    # NaN / Inf / empty must all be summarized safely (this is the logging primitive).
    nan_inf = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0, -2.0])
    s = tk.tensor_stats(nan_inf)
    assert s.nan == 1 and s.inf == 2
    assert not s.finite
    assert math.isfinite(s.mean)  # mean computed over finite entries only

    empty = torch.empty(0)
    se = tk.tensor_stats(empty)
    assert se.numel == 0 and se.nan == 0

    good = torch.arange(10.0)
    sg = tk.tensor_stats(good)
    assert sg.finite and sg.min == 0.0 and sg.max == 9.0
    assert "mean=" in str(sg)  # one-line log format


@pytest.mark.unit
def test_assert_finite_passes_and_fails():
    tk.assert_finite(torch.ones(3), "ones")
    with pytest.raises(AssertionError, match="not finite"):
        tk.assert_finite(torch.tensor([1.0, float("nan")]), "bad")


@pytest.mark.unit
@pytest.mark.golden
def test_golden_roundtrip(tmp_path, monkeypatch):
    # Redirect golden storage to a temp dir so the test is hermetic.
    monkeypatch.setattr(tk, "GOLDEN_DIR", tmp_path / "golden")
    t = torch.linspace(-1, 1, 20).reshape(4, 5)
    # First call bootstraps (writes) the golden and returns without comparing.
    tk.assert_golden("smoke_linspace", t)
    assert (tmp_path / "golden" / "smoke_linspace.npy").exists()
    # Second call compares against the just-written golden -> must pass.
    tk.assert_golden("smoke_linspace", t.clone())
    # A changed tensor must fail the comparison.
    with pytest.raises(AssertionError, match="mismatch"):
        tk.assert_golden("smoke_linspace", t + 1.0)


@pytest.mark.unit
def test_capability_probes_return_bools():
    assert isinstance(tk.cuda_available(), bool)
    assert isinstance(tk.rustbpe_available(), bool)


# ======================================================================= #
# 2. Tiny-model fixtures
# ======================================================================= #
def _logits(out):
    """GPTSynaptic.forward returns (logits, ...) without targets; GPT returns logits."""
    return out[0] if isinstance(out, (tuple, list)) else out


@pytest.mark.unit
def test_tiny_synaptic_model_forward_is_finite(tiny_synaptic_model):
    x = tk.random_tokens(2, 16, vocab=97)
    logits = _logits(tiny_synaptic_model(x))
    assert logits.shape[:2] == (2, 16)
    tk.assert_finite(logits, "synaptic logits")
    assert tk.count_params(tiny_synaptic_model) > 0


@pytest.mark.unit
def test_tiny_synaptic_model_actually_uses_synaptic_attention(tiny_synaptic_model):
    # Guard: the factory must build a GENUINELY synaptic model (a
    # SynapticCausalSelfAttention exists somewhere in the module tree), not silently
    # fall back to vanilla attention — otherwise every "synaptic" test would pass while
    # exercising the wrong code path. (.modules() is robust to the wrapper nesting:
    # Block.attn is a CausalSelfAttention wrapper whose .attn is the synaptic one.)
    from bio_inspired_nanochat.synaptic import SynapticCausalSelfAttention

    assert any(isinstance(m, SynapticCausalSelfAttention) for m in tiny_synaptic_model.modules())


@pytest.mark.unit
def test_tiny_vanilla_model_forward_is_finite(tiny_vanilla_model):
    x = tk.random_tokens(2, 16, vocab=97)
    logits = _logits(tiny_vanilla_model(x))
    assert logits.shape[:2] == (2, 16)
    tk.assert_finite(logits, "vanilla logits")


@pytest.mark.unit
def test_model_factory_accepts_overrides(tiny_synaptic_model_factory):
    m = tiny_synaptic_model_factory(seed=0, n_layer=1)
    assert m.config.n_layer == 1


# ======================================================================= #
# 3. Sample unit test per core bio module (import + a real, cheap op)
# ======================================================================= #
@pytest.mark.unit
def test_module_synaptic_moe_forward():
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

    cfg = SynapticConfig(enable_metabolism=True, enable_hebbian=False, native_genetics=False)
    moe = SynapticMoE(n_embd=8, num_experts=2, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)
    x = torch.randn(1, 6, 8)
    out = moe(x)
    y = out[0] if isinstance(out, (tuple, list)) else out
    assert y.shape == (1, 6, 8)
    tk.assert_finite(y, "moe out")


@pytest.mark.unit
def test_module_gpt_synaptic_config_defaults():
    from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig

    cfg = GPTSynapticConfig(sequence_len=16, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=32)
    assert cfg.n_layer == 1 and cfg.n_embd == 32


@pytest.mark.unit
def test_module_engine_kvcache_shapes():
    from bio_inspired_nanochat.engine import KVCache

    kv = KVCache(batch_size=2, num_heads=2, seq_len=16, head_dim=8, num_layers=1)
    assert kv.get_pos() == 0


@pytest.mark.unit
def test_module_neuroscore_importable():
    # NeuroScore is observability; a smoke import guards against API/import drift.
    import bio_inspired_nanochat.neuroscore as ns

    assert hasattr(ns, "NeuroScore") or hasattr(ns, "NeuroScoreManager")


@pytest.mark.unit
def test_module_splitmerge_importable():
    import bio_inspired_nanochat.synaptic_splitmerge as sm

    # The surgical lifecycle controller must expose a controller entry point.
    assert any(n for n in dir(sm) if "Controller" in n or "SplitMerge" in n)


@pytest.mark.unit
def test_module_kernels_package_imports():
    import bio_inspired_nanochat.kernels as kernels  # noqa: F401

    assert kernels is not None
