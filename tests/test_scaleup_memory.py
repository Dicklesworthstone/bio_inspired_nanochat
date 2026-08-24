"""Memory-budget estimator tests (bead hwxb.2.2 / hwxb.7.2 memory-budget helper).

The estimator's persistent terms (params, buffers, optimizer moment state) are EXACT and
CPU-computable; these tests pin them to a real model + a real stepped optimizer. The
activation/synaptic terms are rough estimates, so we only assert sign + scaling there.
"""
from __future__ import annotations

import json

import pytest

from bio_inspired_nanochat.torch_imports import torch
from _bio_testkit import TINY, make_tiny_synaptic, make_tiny_vanilla
from scripts.scale_memory import (
    BenchmarkResult,
    _parse_int_list,
    buffer_bytes,
    estimate,
    measure_throughput,
    optimizer_moment_bytes,
    param_bytes,
    synaptic_state_bytes_est,
    write_benchmark_jsonl,
)


def _vanilla(**ov):
    m = make_tiny_vanilla(**ov)
    m.init_weights()
    return m


@pytest.mark.unit
def test_param_and_buffer_bytes_are_exact():
    m = _vanilla()
    assert param_bytes(m) == sum(p.numel() * p.element_size() for p in m.parameters())
    assert buffer_bytes(m) == sum(b.numel() * b.element_size() for b in m.buffers())


@pytest.mark.unit
def test_optimizer_moment_bytes_match_a_real_stepped_optimizer():
    torch.manual_seed(0)
    m = _vanilla()
    m.train()
    opts = m.setup_optimizers()
    predicted = optimizer_moment_bytes(m, world_size=1)
    # Take one real step so the moment buffers actually allocate.
    x = torch.randint(0, TINY["vocab_size"], (2, 16))
    y = torch.randint(0, TINY["vocab_size"], (2, 16))
    m(x, y).backward()
    for opt in opts:
        opt.step()
    # Sum the real moment tensors (exclude the negligible per-param int64 `step` scalar).
    real_moments = sum(
        v.numel() * v.element_size()
        for opt in opts
        for st in opt.state.values()
        for k, v in st.items()
        if torch.is_tensor(v) and k != "step"
    )
    assert predicted == real_moments


@pytest.mark.unit
def test_optimizer_bytes_shard_by_world_size():
    m = _vanilla()
    one = optimizer_moment_bytes(m, world_size=1)
    two = optimizer_moment_bytes(m, world_size=2)
    assert two == one // 2


@pytest.mark.unit
def test_tying_cuts_param_and_optimizer_bytes():
    """Tying removes one V×d matrix from BOTH the param count and the optimizer moments."""
    untied = _vanilla(tie_embeddings=False)
    tied = _vanilla(tie_embeddings=True)
    elsize = next(p.element_size() for p in untied.parameters())  # fp32 on CPU
    assert param_bytes(untied) - param_bytes(tied) == TINY["vocab_size"] * TINY["n_embd"] * elsize
    # The shared weight also drops one param's worth of AdamW moments (2×) — tied is smaller.
    assert optimizer_moment_bytes(tied) < optimizer_moment_bytes(untied)


@pytest.mark.unit
def test_budget_estimate_fields_and_scaling():
    mv = _vanilla()
    b1 = estimate(mv, mv.config, batch=2, seq=64, world_size=1)
    b2 = estimate(mv, mv.config, batch=4, seq=64, world_size=1)
    # Activations scale linearly with batch; params/optimizer do not.
    assert b2.activation_bytes_est == 2 * b1.activation_bytes_est
    assert b2.param_bytes == b1.param_bytes
    # Vanilla model carries no synaptic per-key state.
    assert b1.synaptic_state_bytes_est == 0
    # Totals are positive and consistent.
    assert b1.total_est_bytes == b1.persistent_bytes + b1.activation_bytes_est + b1.synaptic_state_bytes_est
    assert all(v >= 0 for v in b1.as_gb().values())


@pytest.mark.unit
def test_synaptic_model_has_nonzero_synaptic_state_estimate():
    ms = make_tiny_synaptic()
    ms.init_weights()
    s = synaptic_state_bytes_est(ms.config, batch=2, seq=64)
    assert s > 0, "synaptic model should carry a per-key state estimate"


@pytest.mark.e2e
def test_measure_throughput_returns_positive_tokps():
    m = _vanilla()
    tp = measure_throughput(m, m.config, batch=2, seq=32, steps=3, warmup=1, device="cpu")
    assert tp["tok_per_sec"] > 0 and tp["step_ms"] > 0


@pytest.mark.e2e
def test_inference_throughput_and_profiler_trace_are_recorded(tmp_path):
    m = _vanilla()
    trace = tmp_path / "trace.json"
    tp = measure_throughput(
        m,
        m.config,
        batch=1,
        seq=16,
        steps=2,
        warmup=1,
        device="cpu",
        mode="infer",
        seed=17,
        profile_path=trace,
    )
    assert tp["tok_per_sec"] > 0 and tp["step_ms"] > 0
    assert tp["peak_allocated_gb"] is None and tp["gpu_utilization_pct"] is None
    assert tp["profile_path"] == str(trace)
    assert trace.exists() and trace.stat().st_size > 0


@pytest.mark.unit
def test_benchmark_jsonl_schema_and_matrix_list_validation(tmp_path):
    row = BenchmarkResult(
        variant="vanilla",
        mode="train",
        batch=2,
        seq=32,
        steps=3,
        warmup=1,
        seed=17,
        device="cpu",
        world_size=1,
        tok_per_sec=123.0,
        step_ms=4.0,
        peak_allocated_gb=None,
        peak_reserved_gb=None,
        gpu_utilization_pct=None,
        profile_path=None,
        provenance={"seed": 17, "git_sha": "abc"},
    )
    path = write_benchmark_jsonl([row], tmp_path / "bench.jsonl")
    try:
        payload = json.JSONDecoder().decode(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(f"benchmark writer emitted invalid JSON: {exc}")
    assert payload["variant"] == "vanilla" and payload["seed"] == 17
    assert payload["tok_per_sec"] == pytest.approx(123.0)
    assert payload["provenance"]["git_sha"] == "abc"
    assert _parse_int_list("1, 4,8", name="batches") == [1, 4, 8]
    with pytest.raises(ValueError, match="positive"):
        _parse_int_list("1,0", name="batches")
