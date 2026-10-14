"""Tests for the Performance-Regression Test Harness and Gates (beads eqyk.15, r2d)."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import bio_inspired_nanochat.perf_regression as perf_regression_module
import pytest
import torch
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.perf_regression import (
    BaselineEntry,
    BenchmarkResult,
    PerfBenchmarkConfig,
    PerfRegressionHarness,
    STANDARD_BENCHMARK_CONFIGS,
    _synchronize_for_timing,
)
from scripts.perf_regression_gate import main as gate_main


def test_perf_harness_decode_uses_one_cache_and_counts_only_generated_tokens(monkeypatch):
    """Prompt prefill is untimed; measured calls are cached Tq=1 decode tokens."""
    calls = []
    timing_boundaries = []
    original_forward = Engine._forward

    def spy_forward(self, ids, kv_cache):
        before = kv_cache.get_pos()
        result = original_forward(self, ids, kv_cache)
        calls.append((tuple(ids.shape), id(kv_cache), before, kv_cache.get_pos()))
        return result

    clock = iter((10.0, 12.0))
    monkeypatch.setattr(Engine, "_forward", spy_forward)
    monkeypatch.setattr(
        perf_regression_module,
        "_synchronize_for_timing",
        lambda device: timing_boundaries.append(device.type),
    )
    monkeypatch.setattr(
        perf_regression_module,
        "time",
        SimpleNamespace(perf_counter=lambda: next(clock)),
    )
    harness = PerfRegressionHarness()
    cfg = PerfBenchmarkConfig(
        name="test_fast_bench",
        mode="decode",
        synaptic=False,
        batch_size=1,
        seq_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=16,
        warmup_steps=1,
        measure_steps=2,
    )
    res = harness.run_benchmark(cfg)
    assert [call[0] for call in calls] == [(1, 8), (1, 1), (1, 1), (1, 1)]
    assert len({call[1] for call in calls}) == 1
    assert [call[2] for call in calls] == [0, 8, 9, 10]
    assert [call[3] for call in calls] == [8, 9, 10, 11]
    assert timing_boundaries == ["cpu", "cpu"]
    assert res.tok_per_sec == 1.0
    assert res.latency_ms == 1000.0
    assert res.peak_memory_mb > 0.0
    assert res.steps_measured == 2


def test_synaptic_cached_decode_uses_inference_semantics(monkeypatch):
    seen_train_modes = []
    original_forward = GPTSynaptic.forward

    @wraps(original_forward)
    def spy_forward(self, *args, **kwargs):
        seen_train_modes.append(kwargs.get("train_mode"))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(GPTSynaptic, "forward", spy_forward)
    result = PerfRegressionHarness().run_benchmark(
        PerfBenchmarkConfig(
            name="synaptic_cached_decode_test",
            mode="decode",
            synaptic=True,
            batch_size=1,
            seq_len=4,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            warmup_steps=1,
            measure_steps=2,
        )
    )
    assert seen_train_modes == [False, False, False, False]
    assert result.tok_per_sec > 0.0


def test_timing_synchronization_targets_cuda_only(monkeypatch):
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))
    _synchronize_for_timing(torch.device("cpu"))
    cuda_device = torch.device("cuda")
    _synchronize_for_timing(cuda_device)
    assert synchronized == [cuda_device]


def test_committed_baselines_cover_every_standard_scenario():
    baselines = PerfRegressionHarness().load_baselines()
    assert {cfg.name for cfg in STANDARD_BENCHMARK_CONFIGS} <= baselines.keys()
    for cfg in STANDARD_BENCHMARK_CONFIGS:
        baseline = baselines[cfg.name]
        assert baseline.device == cfg.device
        assert baseline.benchmark_config == vars(cfg)


def test_standard_cached_decode_uses_stable_long_timing_window():
    decode_configs = [cfg for cfg in STANDARD_BENCHMARK_CONFIGS if cfg.mode == "decode"]
    assert decode_configs
    assert all(cfg.measure_steps >= 200 for cfg in decode_configs)
    assert all(cfg.seq_len + cfg.warmup_steps + cfg.measure_steps >= 200 for cfg in decode_configs)


def test_perf_baselines_save_and_load(tmp_path: Path):
    """Baselines can be saved to disk and reloaded losslessly."""
    baseline_file = tmp_path / "test_baselines.json"
    harness = PerfRegressionHarness(baselines_path=baseline_file)

    mock_results = [
        BenchmarkResult(
            name="bench_1",
            mode="train",
            synaptic=False,
            tok_per_sec=1000.0,
            latency_ms=1.0,
            peak_memory_mb=10.0,
            steps_measured=5,
            batch_size=2,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "bench_1", "device": "cpu"},
        ),
        BenchmarkResult(
            name="bench_2",
            mode="decode",
            synaptic=True,
            tok_per_sec=500.0,
            latency_ms=2.0,
            peak_memory_mb=15.0,
            steps_measured=5,
            batch_size=1,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "bench_2", "device": "cpu"},
        ),
    ]

    harness.save_baselines(mock_results, tolerance=0.20)
    assert baseline_file.exists()

    loaded = harness.load_baselines()
    assert len(loaded) == 2
    assert "bench_1" in loaded
    assert "bench_2" in loaded
    assert loaded["bench_1"].tok_per_sec == 1000.0
    assert loaded["bench_1"].tolerance == 0.20
    assert loaded["bench_1"].device == "cpu"
    assert loaded["bench_1"].benchmark_config == {"name": "bench_1", "device": "cpu"}


def test_perf_gate_regression_detection(tmp_path: Path):
    """Gate fails when observed throughput regresses beyond tolerance."""
    baseline_file = tmp_path / "test_baselines.json"
    harness = PerfRegressionHarness(baselines_path=baseline_file)

    baselines = {
        "bench_fast": BaselineEntry(
            name="bench_fast",
            mode="train",
            tok_per_sec=1000.0,
            peak_memory_mb=10.0,
            device="cpu",
            benchmark_config={"name": "bench_fast", "device": "cpu"},
            tolerance=0.10,
        ),
    }

    # Case 1: Pass (observed is 950 tok/s, regression is 5% <= 10% tol)
    passing_results = [
        BenchmarkResult(
            name="bench_fast",
            mode="train",
            synaptic=False,
            tok_per_sec=950.0,
            latency_ms=1.05,
            peak_memory_mb=10.0,
            steps_measured=5,
            batch_size=2,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "bench_fast", "device": "cpu"},
        )
    ]
    comparisons_pass = harness.evaluate_gates(passing_results, baselines=baselines)
    assert len(comparisons_pass) == 1
    assert comparisons_pass[0].passed

    # Case 2: Fail (observed is 800 tok/s, regression is 20% > 10% tol)
    failing_results = [
        BenchmarkResult(
            name="bench_fast",
            mode="train",
            synaptic=False,
            tok_per_sec=800.0,
            latency_ms=1.25,
            peak_memory_mb=10.0,
            steps_measured=5,
            batch_size=2,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "bench_fast", "device": "cpu"},
        )
    ]
    comparisons_fail = harness.evaluate_gates(failing_results, baselines=baselines)
    assert len(comparisons_fail) == 1
    assert not comparisons_fail[0].passed


def test_perf_gate_fails_closed_for_missing_or_mismatched_baseline(tmp_path: Path):
    result = BenchmarkResult(
        name="bench_fast",
        mode="train",
        synaptic=False,
        tok_per_sec=950.0,
        latency_ms=1.0,
        peak_memory_mb=10.0,
        steps_measured=5,
        batch_size=2,
        seq_len=16,
        device="cpu",
        benchmark_config={"name": "bench_fast", "device": "cpu"},
    )
    harness = PerfRegressionHarness(tmp_path / "missing.json")
    missing = harness.evaluate_gates([result])
    assert not missing[0].passed
    assert "No valid baseline" in missing[0].detail

    mismatch = BaselineEntry(
        name="bench_fast",
        mode="train",
        tok_per_sec=1000.0,
        peak_memory_mb=10.0,
        device="cpu",
        benchmark_config={"name": "bench_fast", "device": "cpu", "measure_steps": 10},
    )
    mismatched = harness.evaluate_gates([result], baselines={"bench_fast": mismatch})
    assert not mismatched[0].passed
    assert "configuration differs" in mismatched[0].detail


def test_cuda_request_does_not_silently_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = PerfBenchmarkConfig(name="cuda_required", device="cuda")
    with pytest.raises(RuntimeError, match="without CUDA"):
        PerfRegressionHarness().run_benchmark(cfg)


def test_perf_gate_cli_fails_closed_without_baselines(tmp_path: Path, monkeypatch):
    result = BenchmarkResult(
        name="bench_fast",
        mode="train",
        synaptic=False,
        tok_per_sec=950.0,
        latency_ms=1.0,
        peak_memory_mb=10.0,
        steps_measured=5,
        batch_size=2,
        seq_len=16,
        device="cpu",
        benchmark_config={"name": "bench_fast", "device": "cpu"},
    )
    monkeypatch.setattr(PerfRegressionHarness, "run_all", lambda self: [result])
    ret_check = gate_main(
        ["--mode", "check", "--baselines", str(tmp_path / "missing.json")]
    )
    assert ret_check == 1


def test_perf_gate_cli_integration(tmp_path: Path):
    """CLI entrypoint works in both check and update modes."""
    baselines_file = tmp_path / "perf_baselines.json"
    output_json = tmp_path / "out_report.json"

    # Step 1: Update baseline file
    ret_update = gate_main([
        "--mode", "update",
        "--baselines", str(baselines_file),
        "--tolerance", "0.40",
    ])
    assert ret_update == 0
    assert baselines_file.exists()

    # Step 2: Check against baseline file with output JSON
    ret_check = gate_main([
        "--mode", "check",
        "--baselines", str(baselines_file),
        "--tolerance", "0.85",
        "--output-json", str(output_json),
    ])
    assert ret_check == 0
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload) >= 4


def test_throughput_budget_planted_3x_slowdown_fails_and_in_budget_passes(
    tmp_path: Path, monkeypatch
):
    """Planted 3x synaptic slowdown fails ThroughputBudget(2.0x train, 1.5x decode)."""
    from bio_inspired_nanochat.perf_regression import ThroughputBudget

    harness = PerfRegressionHarness(baselines_path=tmp_path / "baselines.json")
    planted_results = [
        BenchmarkResult(
            name="standard_transformer_train",
            mode="train",
            synaptic=False,
            tok_per_sec=3000.0,
            latency_ms=1.0,
            peak_memory_mb=1.0,
            steps_measured=5,
            batch_size=2,
            seq_len=32,
            device="cpu",
            benchmark_config={"name": "standard_transformer_train", "device": "cpu"},
        ),
        BenchmarkResult(
            name="synaptic_transformer_train",
            mode="train",
            synaptic=True,
            tok_per_sec=1000.0,  # 3.0x slower than standard (> 2.0x budget)
            latency_ms=3.0,
            peak_memory_mb=1.5,
            steps_measured=5,
            batch_size=2,
            seq_len=32,
            device="cpu",
            benchmark_config={"name": "synaptic_transformer_train", "device": "cpu"},
        ),
        BenchmarkResult(
            name="standard_transformer_cached_tq1_decode",
            mode="decode",
            synaptic=False,
            tok_per_sec=600.0,
            latency_ms=1.67,
            peak_memory_mb=1.0,
            steps_measured=200,
            batch_size=1,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "standard_transformer_cached_tq1_decode", "device": "cpu"},
        ),
        BenchmarkResult(
            name="synaptic_transformer_cached_tq1_decode",
            mode="decode",
            synaptic=True,
            tok_per_sec=200.0,  # 3.0x slower than standard (> 1.5x budget)
            latency_ms=5.0,
            peak_memory_mb=1.5,
            steps_measured=200,
            batch_size=1,
            seq_len=16,
            device="cpu",
            benchmark_config={"name": "synaptic_transformer_cached_tq1_decode", "device": "cpu"},
        ),
    ]
    harness.save_baselines(planted_results, tolerance=0.25)

    budget_comps = harness.evaluate_budget_gates(planted_results, budget=ThroughputBudget())
    assert len(budget_comps) == 2
    assert all(not c.passed for c in budget_comps)
    assert budget_comps[0].slowdown_ratio == pytest.approx(3.0)
    assert budget_comps[1].slowdown_ratio == pytest.approx(3.0)

    # CLI with --check-budget must fail on the planted 3x slowdown even when baseline drift passes
    monkeypatch.setattr(PerfRegressionHarness, "run_all", lambda self: planted_results)
    ret_fail = gate_main([
        "--mode", "check",
        "--baselines", str(tmp_path / "baselines.json"),
        "--check-budget",
    ])
    assert ret_fail == 1

    # When synaptic throughput is within budget (1.8x train <= 2.0x, 1.4x decode <= 1.5x), both pass
    import dataclasses

    passing_results = [
        planted_results[0],
        dataclasses.replace(planted_results[1], tok_per_sec=3000.0 / 1.8),
        planted_results[2],
        dataclasses.replace(planted_results[3], tok_per_sec=600.0 / 1.4),
    ]
    harness.save_baselines(passing_results, tolerance=0.25)
    monkeypatch.setattr(PerfRegressionHarness, "run_all", lambda self: passing_results)
    ret_pass = gate_main([
        "--mode", "check",
        "--baselines", str(tmp_path / "baselines.json"),
        "--check-budget",
    ])
    assert ret_pass == 0

    # Invalid ThroughputBudget ratios and mismatched geometry / non-positive throughput fail closed
    with pytest.raises(ValueError, match="train_ratio_max must be positive"):
        ThroughputBudget(train_ratio_max=0.0)
    with pytest.raises(ValueError, match="decode_ratio_max must be positive"):
        ThroughputBudget(decode_ratio_max=-1.0)

    geom_mismatch = [
        passing_results[0],
        dataclasses.replace(passing_results[1], seq_len=64),
    ]
    geom_comp = harness.evaluate_budget_gates(geom_mismatch)
    assert len(geom_comp) == 1
    assert not geom_comp[0].passed
    assert "geometry mismatch" in geom_comp[0].detail

    zero_tps = [
        dataclasses.replace(passing_results[0], tok_per_sec=0.0),
        passing_results[1],
    ]
    zero_comp = harness.evaluate_budget_gates(zero_tps)
    assert len(zero_comp) == 1
    assert not zero_comp[0].passed
    assert "non-positive throughput" in zero_comp[0].detail


def test_live_throughput_budget_gate_passes():
    """Live CPU benchmarks satisfy ThroughputBudget(train <= 2.0x, decode <= 1.5x) (beads 74f.10, 74f.10.2)."""
    harness = PerfRegressionHarness()
    results = harness.run_all()
    budget_comps = harness.evaluate_budget_gates(results)
    assert len(budget_comps) == 2
    for comp in budget_comps:
        assert comp.passed, comp.detail


