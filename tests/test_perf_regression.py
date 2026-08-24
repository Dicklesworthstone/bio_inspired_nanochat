"""Tests for the Performance-Regression Test Harness and Gates (beads eqyk.15, r2d)."""

from __future__ import annotations

import json
from pathlib import Path


from bio_inspired_nanochat.perf_regression import (
    BaselineEntry,
    BenchmarkResult,
    PerfBenchmarkConfig,
    PerfRegressionHarness,
)
from scripts.perf_regression_gate import main as gate_main


def test_perf_harness_run_single_benchmark():
    """Benchmark runs and measures non-zero throughput and memory."""
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
    assert res.tok_per_sec > 0.0
    assert res.latency_ms > 0.0
    assert res.peak_memory_mb > 0.0
    assert res.steps_measured == 2


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


def test_perf_gate_regression_detection(tmp_path: Path):
    """Gate fails when observed throughput regresses beyond tolerance."""
    baseline_file = tmp_path / "test_baselines.json"
    harness = PerfRegressionHarness(baselines_path=baseline_file)

    baselines = {
        "bench_fast": BaselineEntry(name="bench_fast", mode="train", tok_per_sec=1000.0, peak_memory_mb=10.0, tolerance=0.10),
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
        )
    ]
    comparisons_fail = harness.evaluate_gates(failing_results, baselines=baselines)
    assert len(comparisons_fail) == 1
    assert not comparisons_fail[0].passed


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
        "--tolerance", "0.50",
        "--output-json", str(output_json),
    ])
    assert ret_check == 0
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload) >= 4
