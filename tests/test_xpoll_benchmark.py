"""Tests for Shared Cross-Pollination Benchmark Harness (bead `zc2`).

Verifies comparative benchmarking across Vanilla, Bio-Inspired, Simplicial, and Reversible models:
1. All architectures construct, forward, and train cleanly.
2. Throughput, memory, loss proxy, and gradient stability metrics are recorded.
3. CSV and JSON export formats produce valid, well-formed files.
4. CLI entrypoint runs end-to-end.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.xpoll_benchmark import (
    XpollBenchmarkConfig,
    main as xpoll_main,
    run_xpoll_benchmark,
)


@pytest.mark.unit
def test_xpoll_benchmark_full_run(tmp_path: Path):
    """Run full benchmark across all models and verify report and outputs."""
    cfg = XpollBenchmarkConfig(
        vocab_size=64,
        n_embd=32,
        n_head=2,
        n_kv_head=2,
        n_layer=2,
        sequence_len=16,
        batch_size=2,
        benchmark_steps=3,
        warmup_steps=1,
        device="cpu",
    )
    csv_file = tmp_path / "benchmarks.csv"
    json_file = tmp_path / "benchmarks.json"

    report = run_xpoll_benchmark(cfg, output_csv=csv_file, output_json=json_file, verbose=False)

    assert len(report.results) >= 4
    arch_names = {r.architecture for r in report.results}
    assert "vanilla_gpt" in arch_names
    assert "bio_synaptic" in arch_names
    assert "mgr_simplicial" in arch_names
    assert "mgr_reversible" in arch_names

    for r in report.results:
        assert r.forward_tok_per_sec > 0
        assert r.train_tok_per_sec > 0
        assert r.loss_proxy > 0
        assert r.param_count > 0

    # Check CSV export
    assert csv_file.exists()
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == len(report.results) + 1 # header + rows
        assert rows[0][0] == "Architecture"

    # Check JSON export
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "results" in data
    assert len(data["results"]) == len(report.results)


@pytest.mark.unit
def test_xpoll_benchmark_cli_entrypoint(tmp_path: Path):
    """CLI entrypoint executes and generates CSV file."""
    csv_file = tmp_path / "cli_benchmarks.csv"
    ret = xpoll_main([
        "--output-csv", str(csv_file),
        "--device", "cpu",
        "--steps", "2",
        "--seq-len", "16",
        "--embd-dim", "32",
        "--layers", "2",
        "--batch-size", "2",
        "--seed", "42",
    ])
    assert ret == 0
    assert csv_file.exists()
