"""Tests for the Fast-Weight Programmer & Working Memory Baseline Benchmark (bead `sax.5`).

Verifies:
1. Multi-seed memory suite evaluation across Vanilla, Classical Outer-Product Fast Weights,
   DeltaNet Error-Correcting Fast Weights, and Bio-Inspired Synaptic Transformers.
2. Output shape consistency and numerical stability of fast weight modules.
3. CLI entrypoint and JSON artifact serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.e2e.fast_weight_comparison_bench import (
    ClassicalOuterProductFastWeightBlock,
    DeltaNetFastWeightBlock,
    FastWeightBenchConfig,
    main as fw_main,
    run_fast_weight_benchmark,
)


@pytest.mark.unit
def test_fast_weight_modules_forward_and_shapes():
    """Individual fast weight blocks process sequential inputs with preserved shapes."""
    b, t, d = 2, 8, 16
    x = torch.randn(b, t, d)

    outer_block = ClassicalOuterProductFastWeightBlock(n_embd=d)
    y_outer = outer_block(x)
    assert y_outer.shape == (b, t, d)
    assert not torch.isnan(y_outer).any()

    delta_block = DeltaNetFastWeightBlock(n_embd=d)
    y_delta = delta_block(x)
    assert y_delta.shape == (b, t, d)
    assert not torch.isnan(y_delta).any()


@pytest.mark.unit
def test_fast_weight_benchmark_full_run(tmp_path: Path):
    """The benchmark executes across all 4 architectures and returns structured scores."""
    cfg = FastWeightBenchConfig(
        seeds=(301, 303),
        vocab_size=37,
        sequence_len=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        batch_size=4,
        recall_pairs=(2, 4),
        binding_distractors=(0, 4),
        niah_lengths=(8, 16),
        bootstrap_samples=100,
    )
    report = run_fast_weight_benchmark(cfg, run_dir=tmp_path, verbose=False)

    assert set(report.architectures.keys()) == {
        "vanilla",
        "outer_product_fw",
        "deltanet_fw",
        "bio_synaptic",
    }
    for mode in ("vanilla", "outer_product_fw", "deltanet_fw", "bio_synaptic"):
        arch = report.architectures[mode]
        assert len(arch.scores) == 2
        assert 0.0 <= arch.composite_stats.mean <= 1.0

    assert "outer_product_fw" in report.comparisons_vs_vanilla
    assert "deltanet_fw" in report.comparisons_vs_vanilla
    assert "bio_synaptic" in report.comparisons_vs_vanilla
    assert report.comparisons_bio_vs_deltanet is not None


@pytest.mark.unit
def test_fast_weight_bench_config_validation():
    """Config validation rejects invalid seeds or non-divisible embeddings."""
    with pytest.raises(ValueError, match="at least two"):
        FastWeightBenchConfig(seeds=(1,))
    with pytest.raises(ValueError, match="unique"):
        FastWeightBenchConfig(seeds=(1, 1))
    with pytest.raises(ValueError, match="divisible by 4"):
        FastWeightBenchConfig(n_embd=6)


@pytest.mark.unit
def test_fast_weight_bench_cli_entrypoint(tmp_path: Path):
    """CLI entrypoint runs cleanly and writes structured JSON."""
    json_path = tmp_path / "fw_report.json"
    ret = fw_main([
        "--run-dir", str(tmp_path),
        "--output-json", str(json_path),
        "--seeds", "301", "303",
        "--device", "cpu",
    ])
    assert ret == 0
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "architectures" in data
    assert "comparisons_vs_vanilla" in data
    assert "comparisons_bio_vs_deltanet" in data
