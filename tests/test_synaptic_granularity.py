"""Tests for the synaptic granularity switch and evaluation harness (bead vap.2)."""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticGranularity
from scripts.eval_synaptic_granularity import (
    GranularityBenchConfig,
    run_granularity_arm,
    run_granularity_benchmark,
)


@pytest.mark.unit
def test_granularity_buffer_allocation_scales_appropriately():
    """Verify that buffer allocation in SynapticLinear reflects the configured granularity."""
    cfg_conn = SynapticConfig(granularity=SynapticGranularity.PER_CONNECTION, rank_eligibility=8)
    cfg_neur = SynapticConfig(granularity=SynapticGranularity.PER_NEURON, rank_eligibility=8)
    cfg_exp = SynapticConfig(granularity=SynapticGranularity.PER_EXPERT, rank_eligibility=8)

    gpt_conn = GPTSynaptic(GPTSynapticConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=16, synapses=True, syn_cfg=cfg_conn))
    gpt_neur = GPTSynaptic(GPTSynapticConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=16, synapses=True, syn_cfg=cfg_neur))
    gpt_exp = GPTSynaptic(GPTSynapticConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=16, synapses=True, syn_cfg=cfg_exp))

    # Linear u_buf shape: (in_features, _R)
    # Block MLP fc has in_features=16, out_features=64
    assert gpt_conn.transformer.h[0].mlp.mlp.fc.u_buf.shape[-1] == 8
    assert gpt_neur.transformer.h[0].mlp.mlp.fc.u_buf.shape[-1] == 4
    assert gpt_exp.transformer.h[0].mlp.mlp.fc.u_buf.shape[-1] == 1


@pytest.mark.unit
def test_all_granularity_modes_train_and_evaluate_finitely():
    """Verify that every granularity mode executes forward/backward without NaNs or divergence."""
    bench_cfg = GranularityBenchConfig(
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        sequence_len=8,
        batch_size=2,
        num_steps=3,
        seeds=(42,),
    )

    for gran in (SynapticGranularity.PER_CONNECTION.value, SynapticGranularity.PER_NEURON.value, SynapticGranularity.PER_EXPERT.value):
        res = run_granularity_arm(gran, 42, bench_cfg)
        assert res.passed
        assert len(res.train_losses) == 3
        assert all(torch.isfinite(torch.tensor(loss_val)) for loss_val in res.train_losses)
        assert torch.isfinite(torch.tensor(res.val_loss))
        assert torch.isfinite(torch.tensor(res.val_bpb))
        assert res.tokens_per_sec > 0.0


@pytest.mark.unit
def test_run_granularity_benchmark_aggregated_report():
    """Verify that the benchmark produces a valid multi-seed aggregated report."""
    bench_cfg = GranularityBenchConfig(
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        sequence_len=8,
        batch_size=2,
        num_steps=2,
        seeds=(42, 43),
    )

    report = run_granularity_benchmark(bench_cfg)
    assert report.passed
    assert len(report.arm_results) == 6  # 3 granularities * 2 seeds
    assert len(report.aggregates) == 3

    for agg in report.aggregates:
        assert agg.num_seeds == 2
        assert agg.mean_val_loss > 0.0
        assert agg.mean_throughput > 0.0

    report_dict = report.to_dict()
    assert "arm_results" in report_dict
    assert "aggregates" in report_dict
