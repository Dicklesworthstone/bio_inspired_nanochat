"""Tests for the synaptic granularity switch and evaluation harness (bead vap.2)."""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticGranularity,
    SynapticPresyn,
    build_presyn_state,
)
from scripts.eval_synaptic_granularity import (
    GranularityBenchConfig,
    _build_matched_model,
    run_granularity_arm,
    run_granularity_benchmark,
)


@pytest.mark.unit
def test_granularity_buffer_allocation_scales_appropriately():
    """Verify eligibility and molecular state allocation follows the configured granularity."""
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
    assert gpt_conn.transformer.h[0].mlp.mlp.fc.post.U.shape[-1] == 8
    assert gpt_neur.transformer.h[0].mlp.mlp.fc.post.U.shape[-1] == 4
    assert gpt_exp.transformer.h[0].mlp.mlp.fc.post.U.shape[-1] == 1
    assert gpt_conn.transformer.h[0].mlp.mlp.fc.post.camkii.numel() == 64
    assert gpt_neur.transformer.h[0].mlp.mlp.fc.post.camkii.numel() == 64
    assert gpt_exp.transformer.h[0].mlp.mlp.fc.post.camkii.numel() == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("granularity", "expected_shape"),
    [
        (SynapticGranularity.PER_CONNECTION, (2, 3, 7)),
        (SynapticGranularity.PER_NEURON, (2, 3, 1)),
        (SynapticGranularity.PER_EXPERT, (2, 1, 1)),
    ],
)
def test_presynaptic_state_is_physically_pooled_and_live(granularity, expected_shape):
    cfg = SynapticConfig(
        granularity=granularity,
        stochastic_train_frac=0.0,
    )
    state = build_presyn_state(2, 7, 3, "cpu", torch.float32, cfg)
    assert state["C"].shape == expected_shape
    assert all(
        value.shape == expected_shape
        for name, value in state.items()
        if name != "DELAY"
    )
    assert all(value.shape == expected_shape for value in state["DELAY"])

    before = state["C"].clone()
    drive = torch.tensor(
        [[[[0.1, 0.4, 0.7]], [[0.2, 0.5, 0.8]], [[0.3, 0.6, 0.9]]]]
    ).expand(2, -1, -1, -1)
    idx = torch.tensor([[[[0, 3, 6]]]]).expand(2, 3, -1, -1)
    released = SynapticPresyn(4, cfg).release_canonical(
        state,
        drive,
        idx,
        train=False,
        active_key_count=7,
    )

    assert released.shape == drive.shape
    assert torch.isfinite(released).all()
    assert state["C"].shape == expected_shape
    assert not torch.equal(state["C"], before)
    assert torch.isfinite(state["C"]).all()


@pytest.mark.unit
def test_invalid_direct_granularity_fails_closed():
    cfg = SynapticConfig(granularity="per_synapse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="granularity must be one of"):
        build_presyn_state(1, 4, 2, "cpu", torch.float32, cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("granularity", "expected_shape"),
    [
        (SynapticGranularity.PER_CONNECTION, (1, 2, 2)),
        (SynapticGranularity.PER_NEURON, (1, 2, 1)),
        (SynapticGranularity.PER_EXPERT, (1, 1, 1)),
    ],
)
def test_decode_cache_preserves_physical_granularity(granularity, expected_shape):
    cfg = SynapticConfig(granularity=granularity, stochastic_train_frac=0.0)
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            synapses=True,
            syn_cfg=cfg,
        )
    ).eval()
    cache = KVCache(batch_size=1, num_heads=2, seq_len=8, head_dim=8, num_layers=1)

    with torch.no_grad():
        model(torch.tensor([[1]]), kv_cache=cache, train_mode=False)
        model(torch.tensor([[2]]), kv_cache=cache, train_mode=False)

    assert isinstance(cache.presyn_state, list)
    layer_state = cache.presyn_state[0]
    assert isinstance(layer_state, dict)
    assert layer_state["C"].shape == expected_shape


@pytest.mark.unit
def test_matched_benchmark_initialization_keeps_backbone_identical():
    cfg = GranularityBenchConfig(
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        sequence_len=8,
        batch_size=2,
        num_steps=1,
        seeds=(42,),
    )
    fine = _build_matched_model(SynapticGranularity.PER_CONNECTION, 42, cfg)
    coarse = _build_matched_model(SynapticGranularity.PER_EXPERT, 42, cfg)

    torch.testing.assert_close(fine.transformer.wte.weight, coarse.transformer.wte.weight)
    torch.testing.assert_close(
        fine.transformer.h[0].attn.attn.q_proj.weight,
        coarse.transformer.h[0].attn.attn.q_proj.weight,
    )
    torch.testing.assert_close(
        fine.transformer.h[0].mlp.mlp.fc.w_slow,
        coarse.transformer.h[0].mlp.mlp.fc.w_slow,
    )


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
        assert 0.0 <= res.val_accuracy <= 1.0
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
        assert 0.0 <= agg.mean_val_accuracy <= 1.0

    state_bytes = {agg.granularity: agg.mean_state_bytes for agg in report.aggregates}
    assert state_bytes[SynapticGranularity.PER_CONNECTION.value] > state_bytes[
        SynapticGranularity.PER_NEURON.value
    ]
    assert state_bytes[SynapticGranularity.PER_NEURON.value] > state_bytes[
        SynapticGranularity.PER_EXPERT.value
    ]

    report_dict = report.to_dict()
    assert "arm_results" in report_dict
    assert "aggregates" in report_dict
