"""Tests for the schema'd in-silico patch-clamp API (bead ``odq.1``)."""

import json

import pytest
import torch
from rich.console import Console

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.patch_clamp import PATCH_CLAMP_SCHEMA, PatchClampElectrode
from bio_inspired_nanochat.synaptic import SynapticConfig


def _make_model(*, use_moe: bool = False) -> GPTSynaptic:
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=12,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=use_moe,
        num_experts=2,
        syn_cfg=syn_cfg,
    )
    return GPTSynaptic(gpt_cfg)


def _state_snapshot(model: GPTSynaptic) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_patch_clamp_records_schema_aligned_dense_channels() -> None:
    """Each generated token has presynaptic and postsynaptic channel values."""
    model = _make_model()
    model.train()
    electrode = PatchClampElectrode(model)

    prompt = torch.tensor([[1, 2, 3, 4]])
    trace = electrode.record_generation(prompt, max_new_tokens=3, temperature=0.0)

    assert model.training
    assert trace.schema == PATCH_CLAMP_SCHEMA
    assert trace.time_steps == [0, 1, 2]
    assert trace.recorded_token_ids == trace.generated_token_ids  # ubs:ignore — integer token IDs, not credentials
    assert trace.sample_phase == "after_token_forward"
    assert len(trace.token_ids) == 7
    assert len(trace.generated_token_ids) == 3
    assert len(trace.telemetry_history) == 3
    assert all(snapshot["layers"][0]["attention"] is not None for snapshot in trace.telemetry_history)

    required = {
        "L0.attention.H0.calcium",
        "L0.attention.H1.rrp",
        "L0.attention.H0.energy",
        "L0.dense.fc.camkii",
        "L0.dense.fc.pp1",
        "L0.dense.proj.bdnf",
    }
    assert required <= trace.channels.keys()
    assert trace.channels["L0.attention.H0.calcium"].head_idx == 0
    assert all(len(recording.values) == 3 for recording in trace.channels.values())

    for step, telemetry in enumerate(trace.telemetry_history):
        attention = telemetry["layers"][0]["attention"]
        assert attention is not None
        for telemetry_name, channel_name in {
            "C": "calcium",
            "BUF": "buffer",
            "RRP": "rrp",
            "RES": "reserve_pool",
            "PR": "priming",
            "CL": "clamp",
            "E": "energy",
            "AMP": "amplitude",
        }.items():
            for head_idx, value in enumerate(attention[telemetry_name][0]):
                assert trace.channels[
                    f"L0.attention.H{head_idx}.{channel_name}"
                ].values[step] == pytest.approx(value)

    payload = trace.to_dict()
    assert payload["schema"] == PATCH_CLAMP_SCHEMA
    assert payload["source_schema"] == "bio-telemetry/1"
    assert payload["recorded_token_ids"] == trace.generated_token_ids  # ubs:ignore — integer token IDs, not credentials
    assert payload["generated_token_ids"] == trace.generated_token_ids  # ubs:ignore — integer token IDs, not credentials
    json.dumps(payload)


def test_patch_clamp_records_per_expert_moe_channels() -> None:
    """MoE recordings include energy/fatigue and both sites for every expert."""
    model = _make_model(use_moe=True)
    electrode = PatchClampElectrode(model)

    trace = electrode.record_generation(
        torch.tensor([[2, 4, 6, 8]]),
        max_new_tokens=2,
        temperature=0.0,
    )

    assert len(trace.token_ids) == 6
    for expert_idx in range(2):
        assert f"L0.moe.E{expert_idx}.energy" in trace.channels
        assert f"L0.moe.E{expert_idx}.fatigue" in trace.channels
        assert f"L0.moe.E{expert_idx}.fc1.camkii" in trace.channels
        assert f"L0.moe.E{expert_idx}.fc2.bdnf" in trace.channels
        assert trace.channels[f"L0.moe.E{expert_idx}.energy"].expert_idx == expert_idx
        for step, telemetry in enumerate(trace.telemetry_history):
            mlp = telemetry["layers"][0]["mlp"]
            assert trace.channels[f"L0.moe.E{expert_idx}.energy"].values[
                step
            ] == pytest.approx(mlp["energy"][expert_idx])
            assert trace.channels[f"L0.moe.E{expert_idx}.fatigue"].values[
                step
            ] == pytest.approx(mlp["fatigue"][expert_idx])

    console = Console(record=True, width=120)
    electrode.log_trace_summary(trace, console=console)
    assert "Active Channels" in console.export_text()


def test_patch_clamp_plot_and_request_validation() -> None:
    model = _make_model()
    electrode = PatchClampElectrode(model)
    trace = electrode.record_generation(
        torch.tensor([[1, 3, 5]]),
        max_new_tokens=2,
        temperature=0.0,
    )
    selected = ["L0.attention.H0.calcium", "L0.attention.H0.rrp"]
    figure = electrode.plot_trace(trace, selected)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 2

    from matplotlib import pyplot as plt

    plt.close(figure)
    with pytest.raises(ValueError, match="shape"):
        electrode.record_generation(torch.ones((2, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="context"):
        electrode.record_generation(torch.ones((1, 11), dtype=torch.long), max_new_tokens=2)
    with pytest.raises(ValueError, match="non-negative integer"):
        electrode.record_generation(torch.ones((1, 2), dtype=torch.long), max_new_tokens=True)
    with pytest.raises(ValueError, match="torch.int32 or torch.int64"):
        electrode.record_forward(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="finite, non-negative"):
        electrode.record_generation(
            torch.ones((1, 2), dtype=torch.long), temperature="warm"  # type: ignore[arg-type]
        )


def test_plain_forward_telemetry_and_per_token_forward_trace() -> None:
    """The public read API works after normal forwards and at every token boundary."""
    model = _make_model()
    tokens = torch.tensor([[1, 5, 7, 9]])

    with torch.no_grad():
        model(tokens, train_mode=False)
    telemetry = model.bio_telemetry()
    assert telemetry["layers"][0]["attention"] is not None

    before = _state_snapshot(model)
    trace = PatchClampElectrode(model).record_forward(tokens)
    after = model.state_dict()
    assert trace.recording_kind == "forward"
    assert trace.recorded_token_ids == [1, 5, 7, 9]  # ubs:ignore — integer token IDs, not credentials
    assert trace.generated_token_ids == []  # ubs:ignore — integer token IDs, not credentials
    assert trace.time_steps == [0, 1, 2, 3]
    assert all(len(channel.values) == 4 for channel in trace.channels.values())
    assert before.keys() == after.keys()
    for name, value in before.items():
        torch.testing.assert_close(value, after[name], rtol=0.0, atol=0.0)

    figure = PatchClampElectrode(model).plot_trace(
        trace, ["L0.attention.H0.calcium"]
    )
    assert figure.axes[0].get_xlabel() == "Input token step"
    from matplotlib import pyplot as plt

    plt.close(figure)
    model.reset_sequence_state()
    assert model.bio_telemetry()["layers"][0]["attention"] is None


def test_observational_probe_rejects_deferred_plasticity_without_mutation() -> None:
    """A read-only probe must never flush writes deferred by a training forward."""
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        plasticity_during_training=True,
    )
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            synapses=True,
            syn_cfg=syn_cfg,
        )
    )
    model.train()
    logits, _ = model(torch.tensor([[1, 2, 3]]), train_mode=True)
    logits.sum().backward()
    pending_before = {
        name
        for name, module in model.named_modules()
        if hasattr(module, "_plasticity_pending") and module._plasticity_pending
    }
    assert pending_before
    state_before = _state_snapshot(model)

    with pytest.raises(RuntimeError, match="deferred plasticity writes"):
        PatchClampElectrode(model).record_generation(
            torch.tensor([[4, 5]]), max_new_tokens=1, temperature=0.0
        )

    assert model.training
    pending_after = {
        name
        for name, module in model.named_modules()
        if hasattr(module, "_plasticity_pending") and module._plasticity_pending
    }
    assert pending_after == pending_before
    for name, value in state_before.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0.0, atol=0.0)
