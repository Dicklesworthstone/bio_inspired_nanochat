"""Unit and CPU E2E coverage for the cross-architecture HF bio adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)

from bio_inspired_nanochat.hf_bio_adapter import (
    HFBioLinearAdapter,
    bio_adapter_metrics,
    bio_adapter_parameters,
    inject_bio_adapters,
    iter_bio_adapters,
    load_bio_adapter,
    save_bio_adapter,
    set_bio_adaptation,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


def _tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(11)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=41,
            n_positions=24,
            n_ctx=24,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(17)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=43,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=24,
            attention_dropout=0.0,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "expected_count", "source_kind"),
    [
        (_tiny_gpt2, 2, "transformers.Conv1D"),
        (_tiny_llama, 3, "torch.nn.Linear"),
    ],
)
def test_injection_preserves_logits_across_hf_projection_families(
    factory,
    expected_count: int,
    source_kind: str,
) -> None:
    model = factory().eval()
    tokens = torch.tensor([[1, 5, 8, 13, 3], [2, 7, 11, 4, 9]])
    with torch.no_grad():
        expected = model(tokens).logits

    report = inject_bio_adapters(model)
    set_bio_adaptation(model, False)
    with torch.no_grad():
        actual = model(tokens).logits

    assert report.adapter_count == expected_count
    assert set(report.source_kinds) == {source_kind}
    assert all("attn" not in name and "lm_head" not in name for name in report.adapter_names)
    # Linear's fused addmm and SynapticLinear's matmul-plus-bias may differ by one FP32 ULP.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.unit
def test_injection_fails_closed_when_no_projection_matches() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    with pytest.raises(ValueError, match="no supported feed-forward projections"):
        inject_bio_adapters(model)


@pytest.mark.unit
def test_explicit_pattern_supports_unusual_model_and_toggles_are_neutral() -> None:
    class UnusualModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_projection = nn.Linear(4, 6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.custom_projection(x)

    model = UnusualModel().eval()
    x = torch.randn(3, 4)
    expected = model(x)
    report = inject_bio_adapters(
        model,
        SynapticConfig(
            enable_presyn=False,
            enable_hebbian=False,
            enable_metabolism=False,
        ),
        target_patterns=("custom_projection",),
    )
    actual = model(x)
    adapter = model.custom_projection

    assert report.adapter_names == ("custom_projection",)
    assert isinstance(adapter, HFBioLinearAdapter)
    assert adapter.metrics()["calcium"] == 0.0
    assert adapter.metrics()["energy"] == 1.0
    assert adapter.metrics()["eligibility_norm"] == 0.0
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.e2e
def test_short_adapter_finetune_activates_dynamics_and_roundtrips_bundle(
    tmp_path: Path,
) -> None:
    model = _tiny_gpt2()
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    config = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        enable_metabolism=True,
        stochastic_train_frac=0.0,
        fast_weight_eta=0.02,
        fast_weight_max_norm=0.1,
        post_slow_lr=1e-5,
    )
    report = inject_bio_adapters(model, config)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = bio_adapter_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    tokens = torch.tensor(
        [
            [1, 5, 8, 13, 3, 21, 7, 2],
            [2, 7, 11, 4, 9, 17, 6, 1],
        ]
    )

    model.train()
    set_bio_adaptation(model, True)
    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens).loss
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    metrics = bio_adapter_metrics(model)
    assert report.adapter_count == 2
    assert all(int(values["adaptation_steps"]) >= 3 for values in metrics.values())
    assert all(float(values["calcium"]) > 0.0 for values in metrics.values())
    assert all(float(values["eligibility_norm"]) > 0.0 for values in metrics.values())
    assert any(float(values["fast_weight_norm"]) > 0.0 for values in metrics.values())
    assert all(torch.isfinite(torch.tensor(losses)))

    # One non-adapting inference flushes the final deferred online write before packaging.
    model.eval()
    set_bio_adaptation(model, False)
    with torch.no_grad():
        model(tokens)
    bundle_dir = tmp_path / "adapter_bundle"
    manifest = save_bio_adapter(model, bundle_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_bio_adapter(model, bundle_dir)
    with torch.no_grad():
        expected = model(tokens).logits

    restored = _tiny_gpt2().eval()
    restored.load_state_dict(base_state)
    restored_report = load_bio_adapter(restored, bundle_dir, adaptation_enabled=False)
    with torch.no_grad():
        actual = restored(tokens).logits

    assert manifest["schema_version"] == 1
    assert restored_report.adapter_names == report.adapter_names
    assert len(tuple(iter_bio_adapters(restored))) == report.adapter_count
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
