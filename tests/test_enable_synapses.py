"""End-to-end coverage for pretrained-checkpoint synaptic retrofit (vap.1)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import torch

from bio_inspired_nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from bio_inspired_nanochat.gpt import (
    Block as GPTBlock,
    CausalSelfAttention as GPTAttention,
    GPT,
    GPTConfig,
    MLP as GPTMLP,
)
from bio_inspired_nanochat.synaptic import SynapticLinear, SynapticMoE
from scripts.enable_synapses import build_synaptic, retrofit_checkpoint


def _source_checkpoint(
    root: Path, *, synapses: bool = False, attention_type: str = "standard"
) -> tuple[Path, GPT, GPTConfig]:
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        attention_type=attention_type,
        init_seed=17,
    )
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(-0.03, 0.03, parameter.numel()).reshape_as(parameter)
            parameter.copy_(values + index * 1e-3)
    checkpoint_dir = root / "vanilla"
    save_checkpoint(
        str(checkpoint_dir),
        7,
        model.state_dict(),
        None,
        {"synapses": synapses, "model_config": asdict(config), "provenance": {"run": "test"}},
    )
    return checkpoint_dir, model, config


@pytest.mark.unit
def test_retrofit_copies_pretrained_tensors_into_slow_path(tmp_path):
    source_dir, source, _ = _source_checkpoint(tmp_path)
    output_dir = tmp_path / "synaptic"

    model, report = retrofit_checkpoint(source_dir, output_dir, source_step=-1)

    source_block = cast(GPTBlock, source.blocks[0])
    source_attention = cast(GPTAttention, source_block.attn)
    source_mlp = cast(GPTMLP, source_block.mlp)
    target_block = model.h[0]
    target_dense = target_block.mlp.mlp
    assert torch.equal(model.wte.weight, source.wte.weight)
    assert torch.equal(model.lm_head.weight, source.lm_head.weight)
    assert torch.equal(target_block.attn.attn.q_proj.weight, source_attention.c_q.weight)
    assert torch.equal(target_block.attn.attn.k_proj.weight, source_attention.c_k.weight)
    assert torch.equal(target_block.attn.attn.v_proj.weight, source_attention.c_v.weight)
    assert torch.equal(target_block.attn.attn.o_proj.weight, source_attention.c_proj.weight)
    assert torch.equal(target_dense.fc.w_slow, source_mlp.c_fc.weight.T)
    assert torch.equal(target_dense.proj.w_slow, source_mlp.c_proj.weight.T)
    assert report.source_step == 7
    assert report.copied_tensors == 8
    assert report.expert_copies == 0

    for module in model.modules():
        if not isinstance(module, SynapticLinear):
            continue
        assert module.w_fast is not None and torch.count_nonzero(module.w_fast) == 0
        assert module.u_buf is not None and torch.count_nonzero(module.u_buf) == 0
        assert module.v_buf is not None and torch.count_nonzero(module.v_buf) == 0

    saved_state, _, metadata = load_checkpoint(
        str(output_dir), 0, torch.device("cpu"), load_optimizer=False
    )
    assert bool(metadata["synapses"])
    assert metadata["retrofit"]["source_step"] == 7
    assert metadata["retrofit"]["source_provenance"] == {"run": "test"}
    assert set(saved_state) == set(model.state_dict())


@pytest.mark.unit
def test_retrofit_can_clone_dense_mlp_into_identical_moe_experts(tmp_path):
    source_dir, source, _ = _source_checkpoint(tmp_path)

    model, report = retrofit_checkpoint(
        source_dir,
        tmp_path / "synaptic_moe",
        use_moe=True,
        num_experts=3,
        top_k=2,
    )

    moe = model.h[0].mlp
    assert isinstance(moe, SynapticMoE)
    assert report.expert_copies == 3
    assert torch.count_nonzero(moe.router.weight) == 0
    source_block = cast(GPTBlock, source.blocks[0])
    source_mlp = cast(GPTMLP, source_block.mlp)
    for expert in moe.experts:
        assert torch.equal(expert.fc1.w_slow, source_mlp.c_fc.weight.T)
        assert torch.equal(expert.fc2.w_slow, source_mlp.c_proj.weight.T)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.arange(8).view(1, -1), train_mode=False)
    assert logits.shape == (1, 8, 32)
    assert torch.isfinite(logits).all()


@pytest.mark.unit
def test_retrofit_brief_finetune_activates_dynamics_and_roundtrips(tmp_path):
    source_dir, _, config = _source_checkpoint(tmp_path)
    output_dir = tmp_path / "finetuned"

    model, report = retrofit_checkpoint(
        source_dir,
        output_dir,
        finetune_steps=3,
        finetune_lr=1e-4,
        finetune_seed=23,
    )

    assert report.finetune_steps == 3
    assert report.initial_loss is not None and torch.isfinite(torch.tensor(report.initial_loss))
    assert report.final_loss is not None and torch.isfinite(torch.tensor(report.final_loss))
    assert report.dynamics_active
    assert any(
        module.w_fast is not None and module.w_fast.norm().item() > 0
        for module in model.modules()
        if isinstance(module, SynapticLinear)
    )

    saved_state, _, _ = load_checkpoint(
        str(output_dir), 0, torch.device("cpu"), load_optimizer=False
    )
    rebuilt = build_synaptic(config)
    rebuilt.load_state_dict(saved_state, strict=True)
    rebuilt.eval()
    with torch.no_grad():
        logits, _ = rebuilt(torch.arange(8).view(1, -1), train_mode=False)
    assert torch.isfinite(logits).all()


@pytest.mark.unit
def test_retrofit_rejects_incompatible_or_destructive_inputs(tmp_path):
    source_dir, _, _ = _source_checkpoint(tmp_path)
    output_dir = tmp_path / "synaptic"
    with pytest.raises(ValueError, match="must differ from the source"):
        retrofit_checkpoint(source_dir, source_dir)
    retrofit_checkpoint(source_dir, output_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        retrofit_checkpoint(source_dir, output_dir)

    synaptic_source, _, _ = _source_checkpoint(tmp_path / "already", synapses=True)
    with pytest.raises(ValueError, match="already synaptic"):
        retrofit_checkpoint(synaptic_source, tmp_path / "reject_synaptic")

    ultrametric_source, _, _ = _source_checkpoint(
        tmp_path / "ultrametric", attention_type="ultrametric"
    )
    with pytest.raises(ValueError, match="requires standard Nanochat attention"):
        retrofit_checkpoint(ultrametric_source, tmp_path / "reject_ultrametric")
