import os
import pytest
import torch

from bio_inspired_nanochat.checkpoint_manager import (
    checkpoint_model_config,
    config_provenance,
    load_model,
    save_checkpoint,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig, GPTSynaptic
from bio_inspired_nanochat.synaptic import SynapticConfig


def _create_toy_gpt(use_synapses=False):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    if use_synapses:
        config.synapses = True
        config.syn_cfg = SynapticConfig(
            vesicle_fatigue=True,
            hebbian_fast_weights=True,
        )
        model = GPTSynaptic(config)
    else:
        model = GPT(config)
    return model


def test_chat_sft_checkpoint_roundtrip_vanilla(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    monkeypatch.setattr("bio_inspired_nanochat.common.get_base_dir", lambda: base_dir)
    monkeypatch.setattr("bio_inspired_nanochat.checkpoint_manager.get_base_dir", lambda: base_dir)

    model = _create_toy_gpt(use_synapses=False)
    depth = model.config.n_layer
    model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)

    model_config_kwargs = {
        "sequence_len": model.config.sequence_len,
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
    }
    user_config = {"num_iterations": 10, "lr": 1e-4}
    metrics = {"eval_accuracy": 0.85}

    save_checkpoint(
        checkpoint_dir,
        10,
        model.state_dict(),
        None,
        {
            "step": 10,
            "val_loss": 0.42,
            **metrics,
            "model_config": checkpoint_model_config(model, model_config_kwargs),
            "synapses": False,
            "synaptic_config": None,
            "provenance": None,
            "user_config": user_config,
        },
        rank=0,
    )

    loaded_model, tokenizer, meta = load_model(
        "chatsft",
        torch.device("cpu"),
        phase="eval",
        model_tag=model_tag,
        step=10,
    )
    assert meta["step"] == 10
    assert meta["val_loss"] == 0.42
    assert meta["eval_accuracy"] == 0.85
    assert not meta.get("synapses", False)


def test_chat_rl_checkpoint_roundtrip_synaptic(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    monkeypatch.setattr("bio_inspired_nanochat.common.get_base_dir", lambda: base_dir)
    monkeypatch.setattr("bio_inspired_nanochat.checkpoint_manager.get_base_dir", lambda: base_dir)

    model = _create_toy_gpt(use_synapses=True)
    depth = model.config.n_layer
    model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)

    model_config_kwargs = {
        "sequence_len": model.config.sequence_len,
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
    }
    syn_cfg = getattr(model.config, "syn_cfg", None)
    user_config = {"num_steps": 5, "save_every": 2}

    save_checkpoint(
        checkpoint_dir,
        4,
        model.state_dict(),
        None,
        {
            "step": 4,
            "model_config": checkpoint_model_config(model, model_config_kwargs),
            "synapses": True,
            "synaptic_config": synaptic_config_to_meta(syn_cfg),
            "provenance": config_provenance(syn_cfg),
            "user_config": user_config,
        },
        rank=0,
    )

    loaded_model, tokenizer, meta = load_model(
        "chatrl",
        torch.device("cpu"),
        phase="eval",
        model_tag=model_tag,
        step=4,
    )
    assert meta["step"] == 4
    assert meta.get("synapses") is True
    assert meta.get("synaptic_config") is not None
    assert meta["synaptic_config"]["vesicle_fatigue"] is True
    assert meta["synaptic_config"]["hebbian_fast_weights"] is True
