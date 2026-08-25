import os
import torch

from bio_inspired_nanochat.checkpoint_manager import (
    checkpoint_model_config,
    config_provenance,
    load_model,
    save_checkpoint,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.synaptic import SynapticConfig
from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla


class _MockTokenizer:
    def get_vocab_size(self):
        return 97

    def get_bos_token_id(self):
        return 1


def test_chat_sft_checkpoint_roundtrip_vanilla(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    monkeypatch.setattr("bio_inspired_nanochat.common.get_base_dir", lambda: base_dir)
    monkeypatch.setattr("bio_inspired_nanochat.checkpoint_manager.get_base_dir", lambda: base_dir)
    monkeypatch.setattr("bio_inspired_nanochat.checkpoint_manager.get_tokenizer", lambda: _MockTokenizer())

    model = make_tiny_vanilla()
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
        "sft",
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
    monkeypatch.setattr("bio_inspired_nanochat.checkpoint_manager.get_tokenizer", lambda: _MockTokenizer())

    syn_cfg = SynapticConfig(tau_c=5.0, doc2_gain=0.1)
    model = make_tiny_synaptic(syn_cfg=syn_cfg)
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
        "rl",
        torch.device("cpu"),
        phase="eval",
        model_tag=model_tag,
        step=4,
    )
    assert meta["step"] == 4
    assert meta.get("synapses") is True
    assert meta.get("synaptic_config") is not None
    assert meta["synaptic_config"]["tau_c"] == 5.0
    assert meta["synaptic_config"]["doc2_gain"] == 0.1
