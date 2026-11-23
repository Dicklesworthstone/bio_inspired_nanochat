"""
SynapticConfig checkpoint round-trip — bead vg9.6.

build_model used to rebuild synaptic models with SynapticConfig() DEFAULTS, so a model trained
or tuned with custom bio kinetics silently reloaded as a DIFFERENT model. These tests lock the
fix: the full bio config persists into meta_data (as JSON) and rebuilds EXACTLY, with
forward/back-compat handling and a provenance stamp.

Run:  pytest tests/test_checkpoint_roundtrip.py -v
"""

from __future__ import annotations

import json

import torch

import pytest

from bio_inspired_nanochat.checkpoint_manager import (
    build_model,
    checkpoint_model_config,
    config_hash,
    config_provenance,
    save_checkpoint,
    synaptic_config_from_meta,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


@pytest.mark.parametrize(
    "saved_norm,softcap,tied",
    [(None, 15.0, False), (False, 0.75, False), (True, 0.0, False), (True, 0.75, True)],
    ids=["legacy-head", "unnormalized-softcap", "normalized-uncapped", "normalized-softcap-tied"],
)
def test_serving_and_scoring_reproduce_original_checkpoint_logits(tmp_path, monkeypatch, saved_norm, softcap, tied):
    import tiktoken
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.tokenizer import RustBPETokenizer
    from scripts.eval_matrix import _get_logits, _load_base_train_checkpoint

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    enc = tiktoken.Encoding(
        name="byte-roundtrip", pat_str="(?s).", mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|bos|>": 256},
    )
    tokenizer = RustBPETokenizer(enc, "<|bos|>")
    tokenizer.save(str(tmp_path / "tokenizer"))
    syn_cfg = SynapticConfig()
    cfg = GPTSynapticConfig(sequence_len=8, vocab_size=257, n_layer=1, n_head=1, n_kv_head=1, n_embd=16,
                            init_seed=17, syn_cfg=syn_cfg, final_norm=bool(saved_norm),
                            logit_softcap=softcap, tie_embeddings=tied, rope_base=800.0)
    ids = torch.tensor([tokenizer.encode("abcd")])
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(17)
        original = GPTSynaptic(cfg).eval()
        original.init_weights()
        original.wte.weight.mul_(3)
        if not tied:
            original.lm_head.weight.uniform_(-1, 1)
        expected = original(ids, train_mode=False)[0]
        # A zero head or unit-scale residual could otherwise make this test insensitive.
        cfg.final_norm = not cfg.final_norm
        assert not torch.allclose(expected, original(ids, train_mode=False)[0])
        cfg.final_norm = bool(saved_norm)
        architecture = checkpoint_model_config(original, {})
        assert architecture["logit_softcap"] == softcap
        if saved_norm is None:
            architecture.pop("final_norm")  # historical checkpoints predate the switch
            architecture.pop("logit_softcap")
        directory = tmp_path / "checkpoint"
        save_checkpoint(str(directory), 2, original.state_dict(), None, {
            "model_config": architecture, "synapses": True, "synaptic_config": synaptic_config_to_meta(syn_cfg),
        })
        served, serving_tokenizer, _ = build_model(str(directory), 2, torch.device("cpu"), "eval")
        scored, _, _ = _load_base_train_checkpoint(directory, step=2, preset="bio_all", seed=17, device=torch.device("cpu"))
        assert serving_tokenizer.encode("abcd") == ids[0].tolist()
        for loaded in (served, scored):
            assert (loaded.wte.weight is loaded.lm_head.weight) == tied
            assert loaded.config.logit_softcap == softcap and loaded.config.rope_base == 800.0
        torch.testing.assert_close(served(ids)[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(_get_logits(scored, ids), expected, rtol=0, atol=0)


@pytest.mark.unit
def test_custom_config_round_trips_through_meta_json():
    # The headline acceptance: save a non-default config -> reload -> identical.
    cfg = SynapticConfig(tau_c=9.0, syt_fast_kd=0.7, doc2_gain=0.15, enable_hebbian=False)
    meta = {"synapses": True, "synaptic_config": synaptic_config_to_meta(cfg)}
    meta = json.loads(json.dumps(meta))  # meta_data is persisted as JSON on disk
    assert synaptic_config_from_meta(meta) == cfg, "non-default bio config must round-trip exactly"


@pytest.mark.unit
def test_full_default_config_round_trips():
    cfg = SynapticConfig()
    meta = {"synaptic_config": json.loads(json.dumps(synaptic_config_to_meta(cfg)))}
    assert synaptic_config_from_meta(meta) == cfg


@pytest.mark.unit
def test_missing_config_falls_back_to_defaults():
    # pre-vg9.6 checkpoints (no persisted config) must not crash; they fall back to defaults.
    # They predate hwxb.9, so the fallback keeps Hebbian on (their w_fast/post.* tensors must load).
    assert synaptic_config_from_meta({"synapses": True}) == SynapticConfig(enable_hebbian=True)
    assert synaptic_config_from_meta(None) == SynapticConfig(enable_hebbian=True)


@pytest.mark.unit
def test_unknown_saved_field_is_ignored_forward_compat():
    d = synaptic_config_to_meta(SynapticConfig(tau_c=7.0))
    d["a_field_removed_in_a_future_schema"] = 123
    rebuilt = synaptic_config_from_meta({"synaptic_config": d})
    assert rebuilt.tau_c == 7.0  # known fields applied; the unknown one is dropped, not a crash


@pytest.mark.unit
def test_missing_new_field_takes_default_back_compat():
    d = synaptic_config_to_meta(SynapticConfig())
    removed = d.pop("doc2_gain")  # an old checkpoint that predates a newer field
    rebuilt = synaptic_config_from_meta({"synaptic_config": d})
    assert rebuilt.doc2_gain == SynapticConfig().doc2_gain != removed + 1  # default filled in


@pytest.mark.unit
def test_config_hash_is_stable_and_sensitive():
    a = synaptic_config_to_meta(SynapticConfig(tau_c=6.0))
    b = synaptic_config_to_meta(SynapticConfig(tau_c=6.0))
    c = synaptic_config_to_meta(SynapticConfig(tau_c=9.0))
    assert config_hash(a) == config_hash(b), "same config -> same hash"
    assert config_hash(a) != config_hash(c), "different config -> different hash"


@pytest.mark.unit
def test_provenance_stamp_has_sha_and_config_hash():
    prov = config_provenance(SynapticConfig())
    assert set(prov) == {"git_sha", "synaptic_config_hash"}
    assert len(prov["synaptic_config_hash"]) == 16
    # git_sha is a 40-char hex SHA in a git repo, or None if git is unavailable
    assert prov["git_sha"] is None or len(prov["git_sha"]) == 40


def test_crash_debris_step_is_not_discovered_when_markers_in_use(tmp_path):
    """0qvh regression: a save that crashed between the model/meta writes and the
    commit marker must NOT be auto-selected by find_last_step — resume should
    fall back to the previous complete step instead of dying on missing shards."""
    import os

    from bio_inspired_nanochat.checkpoint_manager import (
        find_last_step,
        list_checkpoint_steps,
        save_checkpoint,
    )

    # Step 5: a complete (marker-bearing) checkpoint.
    save_checkpoint(str(tmp_path), 5, model_data={"w": torch.zeros(1)},
                    optimizer_data={"m": torch.zeros(1)}, meta_data={"step": 5}, rank=0)
    # Step 10: crash debris — model+meta landed, no optim shard, no commit marker.
    (tmp_path / "model_000010.pt").write_bytes(b"partial")
    (tmp_path / "meta_000010.json").write_text("{}")

    assert list_checkpoint_steps(str(tmp_path)) == [5]
    assert find_last_step(str(tmp_path)) == 5
    assert os.path.exists(tmp_path / "commit_000005.json")


def test_legacy_directory_without_markers_stays_discoverable(tmp_path):
    """Back-compat: pre-marker directories keep the old discovery semantics."""
    from bio_inspired_nanochat.checkpoint_manager import list_checkpoint_steps

    (tmp_path / "model_000007.pt").write_bytes(b"x")
    (tmp_path / "meta_000007.json").write_text("{}")
    assert list_checkpoint_steps(str(tmp_path)) == [7]


def test_rustbpe_tokenizer_json_round_trip(tmp_path):
    """fmep: save()/from_directory() must round-trip a tiktoken Encoding through
    lossless JSON (no pickle) with identical encode behavior."""
    import tiktoken

    from bio_inspired_nanochat.tokenizer import RustBPETokenizer

    # tiktoken's byte-fallback requires ALL 256 single-byte tokens present.
    ranks = {bytes([i]): i for i in range(256)}
    ranks.update({b"ab": 256, b"abc": 257})
    enc = tiktoken.Encoding(
        name="test",
        pat_str=r"\s+",
        mergeable_ranks={b"a": 0, b"b": 1, b"c": 2, b"ab": 3, b"abc": 4, b" ": 5},
        special_tokens={"<|bos|>": 6, "<|endoftext|>": 7},
    )
    tok = RustBPETokenizer(enc, "<|bos|>")
    sample = "abc ab c"
    ids_before = tok.encode(sample)

    tok.save(str(tmp_path))
    assert (tmp_path / "tokenizer.json").exists()
    assert not (tmp_path / "tokenizer.pkl").exists(), "new saves must not write pickle"

    tok2 = RustBPETokenizer.from_directory(str(tmp_path))
    assert tok2.encode(sample) == ids_before
    assert tok2.get_vocab_size() == tok.get_vocab_size()
