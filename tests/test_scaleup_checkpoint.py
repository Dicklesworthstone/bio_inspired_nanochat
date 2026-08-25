"""Checkpoint robustness + resume reproducibility tests (bead hwxb.2.6).

Covers the crash-safety + bit-comparable-resume contract for long 2×4090 runs:
- atomic writes leave no partial files and loaders ignore stray ``*.tmp``,
- RNG capture/restore is reproducible,
- keep-last-K + best rotation deletes only superseded checkpoint artifacts,
- and the headline: a synaptic training run resumed from a checkpoint (model + optimizer
  + RNG) continues the EXACT loss trajectory of the uninterrupted run.

The resume test is what makes the RNG persistence worth it: the synaptic forward is
stochastic during training, so without restoring RNG a resume silently diverges.
"""
from __future__ import annotations

import copy
import os

import pytest

from bio_inspired_nanochat import checkpoint_manager as cm
from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.checkpoint_manager import (
    capture_rng_state,
    checkpoint_model_config,
    list_checkpoint_steps,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
    synaptic_config_to_meta,
)

# tests/ is on sys.path (conftest), so this resolves.
from _bio_testkit import make_tiny_synaptic


def test_atomic_write_leaves_no_tmp_and_load_roundtrips(tmp_path):
    d = str(tmp_path)
    model_data = {"w": torch.randn(4, 4)}
    opt_data = {"state": {"x": torch.zeros(2)}}
    meta = {"model_config": {"vocab_size": 97}, "synapses": True}
    save_checkpoint(d, 5, model_data, opt_data, meta, rank=0)
    # No stray .tmp files left behind by the atomic write.
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    # A stray .tmp from a hypothetical crash must NOT be picked up by the loader.
    with open(os.path.join(d, "model_000005.pt.tmp"), "wb") as f:
        f.write(b"garbage-partial-write")
    m, o, meta2 = load_checkpoint(d, 5, torch.device("cpu"), load_optimizer=True)
    assert torch.equal(m["w"], model_data["w"])
    assert meta2["model_config"]["vocab_size"] == 97


def test_save_checkpoint_does_not_mutate_caller_metadata(tmp_path):
    metadata = {"model_config": {"vocab_size": 97}}
    original = copy.deepcopy(metadata)

    save_checkpoint(
        str(tmp_path),
        5,
        {"layer.pre.calcium": torch.zeros(1)},
        None,
        metadata,
    )

    assert metadata == original
    saved = cm.load_checkpoint_metadata(str(tmp_path), 5)
    assert saved["synapses"] is True


def test_find_last_step_ignores_malformed_model_filenames(tmp_path):
    (tmp_path / "model_latest.pt").write_bytes(b"unrelated")
    (tmp_path / "model_123.pt").write_bytes(b"not a padded checkpoint")
    save_checkpoint(
        str(tmp_path),
        42,
        {"w": torch.zeros(1)},
        None,
        {"model_config": {}},
    )

    assert cm.find_last_step(str(tmp_path)) == 42


def test_find_last_step_ignores_model_only_partial_checkpoint(tmp_path):
    save_checkpoint(
        str(tmp_path),
        41,
        {"w": torch.zeros(1)},
        None,
        {"model_config": {}},
    )
    (tmp_path / "model_000042.pt").write_bytes(b"complete-file-but-incomplete-checkpoint")

    assert cm.find_last_step(str(tmp_path)) == 41

    (tmp_path / "meta_000041.json").rename(tmp_path / "meta_000041.json.saved")
    with pytest.raises(FileNotFoundError, match="No complete checkpoints"):
        cm.find_last_step(str(tmp_path))


def test_load_checkpoint_metadata_reports_malformed_json_path(tmp_path):
    metadata_path = tmp_path / "meta_000003.json"
    metadata_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=str(metadata_path)):
        cm.load_checkpoint_metadata(str(tmp_path), 3)

    metadata_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        cm.load_checkpoint_metadata(str(tmp_path), 3)


def test_rng_capture_restore_is_reproducible():
    torch.manual_seed(0)
    _ = torch.randn(10)  # advance RNG
    state = capture_rng_state()
    a = torch.randn(5)
    restore_rng_state(state)
    b = torch.randn(5)
    assert torch.equal(a, b), "restored RNG must reproduce the same draws"


def test_restore_rng_state_rejects_malformed_numpy_state():
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    replacement = torch.Generator().manual_seed(999).get_state()

    with pytest.raises(RuntimeError, match="NumPy RNG state"):
        restore_rng_state(
            {"torch": replacement, "numpy": {"type": "MT19937"}}
        )

    assert torch.equal(torch.get_rng_state(), state_before)


def test_heterogeneous_moe_topology_roundtrips_strictly(tmp_path, monkeypatch):
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
    from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

    syn_cfg = SynapticConfig(enable_hebbian=False)
    config = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=97,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        syn_cfg=syn_cfg,
        use_moe=True,
        num_experts=2,
        moe_experts_per_layer=(2, 3),
        moe_top_k=1,
        moe_hidden_mult=1,
    )
    model = GPTSynaptic(config)
    model.init_weights()
    architecture = checkpoint_model_config(
        model,
        {
            "sequence_len": 16,
            "vocab_size": 97,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_head": 2,
            "n_embd": 8,
        },
    )
    assert architecture["use_moe"] is True
    assert architecture["moe_experts_per_layer"] == [2, 3]
    save_checkpoint(
        str(tmp_path),
        4,
        model.state_dict(),
        None,
        {
            "model_config": architecture,
            "synapses": True,
            "synaptic_config": synaptic_config_to_meta(syn_cfg),
        },
    )

    class _Tokenizer:
        @staticmethod
        def get_vocab_size():
            return 97

    monkeypatch.setattr(cm, "get_tokenizer", lambda: _Tokenizer())
    loaded, _, _ = cm.build_model(str(tmp_path), 4, torch.device("cpu"), "eval")
    counts = [
        block.mlp.num_experts
        for block in loaded.h
        if isinstance(block.mlp, SynapticMoE)
    ]
    assert counts == [2, 3]
    assert loaded.state_dict().keys() == model.state_dict().keys()

    class _WrongTokenizer:
        @staticmethod
        def get_vocab_size():
            return 96

    monkeypatch.setattr(cm, "get_tokenizer", lambda: _WrongTokenizer())
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        cm.build_model(str(tmp_path), 4, torch.device("cpu"), "eval")


def test_build_model_rejects_invalid_phase_before_loading(tmp_path):
    with pytest.raises(ValueError, match="phase must be"):
        cm.build_model(str(tmp_path), 1, torch.device("cpu"), "inference")


def test_prune_keeps_last_k_and_best(tmp_path):
    d = str(tmp_path)
    for s in (10, 20, 30, 40, 50):
        # rank-0 model/meta + two ranks' optim/train shards (mimic a 2-GPU run).
        save_checkpoint(d, s, {"w": torch.zeros(2)}, {"o": torch.zeros(1)},
                        {"model_config": {}}, rank=0, train_state={"rng": capture_rng_state()})
        save_checkpoint(d, s, {"w": torch.zeros(2)}, {"o": torch.zeros(1)},
                        {"model_config": {}}, rank=1, train_state={"rng": capture_rng_state()})
    assert list_checkpoint_steps(d) == [10, 20, 30, 40, 50]
    pruned = prune_checkpoints(d, keep_last=2, best_step=10)
    assert set(pruned) == {20, 30}
    assert list_checkpoint_steps(d) == [10, 40, 50]
    # The pruned steps' artifacts are gone for EVERY rank (no orphaned partial checkpoint).
    for s in (20, 30):
        for r in (0, 1):
            assert not os.path.exists(os.path.join(d, f"optim_{s:06d}_rank{r}.pt"))
            assert not os.path.exists(os.path.join(d, f"train_{s:06d}_rank{r}.pt"))
        assert not os.path.exists(os.path.join(d, f"model_{s:06d}.pt"))
    # Kept steps retain all ranks' artifacts.
    for s in (10, 40, 50):
        assert os.path.exists(os.path.join(d, f"meta_{s:06d}.json"))
        for r in (0, 1):
            assert os.path.exists(os.path.join(d, f"optim_{s:06d}_rank{r}.pt"))


def _train_step(model, opt, x, y):
    """One clean training step (reset per-sequence transient state -> independent seqs)."""
    model.reset_sequence_state()
    _, loss = model(x, y, None, train_mode=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


@pytest.mark.e2e
def test_resume_is_bit_comparable(tmp_path):
    """A run resumed from a checkpoint (model+opt+RNG) reproduces the uninterrupted trajectory.

    The synaptic forward draws from the global RNG during training (stochastic vesicle
    release), so this only holds because the checkpoint restores RNG state. Single-threaded
    to keep CPU reductions bitwise-deterministic.
    """
    torch.set_num_threads(1)
    n_warm, n_after = 8, 6
    # Fixed data pool.
    g = torch.Generator().manual_seed(123)
    pool = [
        (toks[:, :-1].contiguous(), toks[:, 1:].contiguous())
        for toks in (torch.randint(0, 97, (4, 33), generator=g) for _ in range(6))
    ]

    # --- uninterrupted run: warm up, checkpoint, then continue ---
    model = make_tiny_synaptic(seed=0, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for i in range(n_warm):
        _train_step(model, opt, *pool[i % len(pool)])
    # Checkpoint: deep-copy state (so later in-place updates don't mutate the snapshot) + RNG.
    ckpt_model = copy.deepcopy(model.state_dict())
    ckpt_opt = copy.deepcopy(opt.state_dict())
    ckpt_rng = capture_rng_state()
    traj_uninterrupted = [
        _train_step(model, opt, *pool[(n_warm + i) % len(pool)]) for i in range(n_after)
    ]

    # --- resumed run: fresh model+opt, load checkpoint, restore RNG, same data ---
    model_b = make_tiny_synaptic(seed=0, train=True)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=3e-3)
    model_b.load_state_dict(ckpt_model)
    opt_b.load_state_dict(ckpt_opt)
    restore_rng_state(ckpt_rng)
    traj_resumed = [
        _train_step(model_b, opt_b, *pool[(n_warm + i) % len(pool)]) for i in range(n_after)
    ]

    assert traj_resumed == traj_uninterrupted, (
        "resumed trajectory must be bit-identical to the uninterrupted run\n"
        f"  uninterrupted: {traj_uninterrupted}\n  resumed:       {traj_resumed}"
    )


@pytest.mark.e2e
def test_train_state_roundtrips_through_disk(tmp_path):
    """save_checkpoint(train_state=...) -> load_checkpoint(...) restores torch+python+numpy RNG.

    The train-state blob is tensor-encoded (numpy key array as a tensor) so it loads under
    the safe weights_only=True default; this exercises all three RNG streams through disk.
    """
    import random

    import numpy as np

    d = str(tmp_path)
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    # advance each stream so the captured state is non-initial
    [random.random() for _ in range(3)]
    np.random.rand(3)
    _ = torch.randn(3)
    rng = capture_rng_state()
    expected_py = [random.random() for _ in range(4)]
    expected_np = np.random.rand(4).tolist()
    expected_tc = torch.randn(4)

    save_checkpoint(d, 3, {"w": torch.zeros(1)}, {"o": torch.zeros(1)},
                    {"model_config": {}}, rank=0, train_state={"rng": rng, "step": 3})
    _, _, _, train_state = load_checkpoint(
        d, 3, torch.device("cpu"), load_optimizer=True, load_train_state=True
    )
    assert train_state is not None and train_state["step"] == 3
    restore_rng_state(train_state["rng"])
    assert [random.random() for _ in range(4)] == expected_py, "python RNG must round-trip"
    assert np.random.rand(4).tolist() == expected_np, "numpy RNG must round-trip"
    assert torch.equal(torch.randn(4), expected_tc), "torch RNG must round-trip"
