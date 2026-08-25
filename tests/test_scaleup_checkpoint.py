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
    capture_prefetched_batch,
    capture_rng_state,
    checkpoint_model_config,
    list_checkpoint_steps,
    load_checkpoint,
    load_rank_training_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    restore_prefetched_batch,
    restore_optimizer_states,
    restore_rank_model_state,
    save_checkpoint,
    synaptic_config_to_meta,
    validate_exact_resume_payload_step,
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


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_checkpoint_apis_reject_invalid_step_before_io(tmp_path, step):
    with pytest.raises(ValueError, match="non-negative integer"):
        save_checkpoint(
            str(tmp_path),
            step,
            {"w": torch.zeros(1)},
            None,
            {"model_config": {}},
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        load_checkpoint(str(tmp_path), step, torch.device("cpu"))

    assert list(tmp_path.iterdir()) == []


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


def test_explicit_load_rejects_uncommitted_step_in_marker_regime(tmp_path):
    save_checkpoint(
        str(tmp_path),
        1,
        {"w": torch.zeros(1)},
        None,
        {"model_config": {}},
    )
    torch.save({"w": torch.ones(1)}, tmp_path / "model_000002.pt")
    (tmp_path / "meta_000002.json").write_text(
        '{"model_config": {}}',
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="incomplete or uncommitted"):
        load_checkpoint(str(tmp_path), 2, torch.device("cpu"))


def test_failed_same_step_overwrite_invalidates_old_commit_marker(tmp_path, monkeypatch):
    save_checkpoint(
        str(tmp_path),
        5,
        {"w": torch.zeros(1)},
        None,
        {"model_config": {}},
    )
    assert list_checkpoint_steps(str(tmp_path)) == [5]

    def fail_save(_obj, _path):
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(cm, "_atomic_torch_save", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        save_checkpoint(
            str(tmp_path),
            5,
            {"w": torch.ones(1)},
            None,
            {"model_config": {}},
        )

    assert list_checkpoint_steps(str(tmp_path)) == []


def test_distributed_save_invalidates_marker_before_any_rank_artifact(
    tmp_path,
    monkeypatch,
):
    events = []
    original_save = cm._atomic_torch_save

    def tracked_save(obj, path):
        events.append(("save", os.path.basename(path)))
        original_save(obj, path)

    def tracked_barrier():
        events.append(("barrier", None))

    monkeypatch.setattr(cm, "_atomic_torch_save", tracked_save)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", tracked_barrier)

    save_checkpoint(
        str(tmp_path),
        5,
        {"w": torch.zeros(1)},
        {"state": {}},
        {"model_config": {}},
        train_state={"rng": {}},
    )

    artifact_events = [kind for kind, _ in events]
    assert artifact_events == ["barrier", "save", "save", "save", "barrier"]


def test_first_marker_save_failure_preserves_only_preexisting_legacy_steps(
    tmp_path,
    monkeypatch,
):
    torch.save({"w": torch.zeros(1)}, tmp_path / "model_000003.pt")
    (tmp_path / "meta_000003.json").write_text(
        '{"model_config": {}}',
        encoding="utf-8",
    )
    assert list_checkpoint_steps(str(tmp_path)) == [3]

    def fail_save(_obj, _path):
        raise OSError("simulated first marker-era failure")

    monkeypatch.setattr(cm, "_atomic_torch_save", fail_save)
    with pytest.raises(OSError, match="first marker-era failure"):
        save_checkpoint(
            str(tmp_path),
            4,
            {"w": torch.ones(1)},
            None,
            {"model_config": {}},
        )

    assert list_checkpoint_steps(str(tmp_path)) == [3]


def test_complete_marker_fails_closed_when_declared_optimizer_shard_is_missing(tmp_path):
    save_checkpoint(
        str(tmp_path),
        8,
        {"w": torch.zeros(1)},
        {"state": {}},
        {"model_config": {}},
    )
    optimizer_path = tmp_path / "optim_000008_rank0.pt"
    optimizer_path.rename(tmp_path / "optim_000008_rank0.pt.saved")

    assert list_checkpoint_steps(str(tmp_path)) == []
    with pytest.raises(FileNotFoundError, match="incomplete or uncommitted"):
        load_checkpoint(str(tmp_path), 8, torch.device("cpu"))


def test_malformed_commit_marker_schema_fails_closed(tmp_path):
    save_checkpoint(
        str(tmp_path),
        6,
        {"w": torch.zeros(1)},
        None,
        {"model_config": {}},
    )
    (tmp_path / "commit_000006.json").write_text(
        '{"version": 1, "step": 6, "complete": true, "optimizer_shards": "yes"}',
        encoding="utf-8",
    )

    assert list_checkpoint_steps(str(tmp_path)) == []
    with pytest.raises(FileNotFoundError, match="incomplete or uncommitted"):
        load_checkpoint(str(tmp_path), 6, torch.device("cpu"))


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


def test_capture_rng_state_does_not_silently_omit_numpy(monkeypatch):
    import numpy as np

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("simulated NumPy RNG capture failure")

    monkeypatch.setattr(np.random, "get_state", fail_capture)

    with pytest.raises(RuntimeError, match="simulated NumPy RNG capture failure"):
        capture_rng_state()


def test_restore_rng_state_rejects_cuda_checkpoint_before_mutating_cpu_rng(monkeypatch):
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    replacement = torch.Generator().manual_seed(999).get_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        restore_rng_state({"torch": replacement, "cuda": [torch.zeros(8, dtype=torch.uint8)]})

    assert torch.equal(torch.get_rng_state(), state_before)


def test_restore_rng_state_rejects_missing_cuda_payload_on_cuda_host(monkeypatch):
    torch.manual_seed(123)
    state_before = torch.get_rng_state().clone()
    replacement = torch.Generator().manual_seed(999).get_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="no CUDA RNG payload"):
        restore_rng_state({"torch": replacement})

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


def test_rank_training_checkpoint_loads_only_optimizer_and_train_shards(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    save_checkpoint(
        str(tmp_path),
        4,
        {"model": torch.ones(1)},
        {"optimizer": torch.arange(2)},
        {"step": 4, "model_config": {}},
        train_state={"step": 4, "rng": {}},
    )

    original_load = torch.load
    loaded_paths = []

    def recording_load(path, *args, **kwargs):
        loaded_paths.append(os.path.basename(os.fspath(path)))
        if os.path.basename(os.fspath(path)).startswith("model_"):
            raise AssertionError("rank-only restore must not reload model weights")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    optimizer_state, metadata, train_state = load_rank_training_checkpoint(
        str(tmp_path),
        4,
        "cpu",
        rank=0,
        expected_world_size=1,
    )

    assert loaded_paths == ["optim_000004_rank0.pt", "train_000004_rank0.pt"]
    assert torch.equal(optimizer_state["optimizer"], torch.arange(2))
    assert metadata["step"] == train_state["step"] == 4


def test_rank_training_checkpoint_rejects_world_size_before_loading_shards(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    save_checkpoint(
        str(tmp_path),
        5,
        {"model": torch.ones(1)},
        {"optimizer": torch.arange(2)},
        {"step": 5, "model_config": {}},
        train_state={"step": 5},
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("world-size mismatch must fail before loading shards")
        ),
    )

    with pytest.raises(ValueError, match="world size"):
        load_rank_training_checkpoint(
            str(tmp_path),
            5,
            "cpu",
            rank=0,
            expected_world_size=2,
        )


def test_prefetched_batch_roundtrips_through_rank_local_train_state(tmp_path):
    inputs = torch.arange(12, dtype=torch.long).view(3, 4)
    targets = inputs + 1
    save_checkpoint(
        str(tmp_path),
        7,
        {"w": torch.zeros(1)},
        {"o": torch.zeros(1)},
        {"model_config": {}},
        train_state={
            "dataloader_state_dict": {"version": 2, "pq_idx": 0, "rg_idx": 0},
            "prefetched_batch": capture_prefetched_batch(inputs, targets),
        },
    )

    _, _, _, train_state = load_checkpoint(
        str(tmp_path),
        7,
        torch.device("cpu"),
        load_optimizer=True,
        load_train_state=True,
    )
    restored_inputs, restored_targets = restore_prefetched_batch(
        train_state,
        device="cpu",
        expected_shape=(3, 4),
    )

    torch.testing.assert_close(restored_inputs, inputs)
    torch.testing.assert_close(restored_targets, targets)


def test_exact_resume_rejects_checkpoint_without_prefetched_batch():
    with pytest.raises(ValueError, match="prefetched batch"):
        restore_prefetched_batch(
            {"dataloader_state_dict": {"version": 2, "pq_idx": 0, "rg_idx": 0}},
            device="cpu",
            expected_shape=(2, 8),
        )


def test_optimizer_restore_rejects_count_mismatch_before_mutating_any_optimizer():
    class TrackingOptimizer:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state):
            self.loaded.append(state)

    optimizers = [TrackingOptimizer(), TrackingOptimizer()]

    with pytest.raises(ValueError, match="optimizer count"):
        restore_optimizer_states(optimizers, [{"state": {}, "param_groups": []}])

    assert all(optimizer.loaded == [] for optimizer in optimizers)


def test_optimizer_restore_loads_every_saved_state():
    class TrackingOptimizer:
        def __init__(self):
            self.loaded = []

        def load_state_dict(self, state):
            self.loaded.append(state)

    optimizers = [TrackingOptimizer(), TrackingOptimizer()]
    states = [
        {"state": {"a": 1}, "param_groups": []},
        {"state": {"b": 2}, "param_groups": []},
    ]

    restore_optimizer_states(optimizers, states)

    assert optimizers[0].loaded == [states[0]]
    assert optimizers[1].loaded == [states[1]]


def test_rank_local_model_state_overlays_shared_checkpoint_without_replacing_device():
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    original_parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}

    rank_model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        rank_model.weight.fill_(2.0)
        rank_model.bias.fill_(-1.0)

    restore_rank_model_state(
        model,
        {"rank_model_state": rank_model.state_dict()},
        required=True,
    )

    torch.testing.assert_close(model.weight, rank_model.weight)
    torch.testing.assert_close(model.bias, rank_model.bias)
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == original_parameter_ids


def test_distributed_exact_resume_rejects_missing_rank_local_model_state():
    with pytest.raises(ValueError, match="rank-local model state"):
        restore_rank_model_state(torch.nn.Linear(2, 2), {}, required=True)


def test_single_rank_resume_accepts_missing_rank_local_model_state():
    model = torch.nn.Linear(2, 2)
    before = copy.deepcopy(model.state_dict())

    restore_rank_model_state(model, {}, required=False)

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


@pytest.mark.parametrize("saved_step", [None, True, 6, 7.0])
def test_exact_resume_rejects_missing_or_wrong_payload_step(saved_step):
    payload = {} if saved_step is None else {"step": saved_step}

    with pytest.raises(ValueError, match="does not match requested checkpoint"):
        validate_exact_resume_payload_step(
            payload,
            7,
            payload_name="rank-local train state",
        )


def test_exact_resume_accepts_matching_payload_step():
    validate_exact_resume_payload_step(
        {"step": 7},
        7,
        payload_name="checkpoint metadata",
    )


def test_divergence_guard_snapshot_roundtrips_under_safe_checkpoint_loading(tmp_path):
    from bio_inspired_nanochat.divergence_guard import (
        DivergenceGuard,
        DivergenceGuardConfig,
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    guard = DivergenceGuard(
        DivergenceGuardConfig(enable_rollback=True, snapshot_every=1)
    )
    guard.check(torch.tensor(2.0), model, step=0)
    guard.maybe_snapshot(model, optimizer, step=0)
    save_checkpoint(
        str(tmp_path),
        9,
        model.state_dict(),
        optimizer.state_dict(),
        {"model_config": {}},
        train_state={"divergence_guard": guard.state_dict()},
    )

    _, _, _, train_state = load_checkpoint(
        str(tmp_path),
        9,
        torch.device("cpu"),
        load_optimizer=True,
        load_train_state=True,
    )
    restored = DivergenceGuard()
    restored.load_state_dict(train_state["divergence_guard"])

    assert restored.can_rollback()
    assert restored.state_dict()["loss_ema"] == 2.0
