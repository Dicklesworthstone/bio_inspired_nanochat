"""Tests for Persistent Lifelong Synaptic Memory System (bead `re4e.4`)."""

import math

import pytest
import torch

from bio_inspired_nanochat import persistent_memory as persistent_memory_module
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.persistent_memory import PersistentLifelongMemoryManager
from bio_inspired_nanochat.synaptic import SynapticLinear


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    return GPTSynaptic(cfg)


def test_cross_session_memory_persistence_and_consolidation(tmp_path):
    """Verify that facts learned in Session A persist into Session B across offline sleep consolidation."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()

    # Session A: Mount Alice
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None

    # Alice learns an association
    syn_lin.w_fast.data.fill_(0.4)

    # Session A End: Unmount & Sleep Consolidation
    manager.unmount_user(model, "alice", consolidate=True, consolidation_lr=0.5)

    # Fast weights must be cleared in live model
    assert syn_lin.w_fast.norm().item() == pytest.approx(0.0, abs=1e-6)

    # Session B: Fresh Model mounts Alice
    model_b = _make_model()
    syn_lin_b = next(mod for mod in model_b.modules() if isinstance(mod, SynapticLinear))
    w_slow_orig = syn_lin_b.w_slow.data.clone()

    loaded = manager.mount_user(model_b, "alice")
    assert loaded

    # Slow weights in Session B must contain the consolidated delta
    assert not torch.equal(syn_lin_b.w_slow.data, w_slow_orig)


def test_multi_user_isolation(tmp_path):
    """Verify strict isolation between distinct user memory partitions."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()

    # Alice session
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(1.0)
    manager.unmount_user(model, "alice", consolidate=True)

    # Bob session
    model_bob = _make_model()
    bob_loaded = manager.mount_user(model_bob, "bob")
    assert not bob_loaded  # Bob has no prior memory

    syn_lin_bob = next(mod for mod in model_bob.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin_bob.w_fast is not None
    # Bob's model must not contain Alice's fast weights
    assert syn_lin_bob.w_fast.norm().item() == pytest.approx(0.0, abs=1e-6)


def test_right_to_be_forgotten(tmp_path):
    """Verify that forget_user deletes disk artifacts and purges working memory."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()

    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.5)
    manager.unmount_user(model, "alice", consolidate=True)

    # Confirm file exists
    assert len(list(tmp_path.glob("mem_*.pt"))) == 1

    # Forget Alice
    erased = manager.forget_user("alice", model=model)
    assert erased
    assert len(list(tmp_path.glob("mem_*.pt"))) == 0


def test_unmount_restores_base_model_weights(tmp_path):
    """Verify that unmounting a user completely restores the model's base slow weights."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))

    w_slow_pristine = syn_lin.w_slow.data.clone()

    # Mount user Alice who has consolidated memory
    manager.mount_user(model, "alice")
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.8)
    manager.unmount_user(model, "alice", consolidate=True, consolidation_lr=0.5)

    # Base model after unmount must match pristine slow weights
    assert torch.allclose(syn_lin.w_slow.data, w_slow_pristine, atol=1e-6)

    # Re-mount Alice on same model instance -> deltas apply
    manager.mount_user(model, "alice")
    assert not torch.allclose(syn_lin.w_slow.data, w_slow_pristine, atol=1e-6)

    # Unmount Alice again -> returns to pristine
    manager.unmount_user(model, "alice", consolidate=False)
    assert torch.allclose(syn_lin.w_slow.data, w_slow_pristine, atol=1e-6)

    # Partition inspection
    part = manager.load_partition("alice")
    assert part is not None
    assert part.user_id == "alice"


def test_unmount_rejects_cross_user_state_leakage(tmp_path):
    """An inactive user cannot receive or clear the active user's live fast weights."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")

    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.75)
    fast_before = syn_lin.w_fast.detach().clone()

    with pytest.raises(ValueError, match="active user is 'alice'"):
        manager.unmount_user(model, "bob", consolidate=True)

    assert manager.active_user == "alice"
    assert torch.equal(syn_lin.w_fast, fast_before)
    assert manager.load_partition("bob") is None


def test_forget_active_user_requires_model_to_purge_live_state(tmp_path):
    """Erasure cannot report disk success while leaving the active model personalized."""
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")

    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.5)
    manager.unmount_user(model, "alice", consolidate=True)
    assert manager.mount_user(model, "alice")
    syn_lin.w_fast.data.fill_(0.5)

    with pytest.raises(ValueError, match="model is required"):
        manager.forget_user("alice")

    assert manager.active_user == "alice"
    assert torch.count_nonzero(syn_lin.w_fast) > 0
    assert manager.load_partition("alice") is not None


def test_switching_models_unmounts_the_model_that_actually_owns_live_state(tmp_path):
    manager = PersistentLifelongMemoryManager(storage_dir=tmp_path)
    alice_model = _make_model()
    bob_model = _make_model()
    manager.mount_user(alice_model, "alice")
    alice_layer = next(
        mod for mod in alice_model.modules() if isinstance(mod, SynapticLinear)
    )
    bob_layer = next(mod for mod in bob_model.modules() if isinstance(mod, SynapticLinear))
    assert alice_layer.w_fast is not None
    assert bob_layer.w_fast is not None
    alice_layer.w_fast.data.fill_(0.75)

    with pytest.raises(ValueError, match="other than the active model"):
        manager.unmount_user(bob_model, "alice", consolidate=False)
    assert torch.all(alice_layer.w_fast == 0.75)

    assert not manager.mount_user(bob_model, "bob")
    assert manager.active_user == "bob"
    assert torch.count_nonzero(alice_layer.w_fast) == 0
    alice_partition = manager.load_partition("alice")
    assert alice_partition is not None
    assert any(torch.all(weight == 0.75) for weight in alice_partition.fast_weights.values())
    assert torch.count_nonzero(bob_layer.w_fast) == 0


def test_persistent_memory_rejects_invalid_norm_and_consolidation_rate(tmp_path):
    for invalid_norm in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="max_delta_norm"):
            PersistentLifelongMemoryManager(tmp_path, max_delta_norm=invalid_norm)

    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    with pytest.raises(ValueError, match="consolidation_lr"):
        manager.unmount_user(model, "alice", consolidation_lr=math.nan)
    assert manager.active_user == "alice"

    for invalid_user in ("", "   "):
        with pytest.raises(ValueError, match="user_id"):
            manager.load_partition(invalid_user)


def test_invalid_mount_does_not_unmount_the_active_user(tmp_path):
    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.25)

    with pytest.raises(ValueError, match="user_id"):
        manager.mount_user(model, "   ")

    assert manager.active_user == "alice"
    assert torch.all(syn_lin.w_fast == 0.25)
    assert manager.load_partition("alice") is None


def test_corrupt_destination_partition_does_not_unmount_active_user(tmp_path):
    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.25)
    torch.save(
        {"slow_deltas": {0: torch.full_like(syn_lin.w_slow, math.nan)}},
        manager._user_path("bob"),
    )

    with pytest.raises(ValueError, match="finite floating tensors"):
        manager.mount_user(model, "bob")

    assert manager.active_user == "alice"
    assert torch.all(syn_lin.w_fast == 0.25)


def test_unmount_storage_failure_preserves_live_session(tmp_path, monkeypatch):
    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.5)

    def fail_save(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(persistent_memory_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="storage unavailable"):
        manager.unmount_user(model, "alice", consolidate=True)

    assert manager.active_user == "alice"
    assert manager._active_model is model
    assert torch.all(syn_lin.w_fast == 0.5)


def test_interrupted_partition_save_preserves_previous_durable_copy(tmp_path, monkeypatch):
    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.25)
    manager.unmount_user(model, "alice", consolidate=False)

    assert manager.mount_user(model, "alice")
    syn_lin.w_fast.data.fill_(0.75)

    def partial_save_then_fail(_data, path):
        path.write_bytes(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(persistent_memory_module.torch, "save", partial_save_then_fail)
    with pytest.raises(OSError, match="interrupted write"):
        manager.unmount_user(model, "alice", consolidate=False)

    partition = manager.load_partition("alice")
    assert partition is not None
    assert any(torch.all(weight == 0.25) for weight in partition.fast_weights.values())
    assert manager.active_user == "alice"
    assert torch.all(syn_lin.w_fast == 0.75)


def test_zero_consolidation_rate_preserves_fast_memory(tmp_path):
    manager = PersistentLifelongMemoryManager(tmp_path)
    model = _make_model()
    manager.mount_user(model, "alice")
    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.4)

    manager.unmount_user(model, "alice", consolidate=True, consolidation_lr=0.0)

    partition = manager.load_partition("alice")
    assert partition is not None
    assert any(torch.all(weight == 0.4) for weight in partition.fast_weights.values())


def test_same_user_cross_model_remount_fails_without_unmounting(tmp_path):
    manager = PersistentLifelongMemoryManager(tmp_path)
    active_model = _make_model()
    other_model = _make_model()
    manager.mount_user(active_model, "alice")
    active_layer = next(
        mod for mod in active_model.modules() if isinstance(mod, SynapticLinear)
    )
    assert active_layer.w_fast is not None
    active_layer.w_fast.data.fill_(0.6)

    with pytest.raises(ValueError, match="different model"):
        manager.mount_user(other_model, "alice")

    assert manager.active_user == "alice"
    assert manager._active_model is active_model
    assert torch.all(active_layer.w_fast == 0.6)
    assert manager.load_partition("alice") is None
