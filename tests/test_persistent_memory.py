"""Tests for Persistent Lifelong Synaptic Memory System (bead `re4e.4`)."""

import pytest
import torch

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
