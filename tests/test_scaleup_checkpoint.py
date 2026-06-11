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

from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.checkpoint_manager import (
    capture_rng_state,
    list_checkpoint_steps,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
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


def test_rng_capture_restore_is_reproducible():
    torch.manual_seed(0)
    _ = torch.randn(10)  # advance RNG
    state = capture_rng_state()
    a = torch.randn(5)
    restore_rng_state(state)
    b = torch.randn(5)
    assert torch.equal(a, b), "restored RNG must reproduce the same draws"


def test_prune_keeps_last_k_and_best(tmp_path):
    d = str(tmp_path)
    for s in (10, 20, 30, 40, 50):
        save_checkpoint(d, s, {"w": torch.zeros(2)}, {"o": torch.zeros(1)},
                        {"model_config": {}}, rank=0, train_state={"rng": capture_rng_state()})
    assert list_checkpoint_steps(d) == [10, 20, 30, 40, 50]
    pruned = prune_checkpoints(d, keep_last=2, best_step=10, rank=0)
    assert set(pruned) == {20, 30}
    assert list_checkpoint_steps(d) == [10, 40, 50]
    # The pruned steps' optim/meta/train artifacts are gone too.
    for s in (20, 30):
        assert not os.path.exists(os.path.join(d, f"optim_{s:06d}_rank0.pt"))
        assert not os.path.exists(os.path.join(d, f"train_{s:06d}_rank0.pt"))
    # Kept steps retain all artifacts.
    for s in (10, 40, 50):
        assert os.path.exists(os.path.join(d, f"meta_{s:06d}.json"))


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
    """save_checkpoint(train_state=...) -> load_checkpoint(load_train_state=True) restores RNG."""
    d = str(tmp_path)
    torch.manual_seed(7)
    _ = torch.randn(3)
    rng = capture_rng_state()
    expected = torch.randn(4)
    save_checkpoint(d, 3, {"w": torch.zeros(1)}, {"o": torch.zeros(1)},
                    {"model_config": {}}, rank=0, train_state={"rng": rng, "step": 3})
    _, _, _, train_state = load_checkpoint(
        d, 3, torch.device("cpu"), load_optimizer=True, load_train_state=True
    )
    assert train_state is not None and train_state["step"] == 3
    restore_rng_state(train_state["rng"])
    assert torch.equal(torch.randn(4), expected), "RNG restored from disk must reproduce draws"
