"""
Chunked truncated BPTT and exact-gradient checkpoint/replay for the differentiable synaptic
recurrence (beads yw9.2.3 / 0642.1.2.6).

`chunked_recurrence` runs the differentiable presyn recurrence over a sequence of steps, detaching
the carried state every ``chunk_len`` steps. This truncates backprop to within a chunk (bounding
peak memory to ``chunk_len`` steps instead of the whole sequence) while leaving the forward values
untouched — so the differentiable recurrence (yw9.2) becomes usable on long sequences during
training. Detaching changes only the gradient graph, never the values.

These tests lock: gradient flows WITHIN a chunk and is cut at chunk boundaries; full BPTT
(``chunk_len=0``) flows across all steps; forward values are identical for any chunk length; and
non-reentrant checkpoint windows preserve forward state, gradients, EMA, and telemetry while
materially reducing tensors retained for backward.

Run:  pytest tests/test_chunked_tbptt.py -v
"""

from __future__ import annotations

import copy

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
    chunked_recurrence,
)

B, H, T_KEY, K, T, N = 1, 2, 6, 3, 4, 6


def _setup():
    cfg = SynapticConfig(enable_presyn=True)
    presyn = SynapticPresyn(d_head=8, cfg=cfg)
    g = torch.Generator().manual_seed(2)
    drives = [
        (torch.randn(B, H, T, K, generator=g, dtype=torch.float64) * 0.4 + 0.5)
        for _ in range(N)
    ]
    idxs = [torch.randint(0, T_KEY, (B, H, T, K), generator=g) for _ in range(N)]
    return presyn, cfg, drives, idxs


def _run(presyn, cfg, drives, idxs, chunk_len):
    st = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
    presyn.ema_e.fill_(1.0)
    return chunked_recurrence(presyn, st, drives, idxs, chunk_len=chunk_len)


def _grad(outs, drives, j, i):
    g, = torch.autograd.grad(outs[j].sum(), drives[i], retain_graph=True, allow_unused=True)
    return 0.0 if g is None else g.abs().sum().item()


def _state_tensors(state):
    tensors = []
    for value in state.values():
        tensors.extend(value if isinstance(value, list) else [value])
    return tensors


# --------------------------------------------------------------------------- #
# 1. Gradient is TRUNCATED at chunk boundaries, flows WITHIN a chunk
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradient_truncated_across_chunk_boundary():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=3)  # chunks {0,1,2}, {3,4,5}

    assert _grad(outs, drives, 2, 0) > 0, "drive0 -> out2 (same chunk) must carry gradient"
    assert _grad(outs, drives, 5, 3) > 0, "drive3 -> out5 (same chunk) must carry gradient"
    assert _grad(outs, drives, 3, 3) > 0, "same-step dependence is always differentiable"
    assert _grad(outs, drives, 3, 0) == 0.0, "drive0 -> out3 must be CUT by the chunk-3 detach"
    assert _grad(outs, drives, 5, 2) == 0.0, "drive2 -> out5 must be cut across the boundary"


@pytest.mark.unit
def test_configurable_chunk_length_moves_the_boundary():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=2)  # chunks {0,1},{2,3},{4,5}

    assert _grad(outs, drives, 1, 0) > 0, "within chunk {0,1}"
    assert _grad(outs, drives, 2, 1) == 0.0, "cut at the t=2 boundary"
    assert _grad(outs, drives, 3, 2) > 0, "within chunk {2,3}"


# --------------------------------------------------------------------------- #
# 2. Full BPTT (chunk_len=0) flows across the whole sequence
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_chunk_len_zero_is_full_bptt():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=0)
    assert _grad(outs, drives, 5, 0) > 0, "with no truncation, drive0 must reach out5"


# --------------------------------------------------------------------------- #
# 3. Forward values are identical regardless of chunk length (detach ⇒ grad-only)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forward_values_independent_of_chunk_length():
    presyn, cfg, drives, idxs = _setup()
    drives = [d.detach() for d in drives]
    with torch.no_grad():
        full = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=0))
        chunk3 = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=3))
        chunk1 = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=1))
    assert torch.equal(full, chunk3) and torch.equal(full, chunk1), "chunking must not change values"


# --------------------------------------------------------------------------- #
# 4. Non-reentrant checkpoint/replay preserves full BPTT and persistent effects.
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_checkpoint_replay_matches_eager_state_gradients_rng_and_runtime_buffers():
    cfg = SynapticConfig(
        enable_presyn=True,
        learnable_kinetics=True,
        differentiable_recurrence=True,
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
    )
    torch.manual_seed(1)
    eager_presyn = SynapticPresyn(d_head=8, cfg=cfg).double()
    checkpoint_presyn = copy.deepcopy(eager_presyn)
    generator = torch.Generator().manual_seed(2)
    base_drives = [
        torch.randn(B, H, T, K, generator=generator, dtype=torch.float64) * 0.4 + 0.5
        for _ in range(N)
    ]
    idxs = [torch.randint(0, T_KEY, (B, H, T, K), generator=generator) for _ in range(N)]

    def run(presyn, checkpoint_len):
        drives = [drive.clone().requires_grad_() for drive in base_drives]
        state = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
        state["C"][..., 0] = -5.0  # one planted domain breach exercises mixed fallback replay
        torch.manual_seed(22)
        outputs = chunked_recurrence(
            presyn,
            state,
            drives,
            idxs,
            chunk_len=0,
            checkpoint_len=checkpoint_len,
            train=True,
        )
        return drives, state, outputs

    eager_drives, eager_state, eager_outputs = run(eager_presyn, 0)
    checkpoint_drives, checkpoint_state, checkpoint_outputs = run(checkpoint_presyn, 2)
    for eager, replayed in zip(eager_outputs, checkpoint_outputs):
        assert torch.equal(eager, replayed)
    for eager, replayed in zip(_state_tensors(eager_state), _state_tensors(checkpoint_state)):
        assert torch.equal(eager, replayed)
    assert torch.equal(eager_presyn.ema_e, checkpoint_presyn.ema_e)
    assert eager_presyn.get_metriplectic_metrics() == checkpoint_presyn.get_metriplectic_metrics()
    metrics = checkpoint_presyn.get_metriplectic_metrics()
    assert 0 < metrics["fallbacks"] < metrics["steps"]
    cross_window_grad, = torch.autograd.grad(
        checkpoint_outputs[-1].sum(),
        checkpoint_drives[0],
        retain_graph=True,
    )
    assert cross_window_grad.abs().sum().item() > 0

    eager_loss = sum(output.sum() for output in eager_outputs) + sum(
        tensor.sum() for tensor in _state_tensors(eager_state)
    )
    checkpoint_loss = sum(output.sum() for output in checkpoint_outputs) + sum(
        tensor.sum() for tensor in _state_tensors(checkpoint_state)
    )
    eager_loss.backward()
    runtime_after_forward = (
        checkpoint_presyn.ema_e.clone(),
        checkpoint_presyn.metriplectic_steps.clone(),
        checkpoint_presyn.metriplectic_fallbacks.clone(),
        checkpoint_presyn.metriplectic_last_energy_drift.clone(),
        checkpoint_presyn.metriplectic_last_entropy_production.clone(),
        checkpoint_presyn.metriplectic_last_free_energy_delta.clone(),
    )
    torch.manual_seed(999)
    rng_before_backward = torch.get_rng_state().clone()
    checkpoint_loss.backward()

    for eager, replayed in zip(eager_drives, checkpoint_drives):
        assert torch.equal(eager.grad, replayed.grad)
    for eager, replayed in zip(eager_presyn.parameters(), checkpoint_presyn.parameters()):
        assert torch.equal(eager.grad, replayed.grad)
    runtime_after_backward = (
        checkpoint_presyn.ema_e,
        checkpoint_presyn.metriplectic_steps,
        checkpoint_presyn.metriplectic_fallbacks,
        checkpoint_presyn.metriplectic_last_energy_drift,
        checkpoint_presyn.metriplectic_last_entropy_production,
        checkpoint_presyn.metriplectic_last_free_energy_delta,
    )
    for before, after in zip(runtime_after_forward, runtime_after_backward):
        assert torch.equal(before, after), "backward replay must not duplicate persistent effects"
    assert torch.equal(rng_before_backward, torch.get_rng_state())


@pytest.mark.unit
def test_checkpoint_replay_materially_reduces_saved_tensor_storage():
    cfg = SynapticConfig(
        enable_presyn=True,
        learnable_kinetics=True,
        differentiable_recurrence=True,
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
    )

    def saved_bytes(checkpoint_len):
        presyn = SynapticPresyn(d_head=8, cfg=cfg).double()
        generator = torch.Generator().manual_seed(2)
        drives = [
            (
                torch.randn(B, H, T, K, generator=generator, dtype=torch.float64) * 0.4
                + 0.5
            ).requires_grad_()
            for _ in range(16)
        ]
        idxs = [
            torch.randint(0, T_KEY, (B, H, T, K), generator=generator)
            for _ in drives
        ]
        state = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
        nominal = 0
        storages = {}

        def pack(tensor):
            nonlocal nominal
            nominal += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            outputs = chunked_recurrence(
                presyn,
                state,
                drives,
                idxs,
                chunk_len=0,
                checkpoint_len=checkpoint_len,
                train=False,
            )
            loss = sum(output.sum() for output in outputs) + sum(
                tensor.sum() for tensor in _state_tensors(state)
            )
        loss.backward()
        return nominal, sum(storages.values())

    eager = saved_bytes(0)
    checkpointed = saved_bytes(4)

    assert checkpointed[0] <= eager[0] / 8
    assert checkpointed[1] <= eager[1] / 8


@pytest.mark.unit
def test_checkpoint_runtime_rejects_unsupported_sampling_modes():
    cfg = SynapticConfig(
        enable_presyn=True,
        learnable_kinetics=True,
        differentiable_recurrence=True,
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
    )
    presyn = SynapticPresyn(d_head=8, cfg=cfg).double()
    setattr(presyn, "_mc_sampling", True)
    drive = torch.ones(B, H, T, K, dtype=torch.float64, requires_grad=True)
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    state = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)

    with pytest.raises(ValueError, match="MC release sampling"):
        chunked_recurrence(
            presyn,
            state,
            [drive],
            [idx],
            chunk_len=0,
            checkpoint_len=1,
            train=True,
        )


@pytest.mark.unit
def test_checkpoint_runtime_rejects_low_precision_until_replay_parity_exists():
    cfg = SynapticConfig(
        enable_presyn=True,
        learnable_kinetics=True,
        differentiable_recurrence=True,
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
    )
    presyn = SynapticPresyn(d_head=8, cfg=cfg).to(torch.bfloat16)
    drive = torch.ones(B, H, T, K, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    state = build_presyn_state(B, T_KEY, H, "cpu", torch.bfloat16, cfg)

    with pytest.raises(TypeError, match="float32 or float64"):
        chunked_recurrence(
            presyn,
            state,
            [drive],
            [idx],
            chunk_len=0,
            checkpoint_len=1,
            train=True,
        )


@pytest.mark.unit
def test_checkpoint_runtime_upgrades_legacy_heat_and_rejects_mixed_state_dtype():
    cfg = SynapticConfig(
        enable_presyn=True,
        learnable_kinetics=True,
        differentiable_recurrence=True,
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
    )
    presyn = SynapticPresyn(d_head=8, cfg=cfg).double()
    drive = torch.ones(B, H, T, K, dtype=torch.float64, requires_grad=True)
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    legacy_state = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
    del legacy_state["HEAT"]

    outputs = chunked_recurrence(
        presyn,
        legacy_state,
        [drive],
        [idx],
        chunk_len=0,
        checkpoint_len=1,
        train=True,
    )
    assert "HEAT" in legacy_state
    outputs[0].sum().backward()
    assert drive.grad is not None

    mixed_state = build_presyn_state(B, T_KEY, H, "cpu", torch.float32, cfg)
    with pytest.raises(TypeError, match="state dtype"):
        chunked_recurrence(
            presyn,
            mixed_state,
            [drive.detach().requires_grad_()],
            [idx],
            chunk_len=0,
            checkpoint_len=1,
            train=True,
        )
