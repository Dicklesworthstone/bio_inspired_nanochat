"""
Online Hebbian plasticity must run (and be autograd-safe) DURING TRAINING — bead vg9.2.

The #1 correctness gap: all online plasticity (eligibility traces, post.update, w_fast/w_slow
Hebbian writes, hebb_fast/consolidate) was gated behind `if update_mem and not
torch.is_grad_enabled():`. Since grad is enabled during training, the project's headline
"online Hebbian learning" NEVER ran at train time — only under inference/no_grad.

The subtlety the naive fix misses: w_fast/w_slow (SynapticLinear) and post.fast/post.slow
(PostsynapticHebb) are nn.Parameters used in the forward matmuls. Mutating them in place after
the matmul saved them for backward raises "a variable needed for gradient computation has been
modified by an inplace operation". So the fix runs plasticity during training but DEFERS the
four Parameter writes to the top of the NEXT forward (applied before those Parameters are
reused), driven from the persisted eligibility traces (which init to zero -> first apply is a
no-op). Buffer-only updates (traces, CaMKII/PP1/BDNF) happen immediately and are always safe.

These tests lock: (1) plasticity executes with grad enabled, (2) backward + optimizer.step do
NOT crash, (3) deferral applies exactly once (no double-counting), (4) the inference path is
preserved byte-for-byte, (5) the opt-out flag restores legacy inert-during-training behavior.

Run:  pytest tests/test_hebbian_training_plasticity.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import make_tiny_synaptic, random_tokens, set_seed

IN, OUT, B = 16, 16, 4


def _make_lin(**cfg_overrides) -> SynapticLinear:
    set_seed(0)
    cfg = SynapticConfig(enable_hebbian=True, enable_metabolism=True, **cfg_overrides)
    return SynapticLinear(IN, OUT, cfg)


def _signals():
    return torch.ones(B), torch.ones(B)  # calcium, energy (per-row)


# --------------------------------------------------------------------------- #
# 1. THE BUG: plasticity must execute during a grad-enabled (training) forward
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_traces_and_camkii_change_during_training_forward():
    lin = _make_lin().train()
    x = torch.randn(B, IN, requires_grad=True)
    ca, en = _signals()

    u0 = lin.u_buf.clone()
    pp1_0 = lin.post.pp1.clone()
    assert torch.is_grad_enabled(), "this test must run with grad ENABLED (training condition)"

    lin(x, ca, en)

    # Buffer-side plasticity runs immediately even with grad enabled (this was the dead path).
    # (CaMKII only potentiates above camkii_thr; sub-threshold activity exercises the PP1
    # down-regulation path instead, which is the reliable witness that post.update ran.)
    assert not torch.equal(lin.u_buf, u0), "eligibility trace must update during training"
    assert not torch.equal(lin.post.pp1, pp1_0), "PP1/CaMKII machinery must update during training"


@pytest.mark.unit
def test_weight_writes_are_deferred_then_applied_next_forward():
    lin = _make_lin().train()
    x = torch.randn(B, IN)
    ca, en = _signals()
    w_slow0 = lin.w_slow.detach().clone()
    fast0 = lin.post.fast.detach().clone()

    # Step 1 (training): weight writes are DEFERRED -> w_slow unchanged, pending flagged.
    lin(x, ca, en)
    assert lin._plasticity_pending is True, "training step must flag a deferred write"
    assert torch.equal(lin.w_slow, w_slow0), "Parameter write must be deferred, not applied now"
    assert torch.equal(lin.post.fast, fast0), "post.fast write must be deferred too"

    # Step 2: the deferred write is flushed at the TOP, before the matmuls -> weights changed.
    lin(x, ca, en)
    assert (lin.w_slow - w_slow0).abs().sum() > 0, "deferred Hebbian write must land next forward"
    assert (lin.post.fast - fast0).abs().sum() > 0, "deferred post.fast write must land too"


# --------------------------------------------------------------------------- #
# 2. AUTOGRAD SAFETY: backward + optimizer.step must not crash (the real hazard)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_backward_does_not_crash_with_plasticity_during_training():
    lin = _make_lin().train()
    ca, en = _signals()
    opt = torch.optim.SGD(lin.parameters(), lr=1e-2)

    # Mimic gradient accumulation: forward -> backward (immediately) -> repeat -> step.
    for _ in range(4):
        x = torch.randn(B, IN, requires_grad=True)
        y = lin(x, ca, en)
        loss = y.float().pow(2).mean()
        loss.backward()  # MUST NOT raise "modified by an inplace operation"
        assert torch.isfinite(loss)
    opt.step()
    opt.zero_grad(set_to_none=True)
    # And the headline feature genuinely ran: a deferred write is pending after the last fwd.
    assert lin._plasticity_pending is True


@pytest.mark.unit
def test_grads_flow_to_slow_weights():
    lin = _make_lin().train()
    x = torch.randn(B, IN, requires_grad=True)
    ca, en = _signals()
    y = lin(x, ca, en)
    y.float().sum().backward()
    assert lin.w_slow.grad is not None and lin.w_slow.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all(), "grad_x must be finite (not corrupted)"


# --------------------------------------------------------------------------- #
# 3. NO DOUBLE-COUNTING: each step's Hebbian delta is applied exactly once
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_no_double_counting_deferred_write_applied_once():
    lin = _make_lin().train()
    x = torch.randn(B, IN)
    ca, en = _signals()

    # After step 1: traces set, write deferred (0 applications).
    lin(x, ca, en)
    w_before_flush = lin.w_slow.detach().clone()
    traces = (lin.u_buf @ lin.v_buf).clone()
    gate = lin._last_gate_scale
    gscale = gate.item() if gate is not None else 1.0
    expected_delta = lin.cfg.post_slow_lr * gscale * traces  # exactly ONE application

    # Step 2 flushes the single deferred write at the top, then updates traces again.
    lin(x, ca, en)
    applied = lin.w_slow.detach() - w_before_flush
    assert torch.allclose(applied, expected_delta, atol=1e-6), "write must be applied exactly once"


# --------------------------------------------------------------------------- #
# 4. INFERENCE PATH PRESERVED: immediate application under no_grad, no pending
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_inference_applies_immediately_within_same_forward():
    lin = _make_lin().eval()
    x = torch.randn(B, IN)
    ca, en = _signals()
    with torch.no_grad():
        w0 = lin.w_slow.clone()
        lin(x, ca, en)
    # Legacy behavior: applied within the same forward (traces were just updated), no deferral.
    assert lin._plasticity_pending is False, "inference must NOT defer (no backward pending)"
    assert (lin.w_slow - w0).abs().sum() > 0, "inference applies the Hebbian write immediately"


@pytest.mark.unit
def test_eval_with_grad_enabled_does_not_run_plasticity():
    # eval() + grad enabled: not a training step and not no_grad -> plasticity stays inert,
    # matching the original `not torch.is_grad_enabled()` semantics. Must not crash either.
    lin = _make_lin().eval()
    x = torch.randn(B, IN, requires_grad=True)
    ca, en = _signals()
    u0 = lin.u_buf.clone()
    y = lin(x, ca, en)
    y.float().sum().backward()
    assert torch.equal(lin.u_buf, u0), "eval+grad must not mutate plasticity state"
    assert lin._plasticity_pending is False


# --------------------------------------------------------------------------- #
# 5. OPT-OUT: plasticity_during_training=False restores legacy inert behavior
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_flag_off_keeps_training_inert():
    lin = _make_lin(plasticity_during_training=False).train()
    x = torch.randn(B, IN)
    ca, en = _signals()
    w0 = lin.w_slow.detach().clone()
    u0 = lin.u_buf.clone()
    y = lin(x, ca, en)
    y.float().sum().backward()
    lin(x, ca, en)  # second forward — still nothing to flush
    assert torch.equal(lin.w_slow, w0), "flag off -> no Hebbian write during training"
    assert torch.equal(lin.u_buf, u0), "flag off -> traces unchanged during training"
    assert lin._plasticity_pending is False


# --------------------------------------------------------------------------- #
# 6. END-TO-END: a tiny synaptic model trains a full step without crashing
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_tiny_model_training_step_runs_plasticity_without_crash():
    model = make_tiny_synaptic(seed=0, train=True).train()
    lins = [m for m in model.modules() if isinstance(m, SynapticLinear) and m.post is not None]
    assert lins, "tiny synaptic model must contain Hebbian SynapticLinear layers"

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = random_tokens(2, 16, vocab=model.config.vocab_size)

    # Two forward+backward steps so deferred writes get flushed on the second pass.
    for _ in range(2):
        logits, loss = model(x, targets=x)
        assert torch.isfinite(loss)
        loss.backward()  # MUST NOT crash from in-place Parameter mutation
        opt.step()
        opt.zero_grad(set_to_none=True)

    assert any(lin._plasticity_pending for lin in lins), "online plasticity must have run in training"
