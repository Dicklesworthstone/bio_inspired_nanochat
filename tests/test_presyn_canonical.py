"""
Canonical unified presynaptic release function — bead 8j9.2 / subtask 2gjl.

`SynapticPresyn.release_canonical` is the single, faithful, differentiable source of truth
that ports forward()'s biologically-faithful equations (Hill Syt(C)=C/(C+Kd), the calcium
BUFFER ODE, energy->AMPA qamp, the septin distance barrier) onto release()'s top-k,
key-indexed, differentiable scatter structure — while PRESERVING the Doc2 term, the stochastic
STE path, the endocytosis DELAY queue, and EMA normalization.

These are property + differentiability tests (additive: the live path still calls release()).
The formal golden parity vs forward() at K=T is owned by 8j9.4.

Run:  pytest tests/test_presyn_canonical.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)

from _bio_testkit import assert_finite, set_seed

DEV = torch.device("cpu")
DT = torch.float32


def _setup(*, K=4, T=8, B=2, H=4, dh=16, seed=0, requires_grad=False, **cfg_kw):
    set_seed(seed)
    cfg = SynapticConfig(**{"enable_presyn": True, **cfg_kw})
    pre = SynapticPresyn(dh, cfg)
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    drive = torch.randn(B, H, T, K, dtype=DT, requires_grad=requires_grad)
    # Causal top-k indices: each query t selects K keys from [0, t].
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    for t in range(T):
        idx[:, :, t, :] = torch.randint(0, t + 1, (B, H, K))
    return cfg, pre, state, drive, idx


# --------------------------------------------------------------------------- #
# 1. Interface parity with release(): same signature, shape, disabled-path
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_output_shape_matches_release():
    _, pre, state, drive, idx = _setup()
    e = pre.release_canonical(state, drive, idx, train=False)
    assert e.shape == drive.shape
    assert_finite(e, "canonical release")


@pytest.mark.unit
def test_disabled_presyn_returns_ones():
    _, pre, state, drive, idx = _setup(enable_presyn=False)
    e = pre.release_canonical(state, drive, idx, train=False)
    assert torch.equal(e, torch.ones_like(drive)), "disabled presyn must return 1.0 (log->0)"


@pytest.mark.unit
def test_invalid_mask_shape_raises():
    _, pre, state, drive, idx = _setup()
    with pytest.raises(ValueError, match="valid mask must match drive shape"):
        pre.release_canonical(state, drive, idx, train=False, valid=torch.ones(3, 3, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# 2. Differentiability w.r.t. the INPUT drive (the 8j9.2 scope: parity into logits)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_differentiable_wrt_drive():
    _, pre, state, drive, idx = _setup(requires_grad=True)
    e = pre.release_canonical(state, drive, idx, train=False)
    (grad,) = torch.autograd.grad(e.sum(), drive)
    assert grad is not None
    assert_finite(grad, "d e / d drive")
    assert grad.abs().sum() > 0, "release must be differentiable w.r.t. the attention drive"


@pytest.mark.unit
def test_state_recurrence_is_detached():
    # The scope boundary defers differentiable kinetics to yw9: state writes must be detached
    # so no graph is retained across timesteps. The returned bias stays attached to `drive`.
    _, pre, state, drive, idx = _setup(requires_grad=True)
    pre.release_canonical(state, drive, idx, train=False)
    for key in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        assert not state[key].requires_grad, f"state[{key}] must be detached (recurrence is not differentiable)"


# --------------------------------------------------------------------------- #
# 3. State evolves & stays finite; the BUFFER ODE is now ACTIVE (vs release())
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_state_updates_finite_and_calcium_rises():
    _, pre, state, drive, idx = _setup()
    c0 = state["C"].clone()
    pre.release_canonical(state, drive, idx, train=False)
    for key in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        assert_finite(state[key], f"state[{key}]")
    assert (state["C"] - c0).abs().sum() > 0, "calcium must rise at accessed keys"


@pytest.mark.unit
def test_buffer_ode_becomes_active_unlike_release():
    # release() carries BUF unchanged (it's ignored); the canonical drives it via the calcium
    # buffer ODE. BUF is calcium-driven, so it activates on the SECOND step (after C rises).
    cfg, pre, state, drive, idx = _setup()
    pre.release_canonical(state, drive, idx, train=False)  # step 1: C rises, BUF still ~0
    pre.release_canonical(state, drive, idx, train=False)  # step 2: BUF driven by C
    assert state["BUF"].abs().sum() > 0, "calcium BUFFER ODE must be active in the canonical fn"

    # Contrast: release() leaves BUF untouched.
    _, pre2, state2, drive2, idx2 = _setup()
    buf0 = state2["BUF"].clone()
    pre2.release(state2, drive2, idx2, train=False)
    pre2.release(state2, drive2, idx2, train=False)
    assert torch.equal(state2["BUF"], buf0), "release() must leave BUF unused (the gap we close)"


# --------------------------------------------------------------------------- #
# 4. Faithful Hill probability is a valid probability and Doc2 is preserved
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_faithful_prob_in_unit_interval_and_monotone_in_calcium():
    _, pre, _, _, _ = _setup()
    c = torch.linspace(0.0, 5.0, 50)
    pr = torch.full_like(c, 0.7)
    cl = torch.full_like(c, 0.6)
    drive = torch.full_like(c, 2.0)
    p = pre._faithful_release_prob(c, pr, cl, drive)
    assert (p >= 0).all() and (p <= 1).all(), "release probability must lie in [0,1]"
    # More calcium -> more Syt binding -> higher fusion probability (monotone non-decreasing).
    assert (p[1:] - p[:-1] >= -1e-6).all(), "release prob must be non-decreasing in calcium"


@pytest.mark.unit
def test_doc2_term_is_preserved():
    # forward() lacks Doc2; the canonical preserves release()'s Doc2 facilitation. Turning the
    # gain on must change the probability (otherwise the feature was silently dropped).
    c = torch.full((16,), 0.3)
    pr, cl, drive = (torch.full_like(c, v) for v in (0.7, 0.6, 1.0))
    _, pre_off, _, _, _ = _setup(doc2_gain=0.0)
    _, pre_on, _, _, _ = _setup(doc2_gain=0.5)
    p_off = pre_off._faithful_release_prob(c, pr, cl, drive)
    p_on = pre_on._faithful_release_prob(c, pr, cl, drive)
    assert (p_on - p_off).abs().sum() > 0, "Doc2 facilitation must affect the release probability"


# --------------------------------------------------------------------------- #
# 5. Septin distance barrier penalizes long-range edges
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_septin_barrier_penalizes_distant_edges():
    # Single query at position T-1 selecting a NEAR key (T-1) and a FAR key (0), with identical
    # drive and uniform initial state. Only the barrier differs -> near edge gets a larger bias.
    cfg = SynapticConfig(enable_presyn=True, barrier_strength=0.5, stochastic_train_frac=0.0)
    pre = SynapticPresyn(16, cfg)
    B, H, T = 1, 1, 8
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    drive = torch.full((B, H, T, 2), 2.0)
    idx = torch.zeros(B, H, T, 2, dtype=torch.long)
    idx[..., 0] = T - 1            # near key (same as the last query)
    idx[..., 1] = 0                # far key
    e = pre.release_canonical(state, drive, idx, train=False, apply_barrier=True)
    near, far = e[0, 0, T - 1, 0], e[0, 0, T - 1, 1]
    assert near > far, "septin barrier must penalize the longer-range edge"


@pytest.mark.unit
def test_default_applies_no_barrier_so_live_path_is_not_double_penalized():
    # The LIVE attention path calls release_canonical WITHOUT apply_barrier (it applies its own
    # exact logit-level barrier). Guard against regressing the default to True: with the barrier
    # off, two edges at different distances but identical state/drive get IDENTICAL release, so
    # the attention's barrier is never double-counted.
    cfg = SynapticConfig(enable_presyn=True, barrier_strength=0.5, stochastic_train_frac=0.0)
    pre = SynapticPresyn(16, cfg)
    B, H, T = 1, 1, 8
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    drive = torch.full((B, H, T, 2), 2.0)
    idx = torch.zeros(B, H, T, 2, dtype=torch.long)
    idx[..., 0] = T - 1  # near key
    idx[..., 1] = 0      # far key
    e = pre.release_canonical(state, drive, idx, train=False)  # apply_barrier defaults to False
    near, far = e[0, 0, T - 1, 0], e[0, 0, T - 1, 1]
    assert torch.allclose(near, far), "default (barrier off) must NOT penalize distance — the attention does"


@pytest.mark.unit
def test_septin_barrier_respects_query_offset():
    # q_pos lets the live path supply absolute query positions (KV-cache prefix decoding).
    # Shifting queries 10 positions away from key 0 must strengthen the barrier -> less release.
    cfg = SynapticConfig(enable_presyn=True, barrier_strength=0.5, stochastic_train_frac=0.0)
    B, H, T = 1, 1, 4
    drive = torch.full((B, H, T, 1), 2.0)
    idx = torch.zeros(B, H, T, 1, dtype=torch.long)  # every query attends key 0

    pre_a = SynapticPresyn(16, cfg)
    e_default = pre_a.release_canonical(
        build_presyn_state(B, T, H, DEV, DT, cfg), drive, idx, train=False, apply_barrier=True
    )
    pre_b = SynapticPresyn(16, cfg)
    e_offset = pre_b.release_canonical(
        build_presyn_state(B, T, H, DEV, DT, cfg), drive, idx, train=False,
        q_pos=torch.arange(T) + 10, apply_barrier=True,
    )
    assert (e_offset < e_default).all(), "larger query offset must increase the septin penalty"


# --------------------------------------------------------------------------- #
# 5b. Live-path shapes: KV-cache decode (T_query != T_key) and masked (-inf) edges
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_decode_shape_query_count_differs_from_key_extent():
    # KV-cache decode: state spans T_key keys but only T < T_key queries attend (top-k each).
    # The LIVE standard attention path hits this during generation; gather/scatter must handle
    # T != T_key. (None of the other tests exercise this — they use T == T_key.)
    cfg = SynapticConfig(enable_presyn=True)
    pre = SynapticPresyn(16, cfg)
    B, H, Tk, Tq, K = 1, 2, 10, 3, 4
    state = build_presyn_state(B, Tk, H, DEV, DT, cfg)
    drive = torch.randn(B, H, Tq, K, requires_grad=True)
    idx = torch.randint(0, Tk, (B, H, Tq, K))
    e = pre.release_canonical(state, drive, idx, train=False)
    assert e.shape == (B, H, Tq, K)
    assert state["C"].shape == (B, H, Tk), "state must keep the full key extent, not shrink to T"
    (grad,) = torch.autograd.grad(e.sum(), drive)
    assert torch.isfinite(grad).all()
    for k in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        assert_finite(state[k], f"state[{k}] (decode)")


@pytest.mark.unit
def test_neginf_drive_from_masked_topk_stays_finite():
    # Early in a sequence, top-k over masked logits can select -inf edges (valid marks them).
    # softplus/sigmoid(-inf) = 0, so the canonical must stay finite and release nothing there.
    cfg = SynapticConfig(enable_presyn=True)
    pre = SynapticPresyn(16, cfg)
    state = build_presyn_state(1, 6, 2, DEV, DT, cfg)
    drive = torch.randn(1, 2, 3, 4)
    drive[..., -1] = float("-inf")
    valid = torch.isfinite(drive)
    idx = torch.randint(0, 6, (1, 2, 3, 4))
    e = pre.release_canonical(state, drive, idx, train=False, valid=valid)
    assert torch.isfinite(e).all(), "canonical must not NaN on -inf drive from masked top-k"
    assert e[..., -1].abs().max() == 0, "masked (-inf) edges must release nothing"
    for k in ("C", "RRP", "PR", "CL", "E", "BUF"):
        assert_finite(state[k], f"state[{k}] (-inf drive)")


# --------------------------------------------------------------------------- #
# 6. Determinism + stochastic path
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_deterministic_path_is_reproducible():
    out = []
    for _ in range(2):
        _, pre, state, drive, idx = _setup(seed=7)
        out.append(pre.release_canonical(state, drive, idx, train=False))
    assert torch.equal(out[0], out[1]), "deterministic (train=False) path must be reproducible"


@pytest.mark.unit
def test_stochastic_path_runs_and_stays_finite():
    _, pre, state, drive, idx = _setup(seed=1, requires_grad=True, stochastic_train_frac=0.5)
    e = pre.release_canonical(state, drive, idx, train=True)
    assert_finite(e, "stochastic canonical release")
    (grad,) = torch.autograd.grad(e.sum(), drive, allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all(), "STE must keep grads finite"


@pytest.mark.unit
def test_valid_mask_zeros_invalid_edges():
    _, pre, state, drive, idx = _setup()
    valid = torch.ones_like(drive, dtype=torch.bool)
    valid[..., -1] = False  # mask the last selected edge of every query
    e = pre.release_canonical(state, drive, idx, train=False, valid=valid)
    # Masked edges contribute zero release -> e == 0 there (barrier/qamp multiply a zero rel).
    assert e[..., -1].abs().max() == 0, "masked edges must release nothing"
