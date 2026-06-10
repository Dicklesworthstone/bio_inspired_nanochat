"""
Numerical-stability & invariance suite for the synaptic dynamics — bead vg9.8.

The audit found only BDNF and Rust-kernel parity were tested. This suite adds the rigorous,
mechanism-level coverage that was missing:
- torch.autograd.gradcheck on the differentiable release-probability path (the heart of the bias).
- State RANGE invariants (clamps hold over many steps: RRP/RES/PR/CL/E/C/BUF stay in range, finite).
- Vesicle-pool conservation DIRECTION: the pool only depletes (the "fatigue" invariant), bounded
  by the initial total.
- Extreme-input robustness (huge ±drive -> finite output and state).
- STE gradient sanity for all 3 stochastic modes.

Run:  pytest tests/test_synaptic_invariants.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)

from _bio_testkit import set_seed

DEV = torch.device("cpu")
DT = torch.float32


def _setup(*, K=4, T=12, B=2, H=4, dh=16, seed=0, **cfg_kw):
    set_seed(seed)
    cfg = SynapticConfig(**{"enable_presyn": True, **cfg_kw})
    pre = SynapticPresyn(dh, cfg)
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    for t in range(T):
        idx[:, :, t, :] = torch.randint(0, t + 1, (B, H, K))
    return cfg, pre, state, idx


# --------------------------------------------------------------------------- #
# 1. gradcheck on the differentiable release-probability path
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradcheck_faithful_release_probability():
    cfg = SynapticConfig(enable_presyn=True)
    pre = SynapticPresyn(16, cfg)
    # float64 inputs kept away from the [0,1] clamp boundary so the analytic gradient is smooth
    c = (torch.rand(6, dtype=torch.float64) + 0.5).requires_grad_(True)
    pr = (torch.rand(6, dtype=torch.float64) * 0.4 + 0.4).requires_grad_(True)
    cl = (torch.rand(6, dtype=torch.float64) * 0.4 + 0.3).requires_grad_(True)
    drive = torch.randn(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        pre._faithful_release_prob, (c, pr, cl, drive), atol=1e-6, rtol=1e-4
    ), "analytic grad of the Hill/Doc2/fuse release prob must match finite differences"


# --------------------------------------------------------------------------- #
# 2. State RANGE invariants over many steps (clamps hold; everything finite)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_state_stays_in_valid_ranges_over_many_steps():
    cfg, pre, state, idx = _setup(seed=1)
    eps = 1e-4
    for step in range(50):
        drive = torch.randn(2, 4, 12, 4) * 2.0
        pre.release_canonical(state, drive, idx, train=False)
        for k in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
            assert torch.isfinite(state[k]).all(), f"step {step}: state[{k}] not finite"
        assert (state["C"] >= -eps).all(), "calcium must stay non-negative"
        assert (state["BUF"] >= -eps).all() and (state["BUF"] <= 1 + eps).all()
        assert (state["RRP"] >= -eps).all() and (state["RRP"] <= 30 + eps).all()
        assert (state["RES"] >= -eps).all()
        assert (state["PR"] >= -eps).all() and (state["PR"] <= 1 + eps).all()
        assert (state["CL"] >= -eps).all() and (state["CL"] <= 1 + eps).all()
        assert (state["E"] >= -eps).all() and (state["E"] <= cfg.energy_max + eps).all()


# --------------------------------------------------------------------------- #
# 3. Vesicle-pool conservation DIRECTION (the "fatigue" invariant)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_vesicle_pool_is_non_increasing():
    # Vesicles leave RRP on release and only a fraction (rec_rate) re-enters via the endocytosis
    # queue, so the total pool (RRP + RES + in-flight) only DEPLETES -- it must never grow above
    # the initial total. This is the physical basis of the "fatigue/boredom" frequency penalty.
    cfg, pre, state, idx = _setup(seed=2)

    def pool_total() -> float:
        t = state["RRP"] + state["RES"]
        for d in state["DELAY"]:
            t = t + d
        return float(t.sum().item())

    init_total = pool_total()
    prev = init_total
    for _ in range(30):
        pre.release_canonical(state, torch.randn(2, 4, 12, 4) * 2.0, idx, train=False)
        total = pool_total()
        assert total <= init_total + 1e-2, f"vesicle pool grew above initial ({total} > {init_total})"
        assert total <= prev + 1e-2, "vesicle pool must be non-increasing within numerical jitter"
        prev = total


# --------------------------------------------------------------------------- #
# 4. Extreme-input robustness
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("scale", [1.0e3, 1.0e6, -1.0e6])
def test_extreme_drive_stays_finite(scale):
    _, pre, state, idx = _setup(seed=3)
    drive = torch.full((2, 4, 12, 4), float(scale))
    e = pre.release_canonical(state, drive, idx, train=False)
    assert torch.isfinite(e).all(), f"release must stay finite under drive={scale}"
    for k in ("C", "RRP", "RES", "PR", "CL", "E", "BUF"):
        assert torch.isfinite(state[k]).all(), f"state[{k}] not finite under drive={scale}"


@pytest.mark.unit
def test_nan_drive_is_caught_not_silently_swallowed():
    # NaN drive propagates to the output (garbage in, garbage out) -- documenting that the guard
    # (vg9.7), not the release fn, is responsible for catching it.
    _, pre, state, idx = _setup(seed=4)
    drive = torch.randn(2, 4, 12, 4)
    drive[0, 0, 0, 0] = float("nan")
    e = pre.release_canonical(state, drive, idx, train=False)
    assert not torch.isfinite(e).all(), "NaN input should surface (the divergence guard catches it)"


# --------------------------------------------------------------------------- #
# 5. STE gradient sanity for the 3 stochastic modes
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("mode", ["gumbel_sigmoid_ste", "straight_through", "normal_reparam"])
def test_stochastic_ste_gradients_are_finite_and_flow(mode):
    set_seed(5)
    cfg = SynapticConfig(
        enable_presyn=True, stochastic_train_frac=1.0, stochastic_mode=mode, stochastic_tau=1.0
    )
    pre = SynapticPresyn(16, cfg)
    state = build_presyn_state(2, 12, 4, DEV, DT, cfg)
    idx = torch.zeros(2, 4, 12, 4, dtype=torch.long)
    for t in range(12):
        idx[:, :, t, :] = torch.randint(0, t + 1, (2, 4, 4))
    drive = torch.randn(2, 4, 12, 4, requires_grad=True)
    e = pre.release_canonical(state, drive, idx, train=True)
    (grad,) = torch.autograd.grad(e.sum(), drive, allow_unused=True)
    assert grad is not None, f"{mode}: STE must let gradients reach the drive"
    assert torch.isfinite(grad).all(), f"{mode}: STE gradients must be finite"
    assert grad.abs().sum() > 0, f"{mode}: STE must propagate a non-zero gradient"
