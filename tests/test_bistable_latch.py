"""Bistable CaMKII/PP1 consolidation latch — bead sax.2.

The README advertised a "bistable CaMKII/PP1 latch (hysteresis / self-excitation)",
but the live consolidation gate was a static `sigmoid(CaMKII - 0.5)` threshold with PP1
absent (CLAIMS_AUDIT gap #1). This implements the real Lisman-style switch behind
`SynapticConfig.bistable_latch` (default-off): CaMKII autophosphorylation (Hill self-
excitation) + mutual cross-inhibition with PP1 over a basal phosphatase floor, with PP1
folded INTO the gate. These tests lock the three properties the bead requires —
bistability, hysteresis, and PP1-driven erasure — plus that the default path is unchanged.

Run:  pytest tests/test_bistable_latch.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig

pytestmark = pytest.mark.unit

D_V = 8
NEUTRAL = 0.75  # calcium between latch_ltd_thr (0.5) and camkii_thr (1.0): no LTP, no LTD


def _hebb(latch: bool = True, **cfg_kw) -> PostsynapticHebb:
    cfg = SynapticConfig(bistable_latch=latch, **cfg_kw)
    return PostsynapticHebb(d_k=D_V, d_v=D_V, cfg=cfg)


def _drive(post: PostsynapticHebb, ca_value: float, steps: int) -> None:
    ca = torch.full((D_V,), float(ca_value))
    y = torch.zeros(1, D_V)  # `y` is unused by update(); calcium drives the latch
    for _ in range(steps):
        post.update(y, ca)


def _m(post: PostsynapticHebb) -> float:
    return float(post.camkii.mean())


# --------------------------------------------------------------------------- #
# 1. Bistability: two stable states for the SAME (neutral) input
# --------------------------------------------------------------------------- #
def test_two_stable_states_at_neutral_input():
    # From OFF, neutral calcium stays OFF.
    off = _hebb()
    _drive(off, NEUTRAL, 80)
    assert _m(off) < 0.1, f"OFF state must be stable at neutral input (got {_m(off):.3f})"

    # From ON (CaMKII high, PP1 at floor), neutral calcium stays ON.
    on = _hebb()
    with torch.no_grad():
        on.camkii.fill_(1.0)
        on.pp1.fill_(on.cfg.latch_pp1_basal)
    _drive(on, NEUTRAL, 80)
    assert _m(on) > 0.9, f"ON state must be self-sustaining at neutral input (got {_m(on):.3f})"


# --------------------------------------------------------------------------- #
# 2. Hysteresis: supra-threshold latches & persists; sub-threshold relaxes
# --------------------------------------------------------------------------- #
def test_supra_threshold_pulse_latches_and_persists():
    post = _hebb()
    _drive(post, 2.0, 10)        # strong LTP pulse (calcium >> camkii_thr)
    assert _m(post) > 0.9, "supra-threshold input must drive CaMKII high"
    _drive(post, NEUTRAL, 80)    # input drops to neutral
    assert _m(post) > 0.9, "latch must PERSIST after the input drops (hysteresis)"


def test_sub_threshold_pulse_does_not_latch():
    post = _hebb()
    _drive(post, 0.9, 4)         # below camkii_thr=1.0: weak, sub-threshold
    _drive(post, NEUTRAL, 80)
    assert _m(post) < 0.1, "a sub-threshold pulse must relax back to OFF, not latch"


# --------------------------------------------------------------------------- #
# 3. PP1 / LTD flips the latch OFF
# --------------------------------------------------------------------------- #
def test_low_calcium_ltd_erases_the_latch():
    post = _hebb()
    _drive(post, 2.0, 10)        # latch ON
    assert _m(post) > 0.9
    _drive(post, 0.0, 80)        # sustained low calcium -> PP1/LTD
    assert _m(post) < 0.1, "sustained LTD (low calcium) must flip the latch OFF"


# --------------------------------------------------------------------------- #
# 4. PP1 is in the gate only with the latch; the default path is unchanged
# --------------------------------------------------------------------------- #
def _consolidation_delta(post: PostsynapticHebb) -> torch.Tensor:
    # A fixed diagonal eligibility trace so the consolidated delta is deterministic.
    R = post.cfg.rank_eligibility
    U = torch.zeros(D_V, R)
    V = torch.zeros(R, D_V)
    U[:, 0] = 1.0
    V[0, :] = 1.0
    before = post.slow.detach().clone()
    post.consolidate(U, V)
    return post.slow.detach() - before


def test_pp1_enters_the_gate_only_when_latched():
    # Latch ON: gate = sigmoid(beta*(CaMKII - PP1)) depends on PP1, so different PP1
    # levels at the same CaMKII produce different consolidation.
    hi_pp1 = _hebb(latch=True)
    lo_pp1 = _hebb(latch=True)
    for post, p in ((hi_pp1, 0.9), (lo_pp1, 0.3)):
        with torch.no_grad():
            post.camkii.fill_(0.8)
            post.pp1.fill_(p)
    d_hi = _consolidation_delta(hi_pp1).abs().mean()
    d_lo = _consolidation_delta(lo_pp1).abs().mean()
    assert d_lo > d_hi + 1e-6, "with the latch, lower PP1 must open the gate wider"

    # Legacy path: gate = sigmoid(CaMKII - 0.5) - 0.3 ignores PP1 entirely.
    leg_hi = _hebb(latch=False)
    leg_lo = _hebb(latch=False)
    for post, p in ((leg_hi, 0.9), (leg_lo, 0.3)):
        with torch.no_grad():
            post.camkii.fill_(0.8)
            post.pp1.fill_(p)
    assert torch.allclose(
        _consolidation_delta(leg_hi), _consolidation_delta(leg_lo)
    ), "legacy gate must be independent of PP1 (PP1 not in the gate)"


def test_default_off_keeps_legacy_threshold_dynamics():
    # With the latch off, sub-threshold calcium leaves CaMKII untouched (the legacy
    # update only rises when calcium crosses camkii_thr) — i.e. no self-excitation.
    post = _hebb(latch=False)
    _drive(post, 0.9, 20)  # below camkii_thr
    assert _m(post) == 0.0, "legacy CaMKII must not move below threshold (no self-excitation)"
    _drive(post, 2.0, 5)   # above threshold -> rises
    assert _m(post) > 0.0, "legacy CaMKII must rise above threshold"
