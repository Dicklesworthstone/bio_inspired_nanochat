"""
Runtime retention certificate for the cusp latch (bead 0642.2.2.3).

Locks the certificate + deterministic-fallback contract derived in
docs/theory/singular_perturbation.md and implemented in bio_inspired_nanochat/cusp_certificate.py:

  - δ*(a) = (2/3√3)(−a)^{3/2}, 0 for the monostable a ≥ 0 (the closed form);
  - the cusp coefficient a is < 0 (bistable) at the default latch and ≥ 0 when self-excitation is
    off (monostable) — matching the live latch's measured bistability;
  - δ* is MONOTONE in the self-excitation γ (deeper wedge ⟹ wider certified margin), tracking the
    measured hysteresis;
  - the FALLBACK fires (certificate void) when the latch is monostable OR when the fast subsystem is
    not contractive enough (ρ(M_cb) > cusp_eps_max) — fail-closed to the heuristic sax.2 latch;
  - the cusp_latch toggle is default-off and registered with its bistable_latch prerequisite.

Run:  pytest tests/test_cusp_certificate.py -v
"""

from __future__ import annotations

import math

import pytest

from bio_inspired_nanochat import cusp_certificate as cc
from bio_inspired_nanochat.ablation_registry import _BY_FIELD, validate_config
from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig
from bio_inspired_nanochat.torch_imports import torch

pytestmark = pytest.mark.unit


def _cfg(**kw) -> SynapticConfig:
    return SynapticConfig(bistable_latch=True, cusp_latch=True, **kw)


# --------------------------------------------------------------------------- #
# 1. The closed-form δ*(a).
# --------------------------------------------------------------------------- #
def test_delta_star_closed_form_and_monostable_zero():
    assert cc.retention_delta_star(0.0) == 0.0, "a = 0 (cusp point) ⟹ δ* = 0"
    assert cc.retention_delta_star(1.0) == 0.0, "a > 0 (monostable) ⟹ δ* = 0"
    a = -0.75
    assert cc.retention_delta_star(a) == pytest.approx((2.0 / (3 * math.sqrt(3))) * (-a) ** 1.5)
    # Monotone increasing in (−a).
    assert cc.retention_delta_star(-1.0) > cc.retention_delta_star(-0.5) > 0.0


# --------------------------------------------------------------------------- #
# 2. The cusp coefficient sign matches the live latch's bistability.
# --------------------------------------------------------------------------- #
def test_default_latch_is_bistable_a_negative_and_certified():
    cert = cc.certify_retention(_cfg())
    assert cert.bistable and cert.a < 0.0, f"default latch must be bistable, a={cert.a}"
    assert cert.delta_star > 0.0 and cert.certified, cert.reason
    assert not cert.use_heuristic_fallback


def test_self_excitation_off_is_monostable_and_uncertified():
    cert = cc.certify_retention(_cfg(latch_gamma_auto=0.0))
    assert not cert.bistable and cert.a >= 0.0, f"γ=0 must be monostable, a={cert.a}"
    assert cert.delta_star == 0.0 and not cert.certified
    assert cert.use_heuristic_fallback and "monostable" in cert.reason


def test_certificate_bistability_agrees_with_the_live_latch():
    # The certificate says γ ≥ ~0.2 is bistable; confirm the live latch actually RETAINS there and
    # does NOT at γ = 0 (the certificate's a-sign is grounded in the real dynamics, not just algebra).
    # With cusp_latch on (via _cfg), the live latch is the certified cusp gradient-flow (0642.2.2.1),
    # whose write is gradual (monotone, stability-capped) — so it needs a sustained write pulse to
    # climb from deep-OFF to the ON root; retention at the neutral hold is the property under test.
    def retains(gamma: float) -> bool:
        post = PostsynapticHebb(d_k=4, d_v=4, cfg=_cfg(latch_gamma_auto=gamma))
        ca_hi, ca_neutral = torch.full((4,), 2.0), torch.full((4,), 0.75)
        y = torch.zeros(1, 4)
        for _ in range(40):
            post.update(y, ca_hi)        # sustained write pulse (cusp write is graded, not a jump)
        for _ in range(80):
            post.update(y, ca_neutral)   # hold at neutral (inside the retention band |b| < δ*)
        return float(post.camkii.mean()) > 0.5

    assert cc.certify_retention(_cfg(latch_gamma_auto=0.45)).certified and retains(0.45)
    cert0 = cc.certify_retention(_cfg(latch_gamma_auto=0.0))
    assert not cert0.certified and not retains(0.0)


# --------------------------------------------------------------------------- #
# 3. δ* monotone in the self-excitation γ (tracks the measured hysteresis).
# --------------------------------------------------------------------------- #
def test_delta_star_increases_with_self_excitation():
    d_lo = cc.certify_retention(_cfg(latch_gamma_auto=0.30)).delta_star
    d_hi = cc.certify_retention(_cfg(latch_gamma_auto=0.60)).delta_star
    assert 0.0 < d_lo < d_hi, f"δ* must grow with γ (got {d_lo:.4g} then {d_hi:.4g})"


# --------------------------------------------------------------------------- #
# 4. The ε-gauge fallback.
# --------------------------------------------------------------------------- #
def test_epsilon_gauge_is_the_fast_subsystem_spectral_radius():
    eps = cc.epsilon_gauge(SynapticConfig())
    assert 0.0 < eps < 1.0, f"ρ(M_cb) must be a contraction (<1) at defaults, got {eps}"


def test_fallback_fires_when_timescale_separation_is_insufficient():
    # A very slow calcium decay (large tau_c ⟹ ρ_c → 1) closes the contraction gap ⟹ ρ(M_cb) > eps_max.
    cfg = _cfg(latch_gamma_auto=0.45, tau_c=400.0)
    cert = cc.certify_retention(cfg)
    assert cert.eps > cfg.cusp_eps_max, f"ρ(M_cb)={cert.eps} should exceed the cap"
    assert cert.bistable, "the latch is still bistable..."
    assert not cert.certified and cert.use_heuristic_fallback, "...but the certificate is void"
    assert "separation" in cert.reason


# --------------------------------------------------------------------------- #
# 5. Toggle + validation discipline.
# --------------------------------------------------------------------------- #
def test_cusp_latch_default_off_and_registered_with_prerequisite():
    assert SynapticConfig().cusp_latch is False
    assert "cusp_latch" in _BY_FIELD and _BY_FIELD["cusp_latch"].requires == ("bistable_latch",)


def test_validator_flags_cusp_latch_without_bistable_latch():
    errors, _ = validate_config(SynapticConfig(cusp_latch=True, bistable_latch=False))
    assert any("cusp_latch" in e and "bistable_latch" in e for e in errors)


def test_validator_range_checks_eps_max():
    errors, _ = validate_config(SynapticConfig(bistable_latch=True, cusp_latch=True, cusp_eps_max=1.5))
    assert any("cusp_eps_max" in e for e in errors)


def test_default_config_validates_clean():
    assert validate_config(SynapticConfig()) == ([], [])
