"""
Numerical corroboration of the singular-perturbation / cusp theory note (bead 0642.2.1).

These tests check the QUALITATIVE geometry the cusp normal-form reduction predicts, against the
REAL `PostsynapticHebb` CaMKII/PP1 latch (no re-implementation of the dynamics). A normal-form
reduction licenses claims about signs, monotonicities, and the existence of the bistable wedge — not
the exact δ* value (which carries an O(ε) Fenichel correction) — so that is what we assert:

  1. at the default config the latch is BISTABLE — a calcium up/down sweep traces a hysteresis loop
     with the ON-switch calcium strictly above the OFF-switch calcium (a < 0, the fold structure);
  2. the CUSP THRESHOLD in the self-excitation γ — with γ above γ_c the latch retains after a write
     pulse; with self-excitation off (γ = 0, a ≥ 0) it does not retain and the loop collapses;
  3. δ* is MONOTONE in (−a) — increasing γ (deeper into the wedge) widens the hysteresis loop;
  4. the RETENTION margin — after a write, the ON state survives a band of sub-fold inputs.

See docs/theory/singular_perturbation.md.  Run:  pytest tests/test_singular_perturbation_theory.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig

pytestmark = pytest.mark.unit

D = 4                  # value channels (latch is per-channel; mean is the order parameter)
NEUTRAL = 0.75         # between latch_ltd_thr (0.5) and camkii_thr (1.0): no LTP, no LTD


def _latch(**cfg_kw) -> PostsynapticHebb:
    cfg = SynapticConfig(bistable_latch=True, **cfg_kw)
    return PostsynapticHebb(d_k=D, d_v=D, cfg=cfg)


def _equilibrate(post: PostsynapticHebb, ca_value: float, steps: int) -> float:
    ca = torch.full((D,), float(ca_value))
    y = torch.zeros(1, D)
    for _ in range(steps):
        post.update(y, ca)
    return float(post.camkii.mean())


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _hysteresis_loop(cfg_kw: dict, *, lo=0.0, hi=1.8, npts=31, steps=60):
    """Quasi-static calcium sweep up then down through ONE latch; return (c_on, c_off, width).

    c_on  = rising-sweep calcium at which CaMKII first crosses 0.5 (OFF→ON),
    c_off = falling-sweep calcium at which CaMKII drops back below 0.5 (ON→OFF).
    Either is None if the order parameter never crosses 0.5 on that leg.
    """
    post = _latch(**cfg_kw)
    up = _linspace(lo, hi, npts)
    dn = list(reversed(up))
    m_up = [_equilibrate(post, c, steps) for c in up]   # carries state across levels
    m_dn = [_equilibrate(post, c, steps) for c in dn]
    c_on = next((up[i] for i, m in enumerate(m_up) if m > 0.5), None)
    c_off = next((dn[i] for i, m in enumerate(m_dn) if m < 0.5), None)
    width = (c_on - c_off) if (c_on is not None and c_off is not None) else 0.0
    return c_on, c_off, width


# --------------------------------------------------------------------------- #
# 1. Bistability: the default latch traces a hysteresis loop (a < 0).
# --------------------------------------------------------------------------- #
def test_default_latch_is_bistable_with_a_hysteresis_loop():
    c_on, c_off, width = _hysteresis_loop({})
    assert c_on is not None, "latch must switch ON somewhere in the calcium sweep"
    assert c_off is not None, "latch must switch OFF somewhere on the way down"
    # The fold structure: ON-switch calcium strictly ABOVE the OFF-switch calcium. A genuine
    # bistable loop is many grid cells wide (≈0.48 at defaults), well above the ~1-cell width a
    # monostable smooth follow would show.
    assert c_on > c_off, f"hysteresis requires c_on ({c_on:.3f}) > c_off ({c_off:.3f})"
    assert width > 0.2, f"the bistable wedge must give a wide loop, got {width:.3f}"


# --------------------------------------------------------------------------- #
# 2. Cusp threshold in γ: self-excitation must exceed the linear decay (a < 0).
# --------------------------------------------------------------------------- #
def test_self_excitation_off_destroys_retention_and_the_loop():
    # γ = 0 ⟹ a ≥ 0 ⟹ monostable: a write pulse does NOT persist, and the loop collapses.
    post = _latch(latch_gamma_auto=0.0)
    _equilibrate(post, 2.0, 30)                         # strong write pulse
    m_after_pulse = _equilibrate(post, NEUTRAL, 80)     # then drop to neutral
    assert m_after_pulse < 0.5, (
        f"with self-excitation off the state must NOT retain (got {m_after_pulse:.3f})"
    )
    # The monostable "loop" is a ~1-grid-cell artifact of the smooth follow (~0.06); the bistable
    # default loop is several-fold wider (~0.48).
    _, _, width0 = _hysteresis_loop({"latch_gamma_auto": 0.0})
    _, _, width_default = _hysteresis_loop({})
    assert width_default > 3.0 * width0, (
        f"the bistable (default) loop must dominate the monostable one "
        f"({width_default:.3f} vs {width0:.3f})"
    )


def test_default_latch_retains_after_a_write_pulse():
    post = _latch()
    _equilibrate(post, 2.0, 12)                          # supra-threshold write
    assert float(post.camkii.mean()) > 0.9, "write pulse must drive CaMKII high"
    m_held = _equilibrate(post, NEUTRAL, 100)            # input drops to neutral
    assert m_held > 0.9, f"latched state must persist (hysteresis), got {m_held:.3f}"


# --------------------------------------------------------------------------- #
# 3. δ* monotone in (−a): more self-excitation ⟹ wider hysteresis.
# --------------------------------------------------------------------------- #
def test_hysteresis_width_grows_with_self_excitation():
    # Both γ values are in the reversible bistable regime (they erase within the calcium sweep);
    # the deeper one (more negative a) must give the wider loop.
    _, _, width_lo = _hysteresis_loop({"latch_gamma_auto": 0.30})
    _, _, width_hi = _hysteresis_loop({"latch_gamma_auto": 0.60})
    assert width_hi > width_lo, (
        f"δ* must increase with (−a): width(γ=0.60)={width_hi:.3f} should exceed "
        f"width(γ=0.30)={width_lo:.3f}"
    )


# --------------------------------------------------------------------------- #
# 4. Retention margin: the ON state survives a BAND of sub-fold inputs, then collapses.
# --------------------------------------------------------------------------- #
def test_on_state_survives_sub_fold_inputs_and_collapses_past_the_fold():
    # Write ON, then probe a descending band of holding inputs from a fresh ON state each time.
    def held_value(ca_hold: float) -> float:
        post = _latch()
        _equilibrate(post, 2.0, 12)                     # latch ON
        return _equilibrate(post, ca_hold, 80)          # hold at ca_hold

    assert held_value(NEUTRAL) > 0.9, "ON must survive a neutral hold (well inside the fold)"
    assert held_value(0.8) > 0.9, "ON must survive a mild sub-threshold hold"
    # Deep LTD (calcium well below latch_ltd_thr) drives PP1 and erases — past the lower fold.
    assert held_value(0.0) < 0.5, "strong LTD input must cross the fold and erase the latch"
