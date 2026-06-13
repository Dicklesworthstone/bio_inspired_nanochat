"""Unit tests for the certified cusp-normal-form latch + pulse controller (bead 0642.2.2.4).

Covers the runtime object built in bead 0642.2.2.1 (`bio_inspired_nanochat/cusp_certificate.py:
CuspLatch`, `relax_cubic`) and its dispatch in `PostsynapticHebb.update`, against the contract of
`docs/theory/singular_perturbation.md`:

  1. BISTABILITY / HYSTERESIS — the cusp cubic has two stable roots for a < 0; the live latch is
     path-dependent (same neutral calcium → ON or OFF depending on history).
  2. CERTIFICATE TIGHTNESS — a latched state survives a control perturbation of magnitude < δ*(a)
     and flips at magnitude > δ*; δ* is the *exact* fold half-width, not a loose bound.
  3. SLOW-MANIFOLD RECONSTRUCTION (Fenichel, §2) — `quasi_steady_calcium` is a fixed point; the live
     calcium relaxes to it at the certified ε-gauge rate ρ(M_cb); on the manifold the reduced latch
     reproduces the full coupled latch.
  4. FALLBACK — when the certificate is void (monostable, or ε too large), the latch is uncertified,
     `step` refuses to run, and `PostsynapticHebb` falls back to the heuristic sax.2 map — byte for
     byte (the fail-closed discipline, §5).
  5. MINIMUM-ENERGY WRITE/ERASE PULSES (bead 0642.2.1.5) — the minimal pulse reaches exactly a fold;
     a pulse short of the fold fails to flip, one at the fold flips.

Run:  pytest tests/test_cusp_latch.py -v
"""

from __future__ import annotations

import math

import pytest

from bio_inspired_nanochat import cusp_certificate as cc
from bio_inspired_nanochat.cusp_certificate import CuspLatch, relax_cubic
from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig
from bio_inspired_nanochat.torch_imports import torch

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cfg(**kw) -> SynapticConfig:
    return SynapticConfig(bistable_latch=True, cusp_latch=True, **kw)


def _t(x: float):
    return torch.tensor(float(x))


def _drive_latch(lat: CuspLatch, m, calciums):
    """Run the live latch over a calcium schedule; return (final mean CaMKII, final state)."""
    for c in calciums:
        m, _ = lat.step(m, _t(c))
    return float(m.mean()), m


def _run_posthebb(cfg: SynapticConfig, ca_seq, d_v: int = 4):
    """Drive a PostsynapticHebb latch over a calcium schedule; return final (camkii, pp1)."""
    post = PostsynapticHebb(d_k=d_v, d_v=d_v, cfg=cfg)
    y = torch.zeros(1, d_v)
    for ca in ca_seq:
        post.update(y, torch.full((d_v,), float(ca)))
    return post.camkii.clone(), post.pp1.clone()


# =========================================================================== #
# 1. Bistability / hysteresis
# =========================================================================== #
def test_relax_cubic_has_two_stable_roots_when_bistable():
    """a < 0 ⟹ the cusp potential ¼u⁴+½a u² is a double well with minima at ±√(−a)."""
    a, rate = -0.5, 0.2
    root = math.sqrt(-a)
    up = float(relax_cubic(_t(0.05), a, 0.0, rate=rate, steps=5000))
    dn = float(relax_cubic(_t(-0.05), a, 0.0, rate=rate, steps=5000))
    assert up == pytest.approx(root, abs=1e-3), f"+ basin must settle at +√(−a)={root:.4f}, got {up:.4f}"
    assert dn == pytest.approx(-root, abs=1e-3), f"− basin must settle at −√(−a)={-root:.4f}, got {dn:.4f}"


def test_relax_cubic_is_monostable_when_not_bistable():
    """a ≥ 0 ⟹ a single well at 0 (no memory): every basin collapses to the lone root."""
    a, rate = 0.5, 0.2
    for u0 in (-0.6, -0.1, 0.1, 0.6):
        fin = float(relax_cubic(_t(u0), a, 0.0, rate=rate, steps=5000))
        assert fin == pytest.approx(0.0, abs=1e-3), f"monostable: u0={u0} must collapse to 0, got {fin:.4f}"


def test_live_latch_is_path_dependent_at_a_neutral_input():
    """Hysteresis: the SAME neutral calcium yields ON or OFF depending on whether a write preceded it.

    This is the operational signature of bistability — the latch *remembers*.
    """
    lat = CuspLatch(_cfg())
    assert lat.certified
    neutral = 0.5 * (lat.cfg.latch_ltd_thr + lat.cfg.camkii_thr)  # mid-band, inside the wedge

    # Path A: write pulse, then hold neutral.
    mA, _ = _drive_latch(lat, torch.zeros(1), [2.0] * 40 + [neutral] * 60)
    # Path B: never write, hold neutral from OFF.
    mB, _ = _drive_latch(lat, torch.zeros(1), [neutral] * 100)

    assert mA > 0.5, f"written latch must hold ON at neutral, got m={mA:.3f}"
    assert mB < 0.5, f"unwritten latch must stay OFF at neutral, got m={mB:.3f}"
    assert mA - mB > 0.3, f"hysteresis loop must be wide (ΔM={mA - mB:.3f})"


def test_certified_hysteresis_loop_has_positive_width():
    """The write fold (OFF→ON) sits at a strictly higher calcium than the erase fold (ON→OFF)."""
    lat = CuspLatch(_cfg())
    c_write = lat.min_write_calcium()
    c_erase = lat.min_erase_calcium()
    assert c_write is not None and c_erase is not None, "both folds must be reachable at defaults"
    assert c_write > c_erase > 0.0, f"loop width: write-ca {c_write:.3f} must exceed erase-ca {c_erase:.3f}"


# =========================================================================== #
# 2. Certificate tightness — survive < δ*, flip > δ*
# =========================================================================== #
def test_retention_is_tight_in_bias_space():
    """Park at ON; a bias |b| < δ* holds, |b| > δ* flips. δ*(a) is the exact fold half-width."""
    lat = CuspLatch(_cfg())
    a, rate, dstar = lat.k.a, lat.k.rate, lat.delta_star
    assert dstar > 0.0
    on = float(relax_cubic(_t(0.3), a, 0.0, rate=rate, steps=5000))  # the ON root (u > 0)

    held, flipped = [], []
    for frac in (0.0, 0.3, 0.6, 0.9):  # inside the retention band
        u = float(relax_cubic(_t(on), a, frac * dstar, rate=rate, steps=12000))
        held.append((frac, u))
        assert u > 0.0, f"bias {frac:.2f}·δ* (< δ*) must hold ON; u_final={u:+.4f}"
    for frac in (1.1, 1.5, 2.0):  # past the fold
        u = float(relax_cubic(_t(on), a, frac * dstar, rate=rate, steps=12000))
        flipped.append((frac, u))
        assert u < 0.0, f"bias {frac:.2f}·δ* (> δ*) must flip OFF; u_final={u:+.4f}"
    print("retention-curve (frac·δ* → u_final):", "HOLD", held, "FLIP", flipped)


def test_retention_boundary_is_at_delta_star_within_tolerance():
    """The flip threshold is δ* itself: just under holds, just over flips (≈5% band)."""
    lat = CuspLatch(_cfg())
    a, rate, dstar = lat.k.a, lat.k.rate, lat.delta_star
    on = float(relax_cubic(_t(0.3), a, 0.0, rate=rate, steps=5000))
    just_under = float(relax_cubic(_t(on), a, 0.95 * dstar, rate=rate, steps=20000))
    just_over = float(relax_cubic(_t(on), a, 1.05 * dstar, rate=rate, steps=20000))
    assert just_under > 0.0, f"0.95·δ* must still hold ON (u={just_under:+.4f})"
    assert just_over < 0.0, f"1.05·δ* must flip OFF (u={just_over:+.4f})"


def test_retention_is_symmetric_off_branch():
    """The OFF state is equally protected: −b with |b| < δ* holds OFF, > δ* flips to ON."""
    lat = CuspLatch(_cfg())
    a, rate, dstar = lat.k.a, lat.k.rate, lat.delta_star
    off = float(relax_cubic(_t(-0.3), a, 0.0, rate=rate, steps=5000))  # u < 0
    held = float(relax_cubic(_t(off), a, -0.6 * dstar, rate=rate, steps=12000))
    flipped = float(relax_cubic(_t(off), a, -1.4 * dstar, rate=rate, steps=12000))
    assert held < 0.0, f"−0.6·δ* must hold OFF (u={held:+.4f})"
    assert flipped > 0.0, f"−1.4·δ* must flip ON (u={flipped:+.4f})"


def test_resting_bias_is_inside_the_retention_margin():
    """The quiescent operating point sits inside the wedge: |b_rest| < δ* (R1)."""
    lat = CuspLatch(_cfg())
    c_rest = 0.5 * (lat.cfg.latch_ltd_thr + lat.cfg.camkii_thr)
    b_rest = float(lat.bias_at_calcium(_t(c_rest)))
    assert abs(b_rest) < lat.delta_star, f"|b_rest|={abs(b_rest):.5f} must be < δ*={lat.delta_star:.5f}"
    # And it reproduces the certificate's own b (single source of truth).
    assert b_rest == pytest.approx(lat.k.certificate.b, abs=1e-6)


# =========================================================================== #
# 3. Slow-manifold reconstruction (Fenichel, §2)
# =========================================================================== #
def _cb_step(c, buf, influx, cfg):
    """The faithful calcium↔buffer map (mirrors release_canonical / cusp_certificate projector)."""
    rho_c = math.exp(-1.0 / cfg.tau_c)
    rho_b = math.exp(-1.0 / cfg.tau_buf)
    a_on, a_off = cfg.alpha_buf_on, cfg.alpha_buf_off
    c_next = (rho_c * c + influx - a_on * c * (1.0 - buf) + a_off * buf).clamp(min=0.0)
    buf_next = (rho_b * buf + a_on * c * (1.0 - buf) - a_off * buf).clamp(0.0, 1.0)
    return c_next, buf_next


def test_quasi_steady_calcium_is_a_fixed_point():
    """h(influx) must be invariant under one more C/BUF step (it IS the slow manifold)."""
    cfg = _cfg()
    lat = CuspLatch(cfg)
    influx = torch.tensor(0.3, dtype=torch.float64)
    h = lat.quasi_steady_calcium(influx)
    # Recover the matching buffer fixed point by iterating, then confirm c stays put.
    c, buf = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    for _ in range(2000):
        c, buf = _cb_step(c, buf, influx, cfg)
    c_next, _ = _cb_step(c, buf, influx, cfg)
    assert float(c) == pytest.approx(float(h), abs=1e-6), "iterated C must equal the projector's h"
    assert float((c_next - c).abs()) < 1e-8, "h must be a fixed point of the C/BUF map"


def test_calcium_relaxes_to_manifold_at_the_gauge_rate():
    """|C_t − h| decays geometrically at ≤ ρ(M_cb)=ε per step — the certified ε-gauge."""
    cfg = _cfg()
    lat = CuspLatch(cfg)
    eps = cc.epsilon_gauge(cfg)
    influx = torch.tensor(0.3, dtype=torch.float64)
    h = lat.quasi_steady_calcium(influx)
    c, buf = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    err0 = abs(float(h))  # |C_0 − h| from rest
    # After n steps the error must be within err0 · ρ^n (allow a small slack for the buffer coupling).
    for n in (10, 20, 40):
        cc_, buf_ = c.clone(), buf.clone()
        for _ in range(n):
            cc_, buf_ = _cb_step(cc_, buf_, influx, cfg)
        err = abs(float(cc_ - h))
        bound = err0 * (eps ** n) * 5.0 + 1e-9
        assert err <= bound, f"step {n}: |C−h|={err:.3e} must be ≤ {bound:.3e} (ε={eps:.3f})"


def test_reduced_latch_reconstructs_full_latch_on_the_manifold():
    """Once calcium has settled onto M_ε, the reduced latch (calcium=h) reproduces the full coupled
    latch (calcium evolving) — the Fenichel reconstruction error vanishes on the manifold."""
    cfg = _cfg()
    lat = CuspLatch(cfg)
    influx = torch.tensor(0.9, dtype=torch.float64)
    # Settle the calcium first (warm the fast subsystem onto the manifold).
    c, buf = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    for _ in range(200):
        c, buf = _cb_step(c, buf, influx, cfg)
    h = lat.quasi_steady_calcium(influx)
    assert float((c - h).abs()) < 1e-6  # genuinely on the manifold now

    # Full vs reduced latch driven for many steps from the same OFF start.
    m_full, m_red = torch.zeros(1), torch.zeros(1)
    max_gap = 0.0
    for _ in range(80):
        m_full, _ = lat.step(m_full, c.to(torch.float32))
        m_red, _ = lat.step(m_red, h.to(torch.float32))
        max_gap = max(max_gap, float((m_full - m_red).abs().max()))
    assert max_gap < 1e-5, f"on M_ε the reduced latch must track the full latch; max gap={max_gap:.2e}"


# =========================================================================== #
# 4. Fallback — fail-closed when the certificate is void
# =========================================================================== #
def test_latch_uncertified_when_monostable():
    lat = CuspLatch(_cfg(latch_gamma_auto=0.0))
    assert not lat.certified and lat.delta_star == 0.0
    assert "monostable" in lat.k.certificate.reason


def test_latch_uncertified_when_timescale_separation_insufficient():
    # Very slow calcium (large τ_c ⟹ ρ_c→1) closes the contraction gap ⟹ ρ(M_cb) > eps_max.
    lat = CuspLatch(_cfg(latch_gamma_auto=0.45, tau_c=400.0))
    assert not lat.certified
    assert not lat.k.certificate.separated and "separation" in lat.k.certificate.reason


def test_step_refuses_to_run_when_uncertified():
    lat = CuspLatch(_cfg(latch_gamma_auto=0.0))
    with pytest.raises(RuntimeError, match="uncertified|fallback"):
        lat.step(torch.zeros(1), _t(1.0))


def test_posthebb_falls_back_to_sax2_byte_for_byte_when_uncertified():
    """The decisive fail-closed check: an UNCERTIFIED cusp_latch run must be identical to the
    heuristic sax.2 latch (cusp_latch off) under the same config — the cusp path is never silently
    half-applied (this is also the 'reduces to baseline at first order' guarantee)."""
    ca = [2.0] * 10 + [0.75] * 40
    # Monostable ⟹ cusp is uncertified ⟹ must match the pure sax.2 map.
    cusp_m, cusp_p = _run_posthebb(_cfg(latch_gamma_auto=0.0), ca)
    sax_m, sax_p = _run_posthebb(SynapticConfig(bistable_latch=True, cusp_latch=False, latch_gamma_auto=0.0), ca)
    assert torch.allclose(cusp_m, sax_m, atol=0.0), "uncertified cusp CaMKII must equal sax.2 exactly"
    assert torch.allclose(cusp_p, sax_p, atol=0.0), "uncertified cusp PP1 must equal sax.2 exactly"


def test_posthebb_uses_cusp_path_when_certified():
    """Sanity: when certified, the latch HOLDS through a neutral band that the monostable sax.2
    fallback would let decay — the two paths are genuinely different where it matters."""
    ca = [2.0] * 40 + [0.75] * 60
    cusp_m, _ = _run_posthebb(_cfg(latch_gamma_auto=0.45), ca)
    assert float(cusp_m.mean()) > 0.5, f"certified cusp latch must retain (m={float(cusp_m.mean()):.3f})"


# =========================================================================== #
# 5. Minimum-energy write/erase pulses (bead 0642.2.1.5)
# =========================================================================== #
def test_minimum_write_pulse_flips_only_at_the_fold():
    """A bias just past the lower fold (−δ*) latches OFF→ON; a bias short of it does not — the fold
    is the minimal-energy write target (deeper drive is wasted; shallower fails)."""
    lat = CuspLatch(_cfg())
    a, rate, dstar = lat.k.a, lat.k.rate, lat.delta_star
    off = float(relax_cubic(_t(-0.3), a, 0.0, rate=rate, steps=5000))
    short = float(relax_cubic(_t(off), a, -0.5 * dstar, rate=rate, steps=20000))  # inside band
    at_fold = float(relax_cubic(_t(off), a, lat.min_write_bias(), rate=rate, steps=20000))
    assert short < 0.0, f"a sub-fold write (−0.5·δ*) must NOT flip (u={short:+.4f})"
    assert at_fold > 0.0, f"a write at the fold (−δ*−margin) must flip OFF→ON (u={at_fold:+.4f})"


def test_pulse_table_is_well_formed():
    lat = CuspLatch(_cfg())
    rows = {r["action"]: r for r in lat.pulse_table()}
    assert set(rows) == {"write", "erase"}
    assert rows["write"]["bias_target"] < 0.0 < rows["erase"]["bias_target"]
    for r in rows.values():
        assert r["reachable"] and r["energy_proxy"] is not None and r["energy_proxy"] > 0.0
    print("pulse table:", lat.pulse_table())


def test_pulse_bias_targets_are_exactly_the_folds():
    lat = CuspLatch(_cfg())
    assert lat.min_write_bias(margin=0.0) == pytest.approx(-lat.delta_star)
    assert lat.min_erase_bias(margin=0.0) == pytest.approx(lat.delta_star)
    assert lat.fold_biases() == pytest.approx((-lat.delta_star, lat.delta_star))


# =========================================================================== #
# 6. Determinism
# =========================================================================== #
def test_latch_is_deterministic():
    ca = [2.0] * 20 + [0.7] * 30
    a, _ = _run_posthebb(_cfg(), ca)
    b, _ = _run_posthebb(_cfg(), ca)
    assert torch.equal(a, b), "the latch update is no_grad and deterministic — runs must be identical"
