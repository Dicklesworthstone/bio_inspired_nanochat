"""Falsification experiment: certified cusp retention vs the heuristic sigmoid latch (bead 0642.2.3.1).

The end-to-end falsification that closes Thrust F's singular-perturbation/cusp chain. It pits the
certified cusp-normal-form latch (`cusp_latch`, bead 0642.2.2.1) against the heuristic `sax.2`
sigmoid latch (`bistable_latch` alone) on **retention under an erasing perturbation**, and checks the
two falsifiable predictions of `docs/theory/singular_perturbation.md`:

  (A) CERTIFICATE IS TIGHT — in the cusp control coordinate (the bias `b`, the coordinate δ* lives
      in), a latched bit survives any sustained drift of magnitude `< δ*` and flips once the drift
      exceeds `δ*`. The empirical retention half-width equals the closed-form `δ*(a)`. Falsified if a
      sub-δ* drift erased the bit, or a supra-δ* drift failed to.

  (B) CERTIFICATE IS SOUND (conservative) — driving the *full* latch with a physical calcium erase
      ramp, the cusp latch's measured retention is **at least** its certified margin (the bound never
      over-promises). PP1 slaving makes the held ON state even more robust than the basal-p
      certificate, so empirical retention ≥ certified — never the reverse.

  (C) CERTIFIED LEAPFROG — δ* is a *tunable, guaranteed* design dial (δ* grows with the
      self-excitation γ). At an elevated γ (same capacity: identical buffers/compute) the certified
      cusp latch retains under an erase ramp that has already collapsed the uncertified sax.2
      baseline. The baseline has hysteresis but no bound and no dial; the cusp can be *certified* to
      beat it. Falsified if no certified γ out-retained the baseline.

Honest note (logged, not asserted): at the DEFAULT γ the cusp well is shallow (δ*≈0.009) and the two
latches are ~tied on raw erase robustness — the cusp's edge there is the *tight certificate*, not raw
margin. The leapfrog (C) is what raw robustness looks like once the certified dial is turned up.

Runnable:  python tests/test_cusp_falsification.py      # prints the full retention curves (both arms)
Tested:    pytest tests/test_cusp_falsification.py -v   # asserts (A), (B), (C)
"""

from __future__ import annotations

import pathlib
import sys

# Make the repo importable when this file is run directly as a script (pytest uses pyproject's
# pythonpath; a bare `python tests/...` does not). Harmless under pytest.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402

from bio_inspired_nanochat.cusp_certificate import CuspLatch, relax_cubic  # noqa: E402
from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticConfig  # noqa: E402
from bio_inspired_nanochat.torch_imports import torch  # noqa: E402

pytestmark = pytest.mark.e2e

_D_V = 16


# --------------------------------------------------------------------------- #
# Experiment primitives
# --------------------------------------------------------------------------- #
def _cusp_cfg(gamma: float = 0.45) -> SynapticConfig:
    return SynapticConfig(bistable_latch=True, cusp_latch=True, latch_gamma_auto=gamma)


def _sax_cfg(gamma: float = 0.45) -> SynapticConfig:
    return SynapticConfig(bistable_latch=True, cusp_latch=False, latch_gamma_auto=gamma)


def _written_latch(cfg: SynapticConfig, write_ca: float = 2.5, write_steps: int = 120) -> PostsynapticHebb:
    """A PostsynapticHebb latch driven ON by a sustained supra-threshold write pulse."""
    post = PostsynapticHebb(d_k=_D_V, d_v=_D_V, cfg=cfg)
    y = torch.zeros(1, _D_V)
    for _ in range(write_steps):
        post.update(y, torch.full((_D_V,), float(write_ca)))
    return post


def _retention_after_hold(cfg: SynapticConfig, hold_ca: float, hold_steps: int) -> float:
    """Fraction of channels still ON (CaMKII > 0.5) after holding the written latch at `hold_ca`."""
    post = _written_latch(cfg)
    y = torch.zeros(1, _D_V)
    for _ in range(hold_steps):
        post.update(y, torch.full((_D_V,), float(hold_ca)))
    return float((post.camkii > 0.5).float().mean())


def calcium_erase_curve(cfg: SynapticConfig, holds, hold_steps: int = 600):
    """Retention vs a sustained calcium erase ramp. Returns [(hold_ca, retention_fraction), ...]."""
    return [(h, _retention_after_hold(cfg, h, hold_steps)) for h in holds]


def calcium_erase_threshold(cfg: SynapticConfig, hold_steps: int = 600,
                            lo: float = 0.205, hi: float = 0.60, step: float = 0.005):
    """Lowest hold-calcium that still retains (the empirical erase threshold). None ⟹ held throughout."""
    held = hi
    h = hi
    while h >= lo:
        if _retention_after_hold(cfg, h, hold_steps) < 0.5:
            return h, held
        held = h
        h -= step
    return None, held


def bias_drift_retention(lat: CuspLatch, drift: float, hold_steps: int = 4000) -> bool:
    """Does a latched ON bit (normal-form coordinate) survive a sustained bias drift of size `drift`?

    `drift > 0` is the erase direction (pushes toward OFF). Exercises exactly the latch's own update
    rule (`relax_cubic`, what `CuspLatch.step` runs) in the certified bias coordinate.
    """
    a, rate = lat.k.a, lat.k.rate
    on = float(relax_cubic(torch.tensor(0.3), a, 0.0, rate=rate, steps=5000))
    u = float(relax_cubic(torch.tensor(on), a, drift, rate=rate, steps=hold_steps))
    return u > 0.0


def bias_drift_curve(lat: CuspLatch, fracs):
    """Retention vs sustained bias drift, in units of δ*. Returns [(frac, survived_bool), ...]."""
    return [(f, bias_drift_retention(lat, f * lat.delta_star)) for f in fracs]


# =========================================================================== #
# (A) The certificate is TIGHT: survive below δ*, flip above δ*
# =========================================================================== #
def test_retention_half_width_equals_delta_star():
    lat = CuspLatch(_cusp_cfg())
    assert lat.certified and lat.delta_star > 0.0
    curve = bias_drift_curve(lat, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0])
    survived = {f for f, ok in curve if ok}
    flipped = {f for f, ok in curve if not ok}
    # Below δ*: every drift holds. Above δ*: every drift flips. The boundary is δ* itself.
    assert all(f in survived for f in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)), f"sub-δ* drift must hold: {curve}"
    assert all(f in flipped for f in (1.05, 1.1, 1.25, 1.5, 2.0)), f"supra-δ* drift must flip: {curve}"


# =========================================================================== #
# (B) The certificate is SOUND: physical retention ≥ certified margin
# =========================================================================== #
def test_full_latch_retention_is_at_least_the_certified_margin():
    cfg = _cusp_cfg()
    lat = CuspLatch(cfg)
    c_rest = 0.5 * (cfg.latch_ltd_thr + cfg.camkii_thr)
    erase_fold = lat.min_erase_calcium()           # certified erase-collapse calcium (basal-p bound)
    assert erase_fold is not None and erase_fold < c_rest
    # The certificate promises the bit holds for any hold-calcium ABOVE the erase fold.
    held_at_fold_plus = _retention_after_hold(cfg, erase_fold + 0.02, hold_steps=600)
    assert held_at_fold_plus > 0.5, f"latch must retain above the certified erase fold (got {held_at_fold_plus:.2f})"
    # And the actual empirical threshold is at/below the certified bound (PP1 slaving ⟹ conservative).
    flip_ca, _ = calcium_erase_threshold(cfg)
    assert flip_ca is None or flip_ca <= erase_fold + 1e-6, (
        f"empirical erase threshold {flip_ca} must not exceed the certified fold {erase_fold:.3f} "
        f"(the bound must be sound/conservative, never over-promised)"
    )


# =========================================================================== #
# (C) The certified LEAPFROG: a tuned, certified cusp out-retains the baseline
# =========================================================================== #
def test_delta_star_grows_with_self_excitation():
    d = [CuspLatch(_cusp_cfg(g)).delta_star for g in (0.45, 0.8, 1.2)]
    assert d[0] < d[1] < d[2], f"δ* must grow monotonically with γ (the design dial): {d}"
    assert all(CuspLatch(_cusp_cfg(g)).certified for g in (0.45, 0.8, 1.2))


def test_tuned_cusp_outretains_uncertified_baseline():
    """At a hold-calcium where the sax.2 baseline has already collapsed, an elevated-γ certified cusp
    latch still retains — the certified leapfrog past the uncertified baseline (equal capacity)."""
    sax_flip, _ = calcium_erase_threshold(_sax_cfg())
    assert sax_flip is not None, "the sax.2 baseline must have a finite erase threshold to beat"
    # A drift comfortably past where sax.2 fails (and where the DEFAULT cusp also fails).
    c_test = sax_flip - 0.08
    sax_ret = _retention_after_hold(_sax_cfg(), c_test, hold_steps=600)
    tuned = _cusp_cfg(0.8)
    assert CuspLatch(tuned).certified, "the tuned cusp must remain certified to claim a *certified* leapfrog"
    cusp_ret = _retention_after_hold(tuned, c_test, hold_steps=600)
    assert sax_ret < 0.5 <= cusp_ret, (
        f"certified leapfrog at hold={c_test:.3f}: cusp γ=0.8 retention={cusp_ret:.2f} must beat "
        f"the collapsed sax.2 baseline retention={sax_ret:.2f}"
    )


# --------------------------------------------------------------------------- #
# Runnable experiment: print the retention curves for both arms.
# --------------------------------------------------------------------------- #
def _print_curves() -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except Exception:  # pragma: no cover - rich is a project dep, but stay runnable without it
        console = None

    lat = CuspLatch(_cusp_cfg())
    print("\n=== Thrust F falsification — certified cusp latch vs sax.2 sigmoid latch ===")
    print(f"default cusp: a={lat.k.a:.4f}  δ*={lat.delta_star:.5f}  certified={lat.certified}")
    print(f"             certified erase fold (calcium) = {lat.min_erase_calcium()}")

    # Arm A — bias-space tightness around δ*.
    print("\n[A] retention vs sustained bias drift (cusp normal-form coordinate; δ* is the certified fold):")
    a_curve = bias_drift_curve(lat, [0.0, 0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0])
    if console:
        t = Table("drift / δ*", "ON retained?")
        for f, ok in a_curve:
            t.add_row(f"{f:.2f}", "✅ hold" if ok else "❌ flip")
        console.print(t)
    else:
        for f, ok in a_curve:
            print(f"   {f:.2f}·δ*: {'hold' if ok else 'flip'}")

    # Arm C — physical calcium erase ramp, both arms + tuned cusp.
    print("\n[C] retention vs calcium erase ramp (fraction of channels still ON after a long hold):")
    holds = [0.60, 0.55, 0.52, 0.50, 0.45, 0.40, 0.35, 0.30, 0.26]
    arms = {
        "sax.2 (default)": _sax_cfg(),
        "cusp γ=0.45": _cusp_cfg(0.45),
        "cusp γ=0.80": _cusp_cfg(0.80),
        "cusp γ=1.20": _cusp_cfg(1.20),
    }
    curves = {name: dict(calcium_erase_curve(cfg, holds, hold_steps=600)) for name, cfg in arms.items()}
    if console:
        t = Table("hold Ca", *arms.keys())
        for h in holds:
            t.add_row(f"{h:.3f}", *[f"{curves[name][h]:.2f}" for name in arms])
        console.print(t)
    else:
        print("  hold | " + " | ".join(arms))
        for h in holds:
            print(f"  {h:.3f} | " + " | ".join(f"{curves[name][h]:.2f}" for name in arms))
    for name, cfg in arms.items():
        thr, held = calcium_erase_threshold(cfg)
        print(f"   {name:18s}: flips@{('%.3f' % thr) if thr is not None else 'never(≤0.205)':>14s}  (held to {held:.3f})")
    print("\nReading: sax.2 and default cusp flip at ~similar calcium (default δ* is shallow — the cusp's")
    print("edge there is the TIGHT CERTIFICATE); turning the certified γ dial up makes the cusp retain")
    print("far past the uncertified baseline — the certified leapfrog.")


if __name__ == "__main__":
    _print_curves()
