"""
Free-energy deliberation + energy-based decoding (bead r00r.1.1).

Demonstrates the design (docs/theory/free_energy_deliberation.md) on the metriplectic core, proving
the claims the convergence argument rests on:

  - deliberation monotonically reduces F and HALTS (convergence — the safety guarantee);
  - compute SCALES WITH DIFFICULTY: a far-from-equilibrium ("hard") state uses many more iterations
    than a near-equilibrium ("easy") one — adaptive test-time compute;
  - the budget caps the worst case (a hard token with a tiny eps hits max_iters);
  - the effort/confidence signals behave (iters ↑, F_drop ↑ for harder states);
  - the Boltzmann decoder p ∝ exp(−F/kT) prefers the lowest-F candidate, sums to 1, and reduces to
    greedy argmin-F as kT → 0.

Run:  pytest tests/test_free_energy_deliberation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat import metriplectic_integrator as mi

pytestmark = pytest.mark.unit

HARD = np.array([1.5, 1.0, 0.0])   # far from the equilibrium z* = (0,0,E₀)
EASY = np.array([0.02, 0.01, 1.6])  # already near equilibrium


# --------------------------------------------------------------------------- #
# 1. Deliberation converges (monotone F descent + halting).
# --------------------------------------------------------------------------- #
def test_deliberation_reduces_free_energy_and_halts():
    res = mi.deliberate(HARD, dt=0.2, eps=1e-4, max_iters=500)
    assert res.halted_converged, "deliberation must halt on self-consistency, not the budget"
    assert res.F_drop > 0.0, "free energy must strictly drop while pondering a hard state"
    # The relaxed state must sit at lower free energy than where it started.
    assert mi.free_energy(res.z) == pytest.approx(res.F_final)
    assert res.F_final < mi.free_energy(HARD)


def test_each_deliberation_step_is_non_increasing_in_F():
    # Drive the loop one step at a time and confirm F never rises (the Lyapunov safety property).
    z = HARD.copy()
    f_prev = mi.free_energy(z)
    for k in range(50):
        z, rec = mi.guarded_step(z, 0.2, k, mi.GuardThresholds())
        assert rec.F <= f_prev + 1e-12, f"F rose at step {k}: {f_prev} -> {rec.F}"
        f_prev = rec.F


# --------------------------------------------------------------------------- #
# 2. Compute scales with difficulty (the headline capability).
# --------------------------------------------------------------------------- #
def test_compute_scales_with_token_difficulty():
    hard = mi.deliberate(HARD, dt=0.2, eps=1e-4, max_iters=500)
    easy = mi.deliberate(EASY, dt=0.2, eps=1e-4, max_iters=500)
    assert hard.iters > easy.iters, (
        f"a hard state must deliberate longer than an easy one ({hard.iters} vs {easy.iters})"
    )
    assert easy.iters <= 3, "a near-equilibrium state should halt almost immediately"
    assert hard.F_drop > easy.F_drop, "the hard state releases more free energy"


def test_budget_caps_the_worst_case():
    res = mi.deliberate(HARD, dt=0.2, eps=1e-12, max_iters=7)
    assert res.iters == 7 and not res.halted_converged, "an impossible eps must hit the budget"


def test_tighter_eps_deliberates_at_least_as_long():
    loose = mi.deliberate(HARD, dt=0.2, eps=1e-2, max_iters=500)
    tight = mi.deliberate(HARD, dt=0.2, eps=1e-6, max_iters=500)
    assert tight.iters >= loose.iters, "a tighter halting threshold must not deliberate less"


# --------------------------------------------------------------------------- #
# 3. Energy-based (Boltzmann) decoding.
# --------------------------------------------------------------------------- #
def test_boltzmann_prefers_lowest_free_energy_and_normalizes():
    w = mi.boltzmann_weights([1.0, 2.0, 0.5], kT=1.0)
    assert int(np.argmax(w)) == 2, "the lowest-F candidate must get the most mass"
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w > 0.0), "all candidates keep positive probability at finite kT"


def test_boltzmann_reduces_to_greedy_as_temperature_drops():
    w = mi.boltzmann_weights([1.0, 2.0, 0.5], kT=1e-3)
    assert w[2] == pytest.approx(1.0, abs=1e-6), "kT → 0 ⟹ argmin-F greedy"


def test_boltzmann_temperature_must_be_positive():
    with pytest.raises(ValueError):
        mi.boltzmann_weights([1.0, 2.0], kT=0.0)


def test_constraint_energy_term_suppresses_a_candidate():
    # A hard constraint = a large additive energy on the forbidden candidate ⟹ ~0 probability.
    base = [1.0, 1.0, 1.0]
    constrained = [1.0, 1.0 + 50.0, 1.0]  # candidate 1 violates a constraint
    w = mi.boltzmann_weights(constrained, kT=1.0)
    assert w[1] < 1e-6, "a constraint-violating candidate must be suppressed"
    assert np.allclose(mi.boltzmann_weights(base, kT=1.0), 1.0 / 3, atol=1e-9)
