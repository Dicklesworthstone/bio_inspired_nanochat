"""Numerical corroboration of the stochastic-thermodynamics theory note (Thrust E, beads `0642.3.1.*`).

Checks the three falsifiable results of `docs/theory/stochastic_thermodynamics.md` against the
reference Markov-jump model of vesicle release (`bio_inspired_nanochat/stochastic_thermo.py`):

  - `0642.3.1.1` — vesicle release is a driven Markov jump process with entropy production
    `Σ = J·ln(a/b)`; the fluctuation theorems hold (`⟨e^{−Σ}⟩ = 1` exactly; `P(Σ)/P(−Σ) = e^Σ`);
  - `0642.3.1.2` — the TUR `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩` holds for all drives and is tight near equilibrium;
  - `0642.3.1.3` — Crooks/Jarzynski give a calibration guarantee the empirical `Σ` histogram obeys,
    and the check rejects data that does not (so the guarantee is falsifiable, not vacuous).

Far-from-equilibrium identities are verified in **closed form** (the MC estimator converges slowly
there); the Monte-Carlo corroborations use a near-equilibrium regime where both signs of `J` are
well sampled. Run:  pytest tests/test_stochastic_thermo.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bio_inspired_nanochat import stochastic_thermo as st

pytestmark = pytest.mark.unit

# A near-equilibrium regime where the MC fluctuation-theorem estimators converge.
_NEAR_EQ = st.ReleaseRates(a=0.6, b=0.4)


# =========================================================================== #
# 0642.3.1.1 — Markov jump model + entropy production + fluctuation theorems
# =========================================================================== #
def test_affinity_sign_tracks_the_drive():
    assert st.affinity(st.ReleaseRates(a=0.6, b=0.4)) > 0.0      # release-biased ⟹ dissipative
    assert st.affinity(st.ReleaseRates(a=0.4, b=0.4)) == 0.0     # detailed balance ⟹ equilibrium
    assert st.affinity(st.ReleaseRates(a=0.3, b=0.4)) < 0.0      # recovery-biased


def test_mean_entropy_production_is_nonnegative_second_law():
    for a in (0.05, 0.2, 0.4, 0.41, 0.8, 2.4):
        rates = st.ReleaseRates(a=a, b=0.4)
        assert st.mean_entropy_production(rates, steps=10.0) >= -1e-12, f"second law violated at a={a}"
    assert st.mean_entropy_production(st.ReleaseRates(a=0.4, b=0.4), 10.0) == pytest.approx(0.0)


def test_integral_fluctuation_theorem_closed_form_is_exactly_one():
    # ⟨e^{−Σ}⟩ ≡ 1 for every drive and duration (the Skellam MGF identity).
    for a in (0.41, 0.6, 1.5, 2.4):
        for t in (1.0, 5.0, 25.0):
            val = st.integral_ft_closed_form(st.ReleaseRates(a=a, b=0.36), t)
            assert val == pytest.approx(1.0, abs=1e-9), f"FT closed form != 1 at a={a}, t={t}: {val}"


def test_simulator_matches_analytic_moments_and_integral_ft():
    J = st.simulate_currents(_NEAR_EQ, steps=2.0, n_traj=400000, seed=3)
    assert float(J.mean()) == pytest.approx(st.mean_current(_NEAR_EQ, 2.0), abs=0.02)
    assert float(J.var()) == pytest.approx(st.var_current(_NEAR_EQ, 2.0), rel=0.03)
    sig = st.entropy_production_samples(J, _NEAR_EQ)
    assert float(sig.mean()) == pytest.approx(st.mean_entropy_production(_NEAR_EQ, 2.0), abs=0.01)
    assert st.integral_fluctuation_theorem(sig) == pytest.approx(1.0, abs=0.02)  # MC, near-eq


def test_detailed_fluctuation_theorem_ratio():
    # P(J=+k)/P(J=−k) = (a/b)^k = e^{kA}, the detailed FT P(Σ)/P(−Σ)=e^Σ.
    J = st.simulate_currents(_NEAR_EQ, steps=2.0, n_traj=800000, seed=4)
    for k in (1, 2, 3):
        emp, pred = st.detailed_fluctuation_ratio(J, _NEAR_EQ, k)
        assert pred == pytest.approx((_NEAR_EQ.a / _NEAR_EQ.b) ** k)
        assert emp == pytest.approx(pred, rel=0.05), f"detailed FT off at k={k}: {emp} vs {pred}"


def test_rates_from_release_drive_condition():
    driven = st.rates_from_release(p_release=0.4, rec_rate=0.06, pool=6.0)
    assert driven.a > driven.b and st.affinity(driven) > 0.0   # p > rec_rate ⟹ dissipative
    balanced = st.rates_from_release(p_release=0.06, rec_rate=0.06, pool=6.0)
    assert st.affinity(balanced) == pytest.approx(0.0)


# =========================================================================== #
# 0642.3.1.2 — Thermodynamic Uncertainty Relation
# =========================================================================== #
def test_tur_holds_for_all_drives():
    for a in (0.41, 0.5, 0.8, 1.5, 3.0, 6.0):
        cert = st.tur_certificate(st.ReleaseRates(a=a, b=0.4), steps=10.0)
        assert cert.satisfied and cert.slack >= -1e-12, f"TUR violated at a/b={a/0.4:.2f}"
        assert cert.entropy_bound == pytest.approx(2.0 / cert.mean_entropy)


def test_tur_is_tight_near_equilibrium():
    # The relative slack (slack / bound) → 0 as a → b: the TUR is saturated in linear response.
    def rel_slack(a: float) -> float:
        c = st.tur_certificate(st.ReleaseRates(a=a, b=0.4), 10.0)
        return c.slack / c.entropy_bound
    assert rel_slack(0.42) < rel_slack(1.0) < rel_slack(4.0), "TUR must tighten toward equilibrium"
    assert rel_slack(0.42) < 1e-3, "near equilibrium the TUR is essentially saturated"


def test_empirical_tur_from_samples():
    # Use a comfortably-driven regime (not the near-tight near-equilibrium one) so finite-sample noise
    # in the empirical mean/variance cannot dip the estimate below the analytic bound.
    rates = st.rates_from_release(p_release=0.4, rec_rate=0.06, pool=6.0)  # a/b ≈ 6.7, relative slack ~0.3
    J = st.simulate_currents(rates, steps=10.0, n_traj=300000, seed=5)
    cert = st.empirical_tur(J, st.mean_entropy_production(rates, 10.0))
    assert cert.satisfied, "the TUR must hold on sampled currents too"


# =========================================================================== #
# 0642.3.1.3 — Crooks / Jarzynski → calibration guarantee
# =========================================================================== #
def test_jarzynski_recovers_zero_free_energy():
    # Steady-state release: w = kT·Σ, ΔF = 0 — recovered from purely nonequilibrium fluctuations.
    J = st.simulate_currents(_NEAR_EQ, steps=2.0, n_traj=500000, seed=6)
    sig = st.entropy_production_samples(J, _NEAR_EQ)
    assert st.jarzynski_free_energy(sig, kT=1.0) == pytest.approx(0.0, abs=0.02)


def test_crooks_calibration_holds_for_the_real_release():
    J = st.simulate_currents(_NEAR_EQ, steps=2.0, n_traj=800000, seed=7)
    sig = st.entropy_production_samples(J, _NEAR_EQ)
    cal = st.crooks_calibration(sig, n_bins=15, tol=0.25, min_count=50)
    assert cal.calibrated, f"the release Σ histogram must obey the detailed FT (resid={cal.max_abs_residual:.3f})"
    assert cal.bins.size >= 3


def test_crooks_calibration_rejects_misspecified_data():
    # A Σ-like quantity with NO fluctuation-theorem symmetry (Gaussian) must FAIL — the guarantee is
    # falsifiable, not vacuous (the proof-ledger fallback: drop the analytic claim, flag).
    rng = np.random.default_rng(11)
    bad = rng.normal(2.0, 1.0, size=300000)
    cal = st.crooks_calibration(bad, n_bins=15, tol=0.25, min_count=50)
    assert not cal.calibrated and cal.max_abs_residual > 0.25


def test_boltzmann_drive_temperature_relation_smoke():
    # kT enters Jarzynski as the work scale; doubling kT halves Σ-in-work-units but leaves ΔF≈0.
    J = st.simulate_currents(_NEAR_EQ, steps=2.0, n_traj=400000, seed=8)
    sig = st.entropy_production_samples(J, _NEAR_EQ)
    work = 2.0 * sig  # w = kT·Σ with kT = 2
    assert st.jarzynski_free_energy(work, kT=2.0) == pytest.approx(0.0, abs=0.05)


# =========================================================================== #
# 0642.3.1.4 — energy-optimal (Landauer) release temperature
# =========================================================================== #
def test_optimal_exploration_snr_solves_the_stationarity():
    snr = st.optimal_exploration_snr()
    assert snr == pytest.approx(3.9215, abs=1e-3), f"SNR* must be the rate-distortion root, got {snr}"
    # Satisfies 2·SNR/(1+SNR) = ln(1+SNR).
    assert (2 * snr / (1 + snr)) == pytest.approx(math.log1p(snr), abs=1e-9)


def test_bits_per_joule_peaks_at_optimal_snr():
    snr = st.optimal_exploration_snr()
    peak = st.bits_per_joule(snr)
    for delta in (0.5, 1.0, 2.0, 5.0):
        assert st.bits_per_joule(snr + delta) < peak, "bits-per-joule must fall above SNR*"
        assert st.bits_per_joule(max(0.05, snr - delta)) < peak, "bits-per-joule must fall below SNR*"


def test_landauer_temperature_matches_the_uncertainty_scale():
    snr = st.optimal_exploration_snr()
    const = 1.0 / math.sqrt(snr)                      # kT*/σ ≈ 0.505
    for sigma in (0.5, 1.0, 2.0, 4.0):
        kt = st.landauer_optimal_temperature(sigma)
        assert kt == pytest.approx(const * sigma)     # linear in the drive uncertainty
    assert const == pytest.approx(0.505, abs=0.005)
    with pytest.raises(ValueError):
        st.landauer_optimal_temperature(0.0)


def test_ach_coupling_raises_temperature_with_uncertainty():
    base = st.ach_coupled_temperature(1.0, ach_level=0.0)
    hi = st.ach_coupled_temperature(1.0, ach_level=1.0, ach_gain=1.0)
    higher = st.ach_coupled_temperature(1.0, ach_level=3.0, ach_gain=1.0)
    assert base < hi < higher, "more ACh (uncertainty) ⟹ hotter, more-exploratory release"
    assert base == pytest.approx(st.landauer_optimal_temperature(1.0))  # neutral at ACh = 0
    # ACh = 1 doubles the effective uncertainty (gain 1) ⟹ doubles kT*.
    assert hi == pytest.approx(2.0 * base)
