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


# =========================================================================== #
# 0642.3.2.1 — runtime TUR certificate + Crooks calibration monitor
# =========================================================================== #
def test_monitor_certifies_tur_with_a_nonvacuous_bound():
    mon = st.StochasticThermoMonitor()
    driven = st.rates_from_release(p_release=0.4, rec_rate=0.06, pool=6.0)
    for step in range(5):
        rec = mon.record(st.simulate_currents(driven, 10.0, 20000, seed=step), driven, step=step)
        assert rec.tur_satisfied
        assert 0.0 < rec.entropy_bound < float("inf"), "the TUR bound must be finite and positive (non-vacuous)"
        assert rec.tur_slack >= -1e-9
    assert mon.all_currents_satisfy_tur()
    mon.assert_tur()  # must not raise


def test_monitor_tracks_ft_residual_and_calibrates_near_equilibrium():
    mon = st.StochasticThermoMonitor()
    neq = st.ReleaseRates(a=0.6, b=0.4)
    for step in range(4):
        mon.record(st.simulate_currents(neq, 2.0, 200000, seed=step), neq, step=step)
    cal = mon.crooks_calibration(min_count=80)
    assert cal.calibrated, f"the accumulated Σ must obey the detailed FT (residual={cal.max_abs_residual:.3f})"
    assert math.isfinite(mon.ft_residual(min_count=80))


def test_monitor_summary_and_jsonl_well_formed():
    mon = st.StochasticThermoMonitor()
    driven = st.rates_from_release(0.4, 0.06, 6.0)
    for step in range(3):
        mon.record(st.simulate_currents(driven, 10.0, 10000, seed=step), driven, step=step)
    s = mon.summary()
    for key in ("steps", "tur_all_satisfied", "mean_entropy_bound", "ft_residual"):
        assert key in s
    lines = mon.to_jsonl()
    assert len(lines) == 3
    import json
    rec0 = json.loads(lines[0])
    assert {"step", "affinity", "relative_variance", "entropy_bound", "tur_satisfied"} <= set(rec0)


def test_monitor_assert_tur_fires_on_a_violation():
    # The TUR is a theorem for real samples; to exercise the guard, inject a crafted violating record.
    mon = st.StochasticThermoMonitor()
    mon.records.append(st.ThermoStepRecord(
        step=0, n_samples=10, affinity=0.5, mean_current=1.0, relative_variance=0.1,
        mean_entropy=1.0, entropy_bound=2.0, tur_satisfied=False, tur_slack=-1.9,
    ))
    assert not mon.all_currents_satisfy_tur()
    with pytest.raises(AssertionError, match="TUR violated"):
        mon.assert_tur()


# =========================================================================== #
# 0642.3.2.2 / .2.3 — energy-optimal temperature schedule + toggle + fallback
# =========================================================================== #
def test_thermo_uq_toggle_off_is_neutral():
    off = st.ThermoUQController(st.ThermoUQConfig(enabled=False))
    assert off.optimal_temperature(2.0) == 1.0, "disabled ⟹ neutral temperature (baseline path)"
    assert off.temperature_schedule([0.0, 1.0, 5.0]) == [1.0, 1.0, 1.0]


def test_landauer_temperature_schedule_rises_with_ach():
    on = st.ThermoUQController(st.ThermoUQConfig(enabled=True, drive_uncertainty_base=1.0, ach_gain=1.0))
    sched = on.temperature_schedule([0.0, 0.5, 1.0, 2.0])
    assert all(sched[i] < sched[i + 1] for i in range(len(sched) - 1)), "schedule must rise with ACh"
    assert sched[0] == pytest.approx(st.landauer_optimal_temperature(1.0))  # neutral ACh ⟹ base law


def test_calibration_verdict_analytic_for_real_release():
    on = st.ThermoUQController(st.ThermoUQConfig(enabled=True))
    neq = st.ReleaseRates(a=0.6, b=0.4)
    sig = st.entropy_production_samples(st.simulate_currents(neq, 2.0, 300000, seed=0), neq)
    v = on.calibration_verdict(sig, min_count=80)
    assert v.calibrated and v.mode == "analytic_fluctuation_theorem"


def test_fallback_triggers_on_non_markov_rate_misspecification():
    # Ledger E1/E3/R: if Σ is computed with the WRONG affinity (rate misspecification), the empirical
    # FT fails and the controller deterministically drops the analytic claim → empirical-ECE fallback.
    on = st.ThermoUQController(st.ThermoUQConfig(enabled=True, ft_tol=0.25))
    true_rates = st.ReleaseRates(a=0.6, b=0.4)
    J = st.simulate_currents(true_rates, 2.0, 300000, seed=1)
    misspecified = st.ReleaseRates(a=0.6, b=0.2)            # wrong recovery rate ⟹ wrong affinity
    sig_bad = st.entropy_production_samples(J, misspecified)
    v = on.calibration_verdict(sig_bad, min_count=80)
    assert not v.calibrated and v.mode == "empirical_ece_fallback"
    assert "report empirical ECE only" in v.reason


def test_fallback_triggers_on_non_ft_distribution():
    on = st.ThermoUQController(st.ThermoUQConfig(enabled=True, ft_tol=0.25))
    rng = np.random.default_rng(3)
    v = on.calibration_verdict(rng.normal(2.0, 1.0, 300000), min_count=80)  # no FT symmetry at all
    assert not v.calibrated and v.mode == "empirical_ece_fallback"


def test_assess_one_shot_reports_temperature_and_mode():
    on = st.ThermoUQController(st.ThermoUQConfig(enabled=True))
    neq = st.ReleaseRates(a=0.6, b=0.4)
    out = on.assess(st.simulate_currents(neq, 2.0, 200000, seed=2), neq, ach_level=1.0, min_count=80)
    assert {"enabled", "optimal_temperature", "calibration_mode", "ft_calibrated", "ft_residual"} <= set(out)
    assert out["optimal_temperature"] > st.landauer_optimal_temperature(1.0)  # ACh=1 raises it


# =========================================================================== #
# Edge cases / numerical robustness (fresh-eyes round 3)
# =========================================================================== #
def test_integral_ft_closed_form_no_overflow_at_extreme_drive():
    # The exponents sum to exactly 0; computing them separately would overflow at extreme a/b.
    for a in (1e6, 1e12, 1e18):
        assert st.integral_ft_closed_form(st.ReleaseRates(a=a, b=1.0), 1.0) == pytest.approx(1.0, abs=1e-9)
        assert st.integral_ft_closed_form(st.ReleaseRates(a=1.0, b=a), 1.0) == pytest.approx(1.0, abs=1e-9)


def test_empirical_tur_and_monitor_reject_degenerate_batches():
    # A single-sample (or zero-variance) batch makes Var(J) an unreliable estimate that would read as a
    # spurious "TUR violation" of a theorem; empirical_tur must refuse it, and the monitor must treat it
    # as non-informative (satisfied), never a violation. An empty batch is rejected outright.
    rates = st.rates_from_release(p_release=0.4, rec_rate=0.06, pool=6.0)
    with pytest.raises(ValueError):
        st.empirical_tur(np.array([3.0]), 1.0)              # n = 1
    mon = st.StochasticThermoMonitor()
    mon.record(np.array([3.0]), rates)                      # degenerate batch ⟹ non-informative
    mon.record(np.array([2.0, 2.0, 2.0]), rates)            # zero-variance ⟹ non-informative
    assert mon.all_currents_satisfy_tur(), "degenerate batches must NOT register a false TUR violation"
    mon.assert_tur()                                        # must not raise
    with pytest.raises(ValueError):
        mon.record(np.array([]), rates)                     # empty batch rejected


def test_crooks_calibration_handles_min_count_zero():
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 1.0, 5000)
    cal = st.crooks_calibration(sig, n_bins=15, min_count=0)  # must not divide by an empty mirror bin
    assert math.isfinite(cal.max_abs_residual) or cal.max_abs_residual == float("inf")
