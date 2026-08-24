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

import json
import math

import numpy as np
import pytest
import torch

from bio_inspired_nanochat import stochastic_thermo as st
from bio_inspired_nanochat.results_registry import read_records
from scripts.e2e.stochastic_thermo_uq import (
    ExperimentConfig,
    _make_model,
    _mc_dropout_prediction,
    _reset_sequence,
    _softmax_prediction,
    _thermo_prediction,
    append_registry_records,
    binary_auroc,
    binomial_crooks_curve,
    binomial_tur_diagnostic,
    expected_calibration_error,
    run_experiment,
    run_live_release_ft,
    run_multi_seed,
)

pytestmark = pytest.mark.unit

# A near-equilibrium regime where the MC fluctuation-theorem estimators converge.
_NEAR_EQ = st.ReleaseRates(a=0.6, b=0.4)


@pytest.fixture(scope="module")
def multi_seed_report():
    config = ExperimentConfig(
        vocab_size=16,
        seq_len=6,
        batch_size=2,
        pool_size=2,
        eval_pool_size=2,
        train_steps=2,
        n_head=1,
        n_embd=16,
        dropout=0.15,
        mc_samples=2,
        ece_bins=4,
        ft_trajectories=30_000,
        ft_min_count=40,
        ft_tolerance=0.35,
        ft_integral_tolerance=0.07,
    )
    return run_multi_seed(config, [11, 23, 37], bootstrap_samples=250)


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


# =========================================================================== #
# 0642.3.3.1 — live-release FT + ECE/OOD-AUROC falsification
# =========================================================================== #
def test_falsification_metrics_have_exact_known_values():
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.4]])
    targets = torch.tensor([0, 1])
    ece, curve = expected_calibration_error(probabilities, targets, n_bins=2)
    assert ece == pytest.approx(0.25)
    assert sum(point.count for point in curve) == 2
    assert binary_auroc(torch.tensor([0.1, 0.2]), torch.tensor([0.8, 0.9])) == 1.0
    assert binary_auroc(torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2])) == 0.0
    assert binary_auroc(torch.tensor([0.5]), torch.tensor([0.5])) == 0.5
    with pytest.raises(ValueError, match="sum to one"):
        expected_calibration_error(torch.tensor([[0.8, 0.8]]), torch.tensor([0]))
    with pytest.raises(ValueError, match="at least one prediction"):
        expected_calibration_error(torch.empty((0, 2)), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        binary_auroc(torch.tensor([0.1]), torch.tensor([float("nan")]))


def test_exact_binomial_crooks_curve_is_falsifiable():
    rng = np.random.default_rng(31)
    pool_size = 6
    forward_probability = 0.32
    reverse_probability = 0.24
    currents = rng.binomial(pool_size, forward_probability, 120_000) - rng.binomial(
        pool_size, reverse_probability, 120_000
    )
    exact_affinity = math.log(
        forward_probability
        * (1.0 - reverse_probability)
        / (reverse_probability * (1.0 - forward_probability))
    )
    exact = binomial_crooks_curve(
        currents, exact_affinity, pool_size=pool_size, min_count=100
    )
    misspecified = binomial_crooks_curve(
        currents, exact_affinity + 0.5, pool_size=pool_size, min_count=100
    )
    assert len(exact) >= 3
    assert max(abs(point.residual) for point in exact) < 0.25
    assert max(abs(point.residual) for point in misspecified) > 0.25


def test_live_release_trajectories_pass_the_exact_binomial_ft():
    config = ExperimentConfig(
        train_steps=0,
        ft_trajectories=40_000,
        ft_min_count=50,
        ft_tolerance=0.3,
        ft_integral_tolerance=0.06,
    )
    result = run_live_release_ft(config)
    assert result.passed, (
        f"live release failed Crooks/IFT: max_residual={result.max_crooks_residual:.4f}, "
        f"integral_residual={result.integral_ft_residual:.4f}"
    )
    assert result.forward_probability == pytest.approx(config.ft_forward_probability, abs=1e-6)
    assert result.reverse_probability == pytest.approx(config.ft_reverse_probability, abs=1e-6)
    assert len(result.curve) >= 2
    assert result.scope == "one_step_local_detailed_balance"
    assert not result.predictive_distribution_claim


def test_sequence_reset_preserves_backprop_trained_fast_weights():
    model = _make_model(
        ExperimentConfig(
            vocab_size=16,
            seq_len=6,
            batch_size=1,
            pool_size=1,
            eval_pool_size=1,
            n_head=1,
            n_embd=16,
            train_steps=0,
        )
    )
    fast_weight = next(
        module.w_fast
        for module in model.modules()
        if getattr(module, "w_fast", None) is not None
    )
    with torch.no_grad():
        fast_weight.fill_(0.125)
    _reset_sequence(model)
    assert torch.equal(fast_weight, torch.full_like(fast_weight, 0.125))


def test_all_uncertainty_paths_preserve_persistent_model_state():
    config = ExperimentConfig(
        vocab_size=16,
        seq_len=6,
        batch_size=1,
        pool_size=1,
        eval_pool_size=1,
        n_head=1,
        n_embd=16,
        train_steps=0,
        mc_samples=2,
    )
    model = _make_model(config)
    inputs = torch.arange(config.seq_len).reshape(1, -1) % config.vocab_size
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    _softmax_prediction(model, inputs)
    _mc_dropout_prediction(model, inputs, n_samples=config.mc_samples)
    _thermo_prediction(model, inputs, n_samples=config.mc_samples)
    after = model.state_dict()
    assert before.keys() == after.keys()
    for name, expected in before.items():
        assert torch.equal(after[name], expected), f"persistent state mutated: {name}"


@pytest.mark.e2e
def test_thermo_uq_e2e_reports_both_baselines_without_assuming_a_win():
    config = ExperimentConfig(
        vocab_size=16,
        seq_len=6,
        batch_size=2,
        pool_size=2,
        eval_pool_size=2,
        train_steps=2,
        n_head=1,
        n_embd=16,
        dropout=0.15,
        mc_samples=2,
        ece_bins=4,
        ft_trajectories=30_000,
        ft_min_count=40,
        ft_tolerance=0.35,
        ft_integral_tolerance=0.07,
    )
    report = run_experiment(config)
    assert report.live_release_ft.passed
    assert set(report.methods) == {"softmax_entropy", "mc_dropout", "thermo_uq"}
    assert set(report.thermo_deltas) == {"vs_softmax_entropy", "vs_mc_dropout"}
    for metrics in report.methods.values():
        assert math.isfinite(metrics.ece)
        assert 0.0 <= metrics.ood_auroc <= 1.0
        assert sum(point.count for point in metrics.calibration_curve) == (
            config.batch_size * config.eval_pool_size * config.seq_len
        )
    thermo = report.methods["thermo_uq"]
    for baseline_name in ("softmax_entropy", "mc_dropout"):
        baseline = report.methods[baseline_name]
        delta = report.thermo_deltas[f"vs_{baseline_name}"]
        assert delta["ece_delta_lower_is_better"] == pytest.approx(
            thermo.ece - baseline.ece
        )
        assert delta["ood_auroc_delta_higher_is_better"] == pytest.approx(
            thermo.ood_auroc - baseline.ood_auroc
        )
    json.dumps(report.to_dict(), allow_nan=False)
    assert "does not assert an advantage" in report.comparison_policy


def test_live_binomial_tur_is_nonvacuous_but_exposes_poisson_limit_failure():
    forward_probability = 0.32
    reverse_probability = 0.24
    affinity = math.log(
        forward_probability
        * (1.0 - reverse_probability)
        / (reverse_probability * (1.0 - forward_probability))
    )
    diagnostic = binomial_tur_diagnostic(
        pool_size=6,
        forward_probability=forward_probability,
        reverse_probability=reverse_probability,
        affinity=affinity,
    )
    assert diagnostic.nonvacuous
    assert diagnostic.entropy_bound > 0.0
    assert diagnostic.relative_variance == pytest.approx(10.4166666667)
    assert diagnostic.bound_ratio == pytest.approx(0.9972692702)
    assert diagnostic.slack < 0.0
    assert not diagnostic.satisfied
    assert "continuous_time_tur" in diagnostic.scope


def test_multi_seed_report_uses_paired_stats_and_emits_honest_null(multi_seed_report):
    report = multi_seed_report
    assert report.seeds == [11, 23, 37]
    assert report.ft_pass_rate == 1.0
    assert report.live_tur.nonvacuous and not report.live_tur.satisfied
    assert report.verdict == "null"
    assert "TUR bound" in report.verdict_reason
    assert set(report.method_aggregates) == {
        "mc_dropout",
        "softmax_entropy",
        "thermo_uq",
    }
    for metrics in report.method_aggregates.values():
        assert set(metrics) == {"ece", "ood_auroc", "id_accuracy"}
        assert all(stats.n == 3 for stats in metrics.values())
    for comparison in report.paired_comparisons.values():
        assert comparison["ece"].n_pairs == 3
        assert comparison["ood_auroc"].n_pairs == 3
    json.dumps(report.to_dict(), allow_nan=False)


def test_multi_seed_registry_records_are_schema_valid_and_duplicate_safe(
    tmp_path, multi_seed_report
):
    path = tmp_path / "registry.jsonl"
    count = append_registry_records(
        multi_seed_report,
        path,
        artifact="results/stochastic_thermo_uq_test.json",
    )
    assert count == len(multi_seed_report.seeds) * 3
    records = read_records(str(path))
    assert len(records) == count
    assert len({record.run_id for record in records}) == count
    for record in records:
        assert {
            "id_ece",
            "ood_auroc",
            "eval_accuracy",
            "live_ft_max_crooks_residual",
            "live_ft_integral_residual",
            "live_tur_relative_variance",
            "live_tur_entropy_bound",
            "live_tur_slack",
            "live_tur_bound_ratio",
        } == set(record.metrics)
        if "-thermo_uq-" in record.run_id:
            assert record.verdict == multi_seed_report.verdict
        else:
            assert record.verdict is None
        assert not record.eligible_for_best
    with pytest.raises(ValueError, match="registry already contains run IDs"):
        append_registry_records(multi_seed_report, path)
    assert len(read_records(str(path))) == count
