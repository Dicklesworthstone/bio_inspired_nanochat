"""Self-test for the e2e harness + invariant battery (bead hwxb.7.3).

Proves the battery fires correctly: a known-good reduced run passes every invariant,
and a known-bad run (NaN injected mid-training) is caught by the finite/divergence
invariants. Marked ``e2e`` (it runs a real reduced training loop, ~seconds on CPU).
"""
from __future__ import annotations

import pytest

from bio_inspired_nanochat.e2e_harness import E2EConfig, run_e2e


@pytest.mark.e2e
def test_e2e_synaptic_known_good_passes_all_invariants(tmp_path):
    # The full stack, including online Hebbian plasticity (opt-in since hwxb.9).
    cfg = E2EConfig(synapses=True, steps=80, seed=1234, syn_overrides={"enable_hebbian": True})
    report = run_e2e(cfg, run_dir=tmp_path, verbose=False)
    # Every invariant should pass on a healthy synaptic run.
    failures = [r.name for r in report.invariants if not r.passed]
    assert report.passed, f"healthy synaptic run failed invariants: {failures}\n" + "\n".join(
        r.line() for r in report.invariants
    )
    # Sanity: the expected invariant set is present (incl. the synaptic-only ones).
    names = {r.name for r in report.invariants}
    assert {
        "loss_finite", "loss_decreases", "grad_norm_finite_bounded", "params_finite",
        "checkpoint_roundtrip", "generation_nondegenerate", "mechanism_engaged", "mechanism_stable",
    } <= names
    assert report.summary["final_loss"] < report.summary["initial_loss"]


@pytest.mark.e2e
def test_e2e_vanilla_known_good_passes(tmp_path):
    cfg = E2EConfig(synapses=False, steps=80, seed=1234)
    report = run_e2e(cfg, run_dir=tmp_path, verbose=False)
    assert report.passed, "healthy vanilla run failed:\n" + "\n".join(
        r.line() for r in report.invariants
    )
    # Vanilla has no synaptic mechanism invariants.
    names = {r.name for r in report.invariants}
    assert "mechanism_engaged" not in names


@pytest.mark.e2e
def test_e2e_mechanism_engaged_is_load_bearing(tmp_path):
    """With Hebbian on but its training-time writes disabled, mechanism_engaged must FAIL.

    This is the test that proves the invariant actually measures the online Hebbian
    mechanism (the eligibility buffers never move when plasticity_during_training=False)
    rather than the gradient-trained w_fast weights (which AdamW moves regardless). With
    Hebbian off altogether the harness asserts the opposite, that the state stays exactly put
    (``mechanism_off_is_inert``, tests/test_scaleup_ablation_e2e.py).
    """
    cfg = E2EConfig(
        synapses=True, steps=80, seed=1234,
        syn_overrides={"enable_hebbian": True, "plasticity_during_training": False},
    )
    report = run_e2e(cfg, run_dir=tmp_path, verbose=False)
    failed = {r.name for r in report.failures()}
    assert "mechanism_engaged" in failed, (
        "mechanism_engaged should fail when the Hebbian writes never run — otherwise it is not "
        "actually measuring the mechanism. Failures: " + str(failed)
    )
    # The rest of the run is still healthy (SGD still trains the slow weights).
    assert {"loss_finite", "params_finite", "grad_norm_finite_bounded"} & failed == set()
    assert report.summary["final_loss"] < report.summary["initial_loss"]


@pytest.mark.e2e
def test_e2e_injected_nan_is_caught(tmp_path):
    """A NaN injected mid-training must trip the finite/divergence invariants."""
    cfg = E2EConfig(synapses=True, steps=60, inject_nan_at=20, seed=1234)
    report = run_e2e(cfg, run_dir=tmp_path, verbose=False)
    assert not report.passed, "known-bad (NaN-injected) run should NOT pass"
    failed = {r.name for r in report.failures()}
    # The NaN must be caught by at least one of the finiteness invariants.
    assert failed & {"loss_finite", "params_finite", "grad_norm_finite_bounded"}, (
        f"NaN run failed the wrong invariants: {failed}"
    )


@pytest.mark.e2e
def test_e2e_report_assert_passed_raises_on_failure(tmp_path):
    cfg = E2EConfig(synapses=True, steps=40, inject_nan_at=10, seed=7)
    report = run_e2e(cfg, run_dir=tmp_path, verbose=False)
    with pytest.raises(AssertionError):
        report.assert_passed()
