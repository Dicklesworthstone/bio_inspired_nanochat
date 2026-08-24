"""
Runtime monitor + guards + deterministic fallback for the metriplectic integrator
(beads 0642.1.2.2 / 0642.1.2.3 / 0642.1.2.4).

Locks the runtime-certificate contract on top of the discrete-gradient integrator:

  - the free-energy Lyapunov MONITOR records E/S/F + entropy production per step and asserts F is
    non-increasing — the auditable evidence for the stability obligation (0642.1.2.2);
  - the conservation/entropy/degeneracy GUARDS pass for the structural operators, and a
    degeneracy-breaking (learned-style) operator trips the guard and deterministically falls back to
    the clamped-Euler baseline — never corrupting the run (0642.1.2.3);
  - the `metriplectic_integrator` toggle is default-off and registered with its prerequisite
    (0642.1.2.4).

Run:  pytest tests/test_metriplectic_runtime.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bio_inspired_nanochat import metriplectic_integrator as mi
from bio_inspired_nanochat.ablation_registry import _BY_FIELD, validate_config
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state

pytestmark = pytest.mark.unit

Z0 = np.array([1.0, 0.5, 0.0])


# --------------------------------------------------------------------------- #
# 1. The Lyapunov monitor (0642.1.2.2).
# --------------------------------------------------------------------------- #
def test_monitor_records_and_certifies_free_energy_lyapunov():
    traj, mon = mi.run_monitored(Z0, dt=0.1, steps=400)
    assert len(mon.records) == 400
    mon.assert_lyapunov()  # raises if F ever increases beyond tolerance
    s = mon.summary()
    assert s["lyapunov_ok"] is True
    assert s["max_energy_drift"] <= 1e-8, "energy must be conserved within the guard tolerance"
    assert s["min_entropy_production"] >= -1e-10, "entropy must be non-decreasing"
    assert s["n_fallbacks"] == 0, "the structural integrator must never need the fallback"


def test_monitor_detects_a_non_lyapunov_sequence():
    # Hand a fabricated increasing-F record stream to the monitor: it must flag it.
    mon = mi.LyapunovMonitor(tol=1e-9)
    for i, f in enumerate([1.0, 0.9, 0.95]):  # F goes back up at step 2
        mon.append(mi.StepRecord(i, 0, 0, f, 0, 0, 0, 0, False, ""))
    assert not mon.free_energy_nonincreasing()
    with pytest.raises(AssertionError, match="Lyapunov"):
        mon.assert_lyapunov()


# --------------------------------------------------------------------------- #
# 2. The guards + deterministic fallback (0642.1.2.3).
# --------------------------------------------------------------------------- #
def test_structural_operators_pass_all_guards():
    _, rec = mi.guarded_step(Z0, 0.1, 0, mi.GuardThresholds())
    assert rec.breach == "" and not rec.used_fallback
    assert rec.res_L_gradS < 1e-12 and rec.res_M_gradE < 1e-12, "structural degeneracy is exact"
    assert abs(rec.energy_drift) < 1e-10 and rec.entropy_production >= -1e-12


def test_degeneracy_breaking_operator_trips_guard_and_falls_back():
    # A "learned" friction that breaks M·∇E = 0 (an extra non-degenerate term).
    def bad_M(z, gC=mi.GAMMA_C, gB=mi.GAMMA_B):
        return mi.M_op(z, gC, gB) + np.diag([0.3, 0.3, 0.3])  # diag adds M·∇E ≠ 0

    res_ls, res_me = mi.degeneracy_residuals(Z0, M_fn=bad_M)
    assert res_me > 1e-2, "the broken operator must have a large M·∇E residual"

    _, rec = mi.guarded_step(Z0, 0.1, 0, mi.GuardThresholds(), M_fn=bad_M)
    assert rec.breach == "degeneracy" and rec.used_fallback, "must fall back on a degeneracy breach"


def test_energy_drift_guard_trips_on_a_loose_integrator():
    # Force a drift breach with an absurdly tight energy tolerance the exact step still satisfies
    # numerically — so instead inject a bad operator that makes the discrete step drift energy.
    def skew_breaking_L(omega=mi.OMEGA):
        return np.array([[0.2, omega, 0.0], [-omega, 0.0, 0.0], [0.0, 0.0, 0.0]])  # not skew

    _, rec = mi.guarded_step(Z0, 0.3, 0, mi.GuardThresholds(eps_D=1e9), L_fn=skew_breaking_L)
    # eps_D huge so the degeneracy guard does not pre-empt; the non-skew L drifts energy ⟹ fallback.
    assert rec.used_fallback and rec.breach in ("energy_drift", "entropy")


def test_run_monitored_with_a_bad_operator_uses_fallback_and_stays_safe():
    def bad_M(z, gC=mi.GAMMA_C, gB=mi.GAMMA_B):
        return mi.M_op(z, gC, gB) + np.diag([0.3, 0.3, 0.3])

    traj, mon = mi.run_monitored(Z0, dt=0.1, steps=100, M_fn=bad_M)
    s = mon.summary()
    assert s["n_fallbacks"] == 100, "every step must fall back when the operator is degenerate-broken"
    assert np.all(np.isfinite(traj)), "the safe fallback must keep the trajectory finite"


# --------------------------------------------------------------------------- #
# 3. The toggle + registry discipline (0642.1.2.4).
# --------------------------------------------------------------------------- #
def test_metriplectic_integrator_toggle_default_off_and_registered():
    assert SynapticConfig().metriplectic_integrator is False
    assert "metriplectic_integrator" in _BY_FIELD
    assert _BY_FIELD["metriplectic_integrator"].requires == ("enable_presyn",)


def test_validator_flags_integrator_without_presyn():
    errors, _ = validate_config(
        SynapticConfig(metriplectic_integrator=True, enable_presyn=False)
    )
    assert any("metriplectic_integrator" in e and "enable_presyn" in e for e in errors)


def test_default_config_still_validates_clean():
    assert validate_config(SynapticConfig()) == ([], [])


# --------------------------------------------------------------------------- #
# 4. Torch-native LIVE recurrence wiring (0642.1.2 compile bead).
# --------------------------------------------------------------------------- #
def test_torch_step_preserves_energy_produces_entropy_and_is_differentiable():
    calcium = torch.tensor([0.8, 0.3], dtype=torch.float64, requires_grad=True)
    buffer = torch.tensor([0.2, 0.1], dtype=torch.float64)
    heat = torch.tensor([0.0, 0.4], dtype=torch.float64)
    energy0 = 0.5 * (calcium.square() + buffer.square()) + heat

    c_next, b_next, h_next, record = mi.torch_guarded_step(
        calcium, buffer, heat, dt=0.25, omega=-0.1, gC=0.2, gB=0.1
    )
    energy1 = 0.5 * (c_next.square() + b_next.square()) + h_next

    torch.testing.assert_close(energy1, energy0, atol=1e-12, rtol=1e-12)
    assert torch.all(record.entropy_production >= -1e-12)
    assert torch.all(record.free_energy_delta <= 1e-12)
    assert not torch.any(record.fallback_mask)
    c_next.sum().backward()
    assert calcium.grad is not None and torch.all(torch.isfinite(calcium.grad))


def test_torch_step_matches_the_numpy_discrete_gradient_reference():
    z = np.array([0.8, 0.2, 0.1], dtype=np.float64)
    expected = mi.discrete_gradient_step(z, dt=0.1, omega=-0.1, gC=0.2, gB=0.1)
    c_next, b_next, h_next, record = mi.torch_guarded_step(
        torch.tensor(z[0]),
        torch.tensor(z[1]),
        torch.tensor(z[2]),
        dt=0.1,
        omega=-0.1,
        gC=0.2,
        gB=0.1,
    )

    assert expected.converged
    actual = np.array([c_next.item(), b_next.item(), h_next.item()])
    np.testing.assert_allclose(actual, expected.z_next, atol=1e-12, rtol=1e-12)
    assert not record.fallback_mask.item()


def test_torch_guard_selects_the_supplied_live_fallback_exactly():
    calcium = torch.tensor([0.8, 0.4], dtype=torch.float64)
    buffer = torch.tensor([0.2, 0.1], dtype=torch.float64)
    heat = torch.zeros_like(calcium)
    fallback = (
        torch.tensor([0.11, 0.22], dtype=torch.float64),
        torch.tensor([0.33, 0.44], dtype=torch.float64),
        torch.tensor([0.55, 0.66], dtype=torch.float64),
    )

    c_next, b_next, h_next, record = mi.torch_guarded_step(
        calcium,
        buffer,
        heat,
        dt=0.5,
        omega=-0.1,
        gC=-2.0,
        gB=-2.0,
        fallback=fallback,
    )

    assert torch.all(record.fallback_mask)
    assert torch.all(record.breach_code != 0)
    assert torch.equal(c_next, fallback[0])
    assert torch.equal(b_next, fallback[1])
    assert torch.equal(h_next, fallback[2])


def test_release_canonical_wires_metriplectic_heat_and_guard_ledger():
    cfg = SynapticConfig(
        metriplectic_integrator=True,
        stochastic_train_frac=0.0,
        barrier_strength=0.0,
    )
    presyn = SynapticPresyn(d_head=8, cfg=cfg)
    state = build_presyn_state(1, 5, 2, "cpu", torch.float64, cfg)
    drive = torch.full((1, 2, 3, 2), 0.7, dtype=torch.float64)
    idx = torch.tensor([[[[0, 1], [1, 2], [2, 3]]] * 2])

    release = presyn.release_canonical(state, drive, idx, train=False, differentiable=True)
    metrics = presyn.get_metriplectic_metrics()

    assert torch.isfinite(release).all()
    assert "HEAT" in state and torch.all(state["HEAT"] >= 0.0)
    assert torch.any(state["HEAT"] > 0.0), "dissipated calcium energy must enter the heat ledger"
    assert metrics["steps"] == state["C"].numel()
    assert metrics["fallbacks"] == 0
    assert metrics["last_max_energy_drift"] <= 1e-10
    assert metrics["last_min_entropy_production"] >= -1e-10
    assert metrics["last_max_free_energy_delta"] <= 1e-10


def test_live_metriplectic_recurrence_backpropagates_into_learnable_kinetics():
    cfg = SynapticConfig(
        metriplectic_integrator=True,
        learnable_kinetics=True,
        stochastic_train_frac=0.0,
    )
    presyn = SynapticPresyn(d_head=4, cfg=cfg)
    state = build_presyn_state(1, 3, 1, "cpu", torch.float64, cfg)
    drive = torch.full((1, 1, 2, 2), 0.6, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([[[[0, 1], [1, 2]]]])

    presyn.release_canonical(state, drive, idx, train=False, differentiable=True)
    loss = state["C"].sum() + state["BUF"].sum() + state["HEAT"].sum()
    loss.backward()

    assert presyn.kinetics is not None
    grads = [parameter.grad for parameter in presyn.kinetics.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.abs().sum()) for grad in grads if grad is not None) > 0.0


def test_default_state_has_no_heat_cost_and_enabled_old_cache_is_upgraded():
    default_cfg = SynapticConfig(metriplectic_integrator=False)
    assert "HEAT" not in build_presyn_state(1, 3, 1, "cpu", torch.float32, default_cfg)

    enabled_cfg = SynapticConfig(metriplectic_integrator=True, stochastic_train_frac=0.0)
    presyn = SynapticPresyn(d_head=4, cfg=enabled_cfg)
    old_state = build_presyn_state(1, 3, 1, "cpu", torch.float32, default_cfg)
    drive = torch.full((1, 1, 1, 1), 0.5)
    idx = torch.zeros((1, 1, 1, 1), dtype=torch.long)
    presyn.release_canonical(old_state, drive, idx, train=False)

    assert "HEAT" in old_state
    assert old_state["HEAT"].shape == old_state["C"].shape


def test_meta_device_model_initializes_live_guard_telemetry():
    from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        synapses=True,
        syn_cfg=SynapticConfig(
            metriplectic_integrator=True,
            stochastic_train_frac=0.0,
            enable_hebbian=False,
            enable_metabolism=False,
            native_genetics=False,
        ),
    )
    with torch.device("meta"):
        model = GPTSynaptic(cfg)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()

    presyn = next(module for module in model.modules() if isinstance(module, SynapticPresyn))
    assert presyn.get_metriplectic_metrics() == {
        "steps": 0,
        "fallbacks": 0,
        "last_max_energy_drift": 0.0,
        "last_min_entropy_production": 0.0,
        "last_max_free_energy_delta": 0.0,
    }
