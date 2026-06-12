"""
Discrete-gradient structure-preserving integrator for the metriplectic core (bead 0642.1.2.1).

Locks the acceptance: the integrator conserves energy EXACTLY and produces entropy monotonically at
the DISCRETE level (any step), so the Lyapunov certificate of docs/theory/metriplectic.md holds for
the actual code — unlike a naive Euler step.

  1. the Gonzalez discrete gradient satisfies its defining directional property exactly;
  2. discrete energy drift ≤ 1e-10 across a range of steps (acceptance);
  3. discrete entropy is monotone non-decreasing and F = E − T·S is non-increasing (Lyapunov);
  4. the trajectory relaxes to the MaxEnt equilibrium z* = (0, 0, E₀);
  5. it reduces to forward Euler at first order (consistency);
  6. forward Euler (the vg9 baseline) drifts E by ~12 orders of magnitude more;
  7. the implicit step's fixed-point iteration converges in bounded iterations (overhead).

See docs/theory/metriplectic.md §4–§5.  Run:  pytest tests/test_metriplectic_integrator.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat import metriplectic_integrator as mi

pytestmark = pytest.mark.unit

Z0 = np.array([1.0, 0.5, 0.0])
E0 = mi.energy(Z0)


# --------------------------------------------------------------------------- #
# 1. The Gonzalez discrete gradient's defining property.
# --------------------------------------------------------------------------- #
def test_discrete_gradient_directional_property_is_exact():
    rng = np.random.default_rng(0)
    for _ in range(50):
        z, zp = rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
        gbar = mi.discrete_gradient(mi.grad_E, mi.energy, z, zp)
        # (z' − z)·∇̄E must equal E(z') − E(z) exactly.
        assert (zp - z) @ gbar == pytest.approx(mi.energy(zp) - mi.energy(z), abs=1e-12)


def test_discrete_gradient_reduces_to_true_gradient_when_states_coincide():
    z = np.array([0.3, -0.7, 1.2])
    assert np.allclose(mi.discrete_gradient(mi.grad_E, mi.energy, z, z.copy()), mi.grad_E(z))


# --------------------------------------------------------------------------- #
# 2. Exact discrete energy conservation (the acceptance: drift ≤ 1e-10).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1, 0.5])
def test_discrete_energy_conserved_to_machine_precision(dt: float):
    traj = mi.integrate(Z0, dt=dt, steps=400)
    drift = float(np.max(np.abs([mi.energy(z) for z in traj] - E0 * np.ones(len(traj)))))
    assert drift <= 1e-10, f"discrete energy drift {drift:.2e} exceeds 1e-10 at dt={dt}"


# --------------------------------------------------------------------------- #
# 3. Discrete entropy production + free-energy Lyapunov.
# --------------------------------------------------------------------------- #
def test_discrete_entropy_monotone_and_free_energy_nonincreasing():
    traj = mi.integrate(Z0, dt=0.1, steps=500)
    Ss = np.array([mi.entropy(z) for z in traj])
    Fs = np.array([mi.free_energy(z) for z in traj])
    assert np.all(np.diff(Ss) >= -1e-12), "discrete entropy must be non-decreasing"
    assert np.all(np.diff(Fs) <= 1e-12), "discrete F = E − T·S must be non-increasing (Lyapunov)"


# --------------------------------------------------------------------------- #
# 4. Relaxation to the MaxEnt equilibrium z* = (0, 0, E₀).
# --------------------------------------------------------------------------- #
def test_relaxes_to_maxent_equilibrium():
    traj = mi.integrate(Z0, dt=0.2, steps=2000)
    z_star = traj[-1]
    assert abs(z_star[0]) < 1e-3 and abs(z_star[1]) < 1e-3, "calcium must relax to 0"
    assert abs(z_star[2] - E0) < 1e-3, "heat must reach E₀ (MaxEnt on the shell)"


# --------------------------------------------------------------------------- #
# 5. First-order consistency with forward Euler.
# --------------------------------------------------------------------------- #
def test_reduces_to_euler_at_first_order():
    dt = 1e-4
    z_dg = mi.discrete_gradient_step(Z0, dt).z_next
    z_euler = Z0 + dt * mi.field(Z0)
    # The schemes agree to O(dt²); at dt=1e-4 the gap is ~1e-8.
    assert np.max(np.abs(z_dg - z_euler)) < 1e-6, "integrator must match Euler at first order"


# --------------------------------------------------------------------------- #
# 6. The baseline contrast: forward Euler destroys energy conservation.
# --------------------------------------------------------------------------- #
def test_forward_euler_drifts_energy_orders_of_magnitude_more():
    dt, steps = 0.1, 500
    dg = mi.integrate(Z0, dt=dt, steps=steps)
    eu = mi.euler_integrate(Z0, dt=dt, steps=steps)
    dg_drift = float(np.max(np.abs([mi.energy(z) for z in dg] - E0 * np.ones(len(dg)))))
    eu_drift = float(np.max(np.abs([mi.energy(z) for z in eu] - E0 * np.ones(len(eu)))))
    assert eu_drift > 1e-2, f"the vg9-style Euler baseline must drift E noticeably ({eu_drift:.2e})"
    assert eu_drift > 1e6 * dg_drift, (
        f"the discrete-gradient integrator must conserve E far better than Euler "
        f"(dg {dg_drift:.2e} vs euler {eu_drift:.2e})"
    )


# --------------------------------------------------------------------------- #
# 7. Bounded overhead: the implicit fixed-point converges.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5])
def test_fixed_point_converges_in_bounded_iterations(dt: float):
    res = mi.discrete_gradient_step(Z0, dt)
    assert res.converged, f"the implicit step must converge at dt={dt}"
    assert res.iters <= 60, f"convergence took {res.iters} iters at dt={dt} (overhead bound)"
