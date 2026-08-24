# Lean formalization

This directory machine-checks the reduced metriplectic stability result consumed by the Python
runtime. The first certificate is `BioInspiredNanochat.lean`, corresponding to Bead
`bio_inspired_nanochat-r00r.4.1`.

## Scope and mapping

| Lean definition/theorem | Runtime surface | Contract |
| --- | --- | --- |
| `State`, `energy`, `entropy`, `freeEnergy` | `metriplectic_integrator.py::{energy,entropy,free_energy}` | Same reduced `(C, B, h)` functionals |
| `field` | `metriplectic_integrator.py::field` | Same explicit reversible/dissipative rates |
| `poisson_entropy_degeneracy` | `L_op @ grad_S` | `L ∇S = 0` |
| `friction_energy_degeneracy` | `M_op @ grad_E` | `M ∇E = 0` |
| `energyRate_zero`, `entropyRate_nonnegative` | continuous reference dynamics | `dE/dt = 0`, `dS/dt ≥ 0` |
| `freeEnergyRate_nonpositive` | continuous reference dynamics | `dF/dt ≤ 0` for `T ≥ 0` |
| `DiscreteStep` | an accepted `discrete_gradient_step` | Exact-energy and monotone-entropy abstraction |
| `freeEnergy_nonincreasing` | `LyapunovMonitor` | `F(z') ≤ F(z)` |
| `trajectory_component_bounds` | `integrate` / `run_monitored` | `C²+B² ≤ 2E₀`, `0 ≤ h ≤ E₀` |

Mapped runtime regressions are:

- `tests/test_metriplectic_theory.py`
- `tests/test_metriplectic_integrator.py`
- `tests/test_metriplectic_runtime.py`

## Abstraction-to-code gap

The Lean state uses exact real arithmetic. NumPy uses IEEE-754 `float64`, solves the implicit step by
fixed-point iteration, and accepts conservation/production within configured tolerances. Therefore
the proof establishes the mathematics of the explicit field and the `DiscreteStep` contract; it does
not prove that every floating-point proposal satisfies that contract. The runtime closes that gap
operationally with `degeneracy_residuals`, energy/entropy guards, `LyapunovMonitor`, and a deterministic
fallback. A guard-accepted proposal is the concrete witness for `DiscreteStep`; a rejected proposal
is outside this theorem and follows the fallback path.

The formal result also does not prove fixed-point convergence, IEEE rounding-error bounds, strict
entropy production, convergence to the MaxEnt equilibrium, compactness of real-valued energy shells,
or safety of the fallback. It proves the explicit squared-coordinate/heat bounds that are needed for
boundedness once exact energy conservation and nonnegative heat hold. Those omitted concerns remain
runtime-test obligations; the later `r00r.4.3` feedback-loop/CI bead owns automated drift gating.

## Conformance record

- Statement parity: pass for the reduced core and accepted-step contract.
- State parity: pass for `(C, B, h)`; full synaptic state and vesicle Casimir are explicit non-goals.
- Transition parity: pass for `field`; discrete iteration and guard rejection are documented gaps.
- Concurrency parity: not applicable; these functions are pure array transforms.
- Cancellation/drain parity: not applicable.
- Runtime evidence: the three mapped pytest modules cover degeneracy, conservation, entropy,
  Lyapunov monotonicity, bounds, monitoring, and fallback.
- Drift gate: Lean and runtime mappings were checked against the repository state recorded in
  `proof_artifacts.json`.

## Build

The toolchain and Mathlib release are pinned. With `elan`/`lake` installed:

```bash
cd formal/lean
lake update
lake exe cache get
lake build
```

The `#print axioms` commands at the end of the Lean module provide an explicit axiom audit during
compilation. The proof artifact hash and mapped runtime evidence live in `proof_artifacts.json`.
