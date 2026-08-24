# Lean formalization

This directory machine-checks three reduced mathematical contracts consumed by the Python runtime:
metriplectic Lyapunov stability (`r00r.4.1`), cusp retention, and stochastic-thermodynamic
calibration (`r00r.4.2`). They are exact-real certificates for deliberately scoped models, not proofs
of every floating-point runtime path.

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
| `deltaStar`, `fold_discriminant_zero` | `cusp_certificate.py::{retention_delta_star,cusp_constants}` | Runtime formula is the exact fold half-width of the constructed depressed cubic |
| `inside_retention_window_iff_discriminant_pos` | `CuspMonitor.retention_margin` | `|b| < δ*` is exactly the open positive-discriminant wedge |
| `perturbation_below_margin_stays_inside` | `RetentionCertificate` / `CuspMonitor` | Noise below `δ*−|b|` cannot cross a cubic fold |
| `log_rate_tur_core`, `tur_calibration_inequality` | `stochastic_thermo.py::tur_certificate` | Given the analytic driven model's mean, variance, and entropy formulas, `Var(J)/mean(J)² ≥ 2/mean(Σ)` |
| `crooks_log_calibration` | `stochastic_thermo.py::crooks_calibration` | Exact detailed-FT probability ratio implies the log-histogram calibration equation |

Mapped runtime regressions are:

- `tests/test_metriplectic_theory.py`
- `tests/test_metriplectic_integrator.py`
- `tests/test_metriplectic_runtime.py`
- `tests/test_cusp_certificate.py`
- `tests/test_cusp_falsification.py`
- `tests/test_singular_perturbation_theory.py`
- `tests/test_stochastic_thermo.py`

## Abstraction-to-code gap

### Metriplectic model

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

### Cusp-retention model

Lean proves the fold geometry of the constructed depressed cubic `u³+a·u+b`, including the runtime
formula `δ*(a) = 2/(3√3)·(-a)·√(-a)`, the positive-discriminant retention window, and the remaining
margin under additive bias noise. It assumes the singular-perturbation/cusp chart is valid and
`a < 0`.

The physical latch is broader. `cusp_coefficients` freezes PP1 at its configured basal chart value
and estimates derivatives in `float64`; the full two-state latch evolves and clamps both CaMKII and
PP1. Substituting the exact slaving function `p(m)` introduces quadratic terms before a further
coordinate translation, so Lean does not claim that the runtime's frozen-PP1 coefficients are the
translated coefficients of that full model. It also does not prove Fenichel persistence, root
stability/basin invariance, chart membership in `[0,1]`, the epsilon-gauge proxy, finite-difference
accuracy, or IEEE equivalence. The cusp regression and falsification tests remain the evidence for
those runtime obligations.

### Stochastic-thermodynamic model

Lean proves an algebraic TUR inequality for `0 < b < a` and positive observation time after taking
the analytic model's mean-current, variance, and mean-entropy formulas as definitions. The intended
interpretation derives those formulas from independent Poisson forward/reverse jumps, Skellam
moments, and local detailed balance, but that probabilistic derivation is an external assumption,
not represented in Lean. The log-histogram theorem is likewise conditional on the exact detailed
fluctuation ratio. Lean does not universally certify finite-sample `empirical_tur`, nonstationary or
hidden-state release, the live paired-binomial experiment, ECE, or the model's predictive
distribution. Those paths must pass their empirical FT/fallback gates independently.

## Conformance record

- Statement parity: pass for the reduced metriplectic core, constructed cusp cubic, conditional
  driven-model TUR algebra, and Crooks implication.
- State parity: pass for `(C, B, h)`, scalar cubic `(a,b)`, and population moments; full synaptic
  state, evolving PP1, and live binomial trajectories are explicit non-goals.
- Transition parity: pass for `field`; discrete iteration, cusp coefficient extraction, and empirical
  release sampling are documented gaps guarded by mapped runtime tests/fallbacks.
- Concurrency parity: not applicable; these functions are pure array transforms.
- Cancellation/drain parity: not applicable.
- Runtime evidence: the mapped pytest modules cover metriplectic guards, the cusp formula and physical
  conservatism, analytic TUR moments, FT calibration/falsification, and deterministic fallbacks.
- Drift gate: Lean and runtime mappings were checked against the repository state recorded in
  `proof_artifacts.json`.

## Build

The toolchain and Mathlib release are pinned. With `elan`/`lake` installed:

```bash
cd formal/lean
lake update
lake exe cache get
rch exec -- lake build
```

The `#print axioms` commands at the end of the Lean module provide an explicit axiom audit during
compilation. The proof artifact hash and mapped runtime evidence live in `proof_artifacts.json`.
That artifact is consumed by the planned `r00r.4.3` drift gate, prevents theorem/runtime mapping
changes from being silently accepted, and should be retired in favor of CI-generated records once
that gate owns the evidence automatically.
