# The Leapfrog Theory Program — START HERE (bead `0642.9`)

> _Entry point for the theory program (`label: theory`)._ Kept current as thrusts land.

## The unifying picture

The synaptic transformer is best read as a **stochastic, gauge-covariant, metriplectic neural SDE on a
fiber bundle with separated timescales** — the master object formalized in
[master_sde.md](master_sde.md) (capstone `0642.11`), of which every thrust below is a restriction. The
synapse has a hierarchy of strata, each evolving on its own timescale:

```
        calcium  ≪  release  ≪  fast_weights  ≪  slow_weights  ≪  structure
```

Each **thrust** applies the *right* piece of esoteric mathematics to *one* stratum, and the
**composition keystone** (`0642.10`) certifies — at runtime, from the measured timescale separation —
that the strata are decoupled enough for the thrusts to compose without interfering.

## The discipline (every thrust obeys it)

1. **Compile** the math into runtime artifacts (a certificate / monitor / kernel).
2. State a **falsifiable hypothesis** vs a *named* baseline, tested with multi-seed stats (`74f`).
3. Carry an explicit **proof-obligation & assumptions ledger**.
4. Ship a **deterministic fallback** to the engineering baseline, so a failed bet never breaks
   training — behind a **default-off toggle** (`hm4.7`).

## Reading order (by EV and dependency)

1. **Thrust A — Metriplectic/GENERIC dynamics** (the stability keystone; do first).
2. **Thrust F — Singular-perturbation / cusp** (certified memory + the timescale-separation backbone).
3. **The composition keystone** (`0642.10`) — the timescale-separation certificate that licenses
   composing everything else.
4. **Thrust E** (thermo UQ), **Thrust B** (ultrametric memory), then **Thrust C** (topological NAS).
5. Planned: **H** (tropical), **D** (gauge), **G** (sheaf).

## Thrust map

| Thrust | Stratum | Math family | Status | Theory note | Reference module | Headline artifact |
|---|---|---|---|---|---|---|
| **A** | calcium | Metriplectic / GENERIC | ✅ implemented | [metriplectic.md](metriplectic.md) | `metriplectic_integrator.py` | structure-preserving integrator + free-energy Lyapunov monitor |
| **F** | fast_weights | Singular perturbation / cusp catastrophe | ✅ implemented | [singular_perturbation.md](singular_perturbation.md) | `cusp_certificate.py` | certified cusp latch (`δ*` retention) + ε-gauge monitor |
| **E** | release | Stochastic thermodynamics (FT / TUR) | ✅ implemented | [stochastic_thermodynamics.md](stochastic_thermodynamics.md) | `stochastic_thermo.py` | TUR certificate + Crooks calibration monitor + Landauer temperature |
| **B** | slow_weights | Ultrametric / RSB + p-adic | ✅ theory + reference | [ultrametric_memory.md](ultrametric_memory.md) | `ultrametric_memory.py` | p-adic LCP kernel + tree-ness gauge + capacity certificate |
| **C** | structure | Free probability + TDA + optimal transport | ✅ theory + reference | [structural_geometry.md](structural_geometry.md) | `structural_geometry.py` | spectral-conditioning + H0 coverage + OT-barycenter merge |
| **H** | — | Tropical geometry | 🔮 planned | — | — | — |
| **D** | — | Gauge theory | 🔮 planned | — | — | — |
| **G** | — | Sheaf theory | 🔮 planned | — | — | — |

**Capability layer (built on the theory):**

| Capability | Builds on | Note / module |
|---|---|---|
| Free-energy deliberation + energy-based decoding (`r00r.1`) | Thrust A | [free_energy_deliberation.md](free_energy_deliberation.md) · `deliberation.py` |
| Bayesian MC ensembling (`u2t.1`) | Thrust E | `mc_ensemble.py` |

## The capstone (`0642.11`)

[master_sde.md](master_sde.md) writes the single object the thrusts are restrictions of —
`∇^𝒜 z = [L∇E + M∇S]dt + σdW` on a fiber bundle — and gives the explicit **correspondence table**:
**A** is the deterministic drift (`σ→0`), **E** the diffusion (`σσᵀ=2T·M`), **F** the slow-manifold
limit of the drift (`ε→0`), **D** the gauge connection (covariant transport). Reference module
`master_sde.py` + `tests/test_master_sde.py` make each restriction falsifiable (e.g. the `σ→0` drift
is bit-for-bit `metriplectic_integrator.field`). The composition-consistency proof (`0642.11.2`) and
the minimal shared state/API (`0642.11.3`) build on this mapping.

## The composition keystone (`0642.10`)

Before composing multiple math families, the alien-artifact discipline requires a
**timescale-separation statement**. The keystone measures `ε_k = τ_fast / τ_slow` per coupling
(`separation_gauge.py`) and gates each thrust certificate on it (`composition.py`):

- **Eligibility guard** (`0642.10.2`): a thrust *composes* iff its coupling is separated (`ε_k < eps_max`),
  else its deterministic fallback trips. _Honest finding:_ at the **defaults** the
  `release→fast_weights` coupling is not separated (`ε≈1.63`), so the cusp thrust F falls back there.
- **Pairwise interference harness** (`0642.10.3`): two thrusts may co-activate only if every boundary
  *between* their strata is separated; otherwise the **higher-risk** thrust is auto-disabled and the
  lower-risk keystone keeps running.

## How to use these notes

- Each per-thrust note follows the same skeleton: the dynamical system *as it actually is* → the
  results with derivations → a **proof-obligation & assumptions ledger** (with fail-closed fallbacks)
  → **numerical corroboration** against the live code (in `tests/`).
- The reference modules are pure, dependency-light (`numpy`/`torch`), and default-off; nothing here
  changes the live model unless a toggle is set.
- Falsification beads (`0642.*.3`) carry the headline hypotheses; several await trained-model evals.
