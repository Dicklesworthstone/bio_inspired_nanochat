# The Master Neural SDE — Capstone Theory Note (bead `0642.11.1`)

_Grand synthesis: each thrust as a restriction of one geometric object. Author: BeigeSquirrel · 2026-06-14._

## Purpose & scope

The per-thrust notes each cast **one** stratum of the synapse in **one** math family. This note writes
the **single master object** they are all restrictions of, and makes the correspondence **explicit and
falsifiable**: a committed table mapping each thrust `A / D / E / F` to a component, limit, or section
of the master neural SDE, plus a reference module (`bio_inspired_nanochat/master_sde.py`) and tests
(`tests/test_master_sde.py`) that check the restrictions actually reduce to the existing thrust code.

This is the first capstone subtask (`0642.11.1`); it feeds the composition-consistency proof
(`0642.11.2`) and the minimal-shared-state/API definition (`0642.11.3`). It does **not** introduce new
runtime behavior — it is a unifying *reading* of the dynamics already specified by
[metriplectic.md](metriplectic.md) (A), [singular_perturbation.md](singular_perturbation.md) (F),
[stochastic_thermodynamics.md](stochastic_thermodynamics.md) (E), and the (planned) gauge note
`0642.7.1` (D).

---

## 0. The master object

The synaptic transformer is a **stochastic, gauge-covariant, metriplectic neural SDE on a fiber
bundle with separated timescales**. Concretely:

- **Base** `𝔅`: the computational "where" — the sequence/depth index `x = (t, ℓ)` (token position `t`,
  layer `ℓ`). Advancing the base is advancing the forward pass.
- **Fiber** `𝔉`: the **synaptic state** `z` attached to each base point — the calcium core
  `(C, B, h)`, the release state, the fast weights, the slow weights, and the structural state. The
  total space is the bundle `E = 𝔅 × 𝔉` (locally), sections of which are the live model state.
- **Connection** `𝒜`: a gauge connection on the bundle — a rule for **parallel-transporting** the
  fiber across the base (layer→layer, timescale→timescale). Its covariant derivative is
  `∇^𝒜 = d + 𝒜`.
- **Vertical dynamics** on the fiber: a **GENERIC / metriplectic** drift plus a **vesicle-noise**
  diffusion.

The master SDE, written covariantly on a section `z(x)`, is

```
        ∇^𝒜 z  =  [ L(z)·∇E(z) + M(z)·∇S(z) ] dt  +  σ(z)·dW ,          (★)
```

equivalently, expanding the covariant derivative along a base increment `dx`,

```
        dz  =  −𝒜(z; dx)·z  +  [ L(z)·∇E(z) + M(z)·∇S(z) ] dt  +  σ(z)·dW .
        \____ horizontal ___/    \________ vertical drift ________/    \ vert. diffusion /
                 (D)                          (A)                            (E)
```

with the structural side-conditions that make the guarantees hold *by construction*:

| object | condition | gives | thrust |
|---|---|---|---|
| `L` | skew, `L·∇S = 0` (Jacobi) | reversible; `S` a Casimir; `dE/dt\|_rev = 0` | A |
| `M` | symmetric PSD, `M·∇E = 0` | dissipative; `dS/dt ≥ 0` | A |
| `σ` | fluctuation–dissipation `σσᵀ = 2T·M` (on the dissipative coords) | Gibbs stationary `∝ e^{−E/T}`; FT-calibrated noise | E |
| `𝒜` | gauge-covariant: under `z → U(x)z`, `𝒜 → U𝒜U⁻¹ + U dU⁻¹` | transport independent of fiber frame | D |
| `ε` | `ε = τ_fast/τ_slow ≪ 1` (the separation gauge) | normally-hyperbolic slow manifold `M_ε` | F |

Learning is on the *triple* `(brackets L,M ; connection 𝒜 ; diffusion σ)` subject to those
conditions. The point of (★) is that the four thrusts are not four theories — they are four
**restrictions of one**, and (because the conditions are structural) their guarantees can be made to
**compose** rather than collide (`0642.11.2`, gated by the separation certificate `0642.10`).

The free energy `F = E − T·S` is the master Lyapunov functional of the **vertical drift**; the
diffusion adds the matching fluctuations whose **stationary Gibbs** law is `e^{−E/T}` (with `F` its
free energy); the connection transports it across the base; the `ε`-limit restricts it to the slow
manifold. Every thrust is a way
of reading `F`.

---

## 1. Thrust A — the deterministic metriplectic drift  (`σ → 0`, `𝒜 → 0`)

Set the diffusion to zero, take the trivial connection (a single fiber, no transport), and restrict
the state to the calcium core `z = (C, B, h)`. Then (★) collapses to the GENERIC ODE of
[metriplectic.md](metriplectic.md):

```
        dz/dt  =  L(z)·∇E(z) + M(z)·∇S(z),     E = ½C² + ½B² + h,   S = h,
```

with `L` the skew calcium↔buffer rotation and `M = γ_C uuᵀ + γ_B vvᵀ` the PSD friction. The degeneracy
conditions `L∇S = 0`, `M∇E = 0` give `dE/dt = 0`, `dS/dt ≥ 0`, so `F = E − T·S` is non-increasing and
the trajectory is bounded on the compact energy shell `Σ_{E₀}` — the **stability keystone**. The
structure-preserving realization is the discrete-gradient integrator `metriplectic_integrator.py`
(energy conserved + entropy monotone at finite step).

> **Restriction (A):** master SDE with `σ ≡ 0`, `𝒜 ≡ 0`, `z = (C,B,h)`  ⟹  the metriplectic ODE.
> **Falsifiable:** `master_drift(z, σ=0, 𝒜=0)` is bit-for-bit `metriplectic_integrator.field(z)`
> (checked in `test_master_sde.py::test_metriplectic_restriction_matches_reference`).

---

## 2. Thrust E — the vertical diffusion  (`σ ≠ 0`, fluctuation–dissipation)

Keep the diffusion `σ(z)·dW` and restrict to the **release coordinate** (the vesicle pool / current).
Two equivalent readings of the same noise:

- **Diffusion (Langevin) reading.** With the Einstein/fluctuation–dissipation relation
  `σσᵀ = 2T·M` on the dissipative coordinates, the Fokker–Planck stationary density of (★) is the
  **Gibbs measure** `ρ_∞ ∝ e^{−E/T}` (the energy `E` is the Boltzmann potential; `F = E − T·S` is the
  corresponding free energy), and the entropy
  production rate is `≥ 0` — the second law, now for the *noisy* dynamics. For the linearized
  dissipative block `dz = −Mz dt + σ dW`, the stationary covariance solves the Lyapunov balance
  `M·Cov + Cov·Mᵀ = σσᵀ`, which with `σσᵀ = 2T·M` gives `Cov = T·I` — equipartition.
- **Jump (Skellam) reading.** The physical engine is discrete: `K ~ Binomial(N, p)` releases vs
  `rec_rate` recoveries, a `Docked ⇌ Released` Markov jump process with affinity `A = ln(a/b)` and
  entropy production `Σ = J·A` obeying the integral/detailed fluctuation theorems and the TUR
  `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩` — exactly [stochastic_thermodynamics.md](stochastic_thermodynamics.md) /
  `stochastic_thermo.py`. The diffusion `σdW` is the diffusion-limit (van Kampen) of this jump term.

The link `σσᵀ = 2T·M` is what ties Thrust E's noise to Thrust A's dissipation: **same `M`, two faces**
(`σ → 0` recovers A; `σ ≠ 0` is E). `T` is the ACh-set release temperature (`hy8.5`), whose
energy-optimal value is the Landauer `kT* ≈ 0.505·σ_drive`.

> **Restriction (E):** master SDE keeping `σ` with `σσᵀ = 2T·M`  ⟹  Gibbs stationary `∝ e^{−E/T}`
> and (in the jump representation) the FT/TUR thermo of Thrust E.
> **Falsifiable:** the linear-block stationary covariance equals `T·I` under the FDR
> (`test_master_sde.py::test_thermo_restriction_fdr_gibbs`).

---

## 3. Thrust F — the singular (slow-manifold) limit  (`ε → 0`)

Make the **timescale hierarchy** explicit by splitting the fiber `z = (x_fast, y_slow)` and scaling
the fast drift by `1/ε`:

```
        ε · dx = f(x, y) dt + √ε · σ_x dW_x ,        dy = g(x, y) dt + σ_y dW_y ,
```

with `x_fast` = calcium subsystem `(C, BUF)` and `y_slow` = the latch `(m, p)`. As `ε → 0`
(timescale separation), **Fenichel** gives a normally-hyperbolic attracting slow manifold
`M_ε = {x = h(y, ε) = h₀(y) + O(ε)}`; the fast noise averages out (stochastic averaging /
Katzenberger), and the **reduced slow flow** `dy = g(h₀(y), y) dt + …` is the memory dynamics —
on which the latch reduces to the **cusp normal form** `m̃³ + a m̃ + b` with the closed-form retention
half-width `δ*(a) = (2/3√3)(−a)^{3/2}` ([singular_perturbation.md](singular_perturbation.md),
`cusp_certificate.py`). The reduction is valid exactly while the **ε-gauge** `ε̂ = τ_fast/τ_slow` is
small — measured at runtime by `separation_gauge.py` and gated by the composition keystone
(`0642.10`).

> **Restriction (F):** master SDE with the `1/ε` fast-drift scaling, `ε → 0`  ⟹  the Fenichel slow
> manifold + reduced cusp flow of Thrust F.
> **Falsifiable:** at small `ε` the integrated fast variable tracks `h₀(y)` and the slow trajectory
> matches the reduced flow to `O(ε)` (`test_master_sde.py::test_slow_manifold_restriction`).

---

## 4. Thrust D — the gauge connection  (the horizontal part)

The other three thrusts are **vertical** (they move within a fiber). Thrust D is **horizontal**: the
connection `𝒜` and its covariant transport across the base. The synapse must compare and carry state
across layers and across timescales — fast→slow consolidation, layer-to-layer transport — and the
*physics must not depend on the arbitrary frame* chosen for the synaptic state. That is **gauge
covariance**: for a fiber-wise frame change `z → U(x)·z` (with `U` in the structure group, e.g.
orthogonal so the metriplectic `E = ½‖·‖² + h` is preserved), the connection transforms as
`𝒜 → U𝒜U⁻¹ + U·dU⁻¹` and the covariant derivative `∇^𝒜 z` transforms tensorially (`∇^𝒜 z → U∇^𝒜 z`).
So (★) is **form-invariant** under gauge transformations — the guarantees of A/E/F are frame-independent.

Consolidation (fast `W_fast` → slow `W_slow`) and cross-layer transport are the connection's action
along the *timescale* and *depth* directions of the base; the curvature `F_𝒜 = d𝒜 + 𝒜∧𝒜` measures
path-dependence of transport (a nonzero curvature ⟹ consolidation order matters). The full
info-geometry/gauge ledger is the (still-open) bead `0642.7.1`; here D enters the master object as the
**connection component**, characterized structurally with its proof obligations deferred there.

> **Restriction (D):** master SDE keeping only the covariant-transport term `∇^𝒜` ⟹ parallel transport
> of synaptic state across the base, with gauge-covariance the defining invariance.
> **Falsifiable (the part that does not need the unbuilt D module):** for an orthogonal gauge `U`, the
> covariant step **commutes** with the gauge transform — transport-then-evolve equals
> evolve-then-transport (`test_master_sde.py::test_gauge_covariance_commutes`).

---

## 5. The master correspondence table  (the deliverable, `0642.11.1`)

| Thrust | Math family | Stratum | **Restriction of (★)** | Limit / component | Reference note · module |
|---|---|---|---|---|---|
| **A** | Metriplectic / GENERIC | calcium | `σ → 0`, `𝒜 → 0`, core `(C,B,h)` | the **deterministic vertical drift** `L∇E + M∇S` | [metriplectic.md](metriplectic.md) · `metriplectic_integrator.py` |
| **E** | Stochastic thermodynamics | release | keep `σ`, `σσᵀ = 2T·M` | the **vertical diffusion** `σdW` (FDR ↔ A's `M`); jump-limit = Skellam FT/TUR | [stochastic_thermodynamics.md](stochastic_thermodynamics.md) · `stochastic_thermo.py` |
| **F** | Singular perturbation / cusp | fast_weights | `1/ε` fast-drift scaling, `ε → 0` | the **slow-manifold reduction** of the drift (Fenichel + averaging) | [singular_perturbation.md](singular_perturbation.md) · `cusp_certificate.py` |
| **D** | Gauge / info-geometry | (transport) | keep `∇^𝒜`, structure-group frame changes | the **horizontal connection** `𝒜`; covariance under `z→Uz` | `0642.7.1` (planned) · — |

Reading the table: **A is the drift, E is the diffusion, F is the singular limit of the drift, D is
the connection.** A and E share the operator `M` (drift dissipation = noise covariance, via the FDR);
F is A restricted to the slow manifold; D is the geometry on which all three are sections. The vesicle
pool `N` is a **Casimir** of the vertical dynamics (conserved by `L`, `M`, and — in the conservative
limit — `σ`), so every restriction lives on a leaf `{N = const}`.

---

## 6. Proof-obligation & assumptions ledger  → consumed by `0642.11.2`, `0642.11.3`

| # | Assumption (how discharged) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| M1 | **Drift structure** (A1–A2 of metriplectic.md): `L` skew + Jacobi, `M` PSD + `M∇E=0`. | the `σ→0` drift is GENERIC ⟹ A's conservation/Lyapunov. | a learned `L/M` breaks skew/PSD/degeneracy. | project `L,M` to the structural cone; else clamped Euler (per metriplectic A1–A2). |
| M2 | **Fluctuation–dissipation** `σσᵀ = 2T·M` on dissipative coords (this note §2). | the diffusion's stationary law is Gibbs `e^{−E/T}`; noise ties to A's `M`. | a learned `σ` not matched to `M` ⟹ no Gibbs stationary / FT broken. | symmetrize to the FDR-consistent `σ`; or gate on the empirical FT test (`0642.3.3.1`). |
| M3 | **Timescale separation** `ε̂ < eps_max` (separation gauge, `0642.10`). | Fenichel slow manifold exists ⟹ F's reduction valid; restrictions decouple. | `ε̂ ≈ 1` (e.g. `release→fast_weights` at defaults, `ε≈1.63`). | the offending thrust's fallback trips (composition keystone `0642.10.2/.3`). |
| M4 | **Gauge covariance** `𝒜 → U𝒜U⁻¹ + U dU⁻¹` under structure-group `U` (this note §4). | the master object is frame-independent ⟹ guarantees transport. | a non-covariant transport / `U` outside the structure group (e.g. non-orthogonal, breaking `E`). | restrict `U` to the structure group; full ledger deferred to `0642.7.1`. |
| M5 | **Casimir / leaf** `N = RRP+RES+Σdelay` conserved by all of `L,M,σ` (metriplectic A5). | each restriction lives on `{N=const}`; composition stays on one leaf. | a non-conservative noise/refill leaks `N`. | paired-transfer refill (already enforced); discrete-gradient integrator preserves `N`. |

**Composition obligation (handed to `0642.11.2`).** Under M1–M5 *jointly* (the separation certificate
M3 being the binding one), the per-thrust guarantees compose: A's boundedness on the energy shell, E's
FT calibration on the release coordinate, F's retention on the slow manifold, transported gauge-
covariantly by D. The consistency proof — that these do not interfere when each stratum's incoming
coupling is separated — is `0642.11.2`; the minimal state/API each implementation must expose so that
`(L, M, σ, 𝒜, ε̂)` are all readable from one object is `0642.11.3`.

---

## 7. Numerical corroboration

`tests/test_master_sde.py` makes the table falsifiable by checking each restriction against the live
thrust code (or its self-contained signature where the thrust module exists only as math):

- **A** — `master_drift(z, σ=0, 𝒜=0)` equals `metriplectic_integrator.field(z)` on a grid; the
  Euler–Maruyama step with `σ=0` equals the deterministic Euler step; degeneracy `L∇S=0`, `M∇E=0` hold.
- **E** — under the FDR `σσᵀ = 2T·M`, the linear dissipative block's stationary covariance equals
  `T·I` (equipartition / Gibbs), and the sampled entropy-production rate is `≥ 0` (second law).
- **F** — for the fast–slow form at small `ε`, the fast variable tracks the quasi-steady `h₀(y)`
  (slaving) and the slow trajectory matches the reduced flow to `O(ε)`; the error shrinks with `ε`.
- **D** — for an orthogonal gauge `U`, the covariant step commutes with the gauge transform
  (transport∘evolve = evolve∘transport) and preserves `E = ½‖·‖² + h` (structure-group invariance).

These confirm that the four thrusts are *literally* restrictions of one object — the claim the capstone
rests on — rather than four separately-asserted theories.

---

## References

- Grmela, M. & Öttinger, H.C. (1997). *Dynamics and thermodynamics of complex fluids I–II.* Phys. Rev. E 56. — GENERIC (the drift).
- Fenichel, N. (1979). *Geometric singular perturbation theory for ODEs.* J. Diff. Eq. 31. — the slow manifold (F).
- Katzenberger, G.S. (1991). *Solutions of a SDE forced onto a manifold by a large drift.* Ann. Probab. — stochastic averaging onto `M_ε`.
- Seifert, U. (2012). *Stochastic thermodynamics, fluctuation theorems and molecular machines.* Rep. Prog. Phys. — the diffusion/jump thermo (E).
- Kobayashi & Nomizu (1963). *Foundations of Differential Geometry.* — connections on fiber bundles (D).
- Internal: [metriplectic.md](metriplectic.md) (A), [singular_perturbation.md](singular_perturbation.md) (F), [stochastic_thermodynamics.md](stochastic_thermodynamics.md) (E), [README.md](README.md) (the unifying picture), `separation_gauge.py` / `composition.py` (`0642.10`, the timescale-separation certificate).
