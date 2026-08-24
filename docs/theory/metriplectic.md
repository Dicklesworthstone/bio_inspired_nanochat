# Metriplectic / GENERIC Synaptic Dynamics — Theory Note (bead `0642.1.1`)

_Thrust A — stability & conservation by construction. Author: GoldenRiver · 2026-06-12._

## Purpose & scope

This note casts the synaptic relaxation dynamics in **GENERIC / metriplectic** form — the structure
that makes *energy conservation*, *entropy production*, and a *free-energy Lyapunov function*
hold **by construction** rather than by clamping. It fixes the contract the downstream beads build
against: the structure-preserving (discrete-gradient) integrator `0642.1.2.1`, the Lyapunov /
domain-of-attraction subtask `0642.1.1.5`, the reversible-flow O(1)-memory backprop `0642.1.1.6`, the
free-energy *deliberation* capability `r00r.1`, and the capstone master SDE `0642.11`.

A GENERIC system writes the flow of the state `z` as a **reversible** plus an **irreversible** part,

```
            dz/dt = L(z) · ∇E(z)  +  M(z) · ∇S(z),
```

with `L` skew-symmetric (a Poisson/Hamiltonian bracket — conserves energy) and `M` symmetric
positive-semidefinite (a friction/dissipation operator — produces entropy), subject to the two
**degeneracy conditions**

```
            L · ∇S = 0        (entropy is a Casimir of the reversible bracket),
            M · ∇E = 0        (dissipation does no net work on the energy).
```

These give, with no further assumptions, `dE/dt = 0`, `dS/dt ≥ 0`, and hence the free energy
`F = E − T·S` is a Lyapunov function. We (i) identify `E, S, L, M` and the Casimirs for the live
synaptic state, (ii) **prove** the degeneracy conditions for the chosen parameterization, (iii)
derive the conservation/production/Lyapunov chain and the **bounded-trajectory** certificate, and
(iv) corroborate all of it numerically (`tests/test_metriplectic_theory.py`). The **baseline
comparator** is the shipped `vg9` clamped-Euler step (`vg9.5`/`vg9.7`): dissipative-stable (the
`yw9.7` contraction `cb_spectral_radius < 1`) but *not* structure-preserving — it does not exactly
conserve `E` or the vesicle Casimir at finite step. This note specifies the structure the
discrete-gradient integrator preserves exactly.

---

## 0. Where reversibility, dissipation, and conservation live in the synapse

The presynaptic state is `(C, BUF, RRP, RES, DELAY, PR, CL, E_met)` plus opt-in `HEAT`
(`build_presyn_state`, `release_canonical`). Three physically distinct behaviors are present:

- **Reversible exchange** — the **calcium ↔ buffer** shuttle: `αon` moves free calcium `C` into the
  bound store `BUF`, `αoff` releases it back (`docs/stable_recurrence_theory.md` §2). In the
  conservative limit this is a *lossless exchange* of calcium between two forms — the reversible core.
- **Dissipation** — the **leaky decays** `ρc, ρb < 1` (and the energy/complexin/reserve leaks). They
  relax the state toward rest; `yw9.7` proves the calcium↔buffer map is strictly contractive
  (`ρ(M_cb) < 1`). Dissipation is what *produces entropy*.
- **Conservation (Casimir)** — the **vesicle pool** `N = RRP + RES + Σ DELAY` is conserved
  *structurally* by the paired-transfer depletion/refill (`yw9.2.2`, `_vesicle_step`). It is a
  Casimir: untouched by both brackets.

The metriplectic model below is the minimal closed system that carries all three. We use the reduced
calcium core `z = (C, B, h)`: free calcium `C`, buffered calcium `B` (≡ `BUF` content), and a scalar
**heat / entropy reservoir** `h` that books the energy the leaks dissipate. The vesicle pool enters
as the Casimir `N` (§6). This is the faithful reduction: the calcium↔buffer pair is exactly the only
non-trivially-coupled subsystem (`yw9.7` §1), and the pools are conservation-bounded.

---

## 1. Energy `E(z)` and entropy `S(z)`  → subtask `0642.1.1.1`

On `z = (C, B, h)`:

```
            E(z) = ½·C² + ½·B² + h           (stored calcium energy + dissipated heat),
            S(z) = h                          (the heat content is the entropy reservoir).
            ∇E = (C, B, 1),    ∇S = (0, 0, 1).
```

`H(C,B) = ½(C² + B²)` is the **mechanical** (stored-calcium) energy; `h ≥ 0` is the heat the leaks
have produced. `E` is **coercive** (`E → ∞` as `‖z‖ → ∞`, with `h ≥ 0`) — the property that turns
energy conservation into boundedness (§5). `T > 0` is a fixed reference temperature (units relating
heat to entropy).

---

## 2. The skew Poisson operator `L` and the Jacobi identity  → subtask `0642.1.1.2`

The reversible part is the lossless calcium↔buffer exchange, a rotation in the `(C, B)` plane that
leaves the heat untouched:

```
                ⎡  0    ω    0 ⎤
        L  =    ⎢ −ω    0    0 ⎥ ,        ω ∈ ℝ  (signed exchange orientation and rate).
                ⎣  0    0    0 ⎦
```

`L` is **skew-symmetric** (`Lᵀ = −L`) by construction. The associated Poisson bracket
`{f, g} = ∇fᵀ L ∇g` satisfies the **Jacobi identity** trivially: `L` is *constant* (state-independent),
and a constant skew matrix always defines a valid (linear) Poisson structure — the structure
functions `L_ij` have zero derivatives, so the Jacobi closure `Σ (L_il ∂_l L_jk + cyc.) = 0` holds
identically. (When `ω` is later made state-dependent, `0642.1.1.2` must re-verify Jacobi; for the
constant `L` here it is automatic.)

The reversible flow `L∇E = (ω·B, −ω·C, 0)` conserves the mechanical energy:
`dH/dt|_rev = C·(ωB) + B·(−ωC) = 0`, and conserves the total `E` (it does not touch `h`).
The sign fixes the orientation, not the invariants. The live presynaptic reduction uses
`ω = −½(α_buf_on + α_buf_off)`: its negative orientation makes positive free calcium increase the
buffer coordinate under the reduced exchange convention.

---

## 3. The PSD friction operator `M = Bᵀ B`  → subtask `0642.1.1.3`

The dissipation damps `C` and `B` at rates `γ_C, γ_B ≥ 0` (the calcium/buffer leaks `1−ρc, 1−ρb`)
and **deposits the lost energy into the heat** `h`. The operator is

```
                ⎡  γ_C     0     −γ_C·C        ⎤
        M  =    ⎢   0     γ_B    −γ_B·B        ⎥  =  γ_C·u·uᵀ  +  γ_B·v·vᵀ,
                ⎣ −γ_C·C  −γ_B·B  γ_C·C²+γ_B·B² ⎦
                u = (1, 0, −C)ᵀ,   v = (0, 1, −B)ᵀ.
```

Written as `M = γ_C·uuᵀ + γ_B·vvᵀ` it is manifestly **symmetric** and **positive-semidefinite**
(a non-negative combination of rank-1 projectors — the `M = Bᵀ B` form with
`B = [√γ_C·uᵀ ; √γ_B·vᵀ]`). The dissipative flow is

```
        M∇S = M·(0,0,1)ᵀ = ( −γ_C·C, −γ_B·B, γ_C·C² + γ_B·B² )ᵀ,
```

i.e. `Ċ_diss = −γ_C C`, `Ḃ_diss = −γ_B B`, `ḣ_diss = γ_C C² + γ_B B²`: the leaks damp the calcium and
the **exact** energy they remove, `γ_C C² + γ_B B²`, reappears as heat.

---

## 4. Degeneracy conditions ⟹ conservation & production  → subtask `0642.1.1.4`

**Degeneracy (proved for this parameterization).**

```
        L·∇S = L·(0,0,1)ᵀ = (0,0,0)ᵀ = 0.                                  (D1) ✓
        M·∇E = M·(C,B,1)ᵀ:
            row 1:  γ_C·C + 0 − γ_C·C = 0
            row 2:  0 + γ_B·B − γ_B·B = 0
            row 3:  −γ_C·C·C − γ_B·B·B + (γ_C·C² + γ_B·B²) = 0    ⟹   M·∇E = 0.   (D2) ✓
```

(D1) holds because `S` depends only on `h`, which `L` annihilates; (D2) is the algebraic identity
that the heat row exactly balances the damped mechanical rows. Both are **structural** — true for
*every* state `z` and every `ω, γ_C, γ_B ≥ 0`, not just on average.

**The conservation/production theorem.** With `dz/dt = L∇E + M∇S` and (D1)–(D2):

```
  dE/dt = ∇Eᵀ ż = ∇Eᵀ L ∇E + ∇Eᵀ M ∇S
        = 0                  (skew: xᵀLx = 0)
        + (M∇E)ᵀ ∇S = 0      (M symmetric, then D2)            ⟹   dE/dt = 0.

  dS/dt = ∇Sᵀ ż = ∇Sᵀ L ∇E + ∇Sᵀ M ∇S
        = −(L∇S)ᵀ ∇E = 0     (skew, then D1)
        + ∇Sᵀ M ∇S ≥ 0       (M PSD)                            ⟹   dS/dt ≥ 0.
```

Energy is **exactly conserved**; entropy is **non-decreasing**. For the explicit core,
`dE/dt = (Ċ·C + Ḃ·B + ḣ) = (ωCB − γ_C C² − ωCB − γ_B B²) + (γ_C C² + γ_B B²) = 0` and
`dS/dt = ḣ = γ_C C² + γ_B B² ≥ 0`, confirming the abstract result.

---

## 5. `F = E − T·S` is a Lyapunov function ⟹ bounded trajectories  → subtask `0642.1.1.5`

```
        dF/dt = dE/dt − T·dS/dt = 0 − T·(γ_C C² + γ_B B²) ≤ 0       (T > 0).
```

So `F` is **non-increasing** along every trajectory: a Lyapunov function for the relaxation.

**Bounded trajectories (the certificate).** Energy is conserved, so the trajectory is confined to the
level set `Σ_{E₀} = { z : E(z) = E₀ }`. Because `E` is **coercive** and `h ≥ 0`,

```
        ½(C² + B²) + h = E₀,  h ≥ 0   ⟹   C² + B² ≤ 2E₀  and  0 ≤ h ≤ E₀,
```

so `Σ_{E₀}` is **compact**. A trajectory starting on it stays on it (energy conservation) ⟹ **`(C, B,
h)` is bounded for all time**, with the explicit bounds above. No clamp is needed — the bound is a
consequence of the structure.

**Equilibrium & domain of attraction.** `F` decreases until `dF/dt = 0 ⟺ C = B = 0`; the only
invariant set there is `z* = (0, 0, E₀)` — all mechanical energy converted to heat, the **MaxEnt
state on the energy shell** (`S = h = E₀` is maximal subject to `E = E₀`). By LaSalle's invariance
principle on the compact `Σ_{E₀}`, every trajectory on the shell converges to `z*`. The domain of
attraction is the whole shell `Σ_{E₀}` (each energy shell is invariant and has its own `z*`). `F`
attains its minimum `F(z*) = E₀ − T·E₀ = (1−T)E₀` there.

---

## 6. Casimirs — vesicle conservation  → consumed by `0642.1.2.1`, `0642.11.1`

The **vesicle pool** `N = RRP + RES + Σ DELAY` is a **Casimir**: it commutes with the reversible
bracket (`L ∂N = 0` — `N` does not appear in the `(C,B,h)` core) and is conserved by the dissipative
pool dynamics structurally (`yw9.2.2`: every depletion is a *paired transfer*, never a sink). A
structure-preserving runtime (`0642.1.2`) keeps `N` constant to machine precision, exactly
as `_vesicle_step` already does at `rec_rate = 1`. Casimirs foliate the phase space into invariant
leaves `{N = const}`; the metriplectic flow above lives on a single leaf, so the full guarantee is
"bounded on the energy shell **within** the conserved-vesicle leaf."

---

## 7. Proof-obligation & assumptions ledger  → consumed by `0642.1.2`, `0642.10`

| # | Assumption (how discharged) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| A1 | **`L` skew + Jacobi** — structural; constant `L` ⟹ Jacobi automatic (§2). | reversible part conserves `E`; `S` is its Casimir. | a *learned* state-dependent `ω` breaks Jacobi or skewness. | project `L → ½(L − Lᵀ)`; re-verify Jacobi on a grid; else clamped Euler. |
| A2 | **`M = BᵀB` PSD + `M∇E = 0`** — structural (§3–§4, D2). | dissipation gives `dS/dt ≥ 0` and `dE/dt = 0`. | a learned `M` loses PSD or degeneracy. | project to the PSD/degenerate cone (`M ← P M P`, `P = I − ∇E∇Eᵀ/‖∇E‖²`); else `vg9` clamped step. |
| A3 | **`E` coercive**, `h ≥ 0` (§1). | the energy shell `Σ_{E₀}` is compact ⟹ bounded trajectories. | a non-coercive learned `E` (unbounded below) lets `z` escape. | add a quadratic floor to `E`; or clamp the offending channel. |
| A4 | **`T > 0`** (fixed reference temperature). | `F = E − TS` is non-increasing (Lyapunov). | `T ≤ 0`. | fix `T > 0`. |
| A5 | **Casimir exactness** — `N` paired-transfer conserved (`yw9.2.2`). | trajectory stays on `{N = const}`. | a non-conservative refill (`rec_rate ≠ 1`, lossy clamp). | route excess back to reserve (already done); discrete-gradient integrator preserves `N`. |

**Verification protocol** (the bead's "symbolic + grid check"): D1–D2 are checked symbolically (§4)
and on a random grid of states; PSD of `M` and skewness of `L` are checked on the grid; conservation
/ production / Lyapunov / boundedness are checked by integrating the flow. The structural fallbacks
(projections, clamped Euler) are what the runtime takes when a *learned* `L/M/E` violates A1–A3 — the
same fail-closed discipline as `0642.2.1` (the cusp note).

---

## 8. Numerical corroboration

`tests/test_metriplectic_theory.py` checks the construction directly (no hand-waving):

- **Degeneracy** `‖L∇S‖ = 0` and `‖M∇E‖ = 0` at a grid of random states (D1–D2).
- **Structure** `L + Lᵀ = 0` (skew) and `eig(M) ≥ 0` (PSD) on the grid.
- **Conservation & production** — integrating `ż = L∇E + M∇S` (small-step RK4): `E` drift is
  `O(Δt⁴)` and `→ 0` with the step (the *continuous* flow conserves `E`), `S` is monotone
  non-decreasing, and `F` is non-increasing.
- **Boundedness & convergence** — the trajectory stays inside the shell bound `C² + B² ≤ 2E₀` and
  converges to `z* = (0,0,E₀)`.
- **Baseline contrast** — forward Euler (the `vg9`-style step) drifts `E` markedly more than RK4 at
  the same step, and neither conserves `E` *exactly* — motivating the **discrete-gradient**
  integrator of `0642.1.2.1`, which conserves `E` and `N` to machine precision by construction.

These confirm the *exact* algebraic facts (degeneracy, PSD, skew) and the *qualitative* dynamical
facts (conservation in the continuous limit, monotone `S`, Lyapunov `F`, boundedness) that the
structure-preserving integrator realizes at finite step.

---

## 9. Live runtime compilation and fallback  → bead `0642.1.2`

`metriplectic_integrator.torch_guarded_step` is the vectorized torch compilation used by
`SynapticPresyn.release_canonical`. Because `E` is quadratic and `S` is linear, the Gonzalez step is
an implicit-midpoint update with a closed-form 2×2 solve for `(C, BUF)`; there is no Python
fixed-point loop on the live path. `heat` is then closed from the exact energy-shell identity. The
operation remains differentiable, so the existing chunked-BPTT path can carry gradients through it.

The attention drive is an **external forcing**, not part of the closed GENERIC core. The runtime
therefore injects `alpha_ca·softplus(drive)` into calcium first and conserves the resulting energy
shell during the internal relaxation. The certificate is deliberately per relaxation step: it does
not claim that a driven/open synapse conserves energy across input injection.

The feature is behind `SynapticConfig.metriplectic_integrator` and remains default-off. When enabled:

- the live state gains a `HEAT` tensor; old caches acquire a deterministic zero reservoir;
- ordinary/default-off states allocate no `HEAT` tensor and retain the original operation path;
- non-finite proposals, energy drift, negative entropy production, or departure from the live
  physical domain (`C ≥ 0`, `0 ≤ BUF ≤ 1`, `heat ≥ 0`) select the pre-existing clamped-Euler update
  elementwise;
- `SynapticPresyn.get_metriplectic_metrics()` exposes step/fallback counts plus the last energy,
  entropy, and free-energy guard reductions for structured telemetry;
- the vesicle Casimir remains enforced independently by the existing paired-transfer RRP/RES/DELAY
  update, so changing the calcium integrator cannot bypass vesicle conservation.

All guard calculations certify the values after conversion to the live state dtype. The tolerance
floor scales with that dtype's machine epsilon, so fp32/bf16 runs are judged against attainable
precision rather than the fp64 reference tolerance. A fallback receives the already-computed live
clamped-Euler tensors and selects them directly; it does not reimplement or approximate the baseline.

---

## 10. Relationship to the `vg9` baseline & reversible-flow backprop  → subtask `0642.1.1.6`

The shipped dynamics are the **`vg9` clamped-Euler** step: stable (the `yw9.7` contraction) and
conservation-bounded for the pools (`yw9.2.2`), but it enforces stability by **clamping** and does
not exactly conserve `E` or respect the metriplectic split. This note upgrades "stable because we
clamp" to "stable **by construction**": `E` coercive + conserved ⟹ bounded; `M` PSD + degenerate ⟹
entropy production; `F` Lyapunov ⟹ relaxation to the MaxEnt equilibrium.

### 10.1 Exact continuous `L` flow and inverse

Let `x = (C, B)ᵀ`, `J = [[0, 1], [−1, 0]]`, and `θ = ω·Δt`. For constant `ω`, the exact
continuous reversible flow is

```
        x⁺ = R(θ)x,       R(θ) = exp(θJ) = ⎡ cos θ   sin θ ⎤,
                                                 ⎣−sin θ   cos θ ⎦
        h⁺ = h.
```

Because `R(θ)ᵀR(θ) = I` and `det R(θ) = 1`, the map preserves `H`, volume, and orientation. Its
closed-form inverse is

```
        x = R(−θ)x⁺ = R(θ)ᵀx⁺,
        C = cos(θ)·C⁺ − sin(θ)·B⁺,
        B = sin(θ)·C⁺ + cos(θ)·B⁺,        h = h⁺.
```

This statement holds for either sign of `ω`. It assumes `ω` and `Δt` are known and unchanged while
the forward step is reconstructed.

### 10.2 The inverse compatible with the current midpoint discretization

The continuous inverse is **not** the inverse of every discrete `L` integrator. The `L`-only
restriction of the live implicit-midpoint proposal is the Cayley map. With
`q = ω·Δt/2`, `a = (1−q²)/(1+q²)`, and `b = 2q/(1+q²)`, it is

```
        x⁺ = Q(q)x,       Q(q) = ⎡ a   b ⎤,
                                     ⎣−b   a ⎦
```

`a²+b² = 1`, so `Q(q)ᵀQ(q) = I`, `det Q(q) = 1`, and

```
        Q(q)⁻¹ = Q(q)ᵀ = Q(−q),
        C = a·C⁺ − b·B⁺,       B = b·C⁺ + a·B⁺,       h = h⁺.
```

The Cayley rotation angle is `2·atan(q)`, not `ω·Δt`. Applying the trigonometric continuous-flow
inverse to a Cayley forward step would therefore introduce an `O(Δt³)` local reconstruction error.
The implementation bead `0642.1.2.6` must use the Cayley inverse if it retains the current
midpoint forward values; changing to an exact exponential or split forward is a separately reviewed
numerical change.

### 10.3 Reverse-mode contract

For either orthogonal map `y = Qx`, the reconstructed input and input cotangent are
`x = Qᵀy` and `g_x = Qᵀg_y`. For the Cayley map,

```
        da/dq = −4q/(1+q²)²,       db/dq = 2(1−q²)/(1+q²)²,
        dQ/dq = ⎡ da/dq    db/dq ⎤,
                 ⎣−db/dq    da/dq ⎦,
        g_q = g_yᵀ(dQ/dq)x,
        g_ω = (Δt/2)·g_q,          g_Δt = (ω/2)·g_q.
```

Equivalently, `g_q = 2·g_yᵀJy/(1+q²)`. The continuous rotation uses
`g_θ = g_yᵀJy`, `g_ω = Δt·g_θ`, and `g_Δt = ω·g_θ`. Tensor-valued or broadcast parameters must
reduce these cotangents back to their original shapes. A custom backward may evaluate these formulas
directly or reconstruct `x`, detach it, replay one local forward with gradients enabled, and call
autograd on that local graph. `h` is an identity coordinate in the `L` flow.

### 10.4 Where reversibility stops

The current live path does **not** yet implement reversible backprop or an operator-split `L` step.
It solves a combined `L+M` 2×2 midpoint system, converts the proposal to the live dtype, evaluates
guards, and may select a clamped-Euler fallback. Thermodynamic irreversibility of `M` does not by
itself imply non-bijectivity: the fallback-free combined linear map is algebraically invertible when
its reverse solve is nonsingular. Its inverse is nevertheless poorly conditioned under damping. In
the scalar case, reverse error is amplified by `(1+aγ)/(1−aγ)` per step for `a = Δt/2`, so a safe
checkpoint interval must come from measured reconstruction error rather than algebra alone. At the
current `dt=1`, `τ_buf=4` default, the uncoupled buffer mode amplifies reverse error by approximately
`2.43×`, `5.91×`, `34.9×`, and `1221×` over 4, 8, 16, and 32 steps. A span of eight steps is therefore
the initial implementation hypothesis, not a theorem; the dtype drift gate below may shorten it.

The following are hard boundaries of the narrower `L`-reconstruction claim:

- fp32/bf16 rounding is many-to-one; even the condition-one `L` rotation is not bitwise reversible;
- clamps are non-injective, and reconstruction error can change a guard decision; every fallback
  ends a reversible segment and retains its forward mask plus a full boundary checkpoint;
- calcium influx is injected before the closed relaxation, so its drive must be deterministically
  replayed or stored and subtracted after reconstructing the post-influx state;
- top-k indices, validity masks, stochastic draws or RNG state, train-time EMA history, and unchanged
  parameters are replay inputs, not information recoverable from `(C, BUF, HEAT)`;
- `RRP`, `RES`, `DELAY`, `PR`, `CL`, and `E` use clamps, scatter updates, or queue mutation and are
  outside the reversible core; their state policy must be budgeted independently;
- neither the continuous rotation nor the Cayley rotation preserves the physical quadrant in
  general, so fallback-free execution is an assumption to test, not an invariant of the `L` map.

For a future split implementation, define a reversible segment as accepted, deterministic `L`-only
Cayley steps between non-reversible boundaries. A stored dissipative correction
`δ_M = Ψ_M(x) − x` can recover the input to a following `M` update, but storing a dense correction at
every step remains linear in depth even when its numerical magnitude is small. The safe initial
policy is therefore: checkpoint/replay `M` and all exogenous/non-`L` state on an explicitly measured
schedule, and terminate the segment immediately on fallback. An inverse of an accepted `M` step may
replace a checkpoint only after its conditioning and finite-precision drift satisfy the same error
gate.

The existing `recurrence_chunk_len` mechanism is not this policy: it detaches carried state and
therefore truncates gradients. Reversible checkpoints must use a separate setting and preserve the
same full-BPTT gradient as the uncheckpointed control. In particular, bf16 inverse-only
reconstruction cannot recover information lost at the forward cast; it needs deterministic segment
replay from an exact live checkpoint or an explicitly budgeted higher-precision rounding residual.

### 10.5 Memory budget

Let `N` be the number of recurrent substeps, `n` the number of scalar sites in one core invocation,
and `s` the bytes per scalar. For the live recurrence, budget both the full-key surface
`n_key = batch·heads·key_length` and the gathered-edge surface
`n_edge = batch·heads·query_block·top_k`; sums over unequal invocations replace `Nn` below. The table
counts only reduced-core boundary tensors. Autograd intermediates, model activations, replay inputs,
and persistent synaptic state are additional.

| policy | reduced-core snapshot storage | scaling in `N` |
|---|---:|---:|
| standard BPTT lower bound (`C`, `BUF`, `HEAT` at every boundary) | `≥ 3Nns` | `Θ(Nn)` |
| fallback-free `L` reconstruction (terminal state only) | `3ns + O(parameters)` | `Θ(n)`, or `O(1)` only in `N` |
| dense two-plane `M` correction at every step (`HEAT` from the energy shell) | `(3+2N)ns` | `Θ(Nn)` |
| uniform full-core checkpoint/replay windows of length `K` | approximately `(3N/K+3K+3)ns` | minimized at `Θ(ns√N)` |

Thus “O(1)-memory” means a constant number of **`L`-core activation snapshots in `L`-step depth at
fixed tensor shape**. It does not mean constant memory in context length, constant whole-model VRAM,
or longer context for free. If `M`, fallback, or non-`L` checkpoints occur at every step, the total is
again `Θ(Nn)`. Total constant-in-depth storage requires a bounded number of such boundaries or a
constant-memory deterministic recomputation/inversion schedule.

### 10.6 Correctness and falsification plan for `0642.1.2.6`

The implementation bead must meet all of these gates before making a runtime memory claim:

1. **Algebraic identities:** verify `QᵀQ = I`, `det Q = 1`, and `Q(−q)Q(q) = I` in fp64 over both
   signs of `ω`, broadcast shapes, and a wide `Δt` range.
2. **Forward parity:** on guard-free interior states, compare the Cayley forward with the existing
   pure-`L` `discrete_gradient_step` and `torch_guarded_step` (`γ_C = γ_B = 0`). Do not compare a
   Cayley inverse against an exponential forward, or vice versa. A wrapper around the combined live
   step must keep all forward tensors and every `TorchStepRecord` field bit-identical to eager mode.
3. **Round-trip drift:** sweep depths `{1, 16, 256, 4096}` in fp64 and fp32, plus bf16 where the
   backend supports it. Report error growth against machine epsilon; do not require or advertise
   bitwise reconstruction.
4. **Gradient parity:** compare `C`, `BUF`, `HEAT`, `ω`, and `Δt` cotangents with ordinary autograd;
   any combined-step wrapper must also cover `γ_C` and `γ_B`. Then run `torch.autograd.gradcheck`
   in float64 (`eps=1e-6`, `atol=1e-5`, `rtol=1e-3`, `nondet_tol=0`). Add `gradgradcheck` if the
   custom backward promises higher-order derivatives.
5. **Replay inputs:** test deterministic reconstruction with influx, top-k indices, validity masks,
   stochastic samples/RNG state, and EMA history held to their recorded forward values.
6. **Fallback boundary:** plant mixed accepted/fallback masks; accepted elements may reconstruct,
   while rejected elements must use the stored forward mask and checkpointed baseline state. The
   backward pass must never re-decide the branch from reconstructed values.
7. **Memory scaling:** use PyTorch saved-tensor hooks to measure saved bytes versus depth, and use
   separate warmed, synchronized GPU processes for eager and reversible peak-memory measurements.
   Demonstrate a near-zero depth slope for the isolated `L` core against a linear standard-autograd
   control. The full guarded recurrence must fit its declared `O(N/K+K)` checkpoint budget and be
   reported separately, including recompute count and slowdown.

This subtask establishes the inverse and the honest storage/test contract only. The runtime still
uses ordinary autograd through the combined guarded `L+M` midpoint proposal. Custom backward code,
a default-off integration toggle, reconstruction-error checkpoint tuning, and measured memory
claims belong to `0642.1.2.6`.

---

## 11. Falsification and paired multi-seed verdict  → bead `0642.1.3`

The runtime claim is now tested by `scripts/e2e/metriplectic_stability_curve.py`, not inferred from
the algebra alone. The experiment advances the live guarded torch recurrence and its byte-identical
clamped-Euler fallback over the same fixed physical horizon. An independent matrix-exponential
solution supplies the endpoint target. The step-size sweep
`{0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0}` straddles the analytic explicit-Euler boundary: its
spectral radius first exceeds one at `dt=0.5`, and the measured baseline first increases free energy
at that same point. The guarded arm remains finite, physical, fallback-free, energy-conserving, and
entropy-producing through `dt=1.0`.

The paired analysis predeclares two lower-is-better headline metrics. Endpoint loss is the MSE at
`dt=1.0`; divergence rate is the fraction of the analytically unstable stress steps `{0.5, 1.0}`
classified as divergent. Eight fixed seeds independently sample conservative physical batches
(`8` states per seed), and `eval_stats.paired_comparison` supplies the paired t-test, exact Wilcoxon
test, effect size, and 10,000-sample paired-bootstrap interval.

| metric | clamped Euler mean | guarded metriplectic mean | paired Δ (guarded − baseline) | bootstrap 95% CI | paired-t p | Wilcoxon p | favorable |
|---|---:|---:|---:|---:|---:|---:|---:|
| endpoint loss | `2.7084e-2` | `3.7578e-4` | `-2.6709e-2` | `[-2.7658e-2, -2.5724e-2]` | `3.13e-10` | `0.0078125` | `8/8` |
| stress divergence rate | `1.0` | `0.0` | `-1.0` | `[-1.0, -1.0]` | `0` | `0.0078125` | `8/8` |

**Verdict: positive for this closed-system falsification.** Both predeclared metrics improve, both
paired bootstrap intervals exclude zero, and both paired tests meet `α=0.05`. The constant `-1`
divergence-rate shift has zero sample variance, so its mathematical t statistic is infinite; the
strict JSON report records that field as `null` while retaining the exact shift, interval, and
`p=0`.

The durable evidence is joined rather than hand-pasted: per-step and per-seed traces live below the
run's artifact directory, `statistics.json` stores the strict paired report, and each seed/arm
observation is appended to `results/registry.jsonl` under the canonical
`integrator_endpoint_loss` and `integrator_divergence_rate` metrics. The run ID links those records
back to the report.

This is deliberately a narrow claim. The system is a linear, closed calcium/buffer/heat core with
conservative sampled initial states; the analytic instability boundary is state-independent, and
the experiment measures numerical integration error and thermodynamic stability—not language-model
loss, throughput, stochastic synaptic release, or end-to-end training quality. Those broader claims
require separate matched-compute experiments.

---

## References

- Öttinger, H.C. (2005). *Beyond Equilibrium Thermodynamics.* Wiley. — GENERIC, the two-generator
  formalism and the degeneracy conditions.
- Grmela, M. & Öttinger, H.C. (1997). *Dynamics and thermodynamics of complex fluids I–II.* Phys.
  Rev. E 56. — the original GENERIC papers.
- Morrison, P.J. (1986). *A paradigm for joined Hamiltonian and dissipative systems.* Physica D 18. —
  "metriplectic" dynamics.
- McLachlan, Quispel & Robidoux (1999). *Geometric integration using discrete gradients.* Phil.
  Trans. R. Soc. A 357. — structure-preserving integrators (the `0642.1.2.1` target).
- Internal: `docs/stable_recurrence_theory.md` (`yw9.7`, the calcium↔buffer contraction),
  `docs/theory/singular_perturbation.md` (`0642.2.1`, the companion Thrust-F note),
  `tests/test_vesicle_conservation.py` (`yw9.2.2`, the Casimir).
