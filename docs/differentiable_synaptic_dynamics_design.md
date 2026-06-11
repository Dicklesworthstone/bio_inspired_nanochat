# Differentiable Recurrent Synaptic Dynamics — Design Note (bead yw9.1)

_Author: OrangeMill · 2026-06-11 · Flagship epic `yw9` (Differentiable Synaptic Dynamics)._

## Purpose

Specify **how** to make the presynaptic state recurrence (calcium / buffer / RRP / reserve /
SNARE / complexin / energy) differentiable end-to-end, so the kinetic parameters can be
**learned by SGD** instead of hand-tuned or CMA-ES-searched. Today those parameters are fixed
config constants (`tau_c=6.0`, `tau_buf=4.0`, `alpha_ca=0.55`, `alpha_buf_on/off=0.1`,
`prime_rate=0.075`, `nsf_recover=0.08`, `energy_fill/use=0.02`, `endo_delay=3`, …), and the state
update is wrapped in `with torch.no_grad()` (`SynapticPresyn.release_canonical`, the scatter block
at `synaptic.py:451`), so **no gradient flows through the state**. `release()` builds a
differentiable bias w.r.t. the input `drive`, but `forward()` is `@torch.no_grad` and the
recurrence is detached. This note is the math/API contract that `yw9.2` (implementation) and its
subtasks (`yw9.2.1` scan formulation, `yw9.2.2` conservation, `yw9.2.3` chunked TBPTT), `yw9.3`
(learnable kinetics), and `yw9.5` (gradient-correctness) build against.

## 1. The recurrence, as it actually is

State (per `build_presyn_state`, shape `(B, H, T_key)` per scalar variable, plus a `DELAY` list of
`endo_delay` tensors). Per query step the canonical update (`release_canonical`, lines 472–515),
scattered to key positions with `add_vals = Σ released`, `drv_vals = Σ drive`, is:

```
C'   = clamp( ρc·C + αca·softplus(drv)·accessed − αbon·C·(1−BUF) + αboff·BUF , ≥0 )       (calcium)
BUF' = clamp( ρb·BUF + αbon·C·(1−BUF) − αboff·BUF , [0,1] )                                 (buffer)
RRP' = clamp( RRP − add_vals + prime·min(RES,1) , [0,30] )                                  (readily-releasable pool)
RES' = clamp( RES + DELAY[0] − prime·min(RES,1) , ≥0 )                                      (reserve)
DELAY' = DELAY[1:] ++ [add_vals·rec_rate]                                                   (endocytosis shift register)
PR'  = clamp( PR·(1 − unprime·add_vals) + nsf·(1−PR) , [0,1] )                              (SNARE/priming)
CL'  = clamp( 0.995·CL + 0.005 − unprime·add_vals , [0,1] )                                 (complexin clamp)
E'   = clamp( E + fill·(Emax−E) − use·add_vals , [0,Emax] )                                 (energy)
```

with `ρc=exp(−1/τc)`, `ρb=exp(−1/τb)`, and the per-step **release** (the only thing currently
differentiable, w.r.t. `drive`):

```
p    = Hill_Syt(C) · σ(3·syt + 2·PR − 2·(CL+bias)) · σ(drive)            # release probability ∈[0,1]
rel  = p · RRP        (or a binomial STE sample on an ACh-gated fraction)
e    = (rel · σ(qβ(E−0.5))·qmax) / ema_e                                  # returned attention-logit bias
```

## 2. Per-variable linearization analysis (the crux)

Classify each update by how it depends on the variable being updated, holding the *coupling
inputs* (the other variables and `drive`) as exogenous signals `a_t, b_t` — exactly the
"selective state space" view (Mamba/S5), where `a_t, b_t` may be input-dependent yet the scan over
the target variable is still **affine** and therefore associative.

| Variable | Update form (in its own variable) | Class | Scan-able? |
|---|---|---|---|
| **CL** | `CL' = 0.995·CL + (0.005 − unprime·add_vals)` | **Affine, constant decay** (leaky integrator) | ✅ directly |
| **E** | `E' = (1−fill)·E + (fill·Emax − use·add_vals)` | **Affine, constant decay** | ✅ directly |
| **RES** | `RES' = RES + DELAY[0] − prime·min(RES,1)` | Affine except `min(RES,1)` saturation | ✅ (piecewise-affine; subgradient at the knee) |
| **DELAY** | shift register + `add_vals·rec_rate` | **Linear** (pure delay line) | ✅ trivially (fixed-length tensor) |
| **PR** | `PR' = (1 − unprime·add_vals − nsf)·PR + nsf` | **Affine, input-dependent decay** `a_t(add_vals)` | ✅ selective-scan |
| **RRP** | `RRP' = (1−p̄)·RRP + prime·take` (release ∝ `p·RRP`) | **Affine, input-dependent decay** `a_t(p)` | ✅ selective-scan (p exogenous) |
| **C** | `C' = (ρc − αbon·(1−BUF))·C + (αca·softplus(drv)·acc + αboff·BUF)` | **Affine in C, decay depends on BUF** | ✅ selective-scan *if* BUF exogenous |
| **BUF** | `BUF' = (ρb − αboff − αbon·C)·BUF + αbon·C` | **Affine in BUF, decay depends on C** | ✅ selective-scan *if* C exogenous |

Each of C and BUF is affine in *itself* with a coefficient set by the *other* — so neither alone is
the obstruction. The irreducible nonlinearity is their **mutual** coupling (the `C·BUF` product
appears in both): the 2-variable calcium-buffer subsystem must be advanced jointly, and that joint
map is bilinear. Handle this 2-d subsystem with BPTT (or a fixed-point inner solve); scan the other
six channels around it.

**Genuinely nonlinear pieces** (cannot be folded into an affine scan; need BPTT or a correction):
the **C↔BUF bilinear buffering** term `αbon·C·(1−BUF)`, the **Hill/σ release map** `p(C,PR,CL,drive)`,
`softplus(drive)`, the binomial-STE stochastic branch, and every **`clamp`** (a projection giving a
subgradient with dead zones). Everything else is an affine (often leaky, |decay|<1) recurrence.

**Conclusion:** ~6 of the 7 scalar state variables are affine/selective-scan-able; the irreducible
nonlinearity is concentrated in (a) the 2-variable C↔BUF calcium-buffer subsystem and (b) the
pointwise release nonlinearity, which is already differentiable and does not recur.

## 3. Three routes, and the recommendation

**Route 1 — BPTT through an explicit scan.** Drop the `no_grad`, unroll all `T` steps, let autograd
walk the scatter/gather updates. Exact for every nonlinearity. *Memory* `O(T · S)` saved activations
(`S = B·H·T_key` per variable). *Stability* fine because all decays `<1` (§4), but clamps inject
zero-gradient regions. Best as the **correctness reference** (`yw9.5` gradcheck target) and for short
sequences.

**Route 2 — Implicit differentiation at a fixed point.** If the state is run to relaxation (the
metriplectic free-energy descent of Thrust A / `0642.1`, consumed by `r00r.1` deliberation), use the
implicit function theorem at `x* = f(x*,θ)`:  `dx*/dθ = (I − ∂f/∂x)⁻¹ ∂f/∂θ`. *Memory* `O(S)` —
independent of the number of relaxation iterations. Requires a genuine fixed point + a linear solve
(use the contraction/Lyapunov structure so `(I−∂f/∂x)` is well-conditioned). Right tool for the
**deliberation/ponder** regime, **not** for per-token streaming where there is no fixed point.

**Route 3 — Associative/parallel scan (recommended default for training).** Compose the affine
backbone with the parallel-scan operator. An affine recurrence `x_{t+1} = a_t·x_t + b_t` composes
associatively via `(a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)` → Blelloch scan, **`O(log T)` depth, `O(T)`
work**, fully differentiable. Per-step coefficients `a_t,b_t` (which depend on `drive`, `p`, `BUF`,
`add_vals`) are computed in one cheap forward and treated as exogenous inputs to the scan — the
selective-scan trick.

**Recommendation: a HYBRID (Route 3 backbone + Route 1 for the nonlinear core), with Route 2 as an
opt-in for the deliberation regime.**

1. Compute the per-step affine coefficients `(a_t, b_t)` for CL, E, RES, DELAY, PR, RRP in a forward
   pass (these depend only on exogenous signals + the previous step's C/BUF/p).
2. **Parallel-scan** those six affine channels (`O(log T)`).
3. Handle the **C↔BUF** 2-variable bilinear subsystem and the **release nonlinearity** with BPTT
   over the (cheap, 2-d) coupled recurrence — or, for the first implementation, BPTT the whole thing
   and *switch on* the scan as an optimization once gradcheck passes. (Subtask ordering: `yw9.2.1`
   does the affine scan; `yw9.2` ships BPTT-correct first, scan second.)
4. **Chunked TBPTT** (`yw9.2.3`): split `T` into chunks of `chunk_len`, BPTT within a chunk, detach
   the carry across chunks. Caps memory at `O(chunk_len · S)` regardless of total `T`.

## 4. Stability conditions

The recurrence is a bank of **contractive leaky integrators**, which is what makes both the scan and
the gradients well-behaved (no exploding products `Πa_t`):

- `ρc = exp(−1/τc) ∈ (0,1)`, `ρb = exp(−1/τb) ∈ (0,1)`, `CL` decay `0.995 < 1`, `E` decay
  `(1−fill) < 1`, `RRP`/`PR` decays `∈ [0,1)` (since `p ∈ [0,1]`, `nsf,unprime·add_vals` small). All
  spectral radii of the per-channel scalar maps are `< 1` ⇒ BIBO-stable, bounded gradient norms.
- **C↔BUF subsystem**: the linear part is the 2×2 matrix
  `[[ρc − αbon(1−BUF), αboff], [αbon(1−BUF), ρb − αboff]]`. Stability requires its spectral radius
  `<1`; at the defaults this holds for all `BUF∈[0,1]` (trace `< 2`, det `< 1`). The implementation
  must **assert** this and keep it under learning (next point).
- **Stability-preserving parameterization for learnable kinetics (`yw9.3`):** never learn raw
  decays. Parameterize `ρ = σ(θ_ρ) ∈ (0,1)`, rates `α = softplus(θ_α)` with an upper clamp, and the
  energy/fill fractions in `(0,1)` via `σ`. This **guarantees** contraction by construction, so
  learning kinetics can never destabilize the forward pass. The `clamp`s on every state variable
  remain as invariant-preserving projections (calcium ≥0, BUF∈[0,1], pools ≥0, energy ∈[0,Emax]).
- The `clamp` subgradient dead-zones are acceptable (states rarely sit on the boundary mid-trajectory);
  if they bite, swap the hard clamp for a `softplus`/`σ`-squashed invariant in the differentiable path
  only.

## 5. Memory budget vs sequence length

`S = B·H·T_key` per scalar variable; 7 scalars + a length-`endo_delay` delay line.

| Route | Memory in T | Compute depth | When |
|---|---|---|---|
| BPTT explicit scan | `O(T · S)` | `O(T)` | reference / short T / gradcheck |
| Chunked TBPTT | `O(chunk_len · S)` | `O(T)` | **default training** at long T |
| Parallel associative scan | `O(T · S)` (checkpointable to `O(√T·S)`) | `O(log T)` | throughput-bound training |
| Implicit (fixed point) | `O(S)` | `O(#iters)` fwd, `O(1)` in T | deliberation / relaxation (`r00r.1`) |

Recommendation: **chunked TBPTT** as the safe default (bounded memory, exact within chunk), with the
**parallel scan** as the speed path once `yw9.5` confirms gradient parity.

## 6. The API the implementation will expose

```python
# SynapticConfig (yw9.3): kinetics become learnable, stability-preserving Parameters.
differentiable_dynamics: bool = False          # master toggle (default off = today's detached path)
recurrence_mode: Literal["bptt", "scan", "implicit"] = "bptt"   # bptt=reference, scan=fast, implicit=ponder
recurrence_chunk_len: int = 0                  # 0 = full BPTT; >0 = chunked TBPTT (detached carry)
learnable_kinetics: bool = False               # expose τ/α/rates as σ/softplus-parameterized Parameters

# SynapticPresyn.release_canonical(..., differentiable: bool = False)
#   differentiable=False -> exact current behavior (state recurrence under no_grad).
#   differentiable=True  -> the scatter/update block runs WITH grad via the selected backend; the
#                           returned bias AND the advanced state both carry gradient.
#
# New helper (yw9.2.1): affine_scan(a, b, x0) -> associative parallel scan of x_{t+1}=a_t x_t + b_t.
# New invariant (yw9.2.2): vesicle conservation RRP+RES+in-flight(DELAY) is preserved by the
#   differentiable depletion/refill (assert in tests; it is the physical constraint that keeps the
#   pools bounded under learning).
```

Acceptance contract for `yw9.2`: differentiable w.r.t. inputs and (with `learnable_kinetics`)
parameters; `torch.autograd.gradcheck` passes on a small instance (double precision, short T); the
`differentiable=True` forward **value** matches the canonical `release_canonical` reference to fp
tolerance (the gradient is the only thing that changes, not the forward).

## 7. Downstream this unblocks

`yw9.2` (impl) → `yw9.2.1` (affine scan), `yw9.2.2` (conservation), `yw9.2.3` (chunked TBPTT),
`yw9.3` (learnable kinetics), `yw9.5` (gradient correctness); related `jyb.3` (differentiable kernel
backward), `vap.3` (reversible O(1)-memory blocks), and the hybrid optimizer epic `hea` (SGD learns
the differentiable kinetics; evolution searches the discrete/non-differentiable remainder). Route 2
(implicit) is the gradient path for `r00r.1` free-energy deliberation once Thrust A (`0642.1`) lands.
