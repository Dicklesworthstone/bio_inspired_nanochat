# Singular-Perturbation & Catastrophe-Theoretic Working Memory — Theory Note (bead `0642.2.1`)

_Thrust F — certified retention. Author: GoldenRiver · 2026-06-12._

## Purpose & scope

This note gives the mathematical foundation for **certified working-memory retention** in the
CaMKII/PP1 bistable consolidation latch (`sax.2`, `SynapticConfig.bistable_latch`). It establishes
three results and the assumptions they rest on, so the downstream implementation beads
(`0642.2.1.1`–`0642.2.1.6`, the certified cusp latch `0642.2.2.1`) build against a fixed contract:

1. **A fast-slow (singular-perturbation) split** of the synaptic dynamics, with a *runtime ε gauge*
   that measures the timescale separation actually present (§1).
2. **Fenichel persistence** of a normally-hyperbolic slow manifold `M_ε`, on which the slow
   *memory flow* is a regular `O(ε)` perturbation of an explicit reduced flow (§2).
3. **A reduction of the latch to the cusp catastrophe normal form** `m̃³ + a·m̃ + b = 0` (§3), whose
   fold set `4a³ + 27b² = 0` yields a **closed-form hysteresis half-width** `δ*(a) = (2/3√3)·(−a)^{3/2}`
   — the retention certificate: a latched state survives any control perturbation of magnitude `< δ*`
   (§4).

Everything is grounded in the *live* code: the latch map is `PostsynapticHebb.update`
(`bio_inspired_nanochat/synaptic.py`), and the normal-hyperbolicity hypothesis is discharged by the
already-proved calcium↔buffer contraction `cb_spectral_radius < 1` (`docs/stable_recurrence_theory.md`,
bead `yw9.7`). The qualitative claims are corroborated numerically against the real latch in
`tests/test_singular_perturbation_theory.py` (§6). The **baseline** this certifies against is the
heuristic sigmoid latch of `sax.2` (which exhibits hysteresis empirically but carries no retention
bound).

---

## 0. The dynamical system (as it actually is)

The latch state is the pair `(m, p) = (CaMKII, PP1)` per value-channel (`d_v` buffers
`self.camkii`, `self.pp1`). With `bistable_latch=True`, `PostsynapticHebb.update(y, ca_proxy)` applies
the forward-Euler map (interior of the clamps `m∈[0,1]`, `p∈[p₀,1]`):

```
m_{t+1} = m_t + α_ca · d(c) · (1 − m_t) − β_pp1 · p_t · m_t + γ · H(m_t)        (CaMKII)
p_{t+1} = p_t + α_pp1 · e(c) · (1 − p_t) − β_cam · m_t · p_t                    (PP1)
```

with the **BCM calcium drives** (calcium `c = ca_proxy` is the input), the **self-excitation Hill**,
and the **consolidation gate**:

```
d(c) = σ(g·(c − θ_LTP)),   e(c) = σ(g·(θ_LTD − c)),   H(m) = m^n / (kⁿ + mⁿ),
gate = σ(β_gate·(m − p)).
```

Symbol → config map (defaults):

| symbol | config field | default | role |
|---|---|---|---|
| `α_ca` | `latch_alpha_ca` | 0.6 | calcium→CaMKII potentiation |
| `β_pp1` | `latch_beta_pp1` | 1.0 | PP1→CaMKII de-potentiation (cross-inhibition) |
| `γ` | `latch_gamma_auto` | 0.45 | CaMKII autophosphorylation (**self-excitation**) |
| `n`, `k` | `latch_hill_n`, `latch_hill_k` | 6.0, 0.6 | Hill coefficient / half-max of self-excitation |
| `α_pp1` | `latch_alpha_pp1` | 0.5 | low-calcium→PP1 activation |
| `β_cam` | `latch_beta_camkii` | 0.3 | CaMKII→PP1 cross-inhibition |
| `p₀` | `latch_pp1_basal` | 0.3 | basal phosphatase floor (stabilizes OFF) |
| `g` | `latch_input_gain` | 12.0 | sharpness of the BCM sigmoids |
| `θ_LTP`,`θ_LTD` | `camkii_thr`, `latch_ltd_thr` | 1.0, 0.5 | LTP / LTD calcium thresholds |
| `β_gate` | `latch_gate_beta` | 6.0 | consolidation-gate steepness |

The latch sits **downstream of** the presynaptic calcium recurrence (calcium, buffer `BUF`, RRP,
energy — `release_canonical`), which supplies `c`. That coupling is what makes the system genuinely
two-timescale.

---

## 1. The fast-slow split & the runtime ε gauge  → subtask `0642.2.1.1`

Write the full synaptic state as `(x, y)` with **fast** `x` and **slow** `y`:

- **fast `x`** = the presynaptic calcium subsystem `(C, BUF)` (and the per-edge release state). It
  relaxes with per-step retention `ρ_c = e^{−1/τ_c}`, `ρ_b = e^{−1/τ_b}` — a calcium half-life of a
  few steps.
- **slow `y`** = the latch `(m, p)`. Its motion per step is `O(α, γ)` with the effective rates
  `α_ca, α_pp1, γ ≈ 0.5` *gated by* the BCM sigmoids, and — crucially — it only moves while calcium
  sits in an LTP/LTD band, so its **integrated** drift across a sequence is slow.

In standard singular-perturbation form,

```
ε · dx/dt = f(x, y),      dy/dt = g(x, y),        0 < ε ≪ 1,
```

where **ε is the ratio of the slow (latch consolidation) timescale to the fast (calcium relaxation)
timescale**. We do not assume ε; we **measure** it at runtime — this is the *ε gauge*:

```
ε̂ = τ_fast / τ_slow,   τ_fast = 1/(1 − ρ_c)  (calcium relaxation, steps),
                        τ_slow = 1 / (effective per-step |Δ(m,p)|)   (latch drift, steps).
```

Equivalently `ε̂ = (1 − ρ_c)⁻¹` divided by the number of steps the latch takes to traverse `[0,1]`
under a sustained supra-threshold drive. The theory's guarantees are **conditional on `ε̂` being
small** (the proof-obligation ledger, §5, makes the threshold explicit); the gauge is what the
runtime checks before trusting the certificate, and what triggers the fallback when it fails.

---

## 2. Fenichel persistence & the reduced memory flow  → subtask `0642.2.1.2`

**Layer (fast) problem.** Freezing the slow variables, the fast flow `ε dx/dt = f(x, y)` has critical
manifold `M₀ = { (x,y) : f(x,y) = 0 } = { x = h₀(y) }`, where `h₀(y)` is the calcium quasi-steady
state for a given drive. For the calcium↔buffer pair this fixed point is unique and explicit (the
linear `C/BUF` map inverted).

**Normal hyperbolicity (the key hypothesis).** `M₀` is normally hyperbolic iff the fast Jacobian
`∂f/∂x` restricted to `M₀` has eigenvalues bounded away from the imaginary axis, uniformly in `y`.
For the calcium↔buffer subsystem the fast Jacobian is exactly the 2×2 map analyzed in `yw9.7`:

```
J_fast(β) = [[ ρ_c − α_on·β ,   α_off       ],
             [   α_on·β     ,   ρ_b − α_off ]],     β = 1 − BUF ∈ [0,1],
```

and **`cb_spectral_radius(J_fast) < 1` for all β** (proved + asserted in code, `LearnableKinetics.
spectral_radius`). A spectral radius `< 1` for the per-step map is a spectral **gap below the unit
circle**, i.e. all fast eigenvalues are *attracting* and bounded away from the `|λ|=1` boundary —
precisely the discrete-time normal-hyperbolicity (attracting-slow-manifold) condition. So `M₀` is a
uniformly attracting normally hyperbolic critical manifold.

**Fenichel's theorem (1st & 3rd).** Under normal hyperbolicity and smoothness (the maps are `C^∞`),
for `ε` sufficiently small there is a slow manifold

```
M_ε = { x = h(y, ε) },     h(y, ε) = h₀(y) + O(ε),
```

`C^r`-diffeomorphic and `O(ε)`-close to `M₀`, locally invariant, with the same attracting stability,
carrying a smooth **reduced (slow) flow**

```
dy/dt = g(h(y, ε), y) = g(h₀(y), y) + O(ε)        — the "memory flow".
```

**Consequence for this model.** On `M_ε` the calcium is slaved to its quasi-steady value
`c = h(drive)`, so the BCM drives `d(c), e(c)` become functions of the *drive* alone, and the latch
`(m, p)` evolves by the reduced map of §0 with `c` replaced by `h(drive)`. **All of the catastrophe
analysis in §3–§4 is performed on this reduced flow**, and is correct up to `O(ε)` for the true
system by Fenichel. The error is uniform on compact sets where normal hyperbolicity holds; it is
*not* controlled where `ε̂` is not small (§5 failure mode).

---

## 3. Reduction to the cusp normal form  → subtask `0642.2.1.3`

On `M_ε`, equilibria of the reduced latch satisfy (interior of the clamps):

```
(★)   0 = α_ca·d·(1 − m) − β_pp1·p·m + γ·H(m)
(†)   0 = α_pp1·e·(1 − p) − β_cam·m·p     ⟹    p = p(m) = α_pp1·e / (α_pp1·e + β_cam·m).
```

`(†)` **slaves PP1 to CaMKII** (a smooth, monotone `p(m)`); substituting into `(★)` gives a single
scalar equilibrium condition `G(m; d, e) = 0`. The S-shape — hence multistability — comes entirely
from the **self-excitation Hill `H(m)`**, the only nonconvex term.

**Organizing center.** Expand about the inflection of `H`, where `H''(m_*) = 0`:

```
m_* = k · ((n−1)/(n+1))^{1/n},        H'(m_*) = γ · (n² − 1) / (4 n m_*),        H''(m_*) = 0.
```

(For the defaults `n=6, k=0.6`: `m_* ≈ 0.566`, `H'(m_*) ≈ 2.57·γ`.) Put `u = m − m_*`. Because the
**quadratic term of `H` vanishes** at `m_*` and the remaining terms of `(★)` are smooth, the
equilibrium condition Taylor-expands as a *cubic with no quadratic term to leading order* (the
defining feature of a cusp organizing center):

```
G(m) ≈ C₀ + C₁·u + C₃·u³ + O(u⁴),
```

with

```
C₃ = (1/6)·H'''(m_*) < 0            (Hill is concave just past its inflection),
C₁ = H'(m_*) − [ α_ca·d + β_pp1·p(m_*) ]   =  (self-excitation slope) − (linear decay),
C₀ = γ·H(m_*) + α_ca·d·(1 − m_*) − β_pp1·p(m_*)·m_*   (the net bias at the center).
```

Dividing by `|C₃| > 0` and rescaling `m̃ = u` yields the **cusp normal form**

```
            m̃³ + a·m̃ + b = 0,        a = −C₁/|C₃|,     b = −C₀/|C₃|.
```

**Interpretation of the two controls.**

- `a` is the **splitting (bistability) parameter**. `a < 0 ⟺ C₁ > 0 ⟺ H'(m_*) > α_ca·d + β_pp1·p`:
  the synapse is bistable exactly when the **self-excitation slope exceeds the linear decay**. This
  makes the *cusp threshold* explicit — bistability switches on when

  ```
  γ > γ_c := (α_ca·d + β_pp1·p(m_*)) · 4 n m_* / (n² − 1).
  ```

  Below `γ_c` (e.g. `γ = 0`, self-excitation off) the synapse is **monostable — no memory**.
- `b` is the **normal (bias) parameter**: the LTP-drive `d` vs LTD/decay balance. Because
  `b = −C₀/|C₃|` and the LTP drive enters `C₀` with a `+` sign (`+α_ca·d·(1−m_*)`), **writing (high
  drive) pushes `b < 0`** (toward the ON branch `m̃ > 0`); erasing/quiescence pushes `b > 0`.

The cusp point is `a = b = 0` (`C₁ = C₀ = 0`): the codimension-2 organizing center where the two
folds and the bistable wedge are born.

*Discrete-time caveat.* The map of §0 is forward Euler. Its **fixed points coincide exactly** with
the equilibria above (step size cancels), so the cusp geometry is step-size-independent. Only
*stability* picks up a mild step condition — the effective rates must be small enough that the map
does not overshoot a stable fixed point (a CFL-type bound, satisfied at the defaults; checked in §6).

---

## 4. Fold set & the closed-form hysteresis half-width (retention certificate)  → subtask `0642.2.1.4`

The cubic `m̃³ + a·m̃ + b = 0` has a repeated root — a **saddle-node (fold)** where a stable state
annihilates with the separatrix — exactly on the discriminant locus

```
            Δ = −4a³ − 27b² = 0      ⟺      4a³ + 27b² = 0.
```

For `a < 0` (bistable regime) this gives the two symmetric fold biases `b = ± b_f` with

```
            b_f = (2 / (3√3)) · (−a)^{3/2}.
```

Between them, `b ∈ (−b_f, +b_f)`, **three real roots coexist** (ON, OFF, and the unstable separatrix
between): the synapse holds whichever of ON/OFF it was placed in. Outside, only one state survives.
Hence the **closed-form hysteresis half-width**

```
            δ*(a) = b_f = (2 / (3√3)) · (−a)^{3/2},        a < 0,        (δ* = 0 for a ≥ 0).
```

**Write / erase / retention, stated as a certificate.**

- **Write**: to latch ON from OFF, the drive must push `b` past `−δ*` (the lower fold — the OFF state
  disappears, leaving only ON `m̃ > 0`). The *minimal write* is the pulse that reaches `b = −δ*` (the
  minimum-energy pulse is the subject of `0642.2.1.5`).
- **Erase**: to drop ON→OFF, the input must push `b` above `+δ*` (the upper fold).
- **Retention certificate**: once latched, the ON state **persists against any bias perturbation of
  magnitude `< δ*`**. In particular, at quiescence the resting bias `b_rest` (set by basal drive and
  the floor `p₀`) must satisfy `|b_rest| < δ*`; the **retention margin** is `δ* − |b_rest| > 0`. This
  is the precise content of *"cusp ⟹ retention ≥ δ*"*.

`δ*` is **monotone increasing in `(−a)`**: deeper into the bistable wedge (larger self-excitation `γ`
relative to decay) → wider hysteresis → more robust memory. This is the design dial: `δ*` trades
retention robustness against write/erase energy (both grow with `−a`).

Mapping back to the model: `b = −C₀/|C₃|` and `C₀` is affine in the calcium-set drive `d(c)`, so a
bias margin `δ*` corresponds to an explicit **calcium / threshold margin** `Δc* = δ*·|C₃| / (∂C₀/∂c)`,
i.e. how far the resting calcium may drift before the latch is at risk — the runtime-checkable form
of the certificate.

---

## 5. Proof-obligation & assumptions ledger  → consumed by `0642.2.2.1`, `0642.10`

Each result is stated as **(assumption ⟹ statement)**, with its failure mode and the deterministic
fallback the runtime takes when the assumption is not discharged.

| # | Assumption (discharged by) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| F1 | **Normal hyperbolicity**: fast Jacobian eigenvalue gap below `|λ|=1`, uniform in the slow vars — *discharged* by `cb_spectral_radius < 1 ∀β` (`yw9.7`, asserted in code). | The critical calcium manifold `M₀` is uniformly attracting. | A learned/extreme kinetic pushes `ρ(J_fast) → 1` (gap closes). | Clamp kinetics to the certified region; else heuristic latch (no certificate). |
| F2 | **Timescale separation**: `ε̂ = τ_fast/τ_slow ≤ ε_max` (gauge, §1; `ε_max` set with the composition harness `0642.10`). | Fenichel: `M_ε = {x=h(y,ε)}` persists `C^r`, `O(ε)`-close; the reduced memory flow is valid up to `O(ε)`. | `ε̂` not small — calcium and latch co-move; reduction invalid. | Integrate the full coupled system (no reduction); flag certificate as void. |
| C1 | **Smoothness**: latch maps `C^∞` near `m_*` (Hill, sigmoids — true by construction). | Equilibrium condition reduces to the cusp normal form `m̃³+a m̃+b`. | State pinned on a clamp boundary (`m∈{0,1}`, `p=p₀`): the smooth interior model breaks. | Treat the clamp as the boundary of the chart; analyze the active interior branch. |
| C2 | **Above the cusp**: `γ > γ_c` ⟺ `a < 0` (self-excitation slope > linear decay; §3). | Two stable states (ON/OFF) coexist; genuine memory. | `γ ≤ γ_c` (`a ≥ 0`), e.g. self-excitation off: monostable, **no retention**. | Raise `γ` or sharpen `(n,k)` into the bistable wedge; else accept no memory. |
| R1 | **Sub-fold quiescence**: resting bias `|b_rest| < δ*` (calcium margin `Δc*`). | Latched ON state persists; **retention ≥ δ***. | Drift/noise pushes `|b|` past `δ*` (a fold) → state collapses. | Refresh pulse (re-write) before the margin is exhausted; or widen `δ*` (increase `−a`). |

**Composition note** (for `0642.10`/`0642.11.1`): the certificate composes with the other thrusts
only while F1–F2 hold *simultaneously* with the other mechanisms' contraction/Lyapunov hypotheses —
the timescale-separation harness must verify the *joint* `ε̂` with the presyn recurrence and (if on)
the differentiable-recurrence chunking (`yw9.2.3`) active, not the latch in isolation.

---

## 6. Numerical corroboration

`tests/test_singular_perturbation_theory.py` checks the qualitative predictions against the **real**
`PostsynapticHebb` latch (no re-implementation of the dynamics):

- **Bistability / hysteresis exists** — sweeping calcium up then down traces a hysteresis loop: the
  ON-switch calcium (rising) strictly exceeds the OFF-switch calcium (falling); the loop has positive
  width. (Confirms `a < 0` at defaults: prediction C2 + the fold structure §4.)
- **Cusp threshold in `γ`** — at the default `γ` the latch is bistable and *retains* after a write
  pulse; with self-excitation off (`γ = 0`, i.e. `a ≥ 0`) there is no retention and no hysteresis.
  (Confirms the `γ_c` cusp threshold.)
- **`δ*` monotone in `(−a)`** — increasing `γ` (deeper into the wedge) **widens** the measured
  hysteresis loop. (Confirms `δ*(a) = (2/3√3)(−a)^{3/2}` increasing in `−a`.)
- **Retention margin** — after a write, the ON state survives a *band* of sub-fold neutral inputs,
  and collapses once the input crosses the lower fold. (Confirms R1.)

These are corroborations of the *qualitative* geometry (signs, monotonicities, existence of the
wedge), which is what a normal-form reduction licenses; the exact `δ*` value depends on the
`O(ε)` Fenichel correction and the slaving `p(m)`, quantified by the implementation beads.

---

## 7. Baseline & relationship to `sax.2`

The shipped `sax.2` latch **is** the system analyzed here: it exhibits hysteresis empirically
(`tests/test_bistable_latch.py`) but carries **no retention bound**. This note upgrades it from
"observed hysteresis" to a **certified** mechanism: `δ*` is a closed-form, runtime-checkable lower
bound on retention, with explicit assumptions (F1–F2, C1–C2, R1) and a deterministic fallback when
any fails. The implementation bead `0642.2.2.1` compiles `δ*` (and the minimum-energy write/erase
pulses, `0642.2.1.5`) into the runtime, gated by the ε-gauge, with the heuristic `sax.2` latch as the
fallback path.

---

## 8. Runtime verification & falsification results (beads `0642.2.2.*`, `0642.2.3.*`)

The theory above is now **shipped and falsified**. The runtime lives in
`bio_inspired_nanochat/cusp_certificate.py` (the `CuspLatch` update, the minimum-energy pulse
controller, the slow-manifold projector, the `CuspMonitor`) behind the default-off `cusp_latch`
toggle, dispatched from `PostsynapticHebb.update`. What was checked, and what it showed:

### 8.1 The certificate is **tight** (`0642.2.2.4`, `0642.2.3.1`)

The latch update **is** the cusp cubic `m̃ ← m̃ − η(m̃³ + a·m̃ + b(c))` with the certified splitting
parameter `a` fixed and the live calcium bias `b(c) = −C₀(c)/|C₃|` (the *same* `C₀` the certificate
uses — a shared `_residual_taylor`, so implementation and certificate can never drift). Consequently
`δ*(a)` is the **exact** fold half-width, not a loose bound. Sweeping a sustained bias drift in the
control coordinate: the bit holds for drift `≤ 0.95·δ*` and flips for `≥ 1.05·δ*` — the empirical
retention half-width equals the closed-form `δ*`. (`tests/test_cusp_latch.py::
test_retention_boundary_is_at_delta_star_within_tolerance`, `tests/test_cusp_falsification.py::
test_retention_half_width_equals_delta_star`.)

### 8.2 The certificate is **sound / conservative** (`0642.2.3.1`)

Driving the *full* latch with a physical calcium erase ramp, the empirical retention is **at least**
the certified margin — the bound never over-promises. PP1 slaving makes the held ON state strictly
more robust than the basal-`p₀` certificate (the live erase fold sits *below* the certified one), so
empirical ≥ certified, never the reverse. (`test_full_latch_retention_is_at_least_the_certified_margin`.)

### 8.3 ε-gating & deterministic fallback verified across regimes (`0642.2.2.2`, `0642.2.3.2`)

`certified ⟺ bistable (γ > γ_c ⟹ a < 0) AND separated (ρ(M_cb) ≤ cusp_eps_max)`. Verified over a
`(γ, τ_c)` grid: default `(0.45, 6)` and deeper `(0.80, 6)` certify; `γ=0` (monostable) and
`τ_c=400` (ρ_fast → 1, separation lost) do **not**. In **every** uncertified regime the latch reduces
**byte-for-byte** to the heuristic `sax.2` map — the fail-closed contract, verified end to end (no
silent half-application). The `CuspMonitor` emits per-step rich + JSONL traces of the ε gauge, the
retention margin `δ* − |b(c)|` (negative only while a write/erase deliberately crosses a fold), and
the Fenichel slow-manifold reconstruction error `|C_live − h(influx)|` (≈0 on the manifold).
(`test_certificate_gating_is_correct_across_regimes`,
`test_fallback_is_byte_exact_across_all_uncertified_regimes`,
`tests/test_cusp_latch.py::test_monitor_*`.)

### 8.4 The certified leapfrog — multi-seed stats verdict (`0642.2.3.2`)

Two honest readings, both shipped:

- **At the DEFAULT γ the well is shallow** (`δ* ≈ 0.009`) and, under a *noiseless* sustained erase
  drift, the certified cusp and the `sax.2` baseline are ~tied on raw threshold (both flip near
  `Ca ≈ 0.51`–`0.52`). The cusp's edge at default is the **tight certificate**, not raw margin. At a
  mild erase both arms retain the bit (fraction-ON = 1.0).
- **The certified dial buys real robustness.** `δ*` grows monotonically with the self-excitation `γ`
  (a tunable, *guaranteed* margin). Under a **near-critical noisy erase** (`hold=0.54`, zero-mean
  per-channel calcium noise `±0.15`, the regime where `sax.2`'s one-way LTD push collapses and never
  recovers while the cusp's symmetric well returns to the ON root), the certified cusp at `γ=0.8`
  retains essentially all of its CaMKII while `sax.2` loses the bit:

  | arm | retained CaMKII (mean ± sd, 10 seeds) |
  |---|---|
  | `sax.2` (uncertified) | `0.03 ± 0.04` |
  | cusp `γ=0.8` (certified) | `0.90 ± 0.00` |

  Paired across seeds (`eval_stats.paired_comparison`, `74f.3`): `Δ = 0.87`, 95% CI
  `[0.84, 0.89]`, paired-t `p ≈ 1.5e-13`, Wilcoxon `p ≈ 2e-3`, **10/10 seeds favorable**. The verdict
  is **positive and significant**: a certified, equal-capacity cusp latch out-retains the uncertified
  sigmoid baseline under noise. (`tests/test_cusp_falsification.py::
  test_multiseed_stats_certified_cusp_beats_baseline`; run
  `python tests/test_cusp_falsification.py` for the full retention curves.)

**Registry note.** The results-registry (`hm4.1`, `results_registry.py`) schema admits only the
`train`/`eval`/`tune` harnesses with the canonical metric vocabulary; a latch-falsification verdict
fits neither, so the durable record is this note plus the seeded e2e experiment (rather than forcing
an out-of-vocabulary harness/metric into the eval registry).

---

## References

- Fenichel, N. (1979). *Geometric singular perturbation theory for ordinary differential equations.*
  J. Diff. Eq. 31. — slow-manifold persistence (F1–F2).
- Jones, C.K.R.T. (1995). *Geometric singular perturbation theory.* Springer LNM 1609. — modern
  statement of Fenichel.
- Thom, R. (1972); Poston, T. & Stewart, I. (1978), *Catastrophe Theory and its Applications.* —
  the cusp normal form and its universal unfolding `m³ + a m + b`.
- Lisman, J. (1985). *A mechanism for memory storage insensitive to molecular turnover: a bistable
  autophosphorylating kinase.* PNAS. — the CaMKII bistable switch this latch models.
- Internal: `docs/stable_recurrence_theory.md` (`yw9.7`, the `cb_spectral_radius` certificate),
  `docs/differentiable_synaptic_dynamics_design.md` (`yw9.1`), `tests/test_bistable_latch.py` (`sax.2`).
