# Free-Probabilistic / Topological / Optimal-Transport Structural Plasticity — Theory Note (`0642.5.1`)

_Thrust C — spectrally-safe, topology-triggered NAS. Author: BeigeSquirrel · 2026-06-12._

## Purpose & scope

This note replaces the heuristic `health = fatigue × energy` thresholds of the MoE expert lifecycle
(`uta`) with three **principled geometric signals**, each playing one distinct role in
structural plasticity. It establishes three results and the assumptions they rest on, so the runtime
certificates/monitors (`0642.5.2.1`) and the falsification vs the `uta` heuristic (`0642.5.3`) build
against a fixed contract:

1. **Free probability (birth conditioning):** a noisy expert split is controllable so children are
   **well-conditioned by construction** — a certified condition-number bound (§1, `0642.5.1.1`).
2. **Persistent homology (growth trigger):** the routing manifold's **topological holes** (`H0`
   coverage gaps) are a noise-stable signal of where to grow capacity (§2, `0642.5.1.2`).
3. **Optimal transport (merge):** merging two experts is the **Wasserstein barycenter** of their
   weight distributions — the min-cost, spread-preserving merge, not naive averaging (§3, `0642.5.1.3`).

The reference math is `bio_inspired_nanochat/structural_geometry.py` (pure numpy: the Weyl/RMT bound,
`H0` via the MST, the 1D `W2` barycenter via quantiles); the claims are corroborated numerically in
`tests/test_structural_geometry.py` (§5). The **baseline** these beat is the `uta` health-threshold
lifecycle.

---

## 1. Spectrally-safe birth — free-probability conditioning  → subtask `0642.5.1.1`

A function-preserving split clones an expert `W` into the antisymmetric pair `(W + δN, W − δN)` (it
averages back to `W`, output-preserving in the dense regime, as in `uta.3`); the noise `δN` lets the
twins diverge under SGD. The **danger** is that the noise makes a child ill-conditioned (a loss spike).
Free probability controls this: the child singular spectrum is the **free convolution** of the parent
spectrum with the noise spectrum, and in its rigorous, always-valid form **Weyl's inequality** bounds
every singular value's movement:

```
        |σ_i(W ± δN) − σ_i(W)| ≤ ‖δN‖₂    ⟹    κ(W ± δN) ≤ (σ_max + ‖δN‖) / (σ_min − ‖δN‖),
```

valid whenever `‖δN‖ < σ_min` (else a child can be singular — the certificate goes void). This is the
**spectral-conditioning certificate**: it bounds the child `κ` *before* the split, and inverting it
gives `max_noise_for_kappa` — the largest divergence noise that still certifies a target `κ`. So a
birth is **spectrally safe by construction**, never an ad-hoc clone-noise gamble.

---

## 2. Topology-triggered growth — persistent homology  → subtask `0642.5.1.2`

The routing/activation point cloud has a *shape*. Its `H0` persistent homology — computed library-free
as the **minimum spanning tree** of the cloud — exposes coverage: each point is an `H0` feature born at
filtration 0 and dying (merging) at its MST edge length, so the **largest MST edge is the widest gap**
between covered regions, a hole the experts do not serve. The growth signal is that gap's
**persistence ratio** `max_gap / typical_gap`: a ratio `≫ 1` is a region of input space with no expert
coverage — grow capacity there.

This is principled because of the **bottleneck-stability theorem**: perturbing the data by `ε` moves
the persistence diagram by at most `ε` (in bottleneck distance). So a high-persistence gap is a
*genuine* topological feature, robust to noise — unlike a raw density threshold, which is not stable.
(`coverage_signal`; the stability is checked numerically in §5.)

---

## 3. Optimal-transport merge — Wasserstein barycenter  → subtask `0642.5.1.3`

Merging two experts should combine their *distributions*, not their *values*. The correct combination
is the **2-Wasserstein barycenter**: the distribution minimizing `½·W2(·,a)² + ½·W2(·,b)²`. In 1D it
has the closed form of the **quantile (inverse-CDF) interpolation** `(1−t)·F_a^{-1} + t·F_b^{-1}` — the
McCann geodesic midpoint at `t=½`. Being a geodesic midpoint, it **preserves the marginal spread** (for
two Gaussians it yields the average mean *and* the average std).

The naive elementwise average `(a+b)/2`, by contrast, **cancels** the parts where the two experts
disagree and so *shrinks* the variance — a function-destroying merge that throws away capacity. The
**OT-merge certificate** reports both costs and the spread, certifying the barycenter as the min-cost,
shape-preserving merge. (`ot_merge_certificate`.)

---

## 4. Proof-obligation & assumptions ledger  → consumed by `0642.5.2`, `0642.5.3`

| # | Assumption (discharged by) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| S1 | **Bounded noise**: split noise `‖δN‖ < σ_min(W)` (the certificate's precondition). | Weyl ⟹ `κ(child) ≤ (σ_max+‖δN‖)/(σ_min−‖δN‖)` — a spectrally-safe birth. | `‖δN‖ ≥ σ_min` ⟹ a child can be singular; the bound is void. | Shrink the noise (`max_noise_for_kappa`); else use the `uta` clone and flag the lost guarantee. |
| T1 | **Dense sampling**: the routing manifold is sampled densely enough that the MST reflects its topology. | `H0` gaps are real coverage holes; bottleneck-stable under `ε`-perturbation. | Sparse / wrong-metric sampling ⟹ spurious gaps. | Raise the persistence-ratio threshold / use more routing samples; else `uta` utilization signal, flag. |
| O1 | **Comparable supports**: the two experts' weight distributions are over a common 1D coordinate. | The `W2` barycenter is the min-cost, spread-preserving merge. | High-dim coupling ignored (1D marginal only) ⟹ the merge is a marginal approximation. | Per-coordinate / Gaussian-OT barycenter; else naive average, flag. |

**Composition note** (`0642.10`/`0642.11.1`): these signals drive the *same* split/merge controller as
`uta`, so the certificates must be evaluated on the *live* experts (the spectrum and routing cloud as
they actually are), and the tree-ness/coverage gauges checked jointly with the contraction hypotheses
of the other thrusts.

---

## 5. Numerical corroboration

`tests/test_structural_geometry.py` checks the results against the reference implementation:

- **Spectral conditioning** — `κ` matches the construction; the certificate's bound holds for real
  split children (Weyl); `max_noise_for_kappa` hits the target `κ`; the bound goes void when
  `‖δN‖ ≥ σ_min`.
- **Coverage signal** — a two-cluster cloud flags a significant high-persistence hole while a uniform
  cloud does not; the `max_gap` is bottleneck-stable (moves by `~ε` under an `ε`-perturbation).
- **OT merge** — `W2(a,a)=0`, a pure shift gives `W2 = shift`; the Gaussian barycenter averages mean
  and std; the OT merge preserves spread and has lower cost than naive averaging.

These corroborate the three signals — what licenses the runtime certificates (`0642.5.2.1`) and the
falsification vs the heuristic lifecycle (`0642.5.3`).

---

## 6. Equal-work falsification against UTA (`0642.5.3.1`)

The committed protocol `9d8a787bec9e` compares the certificate-driven controller with the UTA
utilization-times-energy threshold controller on eight matched seeds. Every pair begins from the
same four-expert state, keeps four experts and `top_k=2`, and spends exactly 480 expert dispatches
per method. The fixture contains one redundant unhealthy pair, two healthy specialists, and a
sampled routing coverage gap. It is a tiny controlled synthetic test of the lifecycle contract,
**not evidence about language-model scale or wall-clock efficiency**.

All exact obligations held on all eight seeds: the topological controller executed the intended OT
merge plus certified split; both children remained below the recorded Weyl condition-number bound;
the measured H0 maximum-gap change stayed below the `2ε` perturbation bound; the empirical
Wasserstein barycenter cost was no greater than the naive comparator; and removing the live routing
points made the topological controller fall back to a bit-identical UTA state and output.

The loss outcomes favored the topological path on every matched seed:

| Predeclared outcome (topological − UTA) | Mean delta | Paired bootstrap 95% CI | Paired t / Wilcoxon |
|---|---:|---:|---:|
| Final MSE | −0.04776 | [−0.07945, −0.01770] | `p=0.0236` / `p=0.0078` |
| Positive event-loss spike | −0.04783 | [−0.07899, −0.02026] | `p=0.0232` / `p=0.0078` |
| Dead-expert fraction | 0 | [0, 0] | `p=1` / `p=1` |

The predeclared verdict is therefore **null**, not positive: no exact claim failed and the two loss
metrics improved, but the dead-expert endpoint tied at zero for both methods instead of strictly
favoring the topological controller. This is the intended falsification discipline; the result does
not promote an incomplete three-metric win. The full strict-JSON evidence, seed outcomes,
certificates, fallback checks, and statistics are in
`results/structural_falsification_9d8a787bec9e.json`; per-method observations are also recorded in
`results/registry.jsonl` with `eligible_for_best=false` because this fixture is synthetic.

The remaining scale question is separate: a larger routed workload must test whether the
certificate overhead and any dead-expert advantage survive realistic optimization and compute
budgets (`uta.7`).

---

## References

- Voiculescu, D. (1991). *Free probability theory* / free convolution of spectra.
- Weyl, H. (1912). singular-value perturbation inequality `|σ_i(A+E) − σ_i(A)| ≤ ‖E‖`.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology* — persistence; bottleneck stability
  (Cohen-Steiner, Edelsbrunner, Harer 2007).
- Villani, C. (2009). *Optimal Transport: Old and New*; McCann (1997), displacement interpolation.
- Internal: `bio_inspired_nanochat/structural_geometry.py`, `synaptic_splitmerge.py` (`uta`, the heuristic
  baseline), `docs/theory/singular_perturbation.md` (`0642.2.1`, the proof-ledger pattern).
