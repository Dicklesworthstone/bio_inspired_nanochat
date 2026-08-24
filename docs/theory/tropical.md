# Tropical Attention and Routing — Theory Note (bead `0642.6.1.1`)

_Thrust H — exact active-region interpretability in the low-temperature skeleton._

## Purpose and claim boundary

This note isolates the part of attention and expert routing that really is tropical. For a finite
family of affine scores, the zero-temperature limit of softmax is hard selection, the winning-index
regions are polyhedra, and the active IDs index generators on an exposed upper face of a lifted
coefficient polytope. Those facts give an exact, auditable active-vertex fingerprint.

The word **skeleton** matters. The live synaptic attention score also contains a release bias that is
a nonlinear function of the query-key drive, while the live MoE router contains a normalized
alignment term. Their active index is always measurable, but their regions are not globally
polyhedral without an additional approximation certificate. The exact guarantee in this note is
therefore gated on affine scores or on biological terms frozen as state-dependent offsets. A later
runtime certificate (`0642.6.2.1`) must report that scope instead of silently upgrading a local
approximation into a global theorem.

The baseline comparator is ordinary softmax attention or top-k MoE routing plus post-hoc attention
rollout: it reports weights after the fact but does not expose an exact decision cell or a certified
active face.

---

## 1. Score family and tropical limit

Let `x in R^d` be a query or router input and let the eligible choices be `j = 1,...,m`, with
`m >= 2` unless stated otherwise. The affine
skeleton has scores

```text
    z_j(x; h) = a_j^T x + b_j(h),
```

where `h` is the already-observed biological/history state. Holding `h` fixed makes each `b_j(h)` a
constant during this decision. The finite value vectors are likewise conditioned on the current
decision; if they vary with `x`, the score-region theorem still holds but the hard readout is the
corresponding selected value map rather than a fixed vector. The temperature-`tau` softmax and its
**pre-dropout** value readout are

```text
    p_j(tau | x,h) = exp(z_j / tau) / sum_k exp(z_k / tau),       tau > 0,
    y_tau(x,h)     = sum_j p_j(tau | x,h) v_j.
```

The associated max-plus (tropical) polynomial is

```text
    F(x;h) = max_j z_j(x;h) = max_j (a_j^T x + b_j(h)).
```

Its active set is

```text
    A(x;h) = argmax_j z_j(x;h).
```

If the maximizer `j*` is unique and the score gap is

```text
    Delta(x;h) = z_j*(x;h) - max_(k != j*) z_k(x;h) > 0,
```

then

```text
    1 - p_j*(tau | x,h) <= (m - 1) exp(-Delta / tau),
    ||y_tau - v_j*||    <= D_v (m - 1) exp(-Delta / tau),
    D_v := max_k ||v_k - v_j*||.
```

Both bounds follow by dividing every exponential by `exp(z_j*/tau)`. Consequently
`p_j*(tau) -> 1` and `y_tau -> v_j*` exponentially fast as `tau -> 0+`. The most likely softmax
choice is already `j*` for every positive temperature because exponentiation preserves order; the
limit makes the *pre-dropout readout* hard. With a singleton eligible set, use the separate convention
`Delta = +infinity`, `p_1 = 1`, and zero readout error. Attention dropout must be disabled (as in
evaluation) for the realized context to inherit the readout limit: training-time dropout can remove
or rescale the winning mass after softmax.

If several scores tie, the correct limit is set-valued: softmax converges to the uniform mixture on
the exact maximizers (or to a perturbation-dependent mixture if scores approach the tie at different
rates). A tie is therefore an exposed-face fingerprint, not a unique-vertex certificate. Runtime
must return the whole active set or mark uniqueness false.

### Coordinatewise max-plus values are a different map

The shorthand `max_j(z_j + v_j)` is only well-defined coordinatewise,
`[Y_trop]_r = max_j(z_j + v_jr)`, and different output coordinates may choose different `j`.
Ordinary low-temperature attention instead tends to the single selected vector `v_j*`. They agree
only under extra conditions. The runtime artifact must implement hard selection by `argmax z`, not
substitute the coordinatewise max-plus value map and call it attention.

---

## 2. Polyhedral decision regions

For fixed `h`, choice `j` is active on

```text
    R_j(h) = {x : z_j(x;h) >= z_k(x;h) for every k}
           = intersection_(k != j)
             {x : (a_j - a_k)^T x >= b_k(h) - b_j(h)}.
```

Each inequality defines a closed half-space, so every `R_j(h)` is a closed convex polyhedron. If the
cell has nonempty full-dimensional interior and the score term is not duplicated, its interior
contains the unique-winner points. A lower-dimensional cell can consist entirely of ties. Shared
facets satisfy `z_j = z_k`; intersections of several facets are multiway ties. Some affine terms can
be dominated everywhere, in which case their region is empty and they are never active.

This characterization gives three directly checkable artifacts for an input:

1. the active index or tied active set `A(x;h)`;
2. every pairwise slack `z_j* - z_k`, whose minimum is the uniqueness gap `Delta`;
3. the half-space inequalities defining the current cell.

For top-k routing, replace the single index by the ordered top-k set. Its cell is also polyhedral:
for selected `i` and unselected `k`, require `z_i >= z_k`. The ordering inside the selected set can
be retained as additional pairwise inequalities or discarded when only membership matters.

---

## 3. Newton and lifted-coefficient polytopes

The ordinary Newton polytope is `N = conv{a_1, ..., a_m}`. Coefficients affect which terms can be
active, so for the decision fingerprint we lift each affine term to
`u_j = (a_j, b_j(h)) in R^(d+1)` and form the distinct **lifted coefficient polytope**

```text
    N_lift(h) = conv{u_1, ..., u_m}.
```

Evaluating `F` takes the support function of `N_lift(h)` in direction `(x,1)`:

```text
    F(x;h) = max_(u in N_lift(h)) <u, (x,1)>.
```

Therefore `A(x;h)` identifies the choice IDs whose lifted generators lie on the exposed upper face
selected by `(x,1)`. A unique active generator is an exposed vertex of the lifted polytope; an exact
tie can expose an edge or higher-dimensional face. An active exponent `a_j` need not be a vertex of
the ordinary Newton polytope, and duplicate score terms may map several choice IDs to the same lifted
point, so the certificate retains both the choice IDs and the deduplicated lifted-face geometry.

The **interpretability fingerprint** for one decision is thus

```text
    (eligible choice IDs, active IDs, top-k order, score vector,
     pairwise slacks, uniqueness gap, exposed-face dimension, history-state digest).
```

This fingerprint is exact for the declared score vector: recomputing the scores and deterministic
tie rule reproduces it. It does not, by itself, claim that a selected token or expert caused the
final model prediction; that stronger causal claim belongs to the lesion/ablation baseline.

---

## 4. Mapping to the live code

### Synaptic attention

`SynapticCausalSelfAttention.forward` forms the causal base score

```text
    d_j(q) = q^T k_j / sqrt(D)
```

and then adds the selected-edge release term and septin distance barrier:

```text
    S(q) = base-dot-product top-k support,
    clip_c(u) = clamp(u, -c, c) if c > 0, else u,
    z_j = d_j
          + 1[j in S(q)] clip_c(lambda_loge log(epsilon + e_j))
          - barrier_strength |q_pos - k_pos| / T_key.
```

Biology is evaluated only on `S(q)` and scattered back; non-top-k causal edges receive zero
augmentation but still participate in the final softmax. With keys and positions fixed, this is
affine in `q` only after the release offsets are frozen and, separately, either `S(q)` is frozen or
the query is restricted to the relevant base-top-k support cell. The exact region is then the
intersection of that support cell and the augmented-score decision cell. This is
query-conditional geometry: as a function of the entire input sequence, both `q` and `k_j` vary and
their dot product is bilinear, so a global input-space polyhedral claim would require a separate
lifted or local certificate. The causal mask simply removes ineligible choices. The live
`release_canonical` value is not generally frozen: calcium influx uses `softplus(d_j)` and release
probability uses Hill and sigmoid functions of the same drive, while top-k selection changes which
edges receive that bias. Consequently the full augmented score is piecewise smooth and nonlinear
in `q`; its global cells need not be polyhedra. The runtime
certificate must distinguish:

- `exact_affine`: additive release/history offsets were frozen, and the base top-k support was
  either frozen or represented by explicit support-cell inequalities;
- `local_only`: the live nonlinear bias was evaluated at the current query, so the active set is
  exact at that point but the polyhedral-cell interpretation is not certified;
- `invalid`: non-finite scores, an unresolved tie under the requested uniqueness contract, or an
  inconsistent eligible mask.

This formula is specifically the standard branch (`self.flex is None`). The FlexAttention branch
uses its own all-edge score modifier, approximately
`d_j + lambda_loge log(key_factor_j qamp_j sigmoid(d_j) + epsilon) - barrier_j`, and intentionally
uses a different per-edge calcium/normalization approximation. Because that modifier is nonlinear
in `d_j`, its pointwise winner is observable but its geometry is `local_only` under this theorem.
The live implementation also uses `tau = 1`; the temperature-indexed family above is an analysis
contract until a later toggle explicitly divides logits by `tau`.

### Synaptic MoE routing

`SynapticMoE.forward` starts from the linear score `router(x)` and adds gene, metabolism, fatigue,
energy, and lifecycle logit offsets. Those fixed-state offsets fit `b_j(h)`. Its `base_bias` and
`gain_bias` are affine in `mean(x)`. The normalized router-alignment term
`normalize(router_probe(x)) dot normalize(router_embedding_j)` is generally nonlinear wherever the
probe norm varies. Near zero, the epsilon branch of `normalize` can instead be locally linear; that
does not make the global score family affine. The exact global polyhedral claim therefore requires
the term to be disabled, frozen, or explicitly replaced by a certified affine surrogate. The
emitted top-k expert IDs remain an exact pointwise fingerprint either way.

---

## 5. Vesicle depletion as a history-dependent tropical shift

For an attention edge, the biological coefficient enters additively in log-score space:

```text
    beta_j(h) = clip_c(lambda_loge log(epsilon + e_j(h))).
```

If the **measured normalized release** decreases from `e_old` to `e_new <= e_old`, then before clamp
saturation

```text
    Delta beta_j = lambda_loge log((epsilon + e_new) / (epsilon + e_old)) <= 0.
```

This is the precise monotonic score statement: a measured decrease in `e_j` translates choice `j`
downward in max-plus score space.
Equivalently, in the ordinary exponential domain it multiplies that choice's coefficient by
`((epsilon + e_new)/(epsilon + e_old))^(lambda_loge/tau)`. A measured increase moves the offset
upward under the same frozen-state conditions.
The boundary between choices `j` and `k` moves by the relative history shift
`Delta beta_j - Delta beta_k`, so a nonzero relative shift moves that boundary and can change which
polyhedral cell contains the next query while the frozen-history decision remains exactly auditable.

Live recovery can still be outweighed by changes in other state. Clamp saturation is intentionally
visible: once `beta_j` reaches `-c` or `+c`, further biological movement in the saturating direction
does not move the score, although movement in the opposite direction can leave saturation. A
monitor must report the saturation bit because the apparent tropical offset then stops representing
release magnitude faithfully in the saturated direction. When `c = 0`, the live configuration
disables clipping entirely.

Calling that measured decrease **depletion-caused** requires a controlled comparison: hold drive,
calcium/buffer, priming/clamp, energy/qamp, stochastic RNG outcome, selected support, and the shared
adaptive EMA denominator fixed while reducing available RRP. In the live recurrence all of those
quantities can also change `e_j`, so an observational decrease alone does not identify its cause.

---

## 6. Proof obligations and assumptions ledger

| ID | Assumptions | Formal statement | Verification artifact | Failure condition | Conservative fallback |
|---|---|---|---|---|---|
| H1 | At least two eligible finite scores, finite value vectors, `tau > 0`, and pre-dropout/evaluation readout; singleton handled separately with `Delta=+infinity`. | For a unique maximizer with gap `Delta > 0`, `1-p_j* <= (m-1)e^(-Delta/tau)` and `y_tau -> v_j*`. | Soft-versus-hard convergence sweep over decreasing temperature, with bound residuals. | Non-finite score/value, empty eligible set, `tau <= 0`, or dropout applied to the claimed readout. | Keep the ordinary softmax path and emit no tropical certificate. |
| H2 | Each certified score is affine in `x` after conditioning on recorded history `h`. | `R_j(h)` is the half-space intersection in section 2 and is therefore polyhedral. | Score coefficients, offsets, eligible mask, and inequality slacks in the region certificate. | A drive-dependent release/alignment term is active without a proved affine form. | Mark `local_only`; report the pointwise active set without a polyhedral guarantee. |
| H3 | Exact active face uses exact score equality; a numerically unique-winner claim additionally requires `Delta > tie_tol`; deterministic ordering is recorded. | An exactly unique active generator is an exposed vertex of `N_lift(h)`; exact ties expose a face. If `0 < Delta <= tie_tol`, face dimension is withheld as numerically ambiguous. | Exact active IDs, ambiguity candidates, score gap, tie tolerance, lifted-face dimension when licensed, tie rule. | Near-tie, duplicate terms, or non-reproducible ordering. | For exact equality return the tied face; for a near-tie return candidates and refuse vertex/face certification. |
| H4 | Release/support frozen during one decision; `lambda_loge >= 0`; clamp state recorded. A causal depletion claim additionally controls drive, all other biological state, RNG, support, and EMA denominator. | A measured decrease `e_new <= e_old` cannot increase the unclamped offset `beta_j`; under the controlled comparison, lowering RRP induces that use-tax shift. | Old/new release and RRP values, controlled-state digest, offset delta, support, clamp-saturation bit. | Uncontrolled state/RNG/normalizer change, negative gain, or untracked saturation. | Use the measured final scores only; make no depletion-causal claim. |
| H5 | Fingerprint scope is selection, not downstream causality. | Replaying the score vector, mask, and tie rule reproduces the selected active set exactly. | Deterministic replay fixture plus a digest of score-producing state. | Stochastic release is replayed without its RNG state or score inputs are omitted. | Label fingerprint non-replayable; fall back to existing attention-rollout/lesion tools. |

Assumption classes are structural (affine score family and mask), computational (finite arithmetic and
deterministic tie handling), and operational (complete state/RNG logging). Every certificate field is
measurable at runtime. None of H1-H5 licenses a robustness radius; the Lipschitz-to-radius theorem,
temperature/entropy gate, and anneal schedule are the separate `0642.6.1.2` obligation.

---

## 7. Falsifiable verification plan

The runtime and tests that consume this note must check all of the following:

- On random affine score families, the reported cell inequalities contain the input and the active
  IDs equal direct `argmax`/`topk`.
- For unique winners, a temperature sweep makes `1-p_j*` and `||y_tau-v_j*||` stay below the stated
  exponential bounds and converge to zero.
- Exact ties return all maximizers and never produce a unique-vertex certificate.
- Perturbations that retain strictly positive reported pairwise slacks keep a unique active ID;
  reaching a zero-slack facet can expand the active set, and crossing it changes the winner.
- Decreasing a frozen, unclamped release value never raises its augmented score; saturated offsets
  are explicitly flagged.
- Enabling the live drive-dependent release or normalized alignment term changes the scope to
  `local_only` unless a later artifact proves the required affine approximation error bound.

A counterexample to any exact claim invalidates the certificate and routes inference through the
unchanged softmax/top-k baseline. Default behavior remains unchanged: this thrust begins as a
read-only analysis layer, and the optional hard-routing toggle belongs to `0642.6.2.2`.

---

## 8. Transparency card contract

For one decision, a human-facing card should show:

```text
    equation:     z_j = a_j^T x + b_j(h),  j* = argmax_j z_j
    substituted:  top scores, winner, runner-up, and Delta
    geometry:     exact_affine | local_only | invalid; active vertex/face IDs
    intuition:    which token/expert won, its score gap, and (when a norm is declared) boundary distance
    assumptions:  mask, frozen-state status, tie tolerance, clamp saturation
    decision flip: the nearest score equality or failed validity gate
```

The score gap `Delta` is not itself a geometric distance. Under a declared input norm, the distance
to the `j*`/`k` equality hyperplane is
`(z_j* - z_k) / ||a_j* - a_k||_*`. Minimize only over competitors with `a_j* != a_k`:
an equal-slope, lower-offset competitor has no equality hyperplane and infinite boundary distance,
while equal slope and equal offset is a duplicate/tie. The resulting minimum is the nearest-boundary
distance derived in `0642.6.1.2`. The advanced mathematics is inspectable, but the one-line intuition
remains “choice `j*` won by gap `Delta`; the exact polyhedral interpretation is [valid/not valid]
under the recorded score scope.”

---

## References

- Maclagan, D. and Sturmfels, B. *Introduction to Tropical Geometry*. The max-plus polynomial,
  regular subdivision, and Newton-polytope correspondence.
- Maslov, V. P. and Kolokoltsov, V. N. *Idempotent Analysis and Its Applications*. The
  logarithmic/zero-temperature passage from ordinary to max-plus algebra.
- Internal: `bio_inspired_nanochat/synaptic.py` (`SynapticCausalSelfAttention.forward`,
  `release_canonical`, `SynapticMoE.forward`) and `docs/theory/README.md` (Thrust H roadmap status).
