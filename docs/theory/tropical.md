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

## 6. Exact max-plus slopes and a certified active-cell radius

Fix an input norm `||.||` and its dual norm `||.||_*`. For the affine skeleton,

```text
    F(x) = max_j (a_j^T x + b_j).
```

The exact full-space Lipschitz constant is

```text
    L_F = sup_x sup_(g in partial F(x)) ||g||_*
        = max_(j whose region R_j is nonempty) ||a_j||_*.
```

At a tie, `partial F(x)` is the convex hull of the active slopes; a norm achieves its maximum on a
vertex of that hull, which gives the last equality. Keeping dominated terms yields the conservative,
easier bound `max_j ||a_j||_*`. Thus the max-plus slope is computable exactly after empty regions are
removed, and conservatively without solving region feasibility.

`L_F` bounds changes in the maximum score, but it does **not** by itself certify that the maximizing
ID stays fixed. Selection stability uses each pairwise score difference. If `j*` is the unique winner,

```text
    g_j*k(x) = z_j*(x) - z_k(x) > 0,
    L_j*k    = ||a_j* - a_k||_*,
    g_j*k(x + delta) >= g_j*k(x) - L_j*k ||delta||.
```

For every competitor with a different slope, define its facet distance

```text
    r_j*k(x) = g_j*k(x) / L_j*k.
```

Equal slopes require explicit handling: if `a_j* = a_k` and `b_j* > b_k`, choice `k` can never catch
the winner and its distance is `+infinity`; equal slope and equal offset is a duplicate exact tie, so
a unique-winner certificate is invalid. The certified top-1 radius is

```text
    r_top1(x) = min_(k != j*, a_k != a_j*) r_j*k(x).
```

The minimum of an empty set is `+infinity`, matching the case where no eligible competitor has a
different slope.
For every perturbation `||delta|| < r_top1`, all pairwise differences stay positive and `j*`
remains the unique active ID. At the closed boundary `||delta|| = r_top1`, a tie may occur. On the
unconstrained input space this is the exact distance to the nearest score-equality hyperplane; a
production certificate can emit `(1 - safety_fraction) r_top1`, with
`0 < safety_fraction < 1`, to keep a strict numerical margin.

For an unordered top-k set `S`, the corresponding radius is

```text
    r_topk(x) = min_(i in S, k not in S, a_i != a_k)
                (z_i(x) - z_k(x)) / ||a_i - a_k||_*.
```

Apply the same equal-slope rules. This preserves membership of `S`; preserving the order inside `S`
also requires the pairwise constraints among selected choices. The attention or expert **selection**
is constant inside the certified ball. The numerical output is only constant when the selected value
is fixed; if values depend on the perturbed input, their own Lipschitz bound must be composed with the
selection certificate.

Under this note's query-conditional attention threat model—keys, values, history, and mask fixed—the
hard layer output is exactly the same `v_j*` throughout the top-1 ball. Under a full-sequence threat
model the values also move. Likewise, a MoE certificate keeps the selected expert IDs fixed but the
experts still transform the perturbed token; those outputs require the selected value/expert operator
norm before “output stable” can mean more than stable selection.

The standard attention path has two coupled cells: the base-dot-product top-k support determines
where biological offsets are scattered, and the augmented score determines the final winner. If the
support is not frozen independently of `q`, compute `r_support` from the base scores using the top-k
formula above and emit

```text
    r_standard = min(r_support, r_augmented_selection).
```

Omitting `r_support` would allow a perturbation to change which edges receive biological offsets
before reaching the reported augmented-score facet, invalidating the affine expression itself.

These are conditional, per-decision/per-layer guarantees. An end-to-end model radius additionally
needs the input-to-layer operator norms and a final output/class margin; multiplying layer constants
without that margin does not create a predictive robustness certificate. Nonlinear live biological
scores remain `local_only` unless their approximation error and derivative remainder are bounded.

---

## 7. Temperature, score-gap, and entropy validity gate

For a unique winner, define the dimensionless regime gap

```text
    kappa = Delta / tau,
    u     = (m - 1) exp(-kappa),
    p_j*  >= 1 / (1 + u),
    q := 1 - p_j* <= q_hat := u / (1 + u).
```

The entropy is largest when the losing mass `q` is uniform over the other `m-1` choices. With
`h_2(q) = -q log q - (1-q) log(1-q)`, this gives

```text
    H(p) <= h_2(q_hat) + q_hat log(m - 1),
    H_normalized = H(p) / log(m).
```

The bound ranges from `log(m)` at a zero gap to zero as `kappa -> infinity`. For a requested minimum
hard mass `1/m < p_min < 1`, the sufficient gap threshold is

```text
    kappa >= kappa_min := log((m - 1) p_min / (1 - p_min)).
```

These expressions assume `m >= 2`. For a singleton eligible set, use `p_1 = 1` and
`H = H_normalized = 0`; selection is trivially hard, `r = +infinity`, and the gap/`kappa`/entropy
thresholds are bypassed. That certificate is labeled `singleton`, not evidence that annealing reached
a low-temperature regime.

A soft-to-hard **readout** certificate with `m >= 2` is valid only when **all** of these gates pass on
the masked, pre-dropout score distribution:

1. the score scope is `exact_affine`, finite, and replayable;
2. `tau > 0`, `Delta > tie_tol`, and `kappa >= kappa_min`;
3. measured winner mass is at least `p_min` and measured normalized entropy is at most `H_max`;
4. the norm and threat model are declared and `r_top1` or `r_topk` exceeds the configured
   non-vacuity floor;
5. any composition/timescale and provenance gates required by the caller pass.

The gap bound and measured entropy are intentionally both recorded: the analytic bound proves a
sufficient low-temperature regime, while the measured value detects implementation or masking drift.
At high temperature, a region/radius may still describe the hypothetical hard skeleton, but it does
not certify the live soft readout. Report it as `soft_approximation` with `certified=false` and keep
the existing softmax/top-k path.

The protected choice set changes the contract:

- **Attention hard readout:** `m` is every eligible causal edge. Geometry and all temperature/mass/
  entropy gates above apply.
- **MoE top-k membership:** membership is selected before the within-set softmax, so `r_topk`, exact
  scope, non-vacuity, and provenance gates apply; temperature metrics are diagnostic and cannot
  invalidate this membership certificate.
- **MoE hard top-1 readout:** `m` is the current selected set for the softmax gap/mass/entropy gate;
  geometry must additionally keep the selected set and its top-1 ID stable. As `tau -> 0`, this is the
  new claim that the selected mixture collapses to the highest routed expert.

Every emitted certificate states one of these scopes. Reusing the attention temperature gate for
MoE top-k membership would be a category error.

---

## 8. Default-off temperature/septin anneal

Let `s in [0,1]` be persisted schedule progress. A positive geometric cooling schedule avoids zero
temperature:

```text
    tau(s) = tau_start (tau_min / tau_start)^s,
    0 < tau_min <= tau_start.
```

For standard attention, the septin barrier may be ramped independently,

```text
    gamma(s) = gamma_start + s (gamma_end - gamma_start),
```

and substituted for `barrier_strength`. Increasing `gamma` favors nearby keys but is not guaranteed
to increase the winner gap: it can move decision boundaries or change the winner. Therefore schedule
progress never grants validity. Scores, gap, entropy, radius, support, clamp state, and scope are
recomputed after each scheduled update, and only the measured gate in section 7 can issue a
certificate.

The toggle remains default-off. Off means the live `tau = 1` behavior and configured septin barrier
are byte-for-byte unchanged. When enabled, a hard path may enter only after the stricter entry gate
passes for a configured number of consecutive observation windows. It must fall back immediately on
any non-finite value, exact-scope/provenance failure, or violation of the declared exit thresholds.
Entry thresholds may be stricter than exit thresholds to prevent chattering, but the exit thresholds
are the actual certificate assumptions and may never be bypassed by hysteresis.

Every transition records schedule parameters and progress, measured `Delta`, `kappa`, entropy,
winner mass, radius, norm, scope, and the exact gate that passed or failed. The deterministic fallback
is the unchanged softmax/top-k baseline; it may retain diagnostics but must drop the certified label.

---

## 9. Proof obligations and assumptions ledger

| ID | Assumptions | Formal statement | Verification artifact | Failure condition | Conservative fallback |
|---|---|---|---|---|---|
| H1 | At least two eligible finite scores, finite value vectors, `tau > 0`, and pre-dropout/evaluation readout; singleton handled separately with `Delta=+infinity`. | For a unique maximizer with gap `Delta > 0`, `1-p_j* <= (m-1)e^(-Delta/tau)` and `y_tau -> v_j*`. | Soft-versus-hard convergence sweep over decreasing temperature, with bound residuals. | Non-finite score/value, empty eligible set, `tau <= 0`, or dropout applied to the claimed readout. | Keep the ordinary softmax path and emit no tropical certificate. |
| H2 | Each certified score is affine in `x` after conditioning on recorded history `h`. | `R_j(h)` is the half-space intersection in section 2 and is therefore polyhedral. | Score coefficients, offsets, eligible mask, and inequality slacks in the region certificate. | A drive-dependent release/alignment term is active without a proved affine form. | Mark `local_only`; report the pointwise active set without a polyhedral guarantee. |
| H3 | Exact active face uses exact score equality; a numerically unique-winner claim additionally requires `Delta > tie_tol`; deterministic ordering is recorded. | An exactly unique active generator is an exposed vertex of `N_lift(h)`; exact ties expose a face. If `0 < Delta <= tie_tol`, face dimension is withheld as numerically ambiguous. | Exact active IDs, ambiguity candidates, score gap, tie tolerance, lifted-face dimension when licensed, tie rule. | Near-tie, duplicate terms, or non-reproducible ordering. | For exact equality return the tied face; for a near-tie return candidates and refuse vertex/face certification. |
| H4 | Release/support frozen during one decision; `lambda_loge >= 0`; clamp state recorded. A causal depletion claim additionally controls drive, all other biological state, RNG, support, and EMA denominator. | A measured decrease `e_new <= e_old` cannot increase the unclamped offset `beta_j`; under the controlled comparison, lowering RRP induces that use-tax shift. | Old/new release and RRP values, controlled-state digest, offset delta, support, clamp-saturation bit. | Uncontrolled state/RNG/normalizer change, negative gain, or untracked saturation. | Use the measured final scores only; make no depletion-causal claim. |
| H5 | Fingerprint scope is selection, not downstream causality. | Replaying the score vector, mask, and tie rule reproduces the selected active set exactly. | Deterministic replay fixture plus a digest of score-producing state. | Stochastic release is replayed without its RNG state or score inputs are omitted. | Label fingerprint non-replayable; fall back to existing attention-rollout/lesion tools. |
| H6 | Affine score family; declared input norm/dual norm; nonempty active regions identified or conservatively retained. | `F=max_j z_j` has exact full-space constant `max_(R_j nonempty) ||a_j||_*`; retaining all terms is a valid upper bound. | Active-slope ledger and norm-specific constants, checked against adversarial score perturbations. | Unknown norm, nonlinear score, or omitted active slope. | Emit only a conservative local diagnostic; no global Lipschitz claim. |
| H7 | Fixed eligible mask, unique winner/top-k boundary, finite positive score margins, exact affine scope, explicit equal-slope handling, and every score-defining support-cell radius included. | The minimum normalized pairwise margin in section 6 is an exact unconstrained distance to the nearest relevant equality hyperplane and a strict-ball selection radius; an empty finite-boundary set yields `+infinity`. Standard attention takes the minimum of base-support and augmented-selection radii. | Per-competitor margins/slopes/radii, support radius, chosen norm, safety fraction, adversarial boundary back-test. | Tie/near-tie, omitted support boundary, zero/negative/NaN radius, nonlinear term, changing mask, or unsupported threat model. | Keep soft routing; refuse robustness certification. |
| H8 | For soft-to-hard readout with `m>=2`: masked finite pre-dropout scores, `tau>0`, `1/m < p_min < 1`, `0 <= H_max <= 1`, `tie_tol >= 0`, and positive non-vacuity floor. Singleton uses the explicit bypass above; MoE membership uses H7 rather than temperature. | The gap gives the winner-mass and entropy bounds in section 7; attention or MoE top-1 readout is live only while analytic and measured gates pass. MoE top-k membership is temperature-independent. | Certificate scope plus `m`, `tau`, `Delta`, `kappa`, entropy bound/measurement, winner mass, radius, and gate/bypass verdict. | High entropy, insufficient gap/mass/radius, wrong scope, invalid thresholds, dropout, or stale evidence. | Label readout results `soft_approximation`, keep softmax/top-k, and set `certified=false`; retain a separately valid membership certificate if H7 passes. |
| H9 | Default-off schedule, persisted progress/parameters, deterministic score replay, and immediate exit on assumption failure. | Annealing changes temperature/barrier but never self-certifies; only the measured H8 gate enables the optional hard path. | Transition JSONL with before/after parameters, observations, gate, and fallback reason. | Missing schedule provenance, hysteresis bypasses exit gate, or restart changes progress. | Restore the baseline parameters/path and invalidate the certificate. |

Assumption classes are structural (affine score family and mask), computational (finite arithmetic and
deterministic tie handling), and operational (complete state/RNG logging). Every certificate field is
measurable at runtime. H6-H9 license only the declared per-layer selection and soft-to-hard claims;
they do not imply end-to-end predictive robustness without the additional composition described in
section 6.

---

## 10. Falsifiable verification plan

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
- For `L1`, `L2`, and `L-infinity` threat norms, perturbations strictly inside the reported
  dual-norm radius preserve top-1/top-k selection. For finite different-slope competitors, a
  constructed nearest-facet perturbation reaches the predicted tie. Equal-slope/lower-offset terms
  yield `+infinity` and no finite boundary; equal-slope/equal-offset duplicates invalidate uniqueness
  immediately.
- Random softmax distributions obey the analytic winner-mass and entropy bounds; values exactly on
  either configured threshold exercise the documented inclusive/exclusive gate semantics.
- Cooling and septin schedules replay from persisted progress. Schedule completion with a failed
  measured gate never enables hard mode, and any exit-gate violation triggers the baseline in the
  same decision.

A counterexample to any exact claim invalidates the certificate and routes inference through the
unchanged softmax/top-k baseline. Default behavior remains unchanged: this thrust begins as a
read-only analysis layer, and the optional hard-routing toggle belongs to `0642.6.2.2`.

---

## 11. Transparency card contract

For one decision, a human-facing card should show:

```text
    equation:     z_j = a_j^T x + b_j(h),  j* = argmax_j z_j
    substituted:  top scores, winner, runner-up, and Delta
    geometry:     exact_affine | local_only | invalid; active vertex/face IDs
    robustness:   input norm, pairwise slope, certified radius, and safety fraction
    regime:       tau, kappa, winner mass, entropy bound/measurement, gate verdict
    intuition:    which token/expert won and the nearest certified decision boundary
    assumptions:  mask, frozen-state status, tie/clamp state, threat model, provenance
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

## 12. Held-out falsification protocol (`0642.6.3.1`)

The executable experiment is `scripts/e2e/tropical_falsification.py`. This section records the
decision rules before the held-out confirmation seeds are executed. Exploratory development used
seeds `11, 23, 37, 53, 71, 89, 107, 131`; the fixed confirmation seeds are
`149, 167, 181, 199, 223, 241, 263, 281`. The family has six affine choices in two dimensions, a
unique winner, and back-solved offsets. No rejection loop selects successful outcomes.

The named controls are:

1. ordinary `tau=1` soft readout versus the hard readout over the fixed sweep
   `(1, .5, .2, .1, .05, .02, .01)`;
2. a local NumPy argmax oracle, independent of the runtime argmax implementation, queried over
   4,096 angular rays with 48 bisection steps to estimate the first L2 decision flip;
3. 512 random perturbations strictly inside each certified radius; and
4. a frozen-value winner-lesion target versus the active-vertex fingerprint and one-layer rollout
   at the distinct fixed temperature `tau=.5`.

The verdict is **invalidated** on any runtime/oracle argmax disagreement, soft/hard mismatch,
exponential-bound violation, failed temperature/fallback control, non-positive certificate, flip
inside the certificate, certificate exceeding the empirical flip, angular attack error above
`1e-3` relative to the analytic affine boundary, or non-exact lesion-target attribution. A
**positive** verdict additionally requires paired bootstrap intervals plus paired t and Wilcoxon
tests at `alpha=.05` for cold-readout and attribution error. Anything else is an honest **null**.

Certified-to-empirical radius ratio is descriptive formula/attack-resolution conformance, not a
statistical success threshold: the certificate intentionally applies a five-percent safety fraction,
so a well-resolved attack should yield a ratio near `.95`. The test can falsify soundness or attack
resolution, but cannot establish a learned robustness gain.

Each invocation writes to a fresh run-ID subdirectory under `runs/e2e/tropical_falsification/` and
refuses a nonempty explicit directory. Registry rows carry a machine-readable verdict; null and
invalidated rows are ineligible for best-result queries. Run the held-out protocol with:

```bash
uv run python -m scripts.e2e.tropical_falsification
```

### Held-out result

Protocol commit `5f699ae` was pushed before the confirmation seeds were executed. Run
`b6ecc9dcd476` then produced a **positive** verdict:

| Check | Held-out result |
|---|---|
| Runtime vs independent argmax | `8/8` seeds and all seven temperatures matched; exactness mean and 95% CI were `1.0` |
| Soft-to-hard readout | cold-minus-`tau=1` mean L1 delta `-1.45541`, bootstrap 95% CI `[-1.46880, -1.44355]`, paired-t `p=1.42e-14`, Wilcoxon `p=.0078125` |
| Exponential error bound | every temperature at every seed satisfied the declared bound |
| Non-vacuous robustness | certified radii `[.234365, .311701]`, independent attack radii `[.246700, .328106]`, maximum analytic-boundary resolution error `7.73e-8`, and zero flips in 4,096 total interior trials |
| Radius conformance | descriptive certified/attack ratio mean `.94999989`, 95% CI `[.94999981, .94999997]`; empirical-minus-certified margin CI `[.01319, .01515]` |
| Frozen-readout lesion target | active-vertex L1 error `0` for all seeds; active-minus-`tau=.5` rollout mean delta `-1.18344`, 95% CI `[-1.21392, -1.15419]`, paired-t `p=2.91e-11`, Wilcoxon `p=.0078125` |
| Runtime gate and fallback | `8/8` high-temperature decisions returned the supplied baseline object; `8/8` low-temperature decisions authorized the fingerprint-matching hard readout |

The single-run event stream contains 99 records, all carrying run ID `b6ecc9dcd476`. The tracked
strict report is [`results/tropical_falsification_b6ecc9dcd476.json`](../../results/tropical_falsification_b6ecc9dcd476.json);
it includes the full per-seed statistics, protocol SHA, raw-statistics SHA-256, and the 99-record
event-stream SHA-256. The eight committed registry rows carry source SHA `5f699ae`, a
machine-readable positive verdict, and explicit best-query eligibility. Earlier exploratory run
`fef071510c5d` is retained in the append-only corpus, but its mixed event artifact and
pre-implementation SHA make it non-confirmatory; the legacy free-text verdict policy excludes those
rows from best-result queries.

The scope remains the standalone exact-affine, query-conditional selection skeleton. The lesion
target makes the attribution check causal for this frozen hard-selection readout only; it does not
establish causality for a downstream model prediction. The broader lesion and causal-tracing work in
`odq.2` is still required. Live nonlinear attention/MoE adapters must continue to fail closed until
they can supply an exact-affine score ledger or a separately proved approximation-error certificate.

---

## References

- Maclagan, D. and Sturmfels, B. *Introduction to Tropical Geometry*. The max-plus polynomial,
  regular subdivision, and Newton-polytope correspondence.
- Maslov, V. P. and Kolokoltsov, V. N. *Idempotent Analysis and Its Applications*. The
  logarithmic/zero-temperature passage from ordinary to max-plus algebra.
- Internal: `bio_inspired_nanochat/synaptic.py` (`SynapticCausalSelfAttention.forward`,
  `release_canonical`, `SynapticMoE.forward`) and `docs/theory/README.md` (Thrust H roadmap status).
