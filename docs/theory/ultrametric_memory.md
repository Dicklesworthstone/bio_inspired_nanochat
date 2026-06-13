# Ultrametric / RSB Hierarchical Associative Memory — Theory Note (bead `0642.4.1`)

_Thrust B — hierarchical exponential-capacity memory. Author: BeigeSquirrel · 2026-06-12._

## Purpose & scope

This note gives the mathematical foundation for a **hierarchical associative memory** whose attractor
basins nest *ultrametrically* (category → subcategory → instance) rather than flat (one level, as in
standard attention / modern Hopfield). It establishes three results and the assumptions they rest on,
so the runtime (`0642.4.2.1`, the p-adic kernel + depletion descent) and falsification (`0642.4.3`,
vs flat modern-Hopfield) build against a fixed contract:

1. **A p-adic LCP retrieval kernel induces a genuine ultrametric** geometry on memory — the
   strong-triangle (tree) structure of Parisi's RSB pure states (§1).
2. **Exponential capacity in depth**: a depth-`L`, branching-`p` tree stores `p^L` instance memories
   while retrieval stays `O(L)` (§3).
3. **Provable coarse-to-fine retrieval**: a query whose fine digits are corrupted still recovers the
   correct *category*, beating flat retrieval that has no notion of category (§4 — the leapfrog).

Everything is grounded in the project's mechanisms: the fast-weight memory is the attractor net, and
**vesicle depletion is the literal gradient flow *down* the tree** (deplete the coarse attractor,
descend to finer basins); septin lateral inhibition shapes basin boundaries. The reference math is
`bio_inspired_nanochat/ultrametric_memory.py`; the claims are corroborated numerically in
`tests/test_ultrametric_memory.py` (§5). The **baseline** this beats is standard attention / flat
modern-Hopfield retrieval (Ramsauer et al.), which has exponential capacity but only *one* level.

---

## 0. Coordinates & the kernel (as it actually is)

A memory carries a base-`p` hierarchical coordinate of `L` digits, **most-significant first**: digit 0
is the coarsest category, …, digit `L−1` the finest instance — a path from the root of a `p`-ary tree
to a leaf. There are `p^L` leaves. The **longest common prefix** `LCP_p(x,y)` is the depth of the
common ancestor of `x` and `y`. The retrieval similarity is the LCP kernel

```
        sim(x,y) = α^{L − LCP_p(x,y)} ∈ (0,1],     0 < α < 1,
```

`1` iff identical, `α^L` iff a different category at the root — *exponential* contrast per level.
Retrieval is the modern-Hopfield readout `weights ∝ exp(β·sim)` over the stored memories.

---

## 1. The p-adic LCP distance is an ultrametric  → subtask `0642.4.1` (geometry)

Define the tree distance `d(x,y) = p^{−LCP_p(x,y)}` (0 if identical). It is an **ultrametric**: it
satisfies not just the triangle inequality but the *strong* one,

```
        d(x,z) ≤ max( d(x,y), d(y,z) ).                         (strong triangle / ultrametric)
```

*Proof.* `LCP(x,z) ≥ min(LCP(x,y), LCP(y,z))`: if `x,y` agree on their first `a` digits and `y,z` on
their first `b`, then `x,z` agree on their first `min(a,b)`. Hence
`d(x,z) = p^{−LCP(x,z)} ≤ p^{−min(LCP(x,y),LCP(y,z))} = max(d(x,y), d(y,z))`. ∎

Equivalently, **every triangle is isosceles with its two longest sides equal** — the geometric
signature of Parisi's RSB pure-state overlaps and of `p`-adic number fields (`|·|_p` is ultrametric).
The runtime **tree-ness gauge** `ultrametricity_score` measures the empirical fraction of isosceles
triples (`1.0` ⟺ exactly ultrametric); the flat Hamming distance scores well below 1 (it is *not*
ultrametric), which is exactly why it cannot do hierarchy.

---

## 2. Retrieval as gradient flow down the tree

On the LCP-kernel energy landscape, retrieval descends the tree: the coarse digits dominate `sim`
(weight `α^{L−1}` for the root vs `α^0` for the leaf), so the readout first commits to a **category**,
then — as the coarse attractor is depleted (vesicle depletion, the bio drive) — resolves the
subcategory, then the instance. This is a *coarse-to-fine* search whose first decision (category) is
the most robust, the opposite of a flat readout that resolves all levels at once and so has no
fallback when the instance is ambiguous.

---

## 3. Exponential capacity in depth  → subtask `0642.4.1` (capacity certificate)

A depth-`L`, branching-`p` tree stores

```
        leaf capacity = p^L   (instance memories),   resolvable at L coarse-to-fine levels,
```

with retrieval cost `O(L)` digit comparisons. Capacity grows **exponentially in depth** while the
retrieval path is linear — the hierarchical-exponential-capacity claim (`capacity_certificate`). A
flat store over the same `p`-symbol alphabet resolves only `p` states at one level; the tree nests
`p` *per level*. This is the RSB picture: a hierarchy of pure states, exponentially many leaves
organized by their ultrametric overlaps.

---

## 4. Coarse-to-fine retrieval beats flat under corruption  → the leapfrog (`0642.4.3`)

The falsifiable payoff. Store a **sparse** prototype bank (so a query is not itself stored and
retrieval must generalize). Corrupt the `n_fine` finest digits of a clean leaf — destroy the
*instance*, keep the *category* (the coarse prefix). Retrieve:

- **Ultrametric (LCP kernel)**: the never-corrupted coarse prefix dominates `sim`, so the readout
  recovers a prototype of the **correct category** — recall@category `= 1.0` for *any* corruption
  depth (the coarse digit is intact and the bank covers every category).
- **Flat (Hamming modern-Hopfield)**: weights all digits equally, so the corrupted fine digits pull
  the query toward whichever prototype coincides on them — including **wrong-category** prototypes —
  and category recall degrades (empirically to ~0.6 at heavy corruption for `p=4, L=4`).

So the ultrametric memory exhibits **graceful degradation down the tree**: lose the instance, keep the
category; lose the subcategory, keep the category — exactly the compositional, noise-robust recall the
baselines cannot provide. (`leapfrog_recall`.)

---

## 5. Proof-obligation & assumptions ledger  → consumed by `0642.4.2`, `0642.4.3`

| # | Assumption (discharged by) | Statement | Failure mode | Fallback |
|---|---|---|---|---|
| U1 | **Ultrametric coordinates**: memories carry tree (p-adic) coordinates; the distance is `p^{−LCP}` (`ultrametricity_score = 1`). | `d` is an ultrametric; the strong triangle holds; retrieval is hierarchical. | The learned landscape is *not* ultrametric (tree-ness score < 1) — the basins do not nest. | Fall back to flat modern-Hopfield retrieval; flag (no hierarchical guarantee). |
| U2 | **Coarse-weighting**: `sim` weights the coarse prefix exponentially (`α < 1`). | The category decision is the most robust; coarse-to-fine descent. | `α → 1` (flat weighting) erases the hierarchy ⟹ no category robustness. | Keep `α` bounded below 1; else accept flat behavior. |
| U3 | **Category coverage**: the bank stores ≥1 prototype per category at the queried level. | Recall@category `= 1` under instance corruption. | A category has no stored prototype ⟹ it cannot be recovered. | Report recall only over covered categories; flag the gap. |
| C | **Capacity vs ultrametricity tradeoff**: enforcing the tree may cost raw capacity. | `p^L` leaves at depth `L`. | Over-constraining the landscape to be ultrametric reduces storable distinct patterns. | Measure the tree-ness/capacity tradeoff explicitly (the `0642.4.3` capacity curve); choose the depth that fits the task. |

**Composition note** (`0642.10`/`0642.11.1`): the ultrametric retrieval composes with the other
thrusts while U1 holds — the fast-weight landscape that stores the memories is the same one the
Hebbian/consolidation dynamics shape, so the tree-ness gauge must be checked on the *live* landscape,
not just the idealized coordinates.

---

## 6. Numerical corroboration

`tests/test_ultrametric_memory.py` checks the results against the reference implementation:

- **Ultrametric geometry** — `ultrametricity_score = 1.0` for the tree distance (strong triangle on
  every triple), `< 0.9` for the flat Hamming distance; the LCP kernel is monotone in the shared prefix.
- **Capacity** — `leaf_capacity = p^L`, exponential in depth (doubling `L` squares the capacity).
- **The leapfrog** — under instance corruption with a sparse bank, ultrametric recall@category stays
  `≈ 1.0` across corruption depths while the flat baseline degrades; the ultrametric arm never
  underperforms the flat arm.

These corroborate the geometry (an exact ultrametric), the capacity count, and the coarse-to-fine
retrieval advantage — what licenses the runtime kernel (`0642.4.2.1`) and the falsification (`0642.4.3`).

---

## References

- Mézard, M., Parisi, G., Virasoro, M. (1987). *Spin Glass Theory and Beyond.* — RSB ultrametricity.
- Rammal, R., Toulouse, G., Virasoro, M. (1986). *Ultrametricity for physicists.* Rev. Mod. Phys.
- Ramsauer, H., et al. (2021). *Hopfield Networks is All You Need.* — modern Hopfield = attention, flat.
- Khrennikov, A. (1997). *Non-Archimedean Analysis* / p-adic models — `|·|_p` as the ultrametric coordinate.
- Internal: `bio_inspired_nanochat/ultrametric_memory.py`, `docs/theory/singular_perturbation.md`
  (`0642.2.1`, the proof-ledger pattern), `tests/test_engine.py::test_ultrametric_attention_*`.
