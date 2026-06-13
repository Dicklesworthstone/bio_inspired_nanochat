"""Ultrametric / RSB hierarchical associative memory — reference implementation (Thrust B, `0642.4.1`).

Engineers the fast-weight attractor landscape so its basins **nest ultrametrically** (category →
subcategory → instance) instead of being flat (one level, as in standard attention / modern Hopfield).
The organizing tool is the **p-adic longest-common-prefix (LCP) kernel**: memories carry base-`p`
hierarchical coordinates (digit 0 = coarsest category, … , digit `L−1` = finest instance), and the
retrieval similarity is `sim(q,k) = α^{L − LCP_p(q,k)}` — items sharing a longer prefix (a deeper
common ancestor in the tree) are more similar. The induced distance `d(q,k) = p^{−LCP}` is a genuine
**ultrametric** (it obeys the strong triangle inequality `d(x,z) ≤ max(d(x,y), d(y,z))`), which is the
hallmark of Parisi's RSB pure-state geometry.

This buys **provable coarse-to-fine retrieval**: a query whose fine (low-order) digits are corrupted
still retrieves the correct *category* (its high-order prefix), degrading gracefully down the tree —
the leapfrog over flat retrieval, which has no notion of category and is misled by instance noise.
Bio mapping: vesicle depletion is the literal gradient flow *down* the tree (deplete the coarse
attractor, descend to finer basins). This module is the theory + reference math; the live p-adic
retrieval kernel + depletion-driven descent (`0642.4.2.1`) and the falsification vs flat
modern-Hopfield (`0642.4.3`) build on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =========================================================================== #
# §1. p-adic coordinates, LCP, and the ultrametric distance / kernel
# =========================================================================== #
def padic_digits(x: int, p: int, n_levels: int) -> list[int]:
    """The base-`p` digits of `x` over `n_levels`, **most-significant first** (digit 0 = coarsest)."""
    if not 0 <= x < p ** n_levels:
        raise ValueError(f"coordinate {x} out of range [0, {p ** n_levels}) for p={p}, L={n_levels}")
    return [(x // p ** (n_levels - 1 - i)) % p for i in range(n_levels)]


def lcp(x: int, y: int, p: int, n_levels: int) -> int:
    """Longest common prefix length of `x`, `y` in base `p` (the depth of their common ancestor)."""
    dx, dy = padic_digits(x, p, n_levels), padic_digits(y, p, n_levels)
    c = 0
    for a, b in zip(dx, dy):
        if a != b:
            break
        c += 1
    return c


def tree_distance(x: int, y: int, p: int, n_levels: int) -> float:
    """The ultrametric tree distance `d(x,y) = p^{−LCP}` (0 if identical).

    Obeys the **strong triangle inequality** because `LCP(x,z) ≥ min(LCP(x,y), LCP(y,z))` (§1 of the
    note): two coordinates can only share a shorter prefix than the worse of two pairwise prefixes.
    """
    if x == y:
        return 0.0
    return float(p ** (-lcp(x, y, p, n_levels)))


def lcp_kernel(x: int, y: int, p: int, n_levels: int, alpha: float = 0.5) -> float:
    """The ultrametric retrieval similarity `α^{L − LCP} ∈ (0, 1]` (1 iff identical, `α^L` iff different category).

    Monotone *decreasing* in the ultrametric distance: a longer shared prefix ⟹ higher similarity. `α`
    sets how sharply similarity decays per level (smaller `α` ⟹ sharper hierarchical contrast).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    return float(alpha ** (n_levels - lcp(x, y, p, n_levels)))


def distance_matrix(items: list[int], p: int, n_levels: int) -> np.ndarray:
    """The pairwise ultrametric tree-distance matrix for `items`."""
    n = len(items)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = tree_distance(items[i], items[j], p, n_levels)
    return d


def kernel_matrix(items: list[int], p: int, n_levels: int, alpha: float = 0.5) -> np.ndarray:
    """The pairwise LCP-kernel similarity matrix for `items`."""
    n = len(items)
    k = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k[i, j] = lcp_kernel(items[i], items[j], p, n_levels, alpha)
    return k


# =========================================================================== #
# §2. Ultrametricity (tree-ness) diagnostic
# =========================================================================== #
def ultrametricity_score(d: np.ndarray, *, rtol: float = 1e-6) -> float:
    """Fraction of triples whose distances are **isosceles with the two largest equal** — the
    strong-triangle (ultrametric) condition. `1.0` ⟺ a perfectly ultrametric (tree) landscape.

    For a true ultrametric every triple `{i,j,k}` has its two largest pairwise distances equal; the
    score is the empirical rate of that condition, the runtime tree-ness gauge the certificate rests on.
    """
    n = d.shape[0]
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = sorted((d[i, j], d[i, k], d[j, k]))
                tot += 1
                if abs(c - b) <= rtol * max(c, 1e-12):
                    ok += 1
    return ok / tot if tot else 1.0


def is_ultrametric(d: np.ndarray, *, rtol: float = 1e-6) -> bool:
    """True iff the distance matrix is (numerically) ultrametric — every triple is isosceles."""
    return ultrametricity_score(d, rtol=rtol) >= 1.0 - 1e-12


# =========================================================================== #
# §3. Capacity certificate (RSB hierarchical capacity)
# =========================================================================== #
@dataclass(frozen=True)
class CapacityCertificate:
    """The hierarchical storage capacity of a depth-`L`, branching-`p` ultrametric memory."""

    branching: int            # p
    n_levels: int             # L
    leaf_capacity: int        # p^L — storable instance memories (exponential in depth)
    nodes_per_level: tuple    # categories resolvable at each level 1..L
    flat_capacity_ref: int    # a flat one-level store of the same coordinate alphabet (p) for contrast


def capacity_certificate(p: int, n_levels: int) -> CapacityCertificate:
    """The capacity certificate: `p^L` leaf memories over `L` coarse-to-fine levels (note §3).

    Capacity grows **exponentially in depth** `L` while retrieval stays `O(L)` digit comparisons — the
    hierarchical-exponential-capacity claim. A flat store over the same `p`-symbol alphabet resolves
    only `p` states at one level; the tree resolves `p` *per level*, nesting them.
    """
    if p < 2 or n_levels < 1:
        raise ValueError(f"need p >= 2 and n_levels >= 1, got p={p}, L={n_levels}")
    return CapacityCertificate(
        branching=p, n_levels=n_levels, leaf_capacity=p ** n_levels,
        nodes_per_level=tuple(p ** (level + 1) for level in range(n_levels)),
        flat_capacity_ref=p,
    )


# =========================================================================== #
# §4. Hierarchical retrieval + coarse-to-fine descent
# =========================================================================== #
def retrieve(query: int, memories: list[int], p: int, n_levels: int, *,
             alpha: float = 0.5, beta: float = 8.0) -> tuple[int, np.ndarray]:
    """Modern-Hopfield retrieval over the LCP kernel: returns `(argmax_memory, softmax_weights)`.

    `weights ∝ exp(β · sim_LCP(query, memory))` — a one-step attention readout whose energy landscape
    nests ultrametrically. `β` is the inverse temperature (retrieval sharpness).
    """
    sims = np.array([lcp_kernel(query, m, p, n_levels, alpha) for m in memories], dtype=np.float64)
    logits = beta * sims
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    return memories[int(np.argmax(w))], w


def shares_prefix(x: int, y: int, p: int, n_levels: int, level: int) -> bool:
    """Do `x`, `y` agree on their first `level` (coarsest) digits — i.e. share a category at that depth?"""
    return padic_digits(x, p, n_levels)[:level] == padic_digits(y, p, n_levels)[:level]


def corrupt_instance(x: int, p: int, n_levels: int, n_fine: int, rng: np.random.Generator) -> int:
    """Corrupt the `n_fine` finest (low-order) digits of `x`, leaving its coarse category intact."""
    digits = padic_digits(x, p, n_levels)
    for i in range(n_levels - n_fine, n_levels):
        digits[i] = int(rng.integers(0, p))
    return sum(dig * p ** (n_levels - 1 - i) for i, dig in enumerate(digits))


def flat_distance(x: int, y: int, p: int, n_levels: int) -> float:
    """A FLAT (non-hierarchical) baseline distance: Hamming over the digits (all levels equal weight).

    This is the contrast in the falsification — it has no notion of category, so instance corruption
    moves the query as far as a category change and confuses retrieval.
    """
    dx, dy = padic_digits(x, p, n_levels), padic_digits(y, p, n_levels)
    return float(sum(1 for a, b in zip(dx, dy) if a != b))


def retrieve_flat(query: int, memories: list[int], p: int, n_levels: int, *, beta: float = 8.0) -> int:
    """Flat modern-Hopfield baseline: retrieve by the Hamming kernel (no hierarchy)."""
    sims = np.array([-flat_distance(query, m, p, n_levels) for m in memories], dtype=np.float64)
    return memories[int(np.argmax(sims))]


def sparse_prototype_bank(p: int, n_levels: int, n_per_category: int, rng: np.random.Generator) -> list[int]:
    """A sparse memory bank: `n_per_category` random leaves in each top-level category.

    Sparsity is what makes retrieval non-trivial — a corrupted query is *not* itself stored, so the
    net must generalize to the nearest prototype, where the coarse-weighting of the ultrametric kernel
    pays off (a dense all-leaves bank trivially returns the query itself).
    """
    bank: set[int] = set()
    for c in range(p):
        for _ in range(n_per_category):
            digits = [c] + [int(rng.integers(0, p)) for _ in range(n_levels - 1)]
            bank.add(sum(d * p ** (n_levels - 1 - i) for i, d in enumerate(digits)))
    return sorted(bank)


@dataclass(frozen=True)
class LeapfrogResult:
    """The recall@category leapfrog: ultrametric vs flat retrieval under instance corruption."""

    ultrametric_recall: float
    flat_recall: float
    delta: float              # ultrametric − flat (> 0 ⟹ the hierarchical kernel wins)
    trials: int


def leapfrog_recall(p: int, n_levels: int, *, n_per_category: int = 3, n_fine: int = 3,
                    level: int = 1, trials: int = 400, seed: int = 0,
                    alpha: float = 0.5, beta: float = 8.0) -> LeapfrogResult:
    """Recall@category under instance corruption — the falsifiable leapfrog (note §4).

    Build a sparse prototype bank; repeatedly take a clean leaf, corrupt its `n_fine` finest digits,
    retrieve with the ultrametric LCP kernel vs the flat (Hamming) modern-Hopfield baseline, and score
    whether the retrieved prototype kept the clean leaf's category at depth `level`. The ultrametric
    arm weights the coarse prefix exponentially, so it recovers the category even when the instance is
    destroyed; the flat arm is pulled to wrong-category prototypes that coincide on the corrupted digits.
    """
    rng = np.random.default_rng(seed)
    bank = sparse_prototype_bank(p, n_levels, n_per_category, rng)
    cats = [padic_digits(b, p, n_levels)[0] for b in bank]
    if len(set(cats)) < p:
        raise ValueError("bank does not cover every category; increase n_per_category")
    um_ok = flat_ok = 0
    for _ in range(trials):
        c = int(rng.integers(0, p))
        clean = c * p ** (n_levels - 1) + int(rng.integers(0, p ** (n_levels - 1)))
        q = corrupt_instance(clean, p, n_levels, n_fine, rng)
        um, _ = retrieve(q, bank, p, n_levels, alpha=alpha, beta=beta)
        fl = retrieve_flat(q, bank, p, n_levels, beta=beta)
        um_ok += shares_prefix(um, clean, p, n_levels, level)
        flat_ok += shares_prefix(fl, clean, p, n_levels, level)
    um_r, flat_r = um_ok / trials, flat_ok / trials
    return LeapfrogResult(ultrametric_recall=um_r, flat_recall=flat_r, delta=um_r - flat_r, trials=trials)
