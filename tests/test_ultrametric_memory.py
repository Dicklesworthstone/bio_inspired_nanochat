"""Numerical corroboration of the ultrametric/RSB memory theory note (Thrust B, bead `0642.4.1`).

Checks the falsifiable results of `docs/theory/ultrametric_memory.md` against the reference
implementation (`bio_inspired_nanochat/ultrametric_memory.py`):

  - the p-adic LCP tree distance is a genuine **ultrametric** (strong triangle inequality; tree-ness
    score 1.0) while the flat Hamming distance is not (§1–§2);
  - the LCP kernel is monotone in the shared prefix (deeper common ancestor ⟹ more similar);
  - the capacity certificate is **exponential in depth** (`p^L` leaves) (§3);
  - the leapfrog: under instance corruption a sparse ultrametric memory recovers the **category**
    robustly, beating the flat modern-Hopfield baseline (§4).

Run:  pytest tests/test_ultrametric_memory.py -v
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from bio_inspired_nanochat import ultrametric_memory as um

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# §1. p-adic coordinates, LCP, ultrametric distance
# --------------------------------------------------------------------------- #
def test_padic_digits_and_lcp():
    assert um.padic_digits(0, 3, 4) == [0, 0, 0, 0]
    assert um.padic_digits(3 ** 4 - 1, 3, 4) == [2, 2, 2, 2]
    assert um.padic_digits(1, 3, 4) == [0, 0, 0, 1]   # least-significant = finest
    assert um.lcp(0, 1, 3, 4) == 3                     # differ only in the last (finest) digit
    assert um.lcp(0, 3 ** 3, 3, 4) == 0               # differ in the first (coarsest) digit
    with pytest.raises(ValueError):
        um.padic_digits(3 ** 4, 3, 4)                  # out of range


def test_tree_distance_is_an_ultrametric():
    p, levels = 3, 4
    items = list(range(p ** levels))
    d = um.distance_matrix(items, p, levels)
    assert um.ultrametricity_score(d) == 1.0 and um.is_ultrametric(d)
    # explicit strong triangle inequality on every triple of a small sample.
    sample = list(range(0, p ** levels, 5))
    for x, y, z in itertools.combinations(sample, 3):
        dxz = um.tree_distance(x, z, p, levels)
        assert dxz <= max(um.tree_distance(x, y, p, levels), um.tree_distance(y, z, p, levels)) + 1e-12


def test_flat_distance_is_not_ultrametric():
    p, levels = 3, 4
    items = list(range(p ** levels))
    dflat = np.array([[um.flat_distance(a, b, p, levels) for b in items] for a in items], dtype=float)
    assert um.ultrametricity_score(dflat) < 0.9, "Hamming distance must not be ultrametric"


# --------------------------------------------------------------------------- #
# §1b. The LCP kernel
# --------------------------------------------------------------------------- #
def test_lcp_kernel_is_monotone_in_shared_prefix():
    p, levels = 3, 4
    identical = um.lcp_kernel(0, 0, p, levels)
    same_category = um.lcp_kernel(0, 1, p, levels)        # share prefix [0,0,0], differ in last
    different_category = um.lcp_kernel(0, p ** (levels - 1), p, levels)  # differ in the first digit
    assert identical == 1.0
    assert identical > same_category > different_category > 0.0
    with pytest.raises(ValueError):
        um.lcp_kernel(0, 1, p, levels, alpha=1.5)


# --------------------------------------------------------------------------- #
# §3. Capacity certificate
# --------------------------------------------------------------------------- #
def test_capacity_certificate_is_exponential_in_depth():
    cert = um.capacity_certificate(p=3, n_levels=4)
    assert cert.leaf_capacity == 3 ** 4 == 81
    assert cert.nodes_per_level == (3, 9, 27, 81)
    assert cert.flat_capacity_ref == 3
    # exponential growth: doubling depth squares the capacity.
    deep = um.capacity_certificate(p=3, n_levels=8)
    assert deep.leaf_capacity == cert.leaf_capacity ** 2
    with pytest.raises(ValueError):
        um.capacity_certificate(p=1, n_levels=3)


# --------------------------------------------------------------------------- #
# §4. Retrieval + the leapfrog
# --------------------------------------------------------------------------- #
def test_retrieve_recovers_an_exact_stored_pattern():
    p, levels = 3, 4
    bank = list(range(p ** levels))
    for q in (0, 5, 40, 80):
        got, w = um.retrieve(q, bank, p, levels)
        assert got == q and w.argmax() == bank.index(q)


def test_leapfrog_ultrametric_beats_flat_under_corruption():
    res = um.leapfrog_recall(p=4, n_levels=4, n_per_category=3, n_fine=3, level=1, trials=400, seed=0)
    assert res.ultrametric_recall > 0.95, "ultrametric must recover the category via the coarse prefix"
    assert res.delta > 0.2, f"ultrametric must clearly beat flat under corruption (Δ={res.delta:.3f})"
    assert res.flat_recall < res.ultrametric_recall


def test_ultrametric_recall_is_robust_across_corruption_depth():
    # The coarse prefix is never corrupted (corruption hits only fine digits) and the bank covers every
    # category, so the ultrametric arm recovers the category at recall 1.0 for ANY corruption depth,
    # while the flat arm degrades — and ultrametric never underperforms flat.
    prev_flat = 1.1
    for n_fine in (0, 1, 2, 3):
        res = um.leapfrog_recall(p=4, n_levels=4, n_per_category=3, n_fine=n_fine, level=1,
                                 trials=300, seed=1)
        assert res.ultrametric_recall >= 0.99, f"ultrametric category recall must stay ~1 (n_fine={n_fine})"
        assert res.delta >= -1e-9, "ultrametric must never underperform flat"
        prev_flat = res.flat_recall
    assert prev_flat < 0.99, "by full fine corruption the flat baseline must have clearly degraded"


def test_corruption_helpers_preserve_category():
    p, levels = 4, 4
    rng = np.random.default_rng(0)
    clean = 2 * p ** (levels - 1) + 7         # category digit 0 == 2
    q = um.corrupt_instance(clean, p, levels, n_fine=3, rng=rng)
    assert um.padic_digits(q, p, levels)[0] == 2, "corrupting fine digits must keep the coarse category"
    assert um.shares_prefix(q, clean, p, levels, level=1)


def test_corruption_guards_reject_destroying_the_category():
    # Corrupting all (or more than n_levels−level) digits would overwrite the category — must raise,
    # not silently produce a meaningless (negative) leapfrog.
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        um.corrupt_instance(0, 3, 3, n_fine=3, rng=rng)        # n_fine == n_levels
    with pytest.raises(ValueError):
        um.leapfrog_recall(p=3, n_levels=3, n_fine=3, level=1)  # n_fine > n_levels − level (= 2)
    with pytest.raises(ValueError):
        um.leapfrog_recall(p=4, n_levels=4, n_fine=4, level=1)  # would destroy the category
