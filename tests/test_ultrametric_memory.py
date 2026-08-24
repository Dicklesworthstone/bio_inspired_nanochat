"""Numerical corroboration of the ultrametric/RSB memory theory note (Thrust B, bead `0642.4.1`).

Checks the falsifiable results of `docs/theory/ultrametric_memory.md` against the reference
implementation (`bio_inspired_nanochat/ultrametric_memory.py`):

  - the p-adic LCP tree distance is a genuine **ultrametric** (strong triangle inequality; tree-ness
    score 1.0) while the flat Hamming distance is not (§1–§2);
  - the LCP kernel is monotone in the shared prefix (deeper common ancestor ⟹ more similar);
  - the capacity certificate is **exponential in depth** (`p^L` leaves) (§3);
  - the batched Torch kernel resolves exact p-adic prefixes and degenerates exactly to flat retrieval
    at depth one;
  - normalized RRP depletion drives a monotone, resettable coarse-to-fine descent (§4);
  - low or invalid live tree-ness evidence selects the exact flat fallback and emits detailed JSONL;
  - the leapfrog: under instance corruption a sparse ultrametric memory recovers the **category**
    robustly, beating the flat modern-Hopfield baseline (§4).

Run:  pytest tests/test_ultrametric_memory.py -v
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from bio_inspired_nanochat import ultrametric_memory as um
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.torch_imports import torch

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
    assert um.ultrametricity_score(d) == pytest.approx(1.0) and um.is_ultrametric(d)
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
    assert identical == pytest.approx(1.0)
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
# §4. Runtime p-adic kernel + depletion-driven coarse-to-fine descent
# --------------------------------------------------------------------------- #
def test_padic_runtime_kernel_is_batched_normalized_and_level_selective():
    cfg = um.PadicRetrievalConfig(
        enabled=True,
        branching=2,
        n_levels=3,
        alpha=0.25,
        beta=20.0,
    )
    memories = torch.tensor([0, 3, 4, 7])  # 000, 011, 100, 111
    result = um.padic_retrieval_kernel(
        torch.tensor([3, 4]),
        memories,
        config=cfg,
        active_levels=torch.tensor([1, 3]),
    )

    assert result.mode == "padic"
    assert result.weights.shape == (2, 4)
    torch.testing.assert_close(
        result.weights.sum(dim=-1),
        torch.ones(2, dtype=result.weights.dtype),
    )
    assert result.retrieved_coordinates.tolist() == [0, 4]
    assert result.active_levels.tolist() == [1, 3]
    # Resolving all three digits removes the intentional within-category tie for query 011.
    fine = um.padic_retrieval_kernel(
        torch.tensor([3]),
        memories,
        config=cfg,
        active_levels=3,
    )
    assert fine.retrieved_coordinates.item() == 3


def test_batched_kernel_matches_scalar_lcp_reference_at_each_level():
    cfg = um.PadicRetrievalConfig(
        enabled=True,
        branching=3,
        n_levels=4,
        alpha=0.4,
        beta=6.0,
    )
    queries = [5, 40]
    memories = [0, 5, 26, 40, 80]
    for level in range(1, cfg.n_levels + 1):
        got = um.padic_retrieval_kernel(
            torch.tensor(queries),
            torch.tensor(memories),
            config=cfg,
            active_levels=level,
        )
        similarities = np.array(
            [
                [
                    cfg.alpha ** (level - min(um.lcp(query, memory, 3, 4), level))
                    for memory in memories
                ]
                for query in queries
            ],
            dtype=np.float64,
        )
        logits = cfg.beta * similarities
        expected = np.exp(logits - logits.max(axis=-1, keepdims=True))
        expected /= expected.sum(axis=-1, keepdims=True)
        torch.testing.assert_close(
            got.weights,
            torch.from_numpy(expected).to(dtype=got.weights.dtype),
            rtol=1e-6,
            atol=1e-7,
        )


def test_depth_one_padic_kernel_is_exact_flat_retrieval():
    enabled = um.PadicRetrievalConfig(enabled=True, branching=3, n_levels=1, alpha=0.4, beta=7.0)
    disabled = replace(enabled, enabled=False)
    queries = torch.tensor([0, 1, 2])
    memories = torch.tensor([2, 0, 1])
    padic = um.padic_retrieval_kernel(queries, memories, config=enabled)
    flat = um.padic_retrieval_kernel(queries, memories, config=disabled)

    assert padic.mode == flat.mode == "flat"
    torch.testing.assert_close(padic.weights, flat.weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        padic.retrieved_coordinates,
        flat.retrieved_coordinates,
        rtol=0.0,
        atol=0.0,
    )


def test_rrp_depletion_drives_monotone_coarse_to_fine_descent_and_reset():
    cfg = um.PadicRetrievalConfig(
        enabled=True,
        branching=2,
        n_levels=3,
        alpha=0.25,
        beta=20.0,
    )
    retriever = um.DepletionDrivenPadicRetriever(cfg)
    memories = torch.tensor([0, 3, 4, 7])

    coarse = retriever.retrieve(
        torch.tensor([3]), memories, rrp_fraction=1.0, tree_ness_score=1.0
    )
    middle = retriever.retrieve(
        torch.tensor([3]), memories, rrp_fraction=0.6, tree_ness_score=1.0
    )
    leaf = retriever.retrieve(
        torch.tensor([3]), memories, rrp_fraction=0.0, tree_ness_score=1.0
    )
    refilled = retriever.retrieve(
        torch.tensor([3]), memories, rrp_fraction=1.0, tree_ness_score=1.0
    )

    assert [step.active_levels.item() for step in (coarse, middle, leaf, refilled)] == [1, 2, 3, 3]
    assert [step.retrieved_coordinates.item() for step in (coarse, middle, leaf)] == [0, 3, 3]
    assert refilled.active_levels.item() == 3, "refill must not re-ascend within one sequence"
    assert leaf.rrp_fraction is not None and leaf.rrp_fraction.item() == pytest.approx(0.0)

    retriever.reset()
    assert retriever.active_levels is None
    restarted = retriever.retrieve(
        torch.tensor([3]), memories, rrp_fraction=1.0, tree_ness_score=1.0
    )
    assert restarted.active_levels.item() == 1


def test_default_off_retriever_is_flat_and_keeps_no_hierarchy_state():
    retriever = um.DepletionDrivenPadicRetriever(
        um.PadicRetrievalConfig(enabled=False, branching=2, n_levels=3)
    )
    result = retriever.retrieve(
        torch.tensor([3]),
        torch.tensor([0, 3, 4, 7]),
        rrp_fraction=1.0,
    )
    assert result.mode == "flat"
    assert result.retrieved_coordinates.item() == 3
    assert retriever.active_levels is None


def test_runtime_kernel_and_depletion_inputs_fail_closed():
    cfg = um.PadicRetrievalConfig(enabled=True, branching=2, n_levels=3)
    with pytest.raises(ValueError, match="coordinates must be in"):
        um.padic_retrieval_kernel(torch.tensor([8]), torch.tensor([0, 1]), config=cfg)
    with pytest.raises(ValueError, match="must contain integers"):
        um.padic_retrieval_kernel(torch.tensor([1.0]), torch.tensor([0, 1]), config=cfg)
    with pytest.raises(ValueError, match="active_levels"):
        um.padic_retrieval_kernel(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            config=cfg,
            active_levels=torch.tensor([1]),
        )
    for bad_rrp in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="rrp_fraction"):
            um.depletion_levels(bad_rrp, n_levels=3)
    with pytest.raises(ValueError, match="alpha"):
        um.PadicRetrievalConfig(enabled=True, alpha=1.0)
    with pytest.raises(ValueError, match="min_tree_ness"):
        um.PadicRetrievalConfig(enabled=True, min_tree_ness=-0.1)

    retriever = um.DepletionDrivenPadicRetriever(cfg)
    retriever.retrieve(
        torch.tensor([1]),
        torch.tensor([0, 1]),
        rrp_fraction=1.0,
        tree_ness_score=1.0,
    )
    with pytest.raises(ValueError, match="call reset"):
        retriever.retrieve(
            torch.tensor([1, 2]),
            torch.tensor([0, 1, 2]),
            rrp_fraction=1.0,
            tree_ness_score=1.0,
        )


def test_tree_ness_guard_falls_back_exactly_and_clears_descent_state():
    cfg = um.PadicRetrievalConfig(
        enabled=True,
        branching=2,
        n_levels=3,
        min_tree_ness=0.9,
    )
    retriever = um.DepletionDrivenPadicRetriever(cfg)
    queries = torch.tensor([3, 4])
    memories = torch.tensor([0, 3, 4, 7])
    hierarchical = retriever.retrieve(
        queries,
        memories,
        rrp_fraction=torch.tensor([0.5, 0.0]),
        tree_ness_score=0.95,
    )
    assert hierarchical.mode == "padic" and retriever.active_levels is not None

    torch.manual_seed(17)
    guarded = retriever.retrieve(
        queries,
        memories,
        rrp_fraction=torch.tensor([0.5, 0.0]),
        tree_ness_score=0.5,
    )
    torch.manual_seed(17)
    flat = um.padic_retrieval_kernel(queries, memories, config=replace(cfg, enabled=False))

    assert guarded.mode == "flat"
    assert guarded.fallback_used
    assert guarded.fallback_reason == "tree_ness_below_floor"
    assert guarded.tree_ness_passed is not None
    assert not guarded.tree_ness_passed
    assert retriever.active_levels is None
    torch.testing.assert_close(guarded.weights, flat.weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        guarded.retrieved_coordinates,
        flat.retrieved_coordinates,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("score", "reason"),
    [
        (None, "tree_ness_unavailable"),
        (float("nan"), "tree_ness_non_finite"),
        (-0.1, "tree_ness_out_of_range"),
        (1.1, "tree_ness_out_of_range"),
    ],
)
def test_tree_ness_guard_fails_closed_on_invalid_evidence(score, reason):
    retriever = um.DepletionDrivenPadicRetriever(
        um.PadicRetrievalConfig(enabled=True, branching=2, n_levels=3)
    )
    result = retriever.retrieve(
        torch.tensor([3]),
        torch.tensor([0, 3]),
        rrp_fraction=1.0,
        tree_ness_score=score,
    )
    assert result.mode == "flat"
    assert result.fallback_used and result.fallback_reason == reason


def test_tree_ness_floor_is_inclusive_and_malformed_scores_are_rejected():
    retriever = um.DepletionDrivenPadicRetriever(
        um.PadicRetrievalConfig(
            enabled=True,
            branching=2,
            n_levels=3,
            min_tree_ness=0.9,
        )
    )
    result = retriever.retrieve(
        torch.tensor([3]),
        torch.tensor([0, 3]),
        rrp_fraction=1.0,
        tree_ness_score=0.9,
    )
    assert result.mode == "padic" and result.tree_ness_passed

    for malformed in (True, [0.9, 0.8], 1 + 0j):
        with pytest.raises(ValueError, match="numeric scalar"):
            retriever.retrieve(
                torch.tensor([3]),
                torch.tensor([0, 3]),
                rrp_fraction=1.0,
                tree_ness_score=malformed,
            )


def test_default_off_ignores_tree_ness_and_is_exact_flat_identity():
    cfg = um.PadicRetrievalConfig(enabled=False, branching=2, n_levels=3)
    retriever = um.DepletionDrivenPadicRetriever(cfg)
    result = retriever.retrieve(
        torch.tensor([3]),
        torch.tensor([0, 3, 4, 7]),
        rrp_fraction=0.0,
        tree_ness_score=float("nan"),
    )
    direct = um.padic_retrieval_kernel(
        torch.tensor([3]),
        torch.tensor([0, 3, 4, 7]),
        config=cfg,
    )
    assert result.mode == "flat" and not result.fallback_used
    assert result.fallback_reason is None and result.tree_ness_passed is None
    torch.testing.assert_close(result.weights, direct.weights, rtol=0.0, atol=0.0)


def test_tree_ness_fallback_writes_detailed_jsonl(tmp_path):
    retriever = um.DepletionDrivenPadicRetriever(
        um.PadicRetrievalConfig(
            enabled=True,
            branching=2,
            n_levels=3,
            min_tree_ness=0.9,
        )
    )
    with RunLogger(
        tmp_path,
        name="ultrametric_guard",
        console=False,
        provenance={"seed": 17},
    ) as logger:
        passed = retriever.retrieve(
            torch.tensor([3]),
            torch.tensor([0, 3, 4, 7]),
            rrp_fraction=1.0,
            tree_ness_score=0.9,
            run_logger=logger,
            step=6,
        )
        result = retriever.retrieve(
            torch.tensor([3]),
            torch.tensor([0, 3, 4, 7]),
            rrp_fraction=0.4,
            tree_ness_score=0.5,
            run_logger=logger,
            step=7,
        )

    events = logger.read_events()
    records = [event for event in events if event["event"] == "ultrametric_retrieval"]
    run_start = next(event for event in events if event["event"] == "run_start")
    assert run_start["provenance"]["seed"] == 17
    assert len(records) == 2
    passed_record, record = records
    assert passed_record["step"] == 6
    assert passed_record["mode"] == passed.mode == "padic"
    assert passed_record["tree_ness_passed"] and not passed_record["fallback_used"]
    assert record["step"] == 7
    assert record["enabled"] and record["mode"] == result.mode == "flat"
    assert record["tree_ness_score"] == pytest.approx(0.5)
    assert record["tree_ness_floor"] == pytest.approx(0.9)
    assert not record["tree_ness_passed"]
    assert record["fallback_used"]
    assert record["fallback_reason"] == "tree_ness_below_floor"
    assert record["active_levels"] == [3]
    assert record["rrp_fraction"] == pytest.approx([0.4])
    assert record["retrieved_coordinates"] == [3]
    assert len(record["max_weight"]) == len(record["weight_entropy"]) == 1


# --------------------------------------------------------------------------- #
# §5. Scalar retrieval + the leapfrog
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
    # category, so the ultrametric arm recovers the category at recall ≈1.0 for ANY corruption depth and
    # clearly beats the flat baseline at EVERY depth (the leapfrog). Note: flat does NOT need to
    # "degrade with depth" — it is already confused at n_fine=0 because the query is an unstored random
    # leaf — so we assert the per-depth leapfrog, not a depth trend.
    flats = []
    for n_fine in (0, 1, 2, 3):
        res = um.leapfrog_recall(p=4, n_levels=4, n_per_category=3, n_fine=n_fine, level=1,
                                 trials=300, seed=1)
        assert res.ultrametric_recall >= 0.99, f"ultrametric category recall must stay ~1 (n_fine={n_fine})"
        assert res.delta > 0.2, f"ultrametric must clearly beat flat at n_fine={n_fine} (Δ={res.delta:.3f})"
        flats.append(res.flat_recall)
    assert max(flats) < 0.9, f"the flat baseline must be clearly worse at every depth, got {flats}"


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
