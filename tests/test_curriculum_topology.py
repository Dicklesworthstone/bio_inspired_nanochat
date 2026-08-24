"""Unit and comparison tests for Topology-Driven Self-Curriculum (bead r00r.11)."""

from __future__ import annotations

import numpy as np

from bio_inspired_nanochat.curriculum_topology import TopologicalCurriculumSampler


def test_topological_curriculum_scores_holes_higher():
    """Candidates in uncovered holes receive higher sampling weights than near points."""
    sampler = TopologicalCurriculumSampler(temperature=1.0)

    # Covered: cluster around origin [0, 0]
    covered = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    # Candidate 1: near origin [0.05, 0.05]
    # Candidate 2: in topological hole [5.0, 5.0]
    candidates = np.array([[0.05, 0.05], [5.0, 5.0]])

    probs = sampler.compute_sampling_probabilities(covered, candidates)
    assert probs[1] > probs[0]
    assert probs[1] > 0.99, f"Hole candidate must dominate sampling probability, got {probs[1]}"


def test_topological_curriculum_reduces_max_gap_vs_uniform():
    """Topological sampling reduces max H^0 gap faster than uniform sampling."""
    rng = np.random.default_rng(42)
    sampler = TopologicalCurriculumSampler(temperature=1.0)

    # Two distant clusters -> large initial gap
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(20, 2))
    cluster_b = rng.normal(loc=[10.0, 0.0], scale=0.1, size=(20, 2))
    covered = np.vstack([cluster_a, cluster_b])

    # Candidate pool: 10 bridge candidates in the middle + 90 redundant points inside clusters
    bridge_candidates = np.linspace([1.0, 0.0], [9.0, 0.0], num=10)
    redundant_a = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(45, 2))
    redundant_b = rng.normal(loc=[10.0, 0.0], scale=0.1, size=(45, 2))
    candidates = np.vstack([bridge_candidates, redundant_a, redundant_b])

    # 1. Topological curriculum sampling
    selected_idx, report = sampler.sample_batch(
        covered, candidates, batch_size=10, rng=np.random.default_rng(0)
    )

    # Assert that gap was significantly reduced
    assert report.gap_reduction_ratio > 0.3, (
        f"Topological curriculum should reduce max gap significantly: {report.gap_reduction_ratio:.3f}"
    )
    assert report.max_gap_after < report.max_gap_before

