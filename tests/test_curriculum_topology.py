"""Unit and comparison tests for Topology-Driven Self-Curriculum (bead r00r.11)."""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat.curriculum_topology import TopologicalCurriculumSampler
from bio_inspired_nanochat.structural_geometry import coverage_signal


def test_topological_curriculum_scores_bridge_higher_than_remote_outlier():
    """A point splitting an observed gap outranks a farther point that enlarges it."""
    sampler = TopologicalCurriculumSampler(temperature=1.0)

    covered = np.array([[0.0, 0.0], [10.0, 0.0]])
    candidates = np.array([[5.0, 0.0], [100.0, 0.0]])

    probs = sampler.compute_sampling_probabilities(covered, candidates)

    assert probs[0] > 0.99
    assert probs[1] < 0.01


def test_topological_curriculum_never_selects_only_gap_worsening_outliers():
    sampler = TopologicalCurriculumSampler()
    covered = np.array([[0.0, 0.0], [10.0, 0.0]])
    candidates = np.array([[100.0, 0.0], [200.0, 0.0]])

    selected_idx, report = sampler.sample_batch(
        covered,
        candidates,
        batch_size=2,
        rng=np.random.default_rng(0),
    )

    assert selected_idx.size == 0
    assert report.selected_indices == []
    assert report.max_gap_after == report.max_gap_before == 10.0
    assert report.gap_reduction_ratio == 0.0


def test_topological_curriculum_probabilities_are_finite_at_tiny_temperature():
    sampler = TopologicalCurriculumSampler(temperature=1e-300)
    covered = np.array([[0.0, 0.0], [10.0, 0.0]])
    candidates = np.array([[5.0, 0.0], [4.0, 0.0], [100.0, 0.0]])

    probs = sampler.compute_sampling_probabilities(covered, candidates)

    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(1.0)


def test_topological_curriculum_accepts_an_empty_candidate_pool():
    sampler = TopologicalCurriculumSampler()
    covered = np.array([[0.0, 0.0], [10.0, 0.0]])

    selected_idx, report = sampler.sample_batch(
        covered,
        np.empty((0, 2)),
        batch_size=4,
        rng=np.random.default_rng(0),
    )

    assert selected_idx.size == 0
    assert report.max_gap_after == report.max_gap_before


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
    uniform_idx = np.random.default_rng(0).choice(len(candidates), size=10, replace=False)
    uniform_gap = coverage_signal(np.vstack([covered, candidates[uniform_idx]])).max_gap

    assert report.gap_reduction_ratio > 0.3, (
        f"Topological curriculum should reduce max gap significantly: {report.gap_reduction_ratio:.3f}"
    )
    assert report.max_gap_after < report.max_gap_before
    assert report.max_gap_after < uniform_gap


@pytest.mark.parametrize(
    ("covered", "candidates", "match"),
    [
        (np.zeros(2), np.zeros((1, 2)), "shape"),
        (np.zeros((1, 2)), np.zeros((1, 3)), "feature dimension"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2)), "finite"),
    ],
)
def test_topological_curriculum_rejects_invalid_point_clouds(covered, candidates, match):
    sampler = TopologicalCurriculumSampler()
    with pytest.raises(ValueError, match=match):
        sampler.compute_hole_filling_scores(covered, candidates)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"reduction_power": 0.0},
    ],
)
def test_topological_curriculum_rejects_invalid_sampler_configuration(kwargs):
    with pytest.raises(ValueError):
        TopologicalCurriculumSampler(**kwargs)


@pytest.mark.parametrize("batch_size", [-1, 1.5, True])
def test_topological_curriculum_rejects_invalid_batch_size(batch_size):
    sampler = TopologicalCurriculumSampler()
    with pytest.raises(ValueError, match="batch_size"):
        sampler.sample_batch(
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            batch_size=batch_size,
        )
