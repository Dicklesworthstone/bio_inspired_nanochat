"""Topology-Driven Self-Curriculum (bead r00r.11).

Uses persistent-homology coverage signals (H^0 MST gaps, Thrust C) to drive
curriculum sampling that prioritizes data points filling topological holes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from bio_inspired_nanochat.structural_geometry import coverage_signal


@dataclass
class CurriculumStepReport:
    step: int
    max_gap_before: float
    max_gap_after: float
    gap_reduction_ratio: float
    selected_indices: List[int]


class TopologicalCurriculumSampler:
    """Samples training candidates to fill persistent homology coverage holes."""

    def __init__(
        self,
        temperature: float = 1.0,
        distance_power: float = 2.0,
    ) -> None:
        self.temperature = max(1e-4, float(temperature))
        self.distance_power = float(distance_power)

    def compute_hole_filling_scores(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
    ) -> np.ndarray:
        """Compute coverage gap distances from each candidate to the covered manifold.

        Candidates far from all covered points lie in topological holes and receive high scores.
        """
        covered = np.asarray(covered_points, dtype=np.float64)
        candidates = np.asarray(candidate_points, dtype=np.float64)

        if covered.size == 0 or candidates.size == 0:
            return np.ones(len(candidates), dtype=np.float64)

        # Distance from each candidate to closest covered point
        # candidates: (N, D), covered: (M, D) -> min over M
        dists = np.min(
            np.linalg.norm(candidates[:, np.newaxis, :] - covered[np.newaxis, :, :], axis=-1),
            axis=-1,
        )
        return dists ** self.distance_power

    def compute_sampling_probabilities(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
    ) -> np.ndarray:
        """Compute softmax-weighted sampling probabilities over candidates."""
        scores = self.compute_hole_filling_scores(covered_points, candidate_points)
        scaled = scores / self.temperature
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def sample_batch(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, CurriculumStepReport]:
        """Sample a batch of points that maximally reduces topological holes."""
        if rng is None:
            rng = np.random.default_rng()

        covered = np.asarray(covered_points, dtype=np.float64)
        candidates = np.asarray(candidate_points, dtype=np.float64)

        sig_before = coverage_signal(covered)
        max_gap_before = sig_before.max_gap

        probs = self.compute_sampling_probabilities(covered, candidates)
        n_select = min(batch_size, len(candidates))
        selected_idx = rng.choice(len(candidates), size=n_select, replace=False, p=probs)

        # Evaluate augmented point set
        augmented = np.vstack([covered, candidates[selected_idx]])
        sig_after = coverage_signal(augmented)
        max_gap_after = sig_after.max_gap

        reduction = (
            (max_gap_before - max_gap_after) / max(1e-6, max_gap_before)
            if max_gap_before > 0
            else 0.0
        )

        report = CurriculumStepReport(
            step=0,
            max_gap_before=max_gap_before,
            max_gap_after=max_gap_after,
            gap_reduction_ratio=float(reduction),
            selected_indices=selected_idx.tolist(),
        )

        return selected_idx, report
