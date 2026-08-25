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
        reduction_power: float = 2.0,
    ) -> None:
        self.temperature = float(temperature)
        self.reduction_power = float(reduction_power)
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
        if not np.isfinite(self.reduction_power) or self.reduction_power <= 0.0:
            raise ValueError(f"reduction_power must be finite and > 0, got {reduction_power!r}")

    @staticmethod
    def _validate_point_clouds(
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        covered = np.asarray(covered_points, dtype=np.float64)
        candidates = np.asarray(candidate_points, dtype=np.float64)
        if covered.ndim != 2 or candidates.ndim != 2:
            raise ValueError("covered_points and candidate_points must both have shape (N, D)")
        if covered.shape[1] != candidates.shape[1]:
            raise ValueError(
                "covered_points and candidate_points must have the same feature dimension; "
                f"got {covered.shape[1]} and {candidates.shape[1]}"
            )
        if covered.shape[1] == 0:
            raise ValueError("point clouds must have at least one feature dimension")
        if not np.isfinite(covered).all() or not np.isfinite(candidates).all():
            raise ValueError("covered_points and candidate_points must contain only finite values")
        return covered, candidates

    def _gap_reductions(
        self,
        covered: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return powered marginal gap reductions and each candidate's resulting gap."""
        if len(candidates) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        if len(covered) < 2:
            # H0 coverage has no edge before two points exist. Treat candidates
            # uniformly while bootstrapping rather than inventing a reduction.
            return (
                np.ones(len(candidates), dtype=np.float64),
                np.zeros(len(candidates), dtype=np.float64),
            )

        max_gap_before = coverage_signal(covered).max_gap
        max_gaps_after = np.fromiter(
            (
                coverage_signal(np.vstack([covered, candidate[np.newaxis, :]])).max_gap
                for candidate in candidates
            ),
            dtype=np.float64,
            count=len(candidates),
        )
        if not np.isfinite(max_gap_before) or not np.isfinite(max_gaps_after).all():
            raise ValueError("point-cloud distances must remain finite")
        improvements = np.maximum(0.0, max_gap_before - max_gaps_after)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = improvements**self.reduction_power
        if not np.isfinite(scores).all():
            raise ValueError("powered gap-reduction scores must remain finite")
        return scores, max_gaps_after

    def _probabilities_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize positive improvements; zero-improvement candidates wait their turn."""
        if len(scores) == 0:
            return scores
        positive = scores > 0.0
        if not np.any(positive):
            return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)

        probs = np.zeros(len(scores), dtype=np.float64)
        positive_scores = scores[positive]
        # Subtract before dividing so an extremely small, valid temperature
        # cannot overflow a positive score to infinity.
        with np.errstate(over="ignore", under="ignore"):
            scaled = (positive_scores - np.max(positive_scores)) / self.temperature
        exp_scores = np.exp(scaled)
        probs[positive] = exp_scores / np.sum(exp_scores)
        return probs

    def compute_hole_filling_scores(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
    ) -> np.ndarray:
        """Compute each candidate's marginal reduction in the largest H0 gap.

        Distance from the covered cloud is not sufficient: a remote outlier is
        far away but creates a larger MST edge instead of filling an existing
        gap. Scores therefore measure the actual reduction after inserting one
        candidate, clipped at zero and shaped by ``reduction_power``.
        """
        covered, candidates = self._validate_point_clouds(covered_points, candidate_points)
        scores, _ = self._gap_reductions(covered, candidates)
        return scores

    def compute_sampling_probabilities(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
    ) -> np.ndarray:
        """Compute softmax-weighted sampling probabilities over candidates."""
        scores = self.compute_hole_filling_scores(covered_points, candidate_points)
        return self._probabilities_from_scores(scores)

    def sample_batch(
        self,
        covered_points: np.ndarray,
        candidate_points: np.ndarray,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, CurriculumStepReport]:
        """Sample up to ``batch_size`` points without increasing the largest H0 gap."""
        if rng is None:
            rng = np.random.default_rng()
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}")
        if batch_size < 0:
            raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}")

        covered, candidates = self._validate_point_clouds(covered_points, candidate_points)

        sig_before = coverage_signal(covered)
        max_gap_before = sig_before.max_gap

        n_select = min(batch_size, len(candidates))
        remaining = np.arange(len(candidates), dtype=np.int64)
        selected: list[int] = []
        augmented = covered.copy()
        for _ in range(n_select):
            scores, max_gaps_after = self._gap_reductions(augmented, candidates[remaining])
            current_gap = coverage_signal(augmented).max_gap
            eligible = np.ones(len(remaining), dtype=bool)
            if len(augmented) >= 2:
                eligible = max_gaps_after <= current_gap
            if not np.any(eligible):
                break

            eligible_positions = np.flatnonzero(eligible)
            eligible_scores = scores[eligible]
            probabilities = self._probabilities_from_scores(eligible_scores)
            chosen_position = int(rng.choice(eligible_positions, p=probabilities))
            chosen_index = int(remaining[chosen_position])
            selected.append(chosen_index)
            augmented = np.vstack([augmented, candidates[chosen_index]])
            remaining = np.delete(remaining, chosen_position)

        sig_after = coverage_signal(augmented)
        max_gap_after = sig_after.max_gap

        reduction = (
            max(0.0, max_gap_before - max_gap_after) / max_gap_before
            if max_gap_before > 0
            else 0.0
        )

        report = CurriculumStepReport(
            step=0,
            max_gap_before=max_gap_before,
            max_gap_after=max_gap_after,
            gap_reduction_ratio=float(reduction),
            selected_indices=selected,
        )

        return np.asarray(selected, dtype=np.int64), report
