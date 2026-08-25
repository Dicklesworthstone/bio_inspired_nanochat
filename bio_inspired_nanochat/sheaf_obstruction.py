"""Calibrated MVP sheaf obstruction for binding-consistency monitoring.

This is the deliberately conservative first stage specified by ``r00r.5``.  It
computes the normalized quadratic energy of a fixed cellular-sheaf coboundary,
then calibrates that scalar against labeled consistent/inconsistent examples.
It is useful before the exploratory H¹ runtime lands, but it is **not** an H¹
certificate; every result carries that provenance explicitly.

For node stalks ``x_v`` and edge restriction maps ``R_{e,v}``, the residual is

    delta_e(x) = R_{e,tail} x_tail - R_{e,head} x_head

and the obstruction energy is ``||delta(x)||²``.  Zero means the supplied local
bindings agree on every edge.  A bounded, scale-invariant score is obtained by
normalizing against the restricted stalk energy and mapping ``q -> q/(1+q)``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

from bio_inspired_nanochat.torch_imports import Tensor, torch

MVP_CERTIFICATE_KIND = "fixed_sheaf_laplacian_residual_mvp"


class ObstructionAction(StrEnum):
    """Runtime action requested when a calibrated obstruction is detected."""

    FLAG_ONLY = "flag_only"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    DELIBERATE = "deliberate"
    REPAIR = "repair"


@dataclass(frozen=True)
class SheafObstructionResult:
    """One auditable obstruction measurement."""

    available: bool
    score: float
    normalized_residual: float
    quadratic_energy: float
    reference_energy: float
    edge_residual_norms: tuple[float, ...]
    certificate_kind: str = MVP_CERTIFICATE_KIND
    h1_certified: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationStep:
    """One constant segment of the monotone probability calibrator."""

    lower_score: float
    upper_score: float
    probability: float
    count: int
    positives: int


@dataclass(frozen=True)
class ReliabilityBin:
    """One populated point in a reliability diagram."""

    lower_probability: float
    upper_probability: float
    count: int
    mean_probability: float
    observed_frequency: float
    absolute_gap: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Held-out calibration and operating-point metrics."""

    sample_count: int
    positive_count: int
    negative_count: int
    false_positive_rate: float
    true_positive_rate: float
    brier_score: float
    expected_calibration_error: float
    reliability_bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObstructionCalibrator:
    """Monotone probability mapping plus a false-positive-controlled threshold.

    The operating threshold is fit from *negative examples only* using a split-
    conformal upper quantile.  Under exchangeability, this controls marginal
    false-positive probability at ``target_false_positive_rate`` without using
    positive labels to cherry-pick the operating point.  If the negative sample
    is too small to support the requested rate, ``threshold`` is infinite and
    the detector fails closed by flagging nothing.
    """

    steps: tuple[CalibrationStep, ...]
    threshold: float
    target_false_positive_rate: float
    calibration_false_positive_rate: float
    calibration_true_positive_rate: float
    positive_count: int
    negative_count: int
    threshold_protocol: str = "negative_only_split_conformal_quantile"
    certificate_kind: str = MVP_CERTIFICATE_KIND
    h1_certified: bool = False

    def predict_probability(self, score: float) -> float:
        score = _validate_score(score)
        upper_bounds = [step.upper_score for step in self.steps]
        index = min(bisect.bisect_left(upper_bounds, score), len(self.steps) - 1)
        return self.steps[index].probability

    def is_flagged(self, score: float) -> bool:
        return _validate_score(score) >= self.threshold

    def evaluate(
        self,
        scores: Sequence[float],
        labels: Sequence[bool | int],
        *,
        reliability_bins: int = 10,
    ) -> CalibrationEvaluation:
        pairs = _validated_pairs(scores, labels)
        if reliability_bins < 1:
            raise ValueError(
                f"reliability_bins must be >= 1, got {reliability_bins}"
            )

        probabilities = [self.predict_probability(score) for score, _ in pairs]
        bool_labels = [label for _, label in pairs]
        positives = sum(bool_labels)
        negatives = len(bool_labels) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("evaluation requires at least one positive and one negative")

        flags = [self.is_flagged(score) for score, _ in pairs]
        false_positives = sum(flag and not label for flag, label in zip(flags, bool_labels))
        true_positives = sum(flag and label for flag, label in zip(flags, bool_labels))
        brier = sum(
            (probability - float(label)) ** 2
            for probability, label in zip(probabilities, bool_labels)
        ) / len(pairs)
        diagram, ece = _reliability_bins(
            probabilities,
            bool_labels,
            num_bins=reliability_bins,
        )
        return CalibrationEvaluation(
            sample_count=len(pairs),
            positive_count=positives,
            negative_count=negatives,
            false_positive_rate=false_positives / negatives,
            true_positive_rate=true_positives / positives,
            brier_score=brier,
            expected_calibration_error=ece,
            reliability_bins=diagram,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
            "flags_enabled": math.isfinite(self.threshold),
            "target_false_positive_rate": self.target_false_positive_rate,
            "calibration_false_positive_rate": self.calibration_false_positive_rate,
            "calibration_true_positive_rate": self.calibration_true_positive_rate,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "threshold_protocol": self.threshold_protocol,
            "certificate_kind": self.certificate_kind,
            "h1_certified": self.h1_certified,
        }


@dataclass(frozen=True)
class SheafDetectorConfig:
    """Conservative runtime policy for the obstruction detector.

    The detector is default-off.  ``threshold`` is an explicit deployment
    threshold for callers that persist calibration separately; supplying an
    :class:`ObstructionCalibrator` to the detector takes precedence.  With no
    threshold source, the enabled detector fails closed to an exact no-op.
    """

    enabled: bool = False
    action: ObstructionAction = ObstructionAction.FLAG_ONLY
    threshold: float | None = None
    repair_steps: int = 5
    repair_step_size: float = 0.25
    repair_backtracks: int = 8
    repair_min_improvement: float = 1e-9

    def validate(self) -> None:
        try:
            ObstructionAction(self.action)
        except ValueError as error:
            raise ValueError(f"unknown obstruction action: {self.action!r}") from error
        if self.threshold is not None and (
            not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0
        ):
            raise ValueError("threshold must be finite and in [0,1] when supplied")
        if self.repair_steps < 1:
            raise ValueError("repair_steps must be >= 1")
        if not math.isfinite(self.repair_step_size) or self.repair_step_size <= 0.0:
            raise ValueError("repair_step_size must be finite and > 0")
        if self.repair_backtracks < 0:
            raise ValueError("repair_backtracks must be >= 0")
        if (
            not math.isfinite(self.repair_min_improvement)
            or self.repair_min_improvement < 0.0
        ):
            raise ValueError("repair_min_improvement must be finite and non-negative")


@dataclass(frozen=True)
class SheafDetectorDecision:
    """Auditable runtime decision with behavior-neutral output when inactive."""

    enabled: bool
    available: bool
    flagged: bool
    score: float
    score_after: float
    threshold: float | None
    calibrated_probability: float | None
    requested_action: ObstructionAction
    action_taken: str
    output_stalks: Tensor
    edge_residual_norms: tuple[float, ...] = ()
    should_abstain: bool = False
    should_clarify: bool = False
    should_deliberate: bool = False
    repaired: bool = False
    fallback_reason: str | None = None
    certificate_kind: str = MVP_CERTIFICATE_KIND
    h1_certified: bool = False

    def to_event_dict(self) -> dict[str, object]:
        """Return JSON-safe decision metadata, intentionally excluding the tensor."""
        return {
            "enabled": self.enabled,
            "available": self.available,
            "flagged": self.flagged,
            "score": self.score,
            "score_after": self.score_after,
            "threshold": self.threshold,
            "calibrated_probability": self.calibrated_probability,
            "requested_action": self.requested_action.value,
            "action_taken": self.action_taken,
            "edge_residual_norms": list(self.edge_residual_norms),
            "should_abstain": self.should_abstain,
            "should_clarify": self.should_clarify,
            "should_deliberate": self.should_deliberate,
            "repaired": self.repaired,
            "fallback_reason": self.fallback_reason,
            "certificate_kind": self.certificate_kind,
            "h1_certified": self.h1_certified,
        }


class SheafObstructionDetector:
    """Apply a calibrated obstruction policy without changing baseline behavior.

    Disabled, unavailable, uncalibrated, and below-threshold paths return the
    original ``stalks`` tensor object.  Repair uses deterministic gradient
    descent with backtracking and is accepted only when it lowers the same
    normalized obstruction score used for detection.  A failed repair degrades
    to flag-only instead of mutating activations.
    """

    def __init__(
        self,
        config: SheafDetectorConfig | None = None,
        *,
        calibrator: ObstructionCalibrator | None = None,
    ) -> None:
        if config is None:
            config = SheafDetectorConfig()
        config.validate()
        self.config = config
        self.calibrator = calibrator

    def inspect(
        self,
        stalks: Tensor,
        edge_index: Tensor,
        *,
        tail_restrictions: Tensor | None = None,
        head_restrictions: Tensor | None = None,
        edge_weight: Tensor | None = None,
        eps: float = 1e-8,
    ) -> SheafDetectorDecision:
        """Measure the graph and apply the configured action when flagged."""
        action = ObstructionAction(self.config.action)
        if not self.config.enabled:
            return self._noop(
                stalks,
                action=action,
                available=False,
                fallback_reason="disabled",
            )

        result = measure_sheaf_obstruction(
            stalks,
            edge_index,
            tail_restrictions=tail_restrictions,
            head_restrictions=head_restrictions,
            edge_weight=edge_weight,
            eps=eps,
        )
        threshold = self._threshold()
        probability = (
            self.calibrator.predict_probability(result.score)
            if self.calibrator is not None and result.available
            else None
        )
        if not result.available:
            return self._noop(
                stalks,
                action=action,
                available=False,
                fallback_reason=result.fallback_reason,
                threshold=threshold,
            )
        if threshold is None:
            return self._noop(
                stalks,
                action=action,
                available=True,
                score=result.score,
                fallback_reason="calibration_unavailable",
                calibrated_probability=probability,
                edge_residual_norms=result.edge_residual_norms,
            )
        if not math.isfinite(threshold):
            return self._noop(
                stalks,
                action=action,
                available=True,
                score=result.score,
                fallback_reason="calibration_threshold_disabled",
                calibrated_probability=probability,
                edge_residual_norms=result.edge_residual_norms,
            )

        flagged = result.score >= threshold
        if not flagged:
            return SheafDetectorDecision(
                enabled=True,
                available=True,
                flagged=False,
                score=result.score,
                score_after=result.score,
                threshold=threshold,
                calibrated_probability=probability,
                requested_action=action,
                action_taken="below_threshold_noop",
                output_stalks=stalks,
                edge_residual_norms=result.edge_residual_norms,
            )
        if action is not ObstructionAction.REPAIR:
            return SheafDetectorDecision(
                enabled=True,
                available=True,
                flagged=True,
                score=result.score,
                score_after=result.score,
                threshold=threshold,
                calibrated_probability=probability,
                requested_action=action,
                action_taken=action.value,
                output_stalks=stalks,
                edge_residual_norms=result.edge_residual_norms,
                should_abstain=action is ObstructionAction.ABSTAIN,
                should_clarify=action is ObstructionAction.CLARIFY,
                should_deliberate=action is ObstructionAction.DELIBERATE,
            )

        repaired_stalks, repaired_score, repair_reason = _repair_obstruction(
            stalks,
            edge_index,
            tail_restrictions=tail_restrictions,
            head_restrictions=head_restrictions,
            edge_weight=edge_weight,
            eps=eps,
            steps=self.config.repair_steps,
            step_size=self.config.repair_step_size,
            backtracks=self.config.repair_backtracks,
            min_improvement=self.config.repair_min_improvement,
        )
        if repair_reason is not None:
            return SheafDetectorDecision(
                enabled=True,
                available=True,
                flagged=True,
                score=result.score,
                score_after=result.score,
                threshold=threshold,
                calibrated_probability=probability,
                requested_action=action,
                action_taken="repair_failed_flag_only",
                output_stalks=stalks,
                edge_residual_norms=result.edge_residual_norms,
                fallback_reason=repair_reason,
            )
        return SheafDetectorDecision(
            enabled=True,
            available=True,
            flagged=True,
            score=result.score,
            score_after=repaired_score,
            threshold=threshold,
            calibrated_probability=probability,
            requested_action=action,
            action_taken="repair",
            output_stalks=repaired_stalks,
            # Re-measure on the REPAIRED stalks so the reported per-edge norms
            # describe output_stalks; pairing pre-repair norms with post-repair
            edge_residual_norms=measure_sheaf_obstruction(
                repaired_stalks,
                edge_index,
                tail_restrictions=tail_restrictions,
                head_restrictions=head_restrictions,
                edge_weight=edge_weight,
                eps=eps,
            ).edge_residual_norms,
            repaired=True,
        )

    def _threshold(self) -> float | None:
        if self.calibrator is not None:
            return self.calibrator.threshold
        return self.config.threshold

    def _noop(
        self,
        stalks: Tensor,
        *,
        action: ObstructionAction,
        available: bool,
        fallback_reason: str | None,
        score: float = 0.0,
        threshold: float | None = None,
        calibrated_probability: float | None = None,
        edge_residual_norms: tuple[float, ...] = (),
    ) -> SheafDetectorDecision:
        return SheafDetectorDecision(
            enabled=self.config.enabled,
            available=available,
            flagged=False,
            score=score,
            score_after=score,
            threshold=threshold,
            calibrated_probability=calibrated_probability,
            requested_action=action,
            action_taken="noop",
            output_stalks=stalks,
            edge_residual_norms=edge_residual_norms,
            fallback_reason=fallback_reason,
        )


def measure_sheaf_obstruction(
    stalks: Tensor,
    edge_index: Tensor,
    *,
    tail_restrictions: Tensor | None = None,
    head_restrictions: Tensor | None = None,
    edge_weight: Tensor | None = None,
    eps: float = 1e-8,
) -> SheafObstructionResult:
    """Measure a fixed-sheaf Laplacian residual on one token/binding graph.

    ``stalks`` has shape ``(nodes, stalk_dim)`` and ``edge_index`` has shape
    ``(2, edges)``.  Optional tail/head restriction maps both have shape
    ``(edges, restriction_dim, stalk_dim)``.  Omitting both selects identity
    restrictions.  Graphs without positive-weight edges return an explicit,
    behavior-neutral unavailable result rather than a misleading certificate.
    """
    if stalks.ndim != 2 or stalks.shape[0] < 1 or stalks.shape[1] < 1:
        raise ValueError(
            f"stalks must have shape (nodes, stalk_dim), got {tuple(stalks.shape)}"
        )
    if not stalks.is_floating_point():
        raise ValueError(f"stalks must use a floating dtype, got {stalks.dtype}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, edges), got {tuple(edge_index.shape)}")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"edge_index must use an integer dtype, got {edge_index.dtype}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if not bool(torch.isfinite(stalks).all()):
        raise ValueError("stalks must be finite")

    edges = int(edge_index.shape[1])
    if edges == 0:
        return _unavailable("no_edges")
    if bool((edge_index < 0).any()) or bool((edge_index >= stalks.shape[0]).any()):
        raise ValueError("edge_index contains a node outside the stalk array")

    if (tail_restrictions is None) != (head_restrictions is None):
        raise ValueError("tail_restrictions and head_restrictions must be supplied together")

    tail_values = stalks.index_select(0, edge_index[0].to(torch.int64))
    head_values = stalks.index_select(0, edge_index[1].to(torch.int64))
    if tail_restrictions is None:
        restricted_tail = tail_values
        restricted_head = head_values
    else:
        if head_restrictions is None:
            raise RuntimeError("restriction-map validation reached an impossible state")
        expected_prefix = (edges,)
        if (
            tail_restrictions.ndim != 3
            or head_restrictions.ndim != 3
            or tail_restrictions.shape[0:1] != expected_prefix
            or head_restrictions.shape[0:1] != expected_prefix
            or tail_restrictions.shape[2] != stalks.shape[1]
            or head_restrictions.shape[2] != stalks.shape[1]
            or tail_restrictions.shape[1] != head_restrictions.shape[1]
        ):
            raise ValueError(
                "restriction maps must both have shape "
                f"(edges, restriction_dim, {stalks.shape[1]})"
            )
        if not bool(torch.isfinite(tail_restrictions).all()) or not bool(
            torch.isfinite(head_restrictions).all()
        ):
            raise ValueError("restriction maps must be finite")
        tail_maps = tail_restrictions.to(device=stalks.device, dtype=stalks.dtype)
        head_maps = head_restrictions.to(device=stalks.device, dtype=stalks.dtype)
        restricted_tail = torch.einsum("erd,ed->er", tail_maps, tail_values)
        restricted_head = torch.einsum("erd,ed->er", head_maps, head_values)

    if edge_weight is None:
        weights = torch.ones(edges, device=stalks.device, dtype=stalks.dtype)
    else:
        if edge_weight.ndim != 1 or edge_weight.numel() != edges:
            raise ValueError(f"edge_weight must have shape ({edges},)")
        weights = edge_weight.to(device=stalks.device, dtype=stalks.dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
            raise ValueError("edge_weight must be finite and non-negative")
    if float(weights.sum().item()) <= 0.0:
        return _unavailable("no_positive_weight_edges")

    residual = restricted_tail - restricted_head
    residual_sq = residual.square().sum(dim=-1)
    reference_sq = restricted_tail.square().sum(dim=-1) + restricted_head.square().sum(dim=-1)
    quadratic = (weights * residual_sq).sum()
    reference = (weights * reference_sq).sum()
    # A degenerate restriction set (zero maps, or stalks inside every
    # restriction's null space) forces reference == 0 and therefore
    # normalized == bounded_score == 0: a "perfectly gluable" certificate for
    # arbitrarily inconsistent bindings. Report unavailable instead of a vacuous
    # perfect-consistency pass.
    if float(reference.item()) <= eps:
        return _unavailable("degenerate_reference_energy")
    normalized = quadratic / (reference + eps)
    bounded_score = normalized / (1.0 + normalized)
    return SheafObstructionResult(
        available=True,
        score=float(bounded_score.detach().item()),
        normalized_residual=float(normalized.detach().item()),
        quadratic_energy=float(quadratic.detach().item()),
        reference_energy=float(reference.detach().item()),
        edge_residual_norms=tuple(
            float(value) for value in residual_sq.detach().sqrt().cpu().tolist()
        ),
    )


def _repair_obstruction(
    stalks: Tensor,
    edge_index: Tensor,
    *,
    tail_restrictions: Tensor | None,
    head_restrictions: Tensor | None,
    edge_weight: Tensor | None,
    eps: float,
    steps: int,
    step_size: float,
    backtracks: int,
    min_improvement: float,
) -> tuple[Tensor, float, str | None]:
    initial = measure_sheaf_obstruction(
        stalks,
        edge_index,
        tail_restrictions=tail_restrictions,
        head_restrictions=head_restrictions,
        edge_weight=edge_weight,
        eps=eps,
    )
    if not initial.available:
        return stalks, initial.score, initial.fallback_reason or "repair_unavailable"

    working = stalks.detach().clone()
    current_score = initial.score
    accepted_steps = 0
    for _ in range(steps):
        with torch.enable_grad():
            differentiable = working.detach().requires_grad_(True)
            objective = _obstruction_score_tensor(
                differentiable,
                edge_index,
                tail_restrictions=tail_restrictions,
                head_restrictions=head_restrictions,
                edge_weight=edge_weight,
                eps=eps,
            )
            gradient = torch.autograd.grad(objective, differentiable)[0]
        if not bool(torch.isfinite(gradient).all()):
            return stalks, initial.score, "repair_nonfinite_gradient"
        if float(gradient.norm().item()) <= eps:
            break

        candidate_step = step_size
        accepted = False
        for _ in range(backtracks):
            candidate = (working - candidate_step * gradient).detach()
            candidate_result = measure_sheaf_obstruction(
                candidate,
                edge_index,
                tail_restrictions=tail_restrictions,
                head_restrictions=head_restrictions,
                edge_weight=edge_weight,
                eps=eps,
            )
            if candidate_result.score <= current_score - min_improvement:
                working = candidate
                current_score = candidate_result.score
                accepted_steps += 1
                accepted = True
                break
            candidate_step *= 0.5
        if not accepted:
            break

    if accepted_steps == 0:
        return stalks, initial.score, "repair_no_descent_step"
    return working, current_score, None


def _obstruction_score_tensor(
    stalks: Tensor,
    edge_index: Tensor,
    *,
    tail_restrictions: Tensor | None,
    head_restrictions: Tensor | None,
    edge_weight: Tensor | None,
    eps: float,
) -> Tensor:
    tail_values = stalks.index_select(0, edge_index[0].to(torch.int64))
    head_values = stalks.index_select(0, edge_index[1].to(torch.int64))
    if tail_restrictions is None or head_restrictions is None:
        restricted_tail = tail_values
        restricted_head = head_values
    else:
        tail_maps = tail_restrictions.to(device=stalks.device, dtype=stalks.dtype)
        head_maps = head_restrictions.to(device=stalks.device, dtype=stalks.dtype)
        restricted_tail = torch.einsum("erd,ed->er", tail_maps, tail_values)
        restricted_head = torch.einsum("erd,ed->er", head_maps, head_values)
    weights = (
        torch.ones(edge_index.shape[1], device=stalks.device, dtype=stalks.dtype)
        if edge_weight is None
        else edge_weight.to(device=stalks.device, dtype=stalks.dtype)
    )
    residual_energy = (weights * (restricted_tail - restricted_head).square().sum(-1)).sum()
    reference_energy = (
        weights
        * (
            restricted_tail.square().sum(-1)
            + restricted_head.square().sum(-1)
        )
    ).sum()
    normalized = residual_energy / (reference_energy + eps)
    return normalized / (1.0 + normalized)


def fit_obstruction_calibrator(
    scores: Sequence[float],
    labels: Sequence[bool | int],
    *,
    target_false_positive_rate: float = 0.1,
    probability_bins: int = 10,
) -> ObstructionCalibrator:
    """Fit monotone probabilities and a negative-only conformal threshold."""
    pairs = _validated_pairs(scores, labels)
    if not 0.0 < target_false_positive_rate < 1.0:
        raise ValueError(
            "target_false_positive_rate must be strictly between zero and one, got "
            f"{target_false_positive_rate}"
        )
    if probability_bins < 1:
        raise ValueError(f"probability_bins must be >= 1, got {probability_bins}")

    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("calibration requires at least one positive and one negative")

    steps = _fit_monotone_steps(pairs, probability_bins)
    negative_scores = sorted(score for score, label in pairs if not label)
    conformal_rank = math.ceil(
        (len(negative_scores) + 1) * (1.0 - target_false_positive_rate)
    )
    threshold = (
        math.inf
        if conformal_rank > len(negative_scores)
        else math.nextafter(negative_scores[conformal_rank - 1], math.inf)
    )

    false_positives = sum(
        score >= threshold for score, label in pairs if not label
    )
    true_positives = sum(score >= threshold for score, label in pairs if label)
    return ObstructionCalibrator(
        steps=steps,
        threshold=threshold,
        target_false_positive_rate=target_false_positive_rate,
        calibration_false_positive_rate=false_positives / negatives,
        calibration_true_positive_rate=true_positives / positives,
        positive_count=positives,
        negative_count=negatives,
    )


def reliability_diagram_svg(
    evaluation: CalibrationEvaluation,
    *,
    width: int = 640,
    height: int = 480,
) -> str:
    """Render a dependency-free SVG reliability diagram."""
    if width < 320 or height < 240:
        raise ValueError("reliability diagram must be at least 320x240")
    left, right, top, bottom = 72.0, 24.0, 36.0, 66.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(probability: float, frequency: float) -> tuple[float, float]:
        return left + probability * plot_w, top + (1.0 - frequency) * plot_h

    diagonal_start = xy(0.0, 0.0)
    diagonal_end = xy(1.0, 1.0)
    circles = []
    for point in evaluation.reliability_bins:
        x, y = xy(point.mean_probability, point.observed_frequency)
        radius = 4.0 + min(8.0, math.sqrt(point.count))
        tooltip = (
            f"n={point.count}, predicted={point.mean_probability:.4f}, "
            f"observed={point.observed_frequency:.4f}"
        )
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="#5b8ff9" fill-opacity="0.78"><title>{tooltip}</title></circle>'
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Sheaf obstruction reliability</text>',
            f'<line x1="{diagonal_start[0]:.2f}" y1="{diagonal_start[1]:.2f}" '
            f'x2="{diagonal_end[0]:.2f}" y2="{diagonal_end[1]:.2f}" '
            'stroke="#888" stroke-dasharray="6 4"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
            *circles,
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13">Calibrated inconsistency probability</text>',
            f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" '
            f'transform="rotate(-90 18 {top + plot_h / 2:.1f})">Observed frequency</text>',
            f'<text x="{left + 8:.1f}" y="{top + 18:.1f}" font-family="sans-serif" '
            f'font-size="12">ECE={evaluation.expected_calibration_error:.4f}</text>',
            "</svg>",
        ]
    ) + "\n"


def _unavailable(reason: str) -> SheafObstructionResult:
    return SheafObstructionResult(
        available=False,
        score=0.0,
        normalized_residual=0.0,
        quadratic_energy=0.0,
        reference_energy=0.0,
        edge_residual_norms=(),
        fallback_reason=reason,
    )


def _validate_score(score: float) -> float:
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"obstruction score must be finite and in [0,1], got {score!r}")
    return value


def _validated_pairs(
    scores: Sequence[float], labels: Sequence[bool | int]
) -> list[tuple[float, bool]]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    pairs: list[tuple[float, bool]] = []
    for score, label in zip(scores, labels):
        if isinstance(label, bool):
            normalized_label = label
        elif isinstance(label, int) and label in (0, 1):
            normalized_label = bool(label)
        else:
            raise ValueError(f"labels must be bool or integer 0/1, got {label!r}")
        pairs.append((_validate_score(score), normalized_label))
    return pairs


def _fit_monotone_steps(
    pairs: list[tuple[float, bool]], num_bins: int
) -> tuple[CalibrationStep, ...]:
    ordered = sorted(pairs, key=lambda pair: pair[0])
    bins = min(num_bins, len(ordered))
    blocks: list[dict[str, float | int]] = []
    for index in range(bins):
        start = index * len(ordered) // bins
        stop = (index + 1) * len(ordered) // bins
        chunk = ordered[start:stop]
        blocks.append(
            {
                "lower": chunk[0][0],
                "upper": chunk[-1][0],
                "count": len(chunk),
                "positives": sum(label for _, label in chunk),
            }
        )
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            previous_rate = float(previous["positives"]) / int(previous["count"])
            current_rate = float(current["positives"]) / int(current["count"])
            if previous_rate <= current_rate:
                break
            blocks[-2:] = [
                {
                    "lower": previous["lower"],
                    "upper": current["upper"],
                    "count": int(previous["count"]) + int(current["count"]),
                    "positives": int(previous["positives"])
                    + int(current["positives"]),
                }
            ]

    return tuple(
        CalibrationStep(
            lower_score=float(block["lower"]),
            upper_score=float(block["upper"]),
            probability=int(block["positives"]) / int(block["count"]),
            count=int(block["count"]),
            positives=int(block["positives"]),
        )
        for block in blocks
    )


def _reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    num_bins: int,
) -> tuple[tuple[ReliabilityBin, ...], float]:
    points: list[ReliabilityBin] = []
    ece = 0.0
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        selected = [
            sample_index
            for sample_index, probability in enumerate(probabilities)
            if probability >= lower
            and (probability <= upper if index == num_bins - 1 else probability < upper)
        ]
        if not selected:
            continue
        mean_probability = sum(probabilities[i] for i in selected) / len(selected)
        observed_frequency = sum(labels[i] for i in selected) / len(selected)
        gap = abs(mean_probability - observed_frequency)
        ece += gap * len(selected) / len(probabilities)
        points.append(
            ReliabilityBin(
                lower_probability=lower,
                upper_probability=upper,
                count=len(selected),
                mean_probability=mean_probability,
                observed_frequency=observed_frequency,
                absolute_gap=gap,
            )
        )
    return tuple(points), ece
