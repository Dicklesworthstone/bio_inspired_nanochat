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

from bio_inspired_nanochat.torch_imports import Tensor, torch

MVP_CERTIFICATE_KIND = "fixed_sheaf_laplacian_residual_mvp"


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
