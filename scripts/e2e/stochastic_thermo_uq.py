"""Falsify thermo-UQ against softmax entropy and MC-dropout (bead 0642.3.3.1).

This is a runnable, CPU-friendly experiment with two independent checks:

1. Draw *actual* one-step vesicle counts through :meth:`SynapticPresyn.release_canonical`
   under a forward drive and a counter-protocol matched to the configured recovery
   propensity.  For equal-size binomial pools the exact local-detailed-balance affinity is

       A = log(p_f (1 - p_r) / (p_r (1 - p_f))),

   so the observed current ``J = K_f - K_r`` must satisfy the detailed fluctuation
   theorem ``log P(J=k)/P(J=-k) = k A``.  This deliberately tests the live release
   sampler rather than the Poisson reference simulator.  It is an isolated E1/E3 ledger
   check, not a certificate for the recurrent hidden-state dynamics or predictive ensemble;
   the structured report makes that scope explicit.
2. Train one tiny synaptic language model, then evaluate the same weights with
   deterministic softmax entropy, MC-dropout, and stochastic-release thermo-UQ.
   The report contains ID expected calibration error (ECE), OOD AUROC, full ECE
   and selective risk-coverage curves, and matched-seed deltas against both baselines.
   The stochastic-release method is the synaptic MC treatment from ``u2t.1``; the
   thermo-UQ name records its optional evidence collector, not a separate predictor.
   No improvement is assumed: negative results are first-class output.

Run with:

    uv run python -m scripts.e2e.stochastic_thermo_uq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.common import logger
from bio_inspired_nanochat.checkpoint_manager import config_hash as normalized_config_hash
from bio_inspired_nanochat.eval_stats import Aggregate, PairedResult, aggregate, paired_comparison
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.mc_ensemble import mc_sampling
from bio_inspired_nanochat.results_registry import RunRecord, append_record, make_record, read_records
from bio_inspired_nanochat.stochastic_thermo import (
    MultiSeedPredictiveThermoVerdict,
    PredictiveEvidencePolicy,
    PredictiveEvidenceProvenance,
    PredictiveThermoCollector,
    PredictiveThermoEvidence,
    predictive_distribution_verdict,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state
from bio_inspired_nanochat.torch_imports import Tensor, nn, torch


@dataclass(frozen=True)
class ExperimentConfig:
    """All controls needed to reproduce the falsification experiment."""

    seed: int = 42
    device: str = "cpu"
    vocab_size: int = 32
    seq_len: int = 12
    batch_size: int = 4
    pool_size: int = 2
    eval_pool_size: int = 2
    train_steps: int = 24
    learning_rate: float = 3e-3
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.2
    mc_samples: int = 8
    ece_bins: int = 10
    ft_trajectories: int = 80_000
    ft_forward_probability: float = 0.32
    ft_reverse_probability: float = 0.24
    ft_tolerance: float = 0.25
    ft_integral_tolerance: float = 0.04
    ft_min_count: int = 100
    predictive_min_samples: int = 8
    predictive_min_tested_fraction: float = 0.75
    predictive_min_symmetric_bins: int = 2
    predictive_crooks_min_count: int = 5
    predictive_crooks_tolerance: float = 0.35
    predictive_min_tur_bound_ratio: float = 0.95
    predictive_max_events_per_head: int = 100_000

    def predictive_policy(self) -> PredictiveEvidencePolicy:
        return PredictiveEvidencePolicy(
            min_samples=self.predictive_min_samples,
            min_tested_fraction=self.predictive_min_tested_fraction,
            min_symmetric_bins=self.predictive_min_symmetric_bins,
            crooks_min_count=self.predictive_crooks_min_count,
            crooks_tolerance=self.predictive_crooks_tolerance,
            min_tur_bound_ratio=self.predictive_min_tur_bound_ratio,
            max_events_per_head=self.predictive_max_events_per_head,
        )

    def validate(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be >= 4 so ID and OOD token bands are non-empty")
        if (
            self.seq_len < 2
            or self.batch_size < 1
            or self.pool_size < 1
            or self.eval_pool_size < 1
        ):
            raise ValueError(
                "seq_len must be >= 2 and batch_size/pool_size/eval_pool_size must be positive"
            )
        if self.train_steps < 0 or self.mc_samples < 1 or self.ece_bins < 1:
            raise ValueError("train_steps must be nonnegative; mc_samples/ece_bins must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.ft_trajectories < 2 or self.ft_min_count < 1:
            raise ValueError("FT needs at least two trajectories and a positive minimum count")
        self.predictive_policy()
        if not 0.0 < self.ft_reverse_probability < self.ft_forward_probability < 1.0:
            raise ValueError("FT probabilities must satisfy 0 < reverse < forward < 1")
        id_band_size = self.vocab_size // 2
        required_id_starts = self.batch_size * (self.pool_size + self.eval_pool_size)
        if required_id_starts > id_band_size:
            raise ValueError(
                "disjoint train/eval cyclic sequences need "
                f"batch_size*(pool_size+eval_pool_size) <= vocab_size//2, got "
                f"{required_id_starts} > {id_band_size}"
            )
        if self.batch_size * self.eval_pool_size > self.vocab_size - id_band_size:
            raise ValueError("the OOD token band is too small for distinct evaluation starts")


@dataclass(frozen=True)
class CrooksPoint:
    current: int
    positive_count: int
    negative_count: int
    observed_log_ratio: float
    expected_log_ratio: float
    residual: float


@dataclass(frozen=True)
class LiveTURDiagnostic:
    """Exact classic-TUR diagnostic for the finite one-step paired-binomial current.

    The runtime TUR theorem in :mod:`stochastic_thermo` assumes the continuous-time Poisson/Skellam
    limit.  The live falsification harness instead samples two finite binomials.  Reporting both
    sides of the classic bound here makes that assumption testable rather than silently transferring
    the continuous-time certificate to a different process.
    """

    scope: str
    relative_variance: float
    entropy_bound: float
    slack: float
    bound_ratio: float
    nonvacuous: bool
    satisfied: bool


@dataclass(frozen=True)
class LiveReleaseFTResult:
    scope: str
    reverse_transition: str
    predictive_distribution_claim: bool
    passed: bool
    experiment_seed: int
    forward_rng_seed: int
    reverse_rng_seed: int
    release_protocol_config_hash: str
    n_trajectories: int
    pool_size: int
    configured_forward_probability: float
    configured_reverse_probability: float
    forward_drive: float
    reverse_drive: float
    forward_probability: float
    reverse_probability: float
    affinity: float
    current_counts: tuple[int, ...]
    integral_ft: float
    integral_ft_residual: float
    max_crooks_residual: float | None
    crooks_min_count: int
    tolerance: float
    integral_tolerance: float
    curve: list[CrooksPoint]
    tur: LiveTURDiagnostic


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    absolute_gap: float


@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    accepted: int
    errors: int
    risk: float
    uncertainty_threshold: float


@dataclass(frozen=True)
class MethodMetrics:
    ece: float
    ood_auroc: float
    id_accuracy: float
    mean_id_confidence: float
    mean_id_uncertainty: float
    mean_ood_uncertainty: float
    selective_aurc: float
    selective_risk_at_50_coverage: float
    selective_risk_at_80_coverage: float
    calibration_curve: list[CalibrationBin]
    risk_coverage_curve: list[RiskCoveragePoint]


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Task-level ``u2t.2`` result, separate from the stricter thermodynamic claim."""

    bead: str
    treatment_method: str
    baseline_methods: tuple[str, ...]
    ece_relative_improvement: dict[str, float]
    ece_improvement_supported: dict[str, bool]
    ood_auroc_threshold: float
    ood_auroc_ci_low: float
    ood_auroc_threshold_passed: bool
    selective_aurc_improvement_supported: dict[str, bool]
    acceptance_met: bool
    verdict: str
    verdict_reason: str


@dataclass(frozen=True)
class ExperimentReport:
    bead: str
    config: ExperimentConfig
    live_release_ft: LiveReleaseFTResult
    methods: dict[str, MethodMetrics]
    thermo_deltas: dict[str, dict[str, float]]
    training_loss: list[float]
    predictive_thermo_evidence: PredictiveThermoEvidence
    comparison_policy: str = field(
        default=(
            "Negative ECE delta and positive OOD-AUROC delta favor thermo-UQ; "
            "the harness reports measurements and does not assert an advantage."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiSeedReport:
    """Matched-seed statistical verdict for the thermo-UQ falsification experiment."""

    bead: str
    report_id: str
    seeds: list[int]
    reports: list[ExperimentReport]
    method_aggregates: dict[str, dict[str, Aggregate]]
    paired_comparisons: dict[str, dict[str, PairedResult]]
    calibration_evaluation: CalibrationEvaluation
    predictive_distribution: MultiSeedPredictiveThermoVerdict
    ft_pass_rate: float
    ft_aggregates: dict[str, Aggregate | None]
    live_tur: LiveTURDiagnostic
    alpha: float
    bootstrap_samples: int
    bootstrap_seed: int
    verdict: str
    verdict_reason: str
    comparison_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Prediction:
    probabilities: Tensor
    uncertainty: Tensor
    predictive_variance: Tensor


PredictionObserver = Callable[[str, _Prediction, _Prediction], None]


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    n_bins: int = 10,
) -> tuple[float, list[CalibrationBin]]:
    """Return standard top-label ECE plus every populated reliability bin."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a final class dimension")
    flat_probs = probabilities.detach().float().reshape(-1, probabilities.shape[-1])
    flat_targets = targets.detach().reshape(-1)
    if flat_probs.shape[0] == 0 or flat_probs.shape[1] == 0:
        raise ValueError("ECE requires at least one prediction and one class")
    if flat_probs.shape[0] != flat_targets.numel():
        raise ValueError("probabilities and targets contain different numbers of predictions")
    if not torch.isfinite(flat_probs).all() or (flat_probs < 0.0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(
        flat_probs.sum(dim=-1),
        torch.ones(flat_probs.shape[0], device=flat_probs.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("probability rows must sum to one")
    if (flat_targets < 0).any() or (flat_targets >= flat_probs.shape[1]).any():
        raise ValueError("targets must index the probability class dimension")
    confidence, predicted = flat_probs.max(dim=-1)
    correct = predicted.eq(flat_targets).float()
    curve: list[CalibrationBin] = []
    ece = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        mask = confidence.ge(lower) & (
            confidence.le(upper) if index == n_bins - 1 else confidence.lt(upper)
        )
        count = int(mask.sum().item())
        if not count:
            continue
        mean_confidence = float(confidence[mask].mean().item())
        accuracy = float(correct[mask].mean().item())
        gap = abs(mean_confidence - accuracy)
        ece += gap * count / flat_targets.numel()
        curve.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                absolute_gap=gap,
            )
        )
    return ece, curve


def selective_risk_coverage(
    probabilities: Tensor,
    targets: Tensor,
    uncertainty: Tensor,
) -> tuple[float, list[RiskCoveragePoint]]:
    """Return the discrete area under the selective risk-coverage curve.

    Predictions are accepted from lowest to highest uncertainty. At each attainable coverage,
    selective risk is the error fraction among accepted predictions. Stable sorting gives a
    deterministic policy when uncertainty scores tie; the complete curve preserves that choice for
    audit instead of interpolating a more favorable threshold after observing labels.
    """
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a final class dimension")
    flat_probs = probabilities.detach().float().reshape(-1, probabilities.shape[-1])
    flat_targets = targets.detach().reshape(-1)
    flat_uncertainty = uncertainty.detach().float().reshape(-1)
    prediction_count = flat_targets.numel()
    if prediction_count == 0 or flat_probs.shape[1] == 0:
        raise ValueError("risk-coverage requires at least one prediction and one class")
    if flat_probs.shape[0] != prediction_count or flat_uncertainty.numel() != prediction_count:
        raise ValueError("probabilities, targets, and uncertainty must align per prediction")
    if (
        not torch.isfinite(flat_probs).all()
        or not torch.isfinite(flat_uncertainty).all()
        or (flat_probs < 0.0).any()
    ):
        raise ValueError("risk-coverage inputs must be finite and probabilities nonnegative")
    if not torch.allclose(
        flat_probs.sum(dim=-1),
        torch.ones(prediction_count, device=flat_probs.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("probability rows must sum to one")
    if (flat_targets < 0).any() or (flat_targets >= flat_probs.shape[1]).any():
        raise ValueError("targets must index the probability class dimension")

    errors = flat_probs.argmax(dim=-1).ne(flat_targets).to(torch.int64)
    order = torch.argsort(flat_uncertainty, stable=True)
    ranked_errors = errors[order]
    ranked_uncertainty = flat_uncertainty[order]
    cumulative_errors = ranked_errors.cumsum(dim=0)
    accepted = torch.arange(
        1,
        prediction_count + 1,
        device=cumulative_errors.device,
        dtype=torch.int64,
    )
    risks = cumulative_errors.to(torch.float64) / accepted
    curve = [
        RiskCoveragePoint(
            coverage=int(count) / prediction_count,
            accepted=int(count),
            errors=int(error_count),
            risk=float(risk),
            uncertainty_threshold=float(threshold),
        )
        for count, error_count, risk, threshold in zip(
            accepted.cpu().tolist(),
            cumulative_errors.cpu().tolist(),
            risks.cpu().tolist(),
            ranked_uncertainty.cpu().tolist(),
            strict=True,
        )
    ]
    return float(risks.mean().item()), curve


def _risk_at_coverage(curve: Sequence[RiskCoveragePoint], target: float) -> float:
    if not curve:
        raise ValueError("risk-coverage curve must not be empty")
    if not 0.0 < target <= 1.0:
        raise ValueError("target coverage must lie in (0, 1]")
    index = min(math.ceil(target * len(curve)) - 1, len(curve) - 1)
    return curve[index].risk


def binary_auroc(id_scores: Tensor, ood_scores: Tensor) -> float:
    """Exact rank-based AUROC with average ranks for ties and ``O(n)`` extra memory."""
    negative = id_scores.detach().float().reshape(-1).cpu().numpy()
    positive = ood_scores.detach().float().reshape(-1).cpu().numpy()
    if negative.size == 0 or positive.size == 0:
        raise ValueError("AUROC requires at least one ID and one OOD score")
    if not np.isfinite(negative).all() or not np.isfinite(positive).all():
        raise ValueError("AUROC scores must be finite")
    scores = np.concatenate((negative, positive))
    labels = np.concatenate(
        (np.zeros(negative.size, dtype=np.int8), np.ones(positive.size, dtype=np.int8))
    )
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    mann_whitney_u = positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    return mann_whitney_u / (positive.size * negative.size)


def binomial_crooks_curve(
    currents: np.ndarray,
    affinity: float,
    *,
    pool_size: int,
    min_count: int,
) -> list[CrooksPoint]:
    """Measure the detailed-FT line on the exact integer support of live counts."""
    current = np.asarray(currents, dtype=np.int64)
    points: list[CrooksPoint] = []
    for value in range(1, pool_size + 1):
        positive_count = int(np.count_nonzero(current == value))
        negative_count = int(np.count_nonzero(current == -value))
        if positive_count < min_count or negative_count < min_count:
            continue
        observed = math.log(positive_count / negative_count)
        expected = value * affinity
        points.append(
            CrooksPoint(
                current=value,
                positive_count=positive_count,
                negative_count=negative_count,
                observed_log_ratio=observed,
                expected_log_ratio=expected,
                residual=observed - expected,
            )
        )
    return points


def binomial_tur_diagnostic(
    *,
    pool_size: int,
    forward_probability: float,
    reverse_probability: float,
    affinity: float,
) -> LiveTURDiagnostic:
    """Evaluate the classic continuous-time TUR on the exact paired-binomial moments.

    This is intentionally a diagnostic, not a corrected discrete-time bound.  A finite positive
    ``entropy_bound`` proves non-vacuity; ``satisfied`` separately records whether the classic
    Poisson/Skellam inequality actually transfers to the live one-step process.
    """
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if not 0.0 < reverse_probability < forward_probability < 1.0:
        raise ValueError("probabilities must satisfy 0 < reverse < forward < 1")
    mean_current = pool_size * (forward_probability - reverse_probability)
    current_variance = pool_size * (
        forward_probability * (1.0 - forward_probability)
        + reverse_probability * (1.0 - reverse_probability)
    )
    mean_entropy = mean_current * affinity
    relative_variance = current_variance / (mean_current * mean_current)
    entropy_bound = 2.0 / mean_entropy
    slack = relative_variance - entropy_bound
    nonvacuous = bool(math.isfinite(entropy_bound) and entropy_bound > 0.0)
    return LiveTURDiagnostic(
        scope="classic_continuous_time_tur_on_exact_one_step_paired_binomial_moments",
        relative_variance=relative_variance,
        entropy_bound=entropy_bound,
        slack=slack,
        bound_ratio=relative_variance / entropy_bound,
        nonvacuous=nonvacuous,
        satisfied=bool(nonvacuous and slack >= 0.0),
    )


def _fresh_release_state(n_trajectories: int, cfg: SynapticConfig) -> dict[str, Any]:
    return build_presyn_state(
        n_trajectories,
        1,
        1,
        torch.device("cpu"),
        torch.float32,
        cfg,
    )


def _release_probability(presyn: SynapticPresyn, cfg: SynapticConfig, drive: float) -> float:
    state = _fresh_release_state(1, cfg)
    before = state["RRP"].clone()
    presyn._mc_sampling = False
    presyn.release_canonical(
        state,
        torch.full((1, 1, 1, 1), drive),
        torch.zeros((1, 1, 1, 1), dtype=torch.long),
        train=False,
    )
    released = before - state["RRP"]
    return float((released / cfg.init_rrp).item())


def _drive_for_probability(
    presyn: SynapticPresyn,
    cfg: SynapticConfig,
    target: float,
) -> float:
    lower, upper = -12.0, 12.0
    p_lower = _release_probability(presyn, cfg, lower)
    p_upper = _release_probability(presyn, cfg, upper)
    if not p_lower <= target <= p_upper:
        raise ValueError(
            f"target probability {target} is outside live release range [{p_lower}, {p_upper}]"
        )
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if _release_probability(presyn, cfg, middle) < target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _sample_live_counts(
    presyn: SynapticPresyn,
    cfg: SynapticConfig,
    *,
    drive: float,
    n_trajectories: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    state = _fresh_release_state(n_trajectories, cfg)
    before = state["RRP"].clone()
    presyn._mc_sampling = True
    presyn._mc_frac = 1.0
    try:
        presyn.release_canonical(
            state,
            torch.full((n_trajectories, 1, 1, 1), drive),
            torch.zeros((n_trajectories, 1, 1, 1), dtype=torch.long),
            train=False,
        )
    finally:
        presyn._mc_sampling = False
    released = (before - state["RRP"]).round().to(torch.int64)
    return released.reshape(-1).cpu().numpy()


def run_live_release_ft(config: ExperimentConfig) -> LiveReleaseFTResult:
    """Test one-step local detailed balance with live forward/counter-protocol draws.

    This discharges the isolated paired-binomial E1/E3 check only. It does not certify the
    recurrent hidden-state dynamics or the downstream predictive distribution.
    """
    release_cfg = SynapticConfig(
        stochastic_train_frac=1.0,
        stochastic_mode="straight_through",
        stochastic_count_cap=8,
        prime_rate=0.0,
        endo_delay=0,
        init_rrp=6.0,
        rec_rate=config.ft_reverse_probability,
    )
    presyn = SynapticPresyn(d_head=1, cfg=release_cfg)
    forward_drive = _drive_for_probability(
        presyn, release_cfg, config.ft_forward_probability
    )
    reverse_drive = _drive_for_probability(
        presyn, release_cfg, config.ft_reverse_probability
    )
    forward_probability = _release_probability(presyn, release_cfg, forward_drive)
    reverse_probability = _release_probability(presyn, release_cfg, reverse_drive)
    forward_rng_seed = config.seed + 101
    reverse_rng_seed = config.seed + 102
    forward = _sample_live_counts(
        presyn,
        release_cfg,
        drive=forward_drive,
        n_trajectories=config.ft_trajectories,
        seed=forward_rng_seed,
    )
    reverse = _sample_live_counts(
        presyn,
        release_cfg,
        drive=reverse_drive,
        n_trajectories=config.ft_trajectories,
        seed=reverse_rng_seed,
    )
    currents = forward - reverse
    affinity = math.log(
        forward_probability
        * (1.0 - reverse_probability)
        / (reverse_probability * (1.0 - forward_probability))
    )
    sigma = currents.astype(np.float64) * affinity
    pool_size = int(round(release_cfg.init_rrp))
    current_counts = tuple(
        int(count)
        for count in np.bincount(
            currents + pool_size,
            minlength=2 * pool_size + 1,
        )
    )
    integral_ft = float(np.mean(np.exp(-sigma)))
    integral_residual = abs(integral_ft - 1.0)
    curve = binomial_crooks_curve(
        currents,
        affinity,
        pool_size=pool_size,
        min_count=config.ft_min_count,
    )
    max_residual = max((abs(point.residual) for point in curve), default=None)
    tur = binomial_tur_diagnostic(
        pool_size=pool_size,
        forward_probability=forward_probability,
        reverse_probability=reverse_probability,
        affinity=affinity,
    )
    passed = bool(
        curve
        and max_residual is not None
        and max_residual <= config.ft_tolerance
        and integral_residual <= config.ft_integral_tolerance
    )
    return LiveReleaseFTResult(
        scope="one_step_local_detailed_balance",
        reverse_transition=(
            "live binomial counter-protocol matched to the configured recovery propensity; "
            "not a certificate for recurrent hidden-state dynamics"
        ),
        predictive_distribution_claim=False,
        passed=passed,
        experiment_seed=config.seed,
        forward_rng_seed=forward_rng_seed,
        reverse_rng_seed=reverse_rng_seed,
        release_protocol_config_hash=normalized_config_hash(asdict(release_cfg)),
        n_trajectories=config.ft_trajectories,
        pool_size=pool_size,
        configured_forward_probability=config.ft_forward_probability,
        configured_reverse_probability=config.ft_reverse_probability,
        forward_drive=forward_drive,
        reverse_drive=reverse_drive,
        forward_probability=forward_probability,
        reverse_probability=reverse_probability,
        affinity=affinity,
        current_counts=current_counts,
        integral_ft=integral_ft,
        integral_ft_residual=integral_residual,
        max_crooks_residual=max_residual,
        crooks_min_count=config.ft_min_count,
        tolerance=config.ft_tolerance,
        integral_tolerance=config.ft_integral_tolerance,
        curve=curve,
        tur=tur,
    )


def _make_model(config: ExperimentConfig) -> GPTSynaptic:
    synaptic = SynapticConfig(stochastic_mode="straight_through")
    model_config = GPTSynapticConfig(
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        synapses=True,
        syn_cfg=synaptic,
    )
    return GPTSynaptic(model_config).to(config.device)


def _make_pool(
    config: ExperimentConfig,
    *,
    low: int,
    high: int,
    starts: Tensor,
) -> list[tuple[Tensor, Tensor]]:
    """Generate a learnable cyclic successor language from explicitly selected starts."""
    flat_starts = starts.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if flat_starts.numel() % config.batch_size:
        raise ValueError("the number of starts must be divisible by batch_size")
    if (flat_starts < low).any() or (flat_starts >= high).any():
        raise ValueError("cyclic sequence starts must lie inside the requested token band")
    pool: list[tuple[Tensor, Tensor]] = []
    offsets = torch.arange(config.seq_len + 1, dtype=torch.long).reshape(1, -1)
    band_size = high - low
    for batch_starts in flat_starts.reshape(-1, config.batch_size):
        tokens = (
            low + (batch_starts.reshape(-1, 1) - low + offsets) % band_size
        ).to(config.device)
        pool.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return pool


def _train_model(
    model: GPTSynaptic,
    pool: Sequence[tuple[Tensor, Tensor]],
    config: ExperimentConfig,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_history: list[float] = []
    model.train()
    for step in range(config.train_steps):
        inputs, targets = pool[step % len(pool)]
        model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets, train_mode=False)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_history.append(float(loss.detach().item()))
    model.eval()
    return loss_history


def _entropy(probabilities: Tensor) -> Tensor:
    probs = probabilities.float()
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def _reset_sequence(model: GPTSynaptic) -> None:
    # Eligibility traces are sequence-local, but w_fast is a backprop-trained Parameter and the
    # consolidation buffers are part of the learned bio state. Preserve both across evaluations.
    model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)


def _experiment_config_hash(model: GPTSynaptic, config: ExperimentConfig) -> str:
    experiment = asdict(config)
    experiment.pop("seed")
    payload = json.dumps(
        {"experiment": experiment, "model": asdict(model.config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_checkpoint_id(model: GPTSynaptic) -> str:
    """Hash the exact in-memory state that produced the predictive ensemble."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _softmax_prediction(model: GPTSynaptic, inputs: Tensor) -> _Prediction:
    model.eval()
    _reset_sequence(model)
    logits, _ = model(inputs, train_mode=False)
    probabilities = torch.softmax(logits.float(), dim=-1)
    return _Prediction(
        probabilities,
        _entropy(probabilities),
        torch.zeros_like(probabilities),
    )


@contextmanager
def _dropout_sampling(model: GPTSynaptic) -> Iterator[None]:
    dropout_modules = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    prior = [module.training for module in dropout_modules]
    model.eval()
    for module in dropout_modules:
        module.train()
    try:
        yield
    finally:
        for module, was_training in zip(dropout_modules, prior, strict=True):
            module.train(was_training)


@torch.no_grad()
def _mc_dropout_prediction(
    model: GPTSynaptic,
    inputs: Tensor,
    *,
    n_samples: int,
) -> _Prediction:
    probability_sum = None
    probability_sq_sum = None
    with _dropout_sampling(model):
        for _ in range(n_samples):
            _reset_sequence(model)
            logits, _ = model(inputs, train_mode=False)
            probabilities = torch.softmax(logits.float(), dim=-1)
            probability_sum = (
                probabilities if probability_sum is None else probability_sum + probabilities
            )
            probability_sq_sum = (
                probabilities.square()
                if probability_sq_sum is None
                else probability_sq_sum + probabilities.square()
            )
    if probability_sum is None or probability_sq_sum is None:
        raise AssertionError("n_samples validation should make the MC loop non-empty")
    mean_probabilities = probability_sum / n_samples
    predictive_variance = (
        probability_sq_sum / n_samples - mean_probabilities.square()
    ).clamp_min(0.0)
    return _Prediction(
        mean_probabilities,
        _entropy(mean_probabilities),
        predictive_variance,
    )


def _thermo_prediction(
    model: GPTSynaptic,
    inputs: Tensor,
    *,
    n_samples: int,
    evidence_collector: PredictiveThermoCollector | None = None,
) -> _Prediction:
    probability_sum = None
    probability_sq_sum = None
    model.eval()
    with torch.no_grad(), mc_sampling(model, evidence_collector=evidence_collector):
        for sample_index in range(n_samples):
            if evidence_collector is not None:
                evidence_collector.begin_sample(sample_index)
            _reset_sequence(model)
            logits, _ = model(inputs, train_mode=False)
            probabilities = torch.softmax(logits.float(), dim=-1)
            probability_sum = (
                probabilities if probability_sum is None else probability_sum + probabilities
            )
            probability_sq_sum = (
                probabilities.square()
                if probability_sq_sum is None
                else probability_sq_sum + probabilities.square()
            )
    if probability_sum is None or probability_sq_sum is None:
        raise AssertionError("n_samples validation should make the MC loop non-empty")
    mean_probabilities = probability_sum / n_samples
    predictive_variance = (
        probability_sq_sum / n_samples - mean_probabilities.square()
    ).clamp_min(0.0)
    return _Prediction(
        mean_probabilities,
        _entropy(mean_probabilities),
        predictive_variance,
    )


def _method_metrics(
    id_prediction: _Prediction,
    ood_prediction: _Prediction,
    id_targets: Tensor,
    *,
    ece_bins: int,
) -> MethodMetrics:
    ece, curve = expected_calibration_error(
        id_prediction.probabilities,
        id_targets,
        n_bins=ece_bins,
    )
    selective_aurc, risk_coverage_curve = selective_risk_coverage(
        id_prediction.probabilities,
        id_targets,
        id_prediction.uncertainty,
    )
    id_confidence, id_class = id_prediction.probabilities.max(dim=-1)
    return MethodMetrics(
        ece=ece,
        ood_auroc=binary_auroc(id_prediction.uncertainty, ood_prediction.uncertainty),
        id_accuracy=float(id_class.eq(id_targets).float().mean().item()),
        mean_id_confidence=float(id_confidence.mean().item()),
        mean_id_uncertainty=float(id_prediction.uncertainty.mean().item()),
        mean_ood_uncertainty=float(ood_prediction.uncertainty.mean().item()),
        selective_aurc=selective_aurc,
        selective_risk_at_50_coverage=_risk_at_coverage(risk_coverage_curve, 0.5),
        selective_risk_at_80_coverage=_risk_at_coverage(risk_coverage_curve, 0.8),
        calibration_curve=curve,
        risk_coverage_curve=risk_coverage_curve,
    )


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    prediction_observer: PredictionObserver | None = None,
) -> ExperimentReport:
    """Run all uncertainty methods and optionally expose their raw predictions for logging."""
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    live_release_ft = run_live_release_ft(config)
    # The live FT sampler consumes the global torch RNG internally. Reset here so changing only the
    # number of FT trajectories cannot silently change model initialization or benchmark training.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = _make_model(config)
    split = config.vocab_size // 2
    id_generator = torch.Generator().manual_seed(config.seed + 1)
    id_starts = torch.randperm(split, generator=id_generator)
    training_count = config.batch_size * config.pool_size
    evaluation_count = config.batch_size * config.eval_pool_size
    training_starts = id_starts[:training_count]
    evaluation_starts = id_starts[training_count : training_count + evaluation_count]
    ood_generator = torch.Generator().manual_seed(config.seed + 2)
    ood_starts = split + torch.randperm(
        config.vocab_size - split, generator=ood_generator
    )[:evaluation_count]
    if torch.isin(training_starts, evaluation_starts).any():
        raise AssertionError("construction error: ID train/eval sequence starts overlap")

    training_pool = _make_pool(
        config,
        low=0,
        high=split,
        starts=training_starts,
    )
    id_pool = _make_pool(
        config,
        low=0,
        high=split,
        starts=evaluation_starts,
    )
    ood_pool = _make_pool(
        config,
        low=split,
        high=config.vocab_size,
        starts=ood_starts,
    )
    training_loss = _train_model(model, training_pool, config)
    checkpoint_id = _model_checkpoint_id(model)
    config_hash = _experiment_config_hash(model, config)
    synaptic_config_hash = normalized_config_hash(asdict(model.config.syn_cfg))
    id_inputs = torch.cat([batch[0] for batch in id_pool])
    id_targets = torch.cat([batch[1] for batch in id_pool])
    ood_inputs = torch.cat([batch[0] for batch in ood_pool])

    softmax_id = _softmax_prediction(model, id_inputs)
    softmax_ood = _softmax_prediction(model, ood_inputs)
    torch.manual_seed(config.seed + 201)
    dropout_id = _mc_dropout_prediction(model, id_inputs, n_samples=config.mc_samples)
    torch.manual_seed(config.seed + 202)
    dropout_ood = _mc_dropout_prediction(model, ood_inputs, n_samples=config.mc_samples)
    torch.manual_seed(config.seed + 301)
    evidence_collector = PredictiveThermoCollector(
        PredictiveEvidenceProvenance(
            run_id=f"stochastic-thermo-predictive-{config_hash[:12]}-s{config.seed}",
            checkpoint_id=checkpoint_id,
            synaptic_config_hash=synaptic_config_hash,
            config_hash=config_hash,
            rng_seed=config.seed + 301,
        ),
        config.predictive_policy(),
    )
    thermo_id = _thermo_prediction(
        model,
        id_inputs,
        n_samples=config.mc_samples,
        evidence_collector=evidence_collector,
    )
    predictive_evidence = evidence_collector.finalize(
        current_checkpoint_id=_model_checkpoint_id(model),
        current_synaptic_config_hash=normalized_config_hash(
            asdict(model.config.syn_cfg)
        ),
        current_config_hash=_experiment_config_hash(model, config),
        current_rng_seed=config.seed + 301,
    )
    torch.manual_seed(config.seed + 302)
    thermo_ood = _thermo_prediction(model, ood_inputs, n_samples=config.mc_samples)
    predictions = {
        "softmax_entropy": (softmax_id, softmax_ood),
        "mc_dropout": (dropout_id, dropout_ood),
        "thermo_uq": (thermo_id, thermo_ood),
    }
    if prediction_observer is not None:
        for method_name, (id_prediction, ood_prediction) in predictions.items():
            prediction_observer(method_name, id_prediction, ood_prediction)
    methods = {
        "softmax_entropy": _method_metrics(
            softmax_id, softmax_ood, id_targets, ece_bins=config.ece_bins
        ),
        "mc_dropout": _method_metrics(
            dropout_id, dropout_ood, id_targets, ece_bins=config.ece_bins
        ),
        "thermo_uq": _method_metrics(
            thermo_id, thermo_ood, id_targets, ece_bins=config.ece_bins
        ),
    }
    thermo = methods["thermo_uq"]
    deltas: dict[str, dict[str, float]] = {}
    for baseline in ("softmax_entropy", "mc_dropout"):
        reference = methods[baseline]
        deltas[f"vs_{baseline}"] = {
            "ece_delta_lower_is_better": thermo.ece - reference.ece,
            "ood_auroc_delta_higher_is_better": thermo.ood_auroc - reference.ood_auroc,
        }
    return ExperimentReport(
        bead="bio_inspired_nanochat-0642.3.3.1",
        config=config,
        live_release_ft=live_release_ft,
        methods=methods,
        thermo_deltas=deltas,
        training_loss=training_loss,
        predictive_thermo_evidence=predictive_evidence,
    )


def _report_id(base_config: ExperimentConfig, seeds: Sequence[int]) -> str:
    config_payload = asdict(base_config)
    config_payload.pop("seed")
    payload = {
        "protocol_version": "selective-risk-v1",
        "config": config_payload,
        "seeds": list(seeds),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"stochastic-thermo-uq-{digest}"


def _supports_improvement(
    comparison: PairedResult,
    *,
    lower_is_better: bool,
    alpha: float,
) -> bool:
    ci_favorable = (
        comparison.delta_ci_high < 0.0
        if lower_is_better
        else comparison.delta_ci_low > 0.0
    )
    return bool(
        ci_favorable
        and comparison.t_p_value <= alpha
        and comparison.wilcoxon_p_value <= alpha
    )


def _calibration_evaluation(
    method_aggregates: dict[str, dict[str, Aggregate]],
    paired: dict[str, dict[str, PairedResult]],
    *,
    alpha: float,
    ood_auroc_threshold: float = 0.7,
) -> CalibrationEvaluation:
    """Apply the preregistered ``u2t.2`` capability criterion without theory scope creep."""
    treatment_name = "thermo_uq"
    treatment = method_aggregates[treatment_name]
    baseline_names = ("softmax_entropy", "mc_dropout")
    ece_relative_improvement: dict[str, float] = {}
    ece_supported: dict[str, bool] = {}
    aurc_supported: dict[str, bool] = {}
    for baseline in baseline_names:
        baseline_ece = method_aggregates[baseline]["ece"].mean
        ece_relative_improvement[baseline] = (
            (baseline_ece - treatment["ece"].mean) / baseline_ece
            if baseline_ece > 0.0
            else 0.0
        )
        comparison = paired[f"vs_{baseline}"]
        ece_supported[baseline] = bool(
            ece_relative_improvement[baseline] >= 0.10
            and _supports_improvement(
                comparison["ece"],
                lower_is_better=True,
                alpha=alpha,
            )
        )
        aurc_supported[baseline] = _supports_improvement(
            comparison["selective_aurc"],
            lower_is_better=True,
            alpha=alpha,
        )

    auroc_ci_low = treatment["ood_auroc"].ci_low
    auroc_passed = auroc_ci_low > ood_auroc_threshold
    acceptance_met = bool(any(ece_supported.values()) or auroc_passed)
    ece_wins = [name for name, supported in ece_supported.items() if supported]
    if acceptance_met:
        reasons: list[str] = []
        if ece_wins:
            reasons.append(
                "ECE improved by at least 10% with matched-seed support versus "
                + ", ".join(ece_wins)
            )
        if auroc_passed:
            reasons.append(
                f"the treatment OOD-AUROC 95% lower bound {auroc_ci_low:.6f} exceeded "
                f"the {ood_auroc_threshold:.2f} target"
            )
        verdict = "positive"
        verdict_reason = "; ".join(reasons) + (
            ". This satisfies the task threshold, while paired comparisons and selective-risk "
            "statistics remain the authority for claims of superiority over each baseline."
        )
    else:
        verdict = "null"
        verdict_reason = (
            "Neither a statistically supported >=10% ECE improvement nor an OOD-AUROC lower "
            f"confidence bound above {ood_auroc_threshold:.2f} was observed."
        )
    return CalibrationEvaluation(
        bead="bio_inspired_nanochat-u2t.2",
        treatment_method="thermo_uq (synaptic stochastic-release MC)",
        baseline_methods=baseline_names,
        ece_relative_improvement=ece_relative_improvement,
        ece_improvement_supported=ece_supported,
        ood_auroc_threshold=ood_auroc_threshold,
        ood_auroc_ci_low=auroc_ci_low,
        ood_auroc_threshold_passed=auroc_passed,
        selective_aurc_improvement_supported=aurc_supported,
        acceptance_met=acceptance_met,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )


def summarize_multi_seed(
    reports: Sequence[ExperimentReport],
    *,
    bootstrap_samples: int = 10_000,
    alpha: float = 0.05,
    bootstrap_seed: int = 20260824,
) -> MultiSeedReport:
    """Aggregate already-computed matched-seed reports using the shared ``74f.3`` layer."""
    reports = tuple(sorted(reports, key=lambda item: item.config.seed))
    if len(reports) < 2:
        raise ValueError("multi-seed statistics need at least two reports")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    seeds = [report.config.seed for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed reports must have unique seeds")
    method_names = set(reports[0].methods)
    if any(set(report.methods) != method_names for report in reports[1:]):
        raise ValueError("all reports must contain the same uncertainty methods")

    method_aggregates: dict[str, dict[str, Aggregate]] = {}
    for method in sorted(method_names):
        method_aggregates[method] = {
            metric: aggregate(
                [float(getattr(report.methods[method], metric)) for report in reports]
            )
            for metric in (
                "ece",
                "ood_auroc",
                "id_accuracy",
                "selective_aurc",
                "selective_risk_at_50_coverage",
                "selective_risk_at_80_coverage",
            )
        }

    paired: dict[str, dict[str, PairedResult]] = {}
    for baseline in ("softmax_entropy", "mc_dropout"):
        per_metric: dict[str, PairedResult] = {}
        for metric, lower_is_better in (
            ("ece", True),
            ("ood_auroc", False),
            ("selective_aurc", True),
            ("selective_risk_at_80_coverage", True),
        ):
            treatment = {
                report.config.seed: float(getattr(report.methods["thermo_uq"], metric))
                for report in reports
            }
            reference = {
                report.config.seed: float(getattr(report.methods[baseline], metric))
                for report in reports
            }
            comparison = paired_comparison(
                treatment,
                reference,
                lower_is_better=lower_is_better,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed,
            )
            if comparison is None:
                raise AssertionError("validated matched-seed reports produced no paired comparison")
            per_metric[metric] = comparison
        paired[f"vs_{baseline}"] = per_metric

    calibration_evaluation = _calibration_evaluation(
        method_aggregates,
        paired,
        alpha=alpha,
    )

    max_crooks = [
        report.live_release_ft.max_crooks_residual for report in reports
    ]
    ft_aggregates: dict[str, Aggregate | None] = {
        "max_crooks_residual": (
            aggregate([float(value) for value in max_crooks if value is not None])
            if all(value is not None for value in max_crooks)
            else None
        ),
        "integral_ft_residual": aggregate(
            [report.live_release_ft.integral_ft_residual for report in reports]
        ),
    }
    ft_pass_rate = sum(report.live_release_ft.passed for report in reports) / len(reports)
    live_tur = reports[0].live_release_ft.tur
    if any(report.live_release_ft.tur != live_tur for report in reports[1:]):
        raise ValueError("live TUR diagnostics differ across matched protocol configurations")

    comparison_policy = (
        "Positive requires the paired-bootstrap 95% interval to exclude zero in the favorable "
        "direction and both paired t and Wilcoxon p-values <= alpha for ECE and OOD AUROC "
        "against both softmax entropy and MC-dropout; every live FT seed must pass and the "
        "classic TUR must transfer to the live finite-binomial process."
    )
    all_calibration_gates = all(
        _supports_improvement(metrics["ece"], lower_is_better=True, alpha=alpha)
        and _supports_improvement(metrics["ood_auroc"], lower_is_better=False, alpha=alpha)
        for metrics in paired.values()
    )
    predictive_statistics_gate = bool(
        all_calibration_gates
        and all(report.live_release_ft.passed for report in reports)
        and live_tur.satisfied
    )
    predictive_verdict = predictive_distribution_verdict(
        [report.predictive_thermo_evidence for report in reports],
        multi_seed_statistics_passed=predictive_statistics_gate,
    )
    if ft_pass_rate < 1.0:
        verdict = "invalidated"
        verdict_reason = (
            f"Only {ft_pass_rate:.1%} of live one-step FT checks passed; the analytic "
            "local-detailed-balance claim is falsified for this protocol."
        )
    elif predictive_verdict.predictive_distribution_claim:
        verdict = "positive"
        verdict_reason = (
            "Thermo-UQ passed the strict matched-seed ECE/AUROC gate against both baselines, "
            "all live FT checks passed, and the classic TUR held on the live process."
        )
    else:
        verdict = "null"
        limitations: list[str] = []
        if not all_calibration_gates:
            limitations.append("the strict paired ECE/AUROC improvement gate was not met")
        if not live_tur.satisfied:
            limitations.append(
                "the finite-binomial live current had a non-vacuous but slightly violated "
                "classic continuous-time TUR bound"
            )
        if (
            all_calibration_gates
            and live_tur.satisfied
            and not predictive_verdict.predictive_distribution_claim
        ):
            limitations.append("the per-layer/head predictive evidence gate was not met")
        verdict_reason = (
            "All live one-step FT checks passed, but " + " and ".join(limitations) + "."
        )

    return MultiSeedReport(
        bead="bio_inspired_nanochat-0642.3.3.2",
        report_id=_report_id(reports[0].config, seeds),
        seeds=seeds,
        reports=list(reports),
        method_aggregates=method_aggregates,
        paired_comparisons=paired,
        calibration_evaluation=calibration_evaluation,
        predictive_distribution=predictive_verdict,
        ft_pass_rate=ft_pass_rate,
        ft_aggregates=ft_aggregates,
        live_tur=live_tur,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        verdict=verdict,
        verdict_reason=verdict_reason,
        comparison_policy=comparison_policy,
    )


def run_multi_seed(
    base_config: ExperimentConfig,
    seeds: Sequence[int],
    *,
    bootstrap_samples: int = 10_000,
    alpha: float = 0.05,
) -> MultiSeedReport:
    """Run the complete experiment on unique matched seeds and return its statistical verdict."""
    ordered_seeds = [int(seed) for seed in seeds]
    if any(seed < 0 for seed in ordered_seeds):
        raise ValueError("seeds must be nonnegative")
    reports = [run_experiment(replace(base_config, seed=seed)) for seed in ordered_seeds]
    return summarize_multi_seed(
        reports,
        bootstrap_samples=bootstrap_samples,
        alpha=alpha,
    )


def registry_records(
    report: MultiSeedReport,
    *,
    artifact: str | None = None,
) -> list[RunRecord]:
    """Build one schema-valid registry record per method and seed."""
    ece_softmax = report.paired_comparisons["vs_softmax_entropy"]["ece"]
    auroc_softmax = report.paired_comparisons["vs_softmax_entropy"]["ood_auroc"]
    artifact_note = f"; artifact={artifact}" if artifact else ""
    shared_notes = (
        f"experiment=stochastic_thermo_uq; group_verdict={report.verdict}; "
        f"u2t2_verdict={report.calibration_evaluation.verdict}; "
        f"thermo_vs_softmax_ece_delta={ece_softmax.mean_delta:.17g}; "
        f"thermo_vs_softmax_ece_ci=[{ece_softmax.delta_ci_low:.17g},"
        f"{ece_softmax.delta_ci_high:.17g}]; "
        f"thermo_vs_softmax_auroc_delta={auroc_softmax.mean_delta:.17g}; "
        f"thermo_vs_softmax_auroc_ci=[{auroc_softmax.delta_ci_low:.17g},"
        f"{auroc_softmax.delta_ci_high:.17g}]; "
        f"live_tur_bound_ratio={report.live_tur.bound_ratio:.17g}{artifact_note}"
    )
    records: list[RunRecord] = []
    for seed_report in report.reports:
        ft = seed_report.live_release_ft
        if ft.max_crooks_residual is None:
            raise ValueError("registry records require a populated Crooks residual")
        for method, metrics in seed_report.methods.items():
            is_treatment = method == "thermo_uq"
            records.append(
                make_record(
                    "eval",
                    {
                        "id_ece": metrics.ece,
                        "ood_auroc": metrics.ood_auroc,
                        "selective_aurc": metrics.selective_aurc,
                        "selective_risk_at_80_coverage": (
                            metrics.selective_risk_at_80_coverage
                        ),
                        "eval_accuracy": metrics.id_accuracy,
                        "live_ft_max_crooks_residual": ft.max_crooks_residual,
                        "live_ft_integral_residual": ft.integral_ft_residual,
                        "live_tur_relative_variance": ft.tur.relative_variance,
                        "live_tur_entropy_bound": ft.tur.entropy_bound,
                        "live_tur_slack": ft.tur.slack,
                        "live_tur_bound_ratio": ft.tur.bound_ratio,
                    },
                    run_id=(
                        f"{report.report_id}-{method}-s{seed_report.config.seed}"
                    ),
                    config={
                        "experiment": asdict(seed_report.config),
                        "method": method,
                        "group_seeds": report.seeds,
                    },
                    seed=seed_report.config.seed,
                    notes=f"method={method}; {shared_notes}",
                    verdict=(
                        report.calibration_evaluation.verdict if is_treatment else None
                    ),
                    eligible_for_best=False,
                )
            )
    return records


def append_registry_records(
    report: MultiSeedReport,
    path: Path,
    *,
    artifact: str | None = None,
) -> int:
    """Append the multi-seed records once, refusing duplicate run identifiers."""
    records = registry_records(report, artifact=artifact)
    existing = {record.run_id for record in read_records(str(path))}
    duplicates = sorted(record.run_id for record in records if record.run_id in existing)
    if duplicates:
        raise ValueError(f"registry already contains run IDs: {duplicates}")
    for record in records:
        append_record(record, str(path))
    return len(records)


def render_report(report: ExperimentReport, console: Console) -> None:
    ft = report.live_release_ft
    residual = (
        "unavailable"
        if ft.max_crooks_residual is None
        else f"{ft.max_crooks_residual:.4f}"
    )
    console.print(
        f"[bold]One-step local-detailed-balance counter-protocol:[/bold] "
        f"{'[green]PASS[/green]' if ft.passed else '[red]FAIL[/red]'} "
        f"Crooks residual={residual}, "
        f"|<exp(-sigma)>-1|={ft.integral_ft_residual:.4f}"
    )
    console.print(
        f"[dim]Scope={ft.scope}; reverse={ft.reverse_transition}; "
        f"predictive-distribution certificate={ft.predictive_distribution_claim}[/dim]"
    )
    tur_status = "PASS" if ft.tur.satisfied else "LIMITATION"
    tur_style = "green" if ft.tur.satisfied else "yellow"
    console.print(
        f"[bold]Classic TUR on exact live finite-binomial moments:[/bold] "
        f"[{tur_style}]{tur_status}[/{tur_style}] "
        f"relative variance={ft.tur.relative_variance:.6f}, "
        f"bound={ft.tur.entropy_bound:.6f}, ratio={ft.tur.bound_ratio:.6f}, "
        f"non-vacuous={ft.tur.nonvacuous}"
    )
    ft_table = Table(title="Live-release Crooks calibration curve")
    ft_table.add_column("Current J", justify="right")
    ft_table.add_column("N(+J)", justify="right")
    ft_table.add_column("N(-J)", justify="right")
    ft_table.add_column("Observed log ratio", justify="right")
    ft_table.add_column("Expected J*A", justify="right")
    ft_table.add_column("Residual", justify="right")
    for point in ft.curve:
        ft_table.add_row(
            str(point.current),
            str(point.positive_count),
            str(point.negative_count),
            f"{point.observed_log_ratio:.4f}",
            f"{point.expected_log_ratio:.4f}",
            f"{point.residual:+.4f}",
        )
    console.print(ft_table)
    table = Table(title="Uncertainty falsification metrics")
    table.add_column("Method")
    table.add_column("ID ECE", justify="right")
    table.add_column("OOD AUROC", justify="right")
    table.add_column("ID accuracy", justify="right")
    table.add_column("Selective AURC", justify="right")
    table.add_column("Risk@80%", justify="right")
    table.add_column("ID/OOD uncertainty", justify="right")
    for name, metrics in report.methods.items():
        table.add_row(
            name,
            f"{metrics.ece:.4f}",
            f"{metrics.ood_auroc:.4f}",
            f"{metrics.id_accuracy:.4f}",
            f"{metrics.selective_aurc:.4f}",
            f"{metrics.selective_risk_at_80_coverage:.4f}",
            f"{metrics.mean_id_uncertainty:.4f}/{metrics.mean_ood_uncertainty:.4f}",
        )
    console.print(table)
    calibration_table = Table(title="ID reliability curves")
    calibration_table.add_column("Method")
    calibration_table.add_column("Confidence bin")
    calibration_table.add_column("Count", justify="right")
    calibration_table.add_column("Mean confidence", justify="right")
    calibration_table.add_column("Accuracy", justify="right")
    calibration_table.add_column("Absolute gap", justify="right")
    for name, metrics in report.methods.items():
        for point in metrics.calibration_curve:
            calibration_table.add_row(
                name,
                f"[{point.lower:.1f}, {point.upper:.1f}]",
                str(point.count),
                f"{point.mean_confidence:.4f}",
                f"{point.accuracy:.4f}",
                f"{point.absolute_gap:.4f}",
            )
    console.print(calibration_table)
    console.print("[dim]Deltas are measurements, not a predeclared win:[/dim]")
    console.print_json(data=report.thermo_deltas)
    report.predictive_thermo_evidence.render(console)


def render_multi_seed_report(report: MultiSeedReport, console: Console) -> None:
    """Render aggregate estimates, paired tests, theory diagnostics, and the verdict."""
    aggregate_table = Table(title=f"Thermo-UQ matched-seed aggregates (n={len(report.seeds)})")
    aggregate_table.add_column("Method")
    aggregate_table.add_column("Metric")
    aggregate_table.add_column("Mean", justify="right")
    aggregate_table.add_column("Student-t 95% CI", justify="right")
    for method, metrics in report.method_aggregates.items():
        for metric, stats in metrics.items():
            aggregate_table.add_row(
                method,
                metric,
                f"{stats.mean:.6f}",
                f"[{stats.ci_low:.6f}, {stats.ci_high:.6f}]",
            )
    console.print(aggregate_table)

    paired_table = Table(title="Thermo-UQ paired deltas (treatment - baseline)")
    paired_table.add_column("Baseline")
    paired_table.add_column("Metric")
    paired_table.add_column("Mean delta", justify="right")
    paired_table.add_column("Bootstrap 95% CI", justify="right")
    paired_table.add_column("paired-t p", justify="right")
    paired_table.add_column("Wilcoxon p", justify="right")
    paired_table.add_column("Favorable", justify="right")
    for baseline, metrics in report.paired_comparisons.items():
        for metric, comparison in metrics.items():
            paired_table.add_row(
                baseline.removeprefix("vs_"),
                metric,
                f"{comparison.mean_delta:+.6f}",
                f"[{comparison.delta_ci_low:+.6f}, {comparison.delta_ci_high:+.6f}]",
                f"{comparison.t_p_value:.4g}",
                f"{comparison.wilcoxon_p_value:.4g}",
                f"{comparison.n_favorable}/{comparison.n_pairs}",
            )
    console.print(paired_table)
    calibration = report.calibration_evaluation
    calibration_style = "green" if calibration.acceptance_met else "yellow"
    console.print(
        f"[bold]u2t.2 calibration/selective-prediction verdict:[/bold] "
        f"[{calibration_style}]{calibration.verdict.upper()}[/{calibration_style}] — "
        f"{calibration.verdict_reason}"
    )
    console.print(
        f"[bold]Live FT pass rate:[/bold] {report.ft_pass_rate:.1%}; "
        f"[bold]classic live TUR:[/bold] non-vacuous={report.live_tur.nonvacuous}, "
        f"satisfied={report.live_tur.satisfied}, ratio={report.live_tur.bound_ratio:.6f}"
    )
    predictive = report.predictive_distribution
    console.print(
        f"[bold]Predictive-distribution claim:[/bold] "
        f"{predictive.predictive_distribution_claim}; mode={predictive.calibration_mode}; "
        f"local seed pass rate={predictive.local_seed_pass_rate:.1%}; "
        f"reasons={list(predictive.refusal_reasons)}"
    )
    verdict_style = "green" if report.verdict == "positive" else (
        "red" if report.verdict == "invalidated" else "yellow"
    )
    console.print(
        f"[bold]Verdict:[/bold] [{verdict_style}]{report.verdict.upper()}[/{verdict_style}] — "
        f"{report.verdict_reason}"
    )
    console.print(f"[dim]Policy: {report.comparison_policy}[/dim]")


def _parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("--seeds needs at least two integers")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("--seeds must be nonnegative")
    return seeds


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        help="Comma-separated matched seeds; enables the 74f.3 statistical verdict path",
    )
    parser.add_argument("--device", default=ExperimentConfig.device)
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument("--mc-samples", type=int, default=ExperimentConfig.mc_samples)
    parser.add_argument(
        "--ft-trajectories", type=int, default=ExperimentConfig.ft_trajectories
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional results-registry JSONL path; only valid with --seeds",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExperimentConfig(
        seed=args.seed,
        device=args.device,
        train_steps=args.train_steps,
        mc_samples=args.mc_samples,
        ft_trajectories=args.ft_trajectories,
    )
    console = Console()
    multi_report: MultiSeedReport | None = None
    if args.seeds is None:
        if args.registry is not None:
            raise ValueError("--registry requires --seeds so the verdict is statistics-backed")
        logger.info(
            "Starting stochastic-thermo falsification seed=%d device=%s train_steps=%d",
            config.seed,
            config.device,
            config.train_steps,
        )
        report: ExperimentReport | MultiSeedReport = run_experiment(config)
        render_report(report, console)
        exit_code = 0 if report.live_release_ft.passed else 1
    else:
        logger.info(
            "Starting stochastic-thermo multi-seed falsification seeds=%s device=%s train_steps=%d",
            args.seeds,
            config.device,
            config.train_steps,
        )
        multi_report = run_multi_seed(
            config,
            args.seeds,
            bootstrap_samples=args.bootstrap_samples,
            alpha=args.alpha,
        )
        report = multi_report
        render_multi_seed_report(report, console)
        exit_code = 0 if all(
            seed_report.live_release_ft.passed for seed_report in report.reports
        ) else 1
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"Structured report: [cyan]{args.output}[/cyan]")
    if args.registry is not None:
        if multi_report is None:
            raise AssertionError("--registry validation did not select the multi-seed path")
        count = append_registry_records(
            multi_report,
            args.registry,
            artifact=str(args.output) if args.output is not None else None,
        )
        console.print(f"Registry records appended: [cyan]{count}[/cyan] -> {args.registry}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
