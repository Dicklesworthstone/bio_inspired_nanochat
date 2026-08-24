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
   curves, and thermo-UQ deltas against both baselines.  No improvement is assumed:
   negative results are first-class output.

Run with:

    uv run python -m scripts.e2e.stochastic_thermo_uq
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.common import logger
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.mc_ensemble import mc_sampling
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
class LiveReleaseFTResult:
    scope: str
    reverse_transition: str
    predictive_distribution_claim: bool
    passed: bool
    n_trajectories: int
    pool_size: int
    forward_drive: float
    reverse_drive: float
    forward_probability: float
    reverse_probability: float
    affinity: float
    integral_ft: float
    integral_ft_residual: float
    max_crooks_residual: float | None
    tolerance: float
    integral_tolerance: float
    curve: list[CrooksPoint]


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    absolute_gap: float


@dataclass(frozen=True)
class MethodMetrics:
    ece: float
    ood_auroc: float
    id_accuracy: float
    mean_id_confidence: float
    mean_id_uncertainty: float
    mean_ood_uncertainty: float
    calibration_curve: list[CalibrationBin]


@dataclass(frozen=True)
class ExperimentReport:
    bead: str
    config: ExperimentConfig
    live_release_ft: LiveReleaseFTResult
    methods: dict[str, MethodMetrics]
    thermo_deltas: dict[str, dict[str, float]]
    training_loss: list[float]
    comparison_policy: str = field(
        default=(
            "Negative ECE delta and positive OOD-AUROC delta favor thermo-UQ; "
            "the harness reports measurements and does not assert an advantage."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Prediction:
    probabilities: Tensor
    uncertainty: Tensor


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
    forward = _sample_live_counts(
        presyn,
        release_cfg,
        drive=forward_drive,
        n_trajectories=config.ft_trajectories,
        seed=config.seed + 101,
    )
    reverse = _sample_live_counts(
        presyn,
        release_cfg,
        drive=reverse_drive,
        n_trajectories=config.ft_trajectories,
        seed=config.seed + 102,
    )
    currents = forward - reverse
    affinity = math.log(
        forward_probability
        * (1.0 - reverse_probability)
        / (reverse_probability * (1.0 - forward_probability))
    )
    sigma = currents.astype(np.float64) * affinity
    integral_ft = float(np.mean(np.exp(-sigma)))
    integral_residual = abs(integral_ft - 1.0)
    pool_size = int(round(release_cfg.init_rrp))
    curve = binomial_crooks_curve(
        currents,
        affinity,
        pool_size=pool_size,
        min_count=config.ft_min_count,
    )
    max_residual = max((abs(point.residual) for point in curve), default=None)
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
        n_trajectories=config.ft_trajectories,
        pool_size=pool_size,
        forward_drive=forward_drive,
        reverse_drive=reverse_drive,
        forward_probability=forward_probability,
        reverse_probability=reverse_probability,
        affinity=affinity,
        integral_ft=integral_ft,
        integral_ft_residual=integral_residual,
        max_crooks_residual=max_residual,
        tolerance=config.ft_tolerance,
        integral_tolerance=config.ft_integral_tolerance,
        curve=curve,
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


@torch.no_grad()
def _softmax_prediction(model: GPTSynaptic, inputs: Tensor) -> _Prediction:
    model.eval()
    _reset_sequence(model)
    logits, _ = model(inputs, train_mode=False)
    probabilities = torch.softmax(logits.float(), dim=-1)
    return _Prediction(probabilities, _entropy(probabilities))


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
    with _dropout_sampling(model):
        for _ in range(n_samples):
            _reset_sequence(model)
            logits, _ = model(inputs, train_mode=False)
            probabilities = torch.softmax(logits.float(), dim=-1)
            probability_sum = (
                probabilities if probability_sum is None else probability_sum + probabilities
            )
    if probability_sum is None:
        raise AssertionError("n_samples validation should make the MC loop non-empty")
    mean_probabilities = probability_sum / n_samples
    return _Prediction(mean_probabilities, _entropy(mean_probabilities))


def _thermo_prediction(
    model: GPTSynaptic,
    inputs: Tensor,
    *,
    n_samples: int,
) -> _Prediction:
    probability_sum = None
    model.eval()
    with torch.no_grad(), mc_sampling(model):
        for _ in range(n_samples):
            _reset_sequence(model)
            logits, _ = model(inputs, train_mode=False)
            probabilities = torch.softmax(logits.float(), dim=-1)
            probability_sum = (
                probabilities if probability_sum is None else probability_sum + probabilities
            )
    if probability_sum is None:
        raise AssertionError("n_samples validation should make the MC loop non-empty")
    mean_probabilities = probability_sum / n_samples
    return _Prediction(mean_probabilities, _entropy(mean_probabilities))


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
    id_confidence, id_class = id_prediction.probabilities.max(dim=-1)
    return MethodMetrics(
        ece=ece,
        ood_auroc=binary_auroc(id_prediction.uncertainty, ood_prediction.uncertainty),
        id_accuracy=float(id_class.eq(id_targets).float().mean().item()),
        mean_id_confidence=float(id_confidence.mean().item()),
        mean_id_uncertainty=float(id_prediction.uncertainty.mean().item()),
        mean_ood_uncertainty=float(ood_prediction.uncertainty.mean().item()),
        calibration_curve=curve,
    )


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> ExperimentReport:
    """Run the FT check and all three uncertainty methods on identical model weights."""
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
    thermo_id = _thermo_prediction(model, id_inputs, n_samples=config.mc_samples)
    torch.manual_seed(config.seed + 302)
    thermo_ood = _thermo_prediction(model, ood_inputs, n_samples=config.mc_samples)
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
    )


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
    table.add_column("ID/OOD uncertainty", justify="right")
    for name, metrics in report.methods.items():
        table.add_row(
            name,
            f"{metrics.ece:.4f}",
            f"{metrics.ood_auroc:.4f}",
            f"{metrics.id_accuracy:.4f}",
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--device", default=ExperimentConfig.device)
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument("--mc-samples", type=int, default=ExperimentConfig.mc_samples)
    parser.add_argument(
        "--ft-trajectories", type=int, default=ExperimentConfig.ft_trajectories
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
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
    logger.info(
        "Starting stochastic-thermo falsification seed=%d device=%s train_steps=%d",
        config.seed,
        config.device,
        config.train_steps,
    )
    report = run_experiment(config)
    console = Console()
    render_report(report, console)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"Structured report: [cyan]{args.output}[/cyan]")
    return 0 if report.live_release_ft.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
