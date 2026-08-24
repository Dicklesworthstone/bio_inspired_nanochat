"""Stats-backed falsification curve for free-energy deliberation (beads ``r00r.1.4/1.6/1.7``).

The live controller branches a bounded model-top-k candidate set, advances each continuation on an
isolated KV/presynaptic cache and relaxes its state. A frozen pairwise-rank readout is fitted on a
dedicated calibration split, then combines model energy with branch-local synaptic statistics. This
experiment asks a narrow, falsifiable question: after training a tiny synaptic model on a
copy-consistency task, does increasing that calibrated deliberation budget improve held-out
continuation accuracy at controlled extra effort?

Every budget is evaluated on the same model weights, prompts, and sampling seeds.  The report carries
the full compute/quality curve, Student-t confidence intervals, paired t/Wilcoxon tests, paired
bootstrap intervals, and a predeclared verdict at the largest budget.  ``null`` and ``worse`` are
successful experimental outcomes; the command exits nonzero only for an invalid or failed run.

Run with:

    uv run python -m scripts.e2e.deliberation_compute_curve
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.common import logger
from bio_inspired_nanochat.deliberation import (
    CandidateEnergyBatch,
    CandidateEnergyReadout,
    DeliberationConfig,
    DeliberationController,
)
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.eval_stats import (
    Aggregate,
    PairedResult,
    aggregate,
    paired_comparison,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synthetic_tasks import copy_task
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible controls for the matched-seed compute/quality sweep."""

    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    budgets: tuple[int, ...] = (1, 8, 32)
    device: str = "cpu"
    vocab_size: int = 32
    copy_length: int = 4
    train_batch_size: int = 8
    train_steps: int = 128
    calibration_sequences: int = 24
    eval_sequences: int = 8
    learning_rate: float = 3e-3
    temperature: float = 0.8
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 32
    eps: float = 1e-4
    candidate_top_k: int = 8
    candidate_energy_weight: float = 1.0
    readout_l2: float = 1.0
    calibration_seed_offset: int = 8_888
    evaluation_seed_offset: int = 9_999
    alpha: float = 0.05
    bootstrap_samples: int = 10_000
    min_skill_over_chance: float = 0.10

    def validate(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least two unique values for paired statistics")
        if not self.budgets or any(budget < 1 for budget in self.budgets):
            raise ValueError("budgets must contain positive iteration limits")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be unique and strictly increasing")
        if self.vocab_size < 8 or self.copy_length < 2:
            raise ValueError("vocab_size must be >= 8 and copy_length must be >= 2")
        if (
            self.train_batch_size < 1
            or self.calibration_sequences < 1
            or self.eval_sequences < 1
            or self.train_steps < 0
        ):
            raise ValueError("batch/eval sizes must be positive and train_steps must be non-negative")
        if self.learning_rate <= 0.0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.temperature <= 0.0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.eps <= 0.0 or not np.isfinite(self.eps):
            raise ValueError("eps must be finite and positive")
        if self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be positive")
        if not np.isfinite(self.candidate_energy_weight) or self.candidate_energy_weight < 0.0:
            raise ValueError("candidate_energy_weight must be finite and non-negative")
        if not np.isfinite(self.readout_l2) or self.readout_l2 <= 0.0:
            raise ValueError("readout_l2 must be finite and positive")
        if self.calibration_seed_offset == self.evaluation_seed_offset:
            raise ValueError("calibration and evaluation seed offsets must differ")
        if min(self.calibration_seed_offset, self.evaluation_seed_offset) < self.train_steps:
            raise ValueError("calibration/evaluation seed offsets must not overlap training-step seeds")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 <= self.min_skill_over_chance < 1.0:
            raise ValueError("min_skill_over_chance must be in [0, 1)")


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    token_accuracy: float
    exact_match: float
    mean_effort_per_token: float
    deliberation_coverage: float
    generated_tokens: int
    pondered_tokens: int


@dataclass(frozen=True)
class RankingSeedMetrics:
    """Candidate-ranking accuracy on one seed's fixed, held-out candidate sets."""

    seed: int
    total_groups: int
    eligible_groups: int
    candidate_recall: float
    raw_physical_accuracy: float
    raw_total_accuracy: float
    model_only_accuracy: float
    calibrated_accuracy: float


@dataclass(frozen=True)
class RankingReport:
    per_seed: list[RankingSeedMetrics]
    raw_physical_accuracy: Aggregate
    raw_total_accuracy: Aggregate
    model_only_accuracy: Aggregate
    calibrated_accuracy: Aggregate
    calibrated_vs_raw_physical: PairedResult
    calibrated_vs_raw_total: PairedResult


@dataclass(frozen=True)
class CalibrationSummary:
    seed: int
    calibration_seed: int
    evaluation_seed: int
    calibration_sequences: int
    evaluation_sequences: int
    split_overlap: int
    readouts_by_budget: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class CurvePoint:
    label: str
    max_iters: int
    token_accuracy: Aggregate
    exact_match: Aggregate
    mean_effort_per_token: Aggregate
    deliberation_coverage: Aggregate
    token_accuracy_vs_baseline: PairedResult | None
    exact_match_vs_baseline: PairedResult | None
    per_seed: list[SeedMetrics]


@dataclass(frozen=True)
class Verdict:
    outcome: str
    primary_metric: str
    compared_budget: int
    alpha: float
    reason: str
    honest_null_allowed: bool = True


@dataclass(frozen=True)
class ExperimentReport:
    bead: str
    task: str
    mechanism_scope: str
    config: ExperimentConfig
    training_loss_by_seed: dict[int, list[float]]
    calibration: list[CalibrationSummary]
    ranking: RankingReport
    curve: list[CurvePoint]
    verdict: Verdict

    def to_dict(self) -> dict[str, Any]:
        return _strict_json_value(asdict(self))


def _strict_json_value(value: Any) -> Any:
    """Replace non-finite statistical sentinels (for example an infinite t) with JSON null."""
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class _NoToolTokenizer:
    """Engine adapter whose stop/tool ids live outside the model vocabulary."""

    def __init__(self, vocab_size: int) -> None:
        self._special = {
            "<|python_start|>": vocab_size + 1,
            "<|python_end|>": vocab_size + 2,
            "<|output_start|>": vocab_size + 3,
            "<|output_end|>": vocab_size + 4,
            "<|assistant_end|>": vocab_size + 5,
        }
        self._bos = vocab_size + 6

    def encode_special(self, value: str) -> int:
        return self._special[value]

    def get_bos_token_id(self) -> int:
        return self._bos

    def encode(self, _value: str) -> list[int]:
        return []

    def decode(self, _tokens: Sequence[int]) -> str:
        return ""


@dataclass(frozen=True)
class _CandidateObservation:
    model_logits: np.ndarray
    scores: CandidateEnergyBatch
    correct_mask: np.ndarray


class _CandidateCollector(DeliberationController):
    """Record candidate grids against known calibration/evaluation targets."""

    def __init__(
        self,
        cfg: DeliberationConfig,
        *,
        gold_tokens: Sequence[int],
        candidate_readout: CandidateEnergyReadout | None = None,
    ) -> None:
        super().__init__(cfg, candidate_readout=candidate_readout)
        self._gold_tokens = tuple(int(token) for token in gold_tokens)
        self._candidate_index = 0
        self.observations: list[_CandidateObservation] = []

    def candidate_energy_logits(self, logits, candidate_ids, scores: CandidateEnergyBatch):
        if self._candidate_index >= len(self._gold_tokens):
            raise RuntimeError("candidate observation count exceeded the gold continuation length")
        values = torch.as_tensor(logits)
        ids = torch.as_tensor(candidate_ids, device=values.device)
        selected = values.gather(1, ids).detach().to(dtype=torch.float64).cpu().numpy()
        correct = np.equal(
            ids.detach().cpu().numpy(),
            self._gold_tokens[self._candidate_index],
        )
        self.observations.append(
            _CandidateObservation(
                model_logits=selected,
                scores=scores,
                correct_mask=correct,
            )
        )
        self._candidate_index += 1
        return super().candidate_energy_logits(logits, candidate_ids, scores)


def _split_seed(config: ExperimentConfig, seed: int, *, calibration: bool) -> int:
    offset = config.calibration_seed_offset if calibration else config.evaluation_seed_offset
    return seed * 10_000 + offset


def _split_rows(batch, copy_length: int) -> set[tuple[int, ...]]:
    return {
        tuple(int(token) for token in row)
        for row in batch.inputs[:, :copy_length].tolist()
    }


def _collect_candidate_observations(
    model: GPTSynaptic,
    config: ExperimentConfig,
    *,
    seed: int,
    split_seed: int,
    sequences: int,
    max_iters: int,
) -> tuple[list[_CandidateObservation], set[tuple[int, ...]]]:
    """Collect a fixed model-top-k candidate corpus without applying an energy correction."""
    batch = copy_task(
        batch=sequences,
        length=config.copy_length,
        vocab_size=config.vocab_size,
        seed=split_seed,
    )
    expected = batch.inputs[:, : config.copy_length]
    prompts = batch.inputs[:, : config.copy_length + 1]
    engine = Engine(model, _NoToolTokenizer(config.vocab_size))
    observations: list[_CandidateObservation] = []
    for row in range(sequences):
        controller = _CandidateCollector(
            DeliberationConfig(
                enabled=True,
                eps=config.eps,
                max_iters=max_iters,
                candidate_top_k=config.candidate_top_k,
                # Candidate sets/contexts are model-only, so raw and calibrated ranks are compared
                # on the exact same observations rather than on policy-induced distributions.
                candidate_energy_weight=0.0,
            ),
            gold_tokens=expected[row].tolist(),
        )
        list(engine.generate(
            prompts[row].tolist(),
            max_tokens=config.copy_length,
            temperature=0.0,
            seed=seed * 100_000 + row,
            deliberation=controller,
        ))
        if len(controller.observations) != config.copy_length:
            raise RuntimeError("candidate collector did not observe every continuation token")
        observations.extend(controller.observations)
    return observations, _split_rows(batch, config.copy_length)


def _fit_candidate_readout(
    observations: Sequence[_CandidateObservation],
    *,
    l2: float,
) -> CandidateEnergyReadout:
    if not observations:
        raise ValueError("candidate calibration set is empty")
    first_features = observations[0].scores.features
    if first_features is None:
        raise ValueError("candidate calibration scores do not contain synaptic features")
    feature_names = observations[0].scores.feature_names
    for observation in observations:
        if observation.scores.features is None:
            raise ValueError("candidate calibration scores do not contain synaptic features")
        if observation.scores.feature_names != feature_names:
            raise ValueError("candidate calibration feature schemas differ")
    return CandidateEnergyReadout.fit(
        model_logits=np.concatenate([item.model_logits for item in observations], axis=0),
        synaptic_features=np.concatenate(
            [item.scores.features for item in observations if item.scores.features is not None],
            axis=0,
        ),
        correct_mask=np.concatenate([item.correct_mask for item in observations], axis=0),
        feature_names=feature_names,
        l2=l2,
    )


def _make_model(config: ExperimentConfig, seed: int) -> GPTSynaptic:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GPTSynaptic(
        GPTSynapticConfig(
            sequence_len=2 * config.copy_length + 1,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_head,
            n_embd=config.n_embd,
            dropout=0.0,
            init_seed=seed,
        )
    )
    return model.to(config.device)


def _train_model(model: GPTSynaptic, config: ExperimentConfig, seed: int) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    model.train()
    for step in range(config.train_steps):
        batch = copy_task(
            batch=config.train_batch_size,
            length=config.copy_length,
            vocab_size=config.vocab_size,
            seed=seed * 10_000 + step,
        ).to(config.device)
        model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch.inputs, batch.targets, train_mode=False)
        if loss is None or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss for seed={seed}, step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.train(False)
    return losses


def _snapshot_model_state(model: GPTSynaptic) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _restore_model_state(model: GPTSynaptic, frozen_state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(frozen_state, strict=True)
    model.train(False)
    model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=True)


def _mode_label(max_iters: int) -> str:
    return "single_step" if max_iters == 0 else f"deliberation_{max_iters}"


def _evaluate_mode(
    model: GPTSynaptic,
    config: ExperimentConfig,
    *,
    seed: int,
    max_iters: int,
    candidate_readout: CandidateEnergyReadout | None = None,
) -> SeedMetrics:
    """Evaluate one matched seed/budget on held-out copy continuations."""
    batch = copy_task(
        batch=config.eval_sequences,
        length=config.copy_length,
        vocab_size=config.vocab_size,
        seed=seed * 10_000 + 9_999,
    )
    expected = batch.inputs[:, : config.copy_length]
    prompts = batch.inputs[:, : config.copy_length + 1]
    engine = Engine(model, _NoToolTokenizer(config.vocab_size))
    correct_tokens = 0
    exact_sequences = 0
    generated_tokens = 0
    effort = 0
    pondered_tokens = 0
    for row in range(config.eval_sequences):
        controller = None
        if max_iters > 0:
            controller = DeliberationController(
                DeliberationConfig(
                    enabled=True,
                    eps=config.eps,
                    max_iters=max_iters,
                    candidate_top_k=config.candidate_top_k,
                    candidate_energy_weight=config.candidate_energy_weight,
                ),
                candidate_readout=candidate_readout,
            )
        generated: list[int] = []
        stream = engine.generate(
            prompts[row].tolist(),
            max_tokens=config.copy_length,
            temperature=config.temperature,
            seed=seed * 100_000 + row,
            deliberation=controller,
        )
        for token_column, _token_masks in stream:
            generated.append(int(token_column[0]))
        if len(generated) != config.copy_length:
            raise RuntimeError(
                f"generation length mismatch for seed={seed}, budget={max_iters}: "
                f"expected {config.copy_length}, got {len(generated)}"
            )
        target = expected[row].tolist()
        matches = [prediction == truth for prediction, truth in zip(generated, target, strict=True)]
        correct_tokens += sum(matches)
        exact_sequences += int(all(matches))
        generated_tokens += len(generated)
        if controller is not None:
            effort += sum(record.total_effort for record in controller.records)
            pondered_tokens += len(controller.records)
    return SeedMetrics(
        seed=seed,
        token_accuracy=correct_tokens / generated_tokens,
        exact_match=exact_sequences / config.eval_sequences,
        mean_effort_per_token=effort / generated_tokens,
        deliberation_coverage=pondered_tokens / generated_tokens,
        generated_tokens=generated_tokens,
        pondered_tokens=pondered_tokens,
    )


def _ranking_metrics(
    *,
    seed: int,
    observations: Sequence[_CandidateObservation],
    readout: CandidateEnergyReadout,
    raw_weight: float,
) -> RankingSeedMetrics:
    correct_raw_physical = 0
    correct_raw_total = 0
    correct_model_only = 0
    correct_calibrated = 0
    eligible = 0
    for observation in observations:
        correct = observation.correct_mask.reshape(-1)
        if int(correct.sum()) != 1:
            continue
        eligible += 1
        logits = observation.model_logits.reshape(-1)
        raw_physical = observation.scores.F_final.reshape(-1)
        raw_total = -logits + raw_weight * raw_physical
        calibrated = readout.energy(observation.model_logits, observation.scores).reshape(-1)
        correct_raw_physical += int(correct[int(np.argmin(raw_physical))])
        correct_raw_total += int(correct[int(np.argmin(raw_total))])
        correct_model_only += int(correct[int(np.argmax(logits))])
        correct_calibrated += int(correct[int(np.argmin(calibrated))])
    if eligible == 0:
        raise RuntimeError(f"no held-out gold continuations were present in top-k for seed={seed}")
    return RankingSeedMetrics(
        seed=seed,
        total_groups=len(observations),
        eligible_groups=eligible,
        candidate_recall=eligible / len(observations),
        raw_physical_accuracy=correct_raw_physical / eligible,
        raw_total_accuracy=correct_raw_total / eligible,
        model_only_accuracy=correct_model_only / eligible,
        calibrated_accuracy=correct_calibrated / eligible,
    )


def _ranking_report(
    per_seed: list[RankingSeedMetrics],
    config: ExperimentConfig,
) -> RankingReport:
    calibrated = {item.seed: item.calibrated_accuracy for item in per_seed}
    raw_physical = {item.seed: item.raw_physical_accuracy for item in per_seed}
    raw_total = {item.seed: item.raw_total_accuracy for item in per_seed}
    calibrated_vs_raw_physical = paired_comparison(
        calibrated,
        raw_physical,
        lower_is_better=False,
        n_boot=config.bootstrap_samples,
        seed=config.seeds[0] + 20_001,
    )
    calibrated_vs_raw_total = paired_comparison(
        calibrated,
        raw_total,
        lower_is_better=False,
        n_boot=config.bootstrap_samples,
        seed=config.seeds[0] + 20_002,
    )
    if calibrated_vs_raw_physical is None or calibrated_vs_raw_total is None:
        raise RuntimeError("candidate-ranking report requires at least two matched seeds")
    return RankingReport(
        per_seed=per_seed,
        raw_physical_accuracy=aggregate(list(raw_physical.values())),
        raw_total_accuracy=aggregate(list(raw_total.values())),
        model_only_accuracy=aggregate([item.model_only_accuracy for item in per_seed]),
        calibrated_accuracy=aggregate(list(calibrated.values())),
        calibrated_vs_raw_physical=calibrated_vs_raw_physical,
        calibrated_vs_raw_total=calibrated_vs_raw_total,
    )


def classify_verdict(
    comparison: PairedResult | None,
    *,
    budget: int,
    alpha: float,
    baseline_accuracy: float,
    chance_accuracy: float,
    min_skill_over_chance: float,
) -> Verdict:
    """Predeclared two-sided verdict for the largest budget's token accuracy."""
    minimum_valid_accuracy = chance_accuracy + min_skill_over_chance
    if baseline_accuracy < minimum_valid_accuracy:
        return Verdict(
            outcome="inconclusive",
            primary_metric="token_accuracy",
            compared_budget=budget,
            alpha=alpha,
            reason=(
                f"baseline accuracy {baseline_accuracy:.4f} is below the predeclared skill floor "
                f"{minimum_valid_accuracy:.4f}; the task was not learned well enough to test deliberation"
            ),
        )
    if comparison is None:
        return Verdict(
            outcome="null",
            primary_metric="token_accuracy",
            compared_budget=budget,
            alpha=alpha,
            reason="fewer than two matched seeds; no paired verdict is possible",
        )
    significant = comparison.t_p_value < alpha
    if significant and comparison.mean_delta > 0.0 and comparison.delta_ci_low > 0.0:
        outcome = "improved"
        reason = "paired t-test and bootstrap interval support a positive accuracy delta"
    elif significant and comparison.mean_delta < 0.0 and comparison.delta_ci_high < 0.0:
        outcome = "worse"
        reason = "paired t-test and bootstrap interval support a negative accuracy delta"
    else:
        outcome = "null"
        reason = "the paired test is not significant or the bootstrap interval includes zero"
    return Verdict(
        outcome=outcome,
        primary_metric="token_accuracy",
        compared_budget=budget,
        alpha=alpha,
        reason=reason,
    )


def _curve_point(
    label: str,
    max_iters: int,
    values: list[SeedMetrics],
    baseline: list[SeedMetrics],
    config: ExperimentConfig,
) -> CurvePoint:
    token_by_seed = {item.seed: item.token_accuracy for item in values}
    exact_by_seed = {item.seed: item.exact_match for item in values}
    baseline_token = {item.seed: item.token_accuracy for item in baseline}
    baseline_exact = {item.seed: item.exact_match for item in baseline}
    token_comparison = None
    exact_comparison = None
    if max_iters > 0:
        token_comparison = paired_comparison(
            token_by_seed,
            baseline_token,
            lower_is_better=False,
            n_boot=config.bootstrap_samples,
            seed=config.seeds[0] + max_iters,
        )
        exact_comparison = paired_comparison(
            exact_by_seed,
            baseline_exact,
            lower_is_better=False,
            n_boot=config.bootstrap_samples,
            seed=config.seeds[0] + 10_000 + max_iters,
        )
    return CurvePoint(
        label=label,
        max_iters=max_iters,
        token_accuracy=aggregate([item.token_accuracy for item in values]),
        exact_match=aggregate([item.exact_match for item in values]),
        mean_effort_per_token=aggregate([item.mean_effort_per_token for item in values]),
        deliberation_coverage=aggregate([item.deliberation_coverage for item in values]),
        token_accuracy_vs_baseline=token_comparison,
        exact_match_vs_baseline=exact_comparison,
        per_seed=values,
    )


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentReport:
    """Train matched tiny models and return a stats-backed compute/quality curve."""
    if config is None:
        config = ExperimentConfig()
    config.validate()
    modes = (0, *config.budgets)
    by_mode: dict[int, list[SeedMetrics]] = {mode: [] for mode in modes}
    training_loss_by_seed: dict[int, list[float]] = {}
    calibration_summaries: list[CalibrationSummary] = []
    ranking_by_seed: list[RankingSeedMetrics] = []
    for seed in config.seeds:
        model = _make_model(config, seed)
        training_loss_by_seed[seed] = _train_model(model, config, seed)
        frozen_state = _snapshot_model_state(model)

        calibration_seed = _split_seed(config, seed, calibration=True)
        evaluation_seed = _split_seed(config, seed, calibration=False)
        readouts_by_mode: dict[int, CandidateEnergyReadout] = {}
        calibration_rows: set[tuple[int, ...]] | None = None
        for max_iters in config.budgets:
            _restore_model_state(model, frozen_state)
            calibration_observations, rows = _collect_candidate_observations(
                model,
                config,
                seed=seed,
                split_seed=calibration_seed,
                sequences=config.calibration_sequences,
                max_iters=max_iters,
            )
            calibration_rows = rows
            readouts_by_mode[max_iters] = _fit_candidate_readout(
                calibration_observations,
                l2=config.readout_l2,
            )

        _restore_model_state(model, frozen_state)
        ranking_observations, evaluation_rows = _collect_candidate_observations(
            model,
            config,
            seed=seed,
            split_seed=evaluation_seed,
            sequences=config.eval_sequences,
            max_iters=config.budgets[-1],
        )
        if calibration_rows is None:
            raise RuntimeError("no calibration rows were collected")
        split_overlap = len(calibration_rows & evaluation_rows)
        if split_overlap:
            raise RuntimeError(
                f"calibration/evaluation leakage for seed={seed}: {split_overlap} duplicate rows"
            )
        ranking_by_seed.append(_ranking_metrics(
            seed=seed,
            observations=ranking_observations,
            readout=readouts_by_mode[config.budgets[-1]],
            raw_weight=config.candidate_energy_weight,
        ))
        calibration_summaries.append(CalibrationSummary(
            seed=seed,
            calibration_seed=calibration_seed,
            evaluation_seed=evaluation_seed,
            calibration_sequences=config.calibration_sequences,
            evaluation_sequences=config.eval_sequences,
            split_overlap=split_overlap,
            readouts_by_budget={
                budget: readout.to_dict()
                for budget, readout in readouts_by_mode.items()
            },
        ))
        for max_iters in modes:
            _restore_model_state(model, frozen_state)
            by_mode[max_iters].append(
                _evaluate_mode(
                    model,
                    config,
                    seed=seed,
                    max_iters=max_iters,
                    candidate_readout=readouts_by_mode.get(max_iters),
                )
            )
    baseline = by_mode[0]
    curve = [
        _curve_point(_mode_label(max_iters), max_iters, by_mode[max_iters], baseline, config)
        for max_iters in modes
    ]
    highest = curve[-1]
    verdict = classify_verdict(
        highest.token_accuracy_vs_baseline,
        budget=highest.max_iters,
        alpha=config.alpha,
        baseline_accuracy=curve[0].token_accuracy.mean,
        chance_accuracy=1.0 / config.vocab_size,
        min_skill_over_chance=config.min_skill_over_chance,
    )
    return ExperimentReport(
        bead="bio_inspired_nanochat-r00r.1.7",
        task="held-out copy-continuation consistency",
        mechanism_scope=(
            "bounded model-top-k candidate continuations are advanced on isolated cache branches; "
            "a frozen pairwise-rank readout calibrated on disjoint sequences maps model energy plus "
            "branch-local synaptic statistics into the actual decode logits on every generated token"
        ),
        config=config,
        training_loss_by_seed=training_loss_by_seed,
        calibration=calibration_summaries,
        ranking=_ranking_report(ranking_by_seed, config),
        curve=curve,
        verdict=verdict,
    )


def render_report(report: ExperimentReport, console: Console) -> None:
    ranking_table = Table(title="Held-out candidate-energy ranking")
    ranking_table.add_column("Seed")
    ranking_table.add_column("Gold in top-k", justify="right")
    ranking_table.add_column("Raw F", justify="right")
    ranking_table.add_column("Raw total", justify="right")
    ranking_table.add_column("Model only", justify="right")
    ranking_table.add_column("Calibrated", justify="right")
    for item in report.ranking.per_seed:
        ranking_table.add_row(
            str(item.seed),
            f"{item.candidate_recall:.1%}",
            f"{item.raw_physical_accuracy:.4f}",
            f"{item.raw_total_accuracy:.4f}",
            f"{item.model_only_accuracy:.4f}",
            f"{item.calibrated_accuracy:.4f}",
        )
    ranking_table.add_row(
        "mean",
        "—",
        f"{report.ranking.raw_physical_accuracy.mean:.4f}",
        f"{report.ranking.raw_total_accuracy.mean:.4f}",
        f"{report.ranking.model_only_accuracy.mean:.4f}",
        f"{report.ranking.calibrated_accuracy.mean:.4f}",
    )
    console.print(ranking_table)
    comparison = report.ranking.calibrated_vs_raw_physical
    console.print(
        "[dim]Calibrated − raw F: "
        f"{comparison.mean_delta:+.4f} "
        f"[{comparison.delta_ci_low:+.4f}, {comparison.delta_ci_high:+.4f}], "
        f"paired t p={comparison.t_p_value:.4g}[/dim]"
    )

    table = Table(title="Free-energy deliberation compute/quality curve")
    table.add_column("Mode")
    table.add_column("Effort/token", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Token accuracy (95% CI)", justify="right")
    table.add_column("Exact match (95% CI)", justify="right")
    table.add_column("Δ token accuracy", justify="right")
    table.add_column("paired p", justify="right")
    for point in report.curve:
        comparison = point.token_accuracy_vs_baseline
        delta = (
            "baseline"
            if comparison is None
            else (
                f"{comparison.mean_delta:+.4f} "
                f"[{comparison.delta_ci_low:+.4f}, {comparison.delta_ci_high:+.4f}]"
            )
        )
        p_value = (
            "—"
            if comparison is None
            else f"t={comparison.t_p_value:.4f} / W={comparison.wilcoxon_p_value:.4f}"
        )
        table.add_row(
            point.label,
            f"{point.mean_effort_per_token.mean:.2f}",
            f"{point.deliberation_coverage.mean:.1%}",
            f"{point.token_accuracy.mean:.4f} "
            f"[{point.token_accuracy.ci_low:.4f}, {point.token_accuracy.ci_high:.4f}]",
            f"{point.exact_match.mean:.4f} "
            f"[{point.exact_match.ci_low:.4f}, {point.exact_match.ci_high:.4f}]",
            delta,
            p_value,
        )
    console.print(table)
    color = {
        "improved": "green",
        "null": "yellow",
        "worse": "red",
        "inconclusive": "cyan",
    }[report.verdict.outcome]
    console.print(
        f"[bold]Verdict at max_iters={report.verdict.compared_budget}:[/bold] "
        f"[{color}]{report.verdict.outcome.upper()}[/{color}] — {report.verdict.reason}"
    )
    console.print(f"[dim]Scope: {report.mechanism_scope}[/dim]")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_int_tuple, default=ExperimentConfig.seeds)
    parser.add_argument("--budgets", type=_parse_int_tuple, default=ExperimentConfig.budgets)
    parser.add_argument("--device", default=ExperimentConfig.device)
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument(
        "--calibration-sequences",
        type=int,
        default=ExperimentConfig.calibration_sequences,
    )
    parser.add_argument("--eval-sequences", type=int, default=ExperimentConfig.eval_sequences)
    parser.add_argument("--readout-l2", type=float, default=ExperimentConfig.readout_l2)
    parser.add_argument("--output", type=Path, help="Optional strict-JSON report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExperimentConfig(
        seeds=args.seeds,
        budgets=args.budgets,
        device=args.device,
        train_steps=args.train_steps,
        calibration_sequences=args.calibration_sequences,
        eval_sequences=args.eval_sequences,
        readout_l2=args.readout_l2,
    )
    logger.info(
        "Starting deliberation falsification seeds=%s budgets=%s device=%s train_steps=%d calibration=%d",
        config.seeds,
        config.budgets,
        config.device,
        config.train_steps,
        config.calibration_sequences,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
