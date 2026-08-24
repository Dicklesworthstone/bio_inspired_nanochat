"""Stats-backed falsification curve for free-energy deliberation (beads ``r00r.1.4/1.6``).

The live controller branches a bounded model-top-k candidate set, advances each continuation on an
isolated KV/presynaptic cache, relaxes its state, and adds the resulting physical free energy to the
candidate's model energy. This experiment asks a narrow, falsifiable question: after training a tiny
synaptic model on a copy-consistency task, does increasing that deliberation budget improve held-out
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
from bio_inspired_nanochat.deliberation import DeliberationConfig, DeliberationController
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.eval_stats import Aggregate, PairedResult, aggregate, paired_comparison
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
    eval_sequences: int = 8
    learning_rate: float = 3e-3
    temperature: float = 0.8
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 32
    eps: float = 1e-4
    candidate_top_k: int = 8
    candidate_energy_weight: float = 1.0
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
        if self.train_batch_size < 1 or self.eval_sequences < 1 or self.train_steps < 0:
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


def _mode_label(max_iters: int) -> str:
    return "single_step" if max_iters == 0 else f"deliberation_{max_iters}"


def _evaluate_mode(
    model: GPTSynaptic,
    config: ExperimentConfig,
    *,
    seed: int,
    max_iters: int,
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
                )
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
    for seed in config.seeds:
        model = _make_model(config, seed)
        training_loss_by_seed[seed] = _train_model(model, config, seed)
        for max_iters in modes:
            by_mode[max_iters].append(
                _evaluate_mode(model, config, seed=seed, max_iters=max_iters)
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
        bead="bio_inspired_nanochat-r00r.1.6",
        task="held-out copy-continuation consistency",
        mechanism_scope=(
            "bounded model-top-k candidate continuations are advanced on isolated cache branches; "
            "their relaxed free energy is added to the actual decode logits on every generated token"
        ),
        config=config,
        training_loss_by_seed=training_loss_by_seed,
        curve=curve,
        verdict=verdict,
    )


def render_report(report: ExperimentReport, console: Console) -> None:
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
    parser.add_argument("--eval-sequences", type=int, default=ExperimentConfig.eval_sequences)
    parser.add_argument("--output", type=Path, help="Optional strict-JSON report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ExperimentConfig(
        seeds=args.seeds,
        budgets=args.budgets,
        device=args.device,
        train_steps=args.train_steps,
        eval_sequences=args.eval_sequences,
    )
    logger.info(
        "Starting deliberation falsification seeds=%s budgets=%s device=%s train_steps=%d",
        config.seeds,
        config.budgets,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
