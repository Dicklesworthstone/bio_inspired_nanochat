"""Calibrate the MVP sheaf obstruction on held-out binding inconsistencies.

This CPU-only harness builds a deterministic labeled corpus of locally
consistent token graphs and graphs with one corrupted binding.  It fits the
probability map and false-positive-controlled threshold on one split, evaluates
on a disjoint split, writes a reliability SVG, and records structured JSONL
evidence.  It does not claim natural-language hallucination validity or H¹
certification; those remain the explicit scope of ``r00r.5.3`` and Thrust G.

Run with:

    uv run python -m scripts.e2e.sheaf_obstruction_calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.sheaf_obstruction import (
    MVP_CERTIFICATE_KIND,
    CalibrationEvaluation,
    fit_obstruction_calibrator,
    measure_sheaf_obstruction,
    reliability_diagram_svg,
)
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class SheafCalibrationConfig:
    seed: int = 51051
    calibration_examples: int = 256
    evaluation_examples: int = 256
    nodes: int = 6
    stalk_dim: int = 8
    consistent_noise: float = 0.025
    corruption_min: float = 0.8
    corruption_max: float = 2.0
    target_false_positive_rate: float = 0.1
    probability_bins: int = 12
    reliability_bins: int = 10

    def validate(self) -> None:
        if self.calibration_examples < 20 or self.evaluation_examples < 20:
            raise ValueError("calibration and evaluation splits must each contain >= 20 examples")
        if self.calibration_examples % 2 or self.evaluation_examples % 2:
            raise ValueError("split sizes must be even so labels are balanced exactly")
        if self.nodes < 3 or self.stalk_dim < 2:
            raise ValueError("synthetic binding graphs require >=3 nodes and stalk_dim >=2")
        if not math.isfinite(self.consistent_noise) or self.consistent_noise < 0.0:
            raise ValueError("consistent_noise must be finite and non-negative")
        if not 0.0 < self.corruption_min <= self.corruption_max:
            raise ValueError("corruption bounds must satisfy 0 < min <= max")
        if not 0.0 < self.target_false_positive_rate < 1.0:
            raise ValueError("target_false_positive_rate must be in (0,1)")
        if self.probability_bins < 1 or self.reliability_bins < 1:
            raise ValueError("probability_bins and reliability_bins must be positive")


@dataclass(frozen=True)
class SheafCalibrationReport:
    run_id: str
    calibration: dict[str, Any]
    evaluation: CalibrationEvaluation
    calibration_score_summary: dict[str, float]
    evaluation_score_summary: dict[str, float]
    summary_path: str
    reliability_path: str
    events_path: str
    passed: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "calibration": self.calibration,
            "evaluation": self.evaluation.to_dict(),
            "calibration_score_summary": self.calibration_score_summary,
            "evaluation_score_summary": self.evaluation_score_summary,
            "summary_path": self.summary_path,
            "reliability_path": self.reliability_path,
            "events_path": self.events_path,
            "limitations": list(self.limitations),
        }

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                "sheaf obstruction calibration failed its predeclared synthetic MVP gates: "
                + json.dumps(self.to_dict(), sort_keys=True)
            )


def run_sheaf_calibration(
    config: SheafCalibrationConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    verbose: bool = True,
) -> SheafCalibrationReport:
    """Fit on one deterministic split and report disjoint-split reliability."""
    if config is None:
        config = SheafCalibrationConfig()
    config.validate()
    run_id = _run_id(config)
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs") / "e2e" / "sheaf_obstruction_calibration" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_scores, calibration_labels = _labeled_scores(
        config,
        seed=config.seed,
        examples=config.calibration_examples,
    )
    evaluation_scores, evaluation_labels = _labeled_scores(
        config,
        seed=config.seed + 1,
        examples=config.evaluation_examples,
    )
    calibrator = fit_obstruction_calibrator(
        calibration_scores,
        calibration_labels,
        target_false_positive_rate=config.target_false_positive_rate,
        probability_bins=config.probability_bins,
    )
    evaluation = calibrator.evaluate(
        evaluation_scores,
        evaluation_labels,
        reliability_bins=config.reliability_bins,
    )
    calibration_summary = _score_summary(calibration_scores, calibration_labels)
    evaluation_summary = _score_summary(evaluation_scores, evaluation_labels)

    # The threshold's calibration FPR is the guaranteed design target.  The
    # disjoint synthetic split gets a small predeclared sampling margin rather
    # than being silently tuned after seeing its labels.
    passed = (
        math.isfinite(calibrator.threshold)
        and calibrator.calibration_false_positive_rate
        <= config.target_false_positive_rate + 1e-12
        and evaluation.false_positive_rate
        <= config.target_false_positive_rate + 0.05
        and evaluation.true_positive_rate >= 0.9
        and 0.0 <= evaluation.expected_calibration_error <= 0.2
    )
    limitations = (
        "Labels are synthetic binding corruptions, not natural-language hallucinations.",
        "The score is a fixed sheaf-Laplacian residual MVP, not an H^1 certificate.",
        "AUROC/baseline comparison and real faithfulness evaluation remain r00r.5.3 scope.",
    )

    reliability_path = output_dir / "reliability.svg"
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.jsonl"
    reliability_path.write_text(reliability_diagram_svg(evaluation), encoding="utf-8")

    with RunLogger(
        output_dir,
        name="sheaf_obstruction_calibration",
        run_id=run_id,
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-r00r.5.1",
            "config": asdict(config),
            "certificate_kind": MVP_CERTIFICATE_KIND,
            "h1_certified": False,
        },
    ) as logger:
        logger.event(
            "sheaf_obstruction_calibration_split",
            split="calibration",
            scores=calibration_scores,
            labels=calibration_labels,
            score_summary=calibration_summary,
        )
        logger.event(
            "sheaf_obstruction_calibration_fit",
            calibration=calibrator.to_dict(),
        )
        logger.event(
            "sheaf_obstruction_calibration_split",
            split="evaluation",
            scores=evaluation_scores,
            labels=evaluation_labels,
            score_summary=evaluation_summary,
        )
        logger.event(
            "sheaf_obstruction_reliability",
            evaluation=evaluation.to_dict(),
            reliability_path=str(reliability_path),
        )
        logger.event(
            "sheaf_obstruction_calibration_verdict",
            passed=passed,
            limitations=limitations,
        )

    report = SheafCalibrationReport(
        run_id=run_id,
        calibration=calibrator.to_dict(),
        evaluation=evaluation,
        calibration_score_summary=calibration_summary,
        evaluation_score_summary=evaluation_summary,
        summary_path=str(summary_path),
        reliability_path=str(reliability_path),
        events_path=str(events_path),
        passed=passed,
        limitations=limitations,
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bead": "bio_inspired_nanochat-r00r.5.1",
                "config": asdict(config),
                **report.to_dict(),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if verbose:
        _render_report(report)
    return report


def _labeled_scores(
    config: SheafCalibrationConfig,
    *,
    seed: int,
    examples: int,
) -> tuple[list[float], list[int]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tail = torch.arange(config.nodes, dtype=torch.long)
    head = torch.roll(tail, shifts=-1)
    edge_index = torch.stack((tail, head))
    scores: list[float] = []
    labels: list[int] = []

    for index in range(examples):
        inconsistent = bool(index % 2)
        latent = torch.randn(config.stalk_dim, generator=generator)
        latent = latent / latent.norm().clamp_min(1e-8)
        stalks = latent.expand(config.nodes, -1).clone()
        stalks.add_(
            config.consistent_noise
            * torch.randn(
                config.nodes,
                config.stalk_dim,
                generator=generator,
            )
        )
        if inconsistent:
            direction = torch.randn(config.stalk_dim, generator=generator)
            direction = direction / direction.norm().clamp_min(1e-8)
            severity = config.corruption_min + (
                config.corruption_max - config.corruption_min
            ) * float(torch.rand((), generator=generator).item())
            stalks[index % config.nodes].add_(severity * direction)

        result = measure_sheaf_obstruction(stalks, edge_index)
        if not result.available:
            raise RuntimeError(f"synthetic ring unexpectedly unavailable: {result.fallback_reason}")
        scores.append(result.score)
        labels.append(int(inconsistent))
    return scores, labels


def _score_summary(scores: list[float], labels: list[int]) -> dict[str, float]:
    negative = [score for score, label in zip(scores, labels) if not label]
    positive = [score for score, label in zip(scores, labels) if label]
    return {
        "consistent_mean": sum(negative) / len(negative),
        "consistent_max": max(negative),
        "inconsistent_mean": sum(positive) / len(positive),
        "inconsistent_min": min(positive),
    }


def _run_id(config: SheafCalibrationConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return f"sheaf-obstruction-{digest}-{uuid.uuid4().hex[:8]}"


def _render_report(report: SheafCalibrationReport) -> None:
    console = Console()
    table = Table(title=f"Sheaf obstruction calibration — {report.run_id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("threshold", str(report.calibration["threshold"]))
    table.add_row(
        "calibration FPR",
        f"{float(report.calibration['calibration_false_positive_rate']):.4f}",
    )
    table.add_row("held-out FPR", f"{report.evaluation.false_positive_rate:.4f}")
    table.add_row("held-out TPR", f"{report.evaluation.true_positive_rate:.4f}")
    table.add_row("held-out ECE", f"{report.evaluation.expected_calibration_error:.4f}")
    table.add_row("verdict", "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]")
    console.print(table)
    console.print(f"Reliability diagram: [cyan]{report.reliability_path}[/cyan]")
    for limitation in report.limitations:
        console.print(f"[yellow]Limitation:[/yellow] {limitation}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the MVP sheaf obstruction with held-out reliability evidence"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SheafCalibrationConfig.seed)
    parser.add_argument(
        "--calibration-examples",
        type=int,
        default=SheafCalibrationConfig.calibration_examples,
    )
    parser.add_argument(
        "--evaluation-examples",
        type=int,
        default=SheafCalibrationConfig.evaluation_examples,
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=SheafCalibrationConfig.target_false_positive_rate,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SheafCalibrationConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        evaluation_examples=args.evaluation_examples,
        target_false_positive_rate=args.target_fpr,
    )
    report = run_sheaf_calibration(config, run_dir=args.run_dir)
    report.assert_passed()


if __name__ == "__main__":
    main()
