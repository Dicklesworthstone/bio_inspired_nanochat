"""Dedicated uncertainty/calibration E2E with detailed evidence (bead ``eqyk.20``).

This wrapper runs the real stochastic-release MC, MC-dropout, and softmax-entropy paths from
``stochastic_thermo_uq`` on identical model weights.  It adds the testing-epic contract around that
experiment: complete predictive distributions, per-prediction MC variance, calibration and
risk-coverage curves, fluctuation-theorem residuals, an uncertainty-decoding action, strict
invariants, and a machine-readable validation-report fragment.

Run with:

    uv run python -m scripts.e2e.uncertainty_calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.adaptive_compute import (
    UncertaintyDecodingConfig,
    uncertainty_decode_action,
)
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.run_logging import RunLogger
from scripts.e2e.stochastic_thermo_uq import (
    ExperimentConfig,
    ExperimentReport,
    _Prediction,
    run_experiment,
)


@dataclass(frozen=True)
class UncertaintyE2EConfig:
    """Configuration for one real, CPU-friendly uncertainty pipeline run."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    target_coverage: float = 0.8

    def validate(self) -> None:
        self.experiment.validate()
        if not math.isfinite(self.target_coverage) or not 0.0 < self.target_coverage < 1.0:
            raise ValueError(
                "target_coverage must be finite and strictly between zero and one, got "
                f"{self.target_coverage!r}"
            )


@dataclass(frozen=True)
class UncertaintyE2EReport:
    """Strict outcome consumed by pytest, nightly artifacts, and ``eqyk.16``."""

    run_id: str
    invariants: list[InvariantResult]
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.invariants)

    @property
    def failures(self) -> list[InvariantResult]:
        return [result for result in self.invariants if not result.passed]

    def assert_passed(self) -> None:
        if self.passed:
            return
        lines = "\n".join(result.line() for result in self.failures)
        raise AssertionError(f"uncertainty/calibration e2e FAILED:\n{lines}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "invariants": [asdict(result) for result in self.invariants],
            "summary": self.summary,
        }


def _run_id(config: UncertaintyE2EConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"uncertainty-calibration-{digest}-{uuid.uuid4().hex[:8]}"


def _method_summary(report: ExperimentReport) -> dict[str, dict[str, float]]:
    return {
        method: {
            "ece": metrics.ece,
            "ood_auroc": metrics.ood_auroc,
            "id_accuracy": metrics.id_accuracy,
            "selective_aurc": metrics.selective_aurc,
            "selective_risk_at_80_coverage": metrics.selective_risk_at_80_coverage,
        }
        for method, metrics in report.methods.items()
    }


def _write_validation_fragment(
    path: Path,
    report: UncertaintyE2EReport,
    config: UncertaintyE2EConfig,
) -> None:
    fragment = {
        "schema_version": 1,
        "strict": True,
        "validation_suite": "bio_inspired_nanochat-eqyk.16",
        "subsystem": "uncertainty_calibration",
        "bead": "bio_inspired_nanochat-eqyk.20",
        "config": asdict(config),
        **report.to_dict(),
    }
    path.write_text(
        json.dumps(fragment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_uncertainty_e2e(
    config: UncertaintyE2EConfig = UncertaintyE2EConfig(),
    *,
    run_dir: str | Path | None = None,
    verbose: bool = True,
) -> UncertaintyE2EReport:
    """Run the real synthetic uncertainty pipeline and assert its evidence contract."""
    config.validate()
    run_id = _run_id(config)
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs") / "e2e" / "uncertainty_calibration" / run_id
    )
    predictions: dict[str, dict[str, _Prediction]] = {}
    prediction_event_count = 0

    with RunLogger(
        output_dir,
        name="uncertainty_calibration",
        run_id=run_id,
        console=False,
        provenance={"bead": "bio_inspired_nanochat-eqyk.20", "config": asdict(config)},
    ) as logger:

        def observe(
            method: str,
            id_prediction: _Prediction,
            ood_prediction: _Prediction,
        ) -> None:
            nonlocal prediction_event_count
            predictions[method] = {"id": id_prediction, "ood": ood_prediction}
            for split, prediction in predictions[method].items():
                probabilities = prediction.probabilities.detach().float().cpu()
                uncertainty = prediction.uncertainty.detach().float().cpu()
                predictive_variance = (
                    prediction.predictive_variance.detach().float().mean(dim=-1).cpu()
                )
                logger.event(
                    "uncertainty_prediction_batch",
                    method=method,
                    split=split,
                    predictive_distribution=probabilities.tolist(),
                    predictive_entropy=uncertainty.tolist(),
                    predictive_variance=predictive_variance.tolist(),
                    nonzero_variance_count=int(
                        prediction.predictive_variance.gt(0.0).sum().item()
                    ),
                )
                prediction_event_count += 1

        experiment_report = run_experiment(
            config.experiment,
            prediction_observer=observe,
        )
        for method, metrics in experiment_report.methods.items():
            logger.event(
                "uncertainty_calibration_method",
                method=method,
                ece=metrics.ece,
                ood_auroc=metrics.ood_auroc,
                id_accuracy=metrics.id_accuracy,
                selective_aurc=metrics.selective_aurc,
                selective_risk_at_80_coverage=metrics.selective_risk_at_80_coverage,
                calibration_curve=[asdict(point) for point in metrics.calibration_curve],
                risk_coverage_curve=[asdict(point) for point in metrics.risk_coverage_curve],
            )

        thermo_metrics = experiment_report.methods["thermo_uq"]
        full_point = thermo_metrics.risk_coverage_curve[-1]
        selected_point = next(
            point
            for point in thermo_metrics.risk_coverage_curve
            if point.coverage >= config.target_coverage
        )
        policy = UncertaintyDecodingConfig(
            enabled=True,
            max_predictive_entropy_nats=selected_point.uncertainty_threshold,
            terminal_action="abstain",
        )
        thermo_uncertainty = predictions["thermo_uq"]["id"].uncertainty
        decode_actions = [
            uncertainty_decode_action(float(value), policy)
            for value in thermo_uncertainty.reshape(-1).tolist()
        ]
        abstained = decode_actions.count("abstain")
        logger.event(
            "uncertainty_decoding_action",
            action="abstain",
            policy="uncertainty_decode_action",
            threshold_nats=policy.max_predictive_entropy_nats,
            target_coverage=config.target_coverage,
            attained_coverage=selected_point.coverage,
            accepted=selected_point.accepted,
            full_accepted=full_point.accepted,
            abstained=abstained,
            selected_errors=selected_point.errors,
            full_errors=full_point.errors,
            selected_risk=selected_point.risk,
            full_risk=full_point.risk,
        )

        live_ft = experiment_report.live_release_ft
        logger.event(
            "uncertainty_thermodynamic_residual",
            integral_ft=live_ft.integral_ft,
            integral_ft_residual=live_ft.integral_ft_residual,
            max_crooks_residual=live_ft.max_crooks_residual,
            predictive_thermo_evidence=asdict(
                experiment_report.predictive_thermo_evidence
            ),
        )

        method_names = set(experiment_report.methods)
        metrics_finite = all(
            math.isfinite(value)
            for metrics in experiment_report.methods.values()
            for value in (
                metrics.ece,
                metrics.ood_auroc,
                metrics.id_accuracy,
                metrics.selective_aurc,
            )
        )
        metrics_bounded = all(
            0.0 <= metrics.ece <= 1.0
            and 0.0 <= metrics.ood_auroc <= 1.0
            and 0.0 <= metrics.id_accuracy <= 1.0
            and 0.0 <= metrics.selective_aurc <= 1.0
            for metrics in experiment_report.methods.values()
        )
        thermo_variance = predictions["thermo_uq"]["id"].predictive_variance
        max_thermo_variance = float(thermo_variance.max().item())
        ft_residuals_finite = math.isfinite(live_ft.integral_ft_residual) and (
            live_ft.max_crooks_residual is not None
            and math.isfinite(live_ft.max_crooks_residual)
        )
        evidence = experiment_report.predictive_thermo_evidence
        invariants = [
            InvariantResult(
                "all_uncertainty_methods_executed",
                method_names == {"softmax_entropy", "mc_dropout", "thermo_uq"},
                sorted(method_names),
                f"methods={sorted(method_names)}",
            ),
            InvariantResult(
                "calibration_metrics_finite_and_bounded",
                metrics_finite and metrics_bounded,
                _method_summary(experiment_report),
                "ECE, OOD AUROC, accuracy, and selective AURC are finite in [0,1]",
            ),
            InvariantResult(
                "predictive_distributions_and_variance_logged",
                prediction_event_count == 6,
                prediction_event_count,
                f"logged {prediction_event_count}/6 method×split prediction batches",
            ),
            InvariantResult(
                "synaptic_mc_variance_is_live",
                math.isfinite(max_thermo_variance) and max_thermo_variance > 0.0,
                max_thermo_variance,
                f"maximum ID predictive variance={max_thermo_variance:.6g}",
            ),
            InvariantResult(
                "uncertainty_decoding_action_fires",
                abstained > 0 and selected_point.accepted < full_point.accepted,
                abstained,
                f"abstained={abstained}, accepted={selected_point.accepted}/"
                f"{full_point.accepted} at coverage={selected_point.coverage:.3f}",
            ),
            InvariantResult(
                "selective_risk_does_not_increase",
                selected_point.risk <= full_point.risk,
                selected_point.risk - full_point.risk,
                f"selected risk={selected_point.risk:.6f}, full risk={full_point.risk:.6f}",
            ),
            InvariantResult(
                "fluctuation_residuals_logged",
                ft_residuals_finite,
                {
                    "integral": live_ft.integral_ft_residual,
                    "crooks": live_ft.max_crooks_residual,
                },
                "live integral-FT and Crooks residuals are finite",
            ),
            InvariantResult(
                "predictive_thermo_evidence_observed",
                evidence.observed_events == evidence.tested_events > 0,
                evidence.observed_events,
                f"observed/tested predictive events={evidence.observed_events}/"
                f"{evidence.tested_events}",
            ),
        ]
        summary = {
            "bead": "bio_inspired_nanochat-eqyk.20",
            "experiment_bead": experiment_report.bead,
            "methods": _method_summary(experiment_report),
            "selective_action": {
                "threshold_nats": policy.max_predictive_entropy_nats,
                "target_coverage": config.target_coverage,
                "attained_coverage": selected_point.coverage,
                "accepted": selected_point.accepted,
                "full_accepted": full_point.accepted,
                "abstained": abstained,
                "selected_errors": selected_point.errors,
                "full_errors": full_point.errors,
            },
            "max_thermo_predictive_variance": max_thermo_variance,
            "ft_residuals": {
                "integral": live_ft.integral_ft_residual,
                "crooks": live_ft.max_crooks_residual,
            },
            "events_path": str(output_dir / "events.jsonl"),
            "validation_fragment": str(output_dir / "validation-summary.json"),
        }
        report = UncertaintyE2EReport(
            run_id=run_id,
            invariants=invariants,
            summary=summary,
        )
        for result in invariants:
            logger.event(
                "uncertainty_invariant",
                name=result.name,
                passed=result.passed,
                observed=result.observed,
                detail=result.detail,
            )
        _write_validation_fragment(
            output_dir / "validation-summary.json",
            report,
            config,
        )

    if verbose:
        _render_report(report)
    return report


def _render_report(report: UncertaintyE2EReport) -> None:
    console = Console()
    table = Table(title=f"Uncertainty/calibration E2E — {report.run_id}")
    table.add_column("Invariant")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    for result in report.invariants:
        table.add_row(
            result.name,
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
            result.detail,
        )
    console.print(table)
    console.print(
        f"[bold]{'PASS' if report.passed else 'FAIL'}[/bold] — "
        f"events: {report.summary['events_path']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uncertainty/calibration E2E + JSONL evidence")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument("--mc-samples", type=int, default=ExperimentConfig.mc_samples)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment = ExperimentConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        mc_samples=args.mc_samples,
    )
    report = run_uncertainty_e2e(
        UncertaintyE2EConfig(
            experiment=experiment,
            target_coverage=args.target_coverage,
        ),
        run_dir=args.run_dir,
        verbose=True,
    )
    report.assert_passed()


if __name__ == "__main__":
    main()
