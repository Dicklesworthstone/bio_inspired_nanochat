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
                0 < evidence.tested_events <= evidence.observed_events
                and evidence.tested_events + evidence.degenerate_events
                == evidence.observed_events,
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


def run_checkpoint_abstention(tag: str, step: int) -> dict:
    """Evaluate a fixed threshold policy on a fully enumerated finite distribution."""
    import pyarrow.parquet as pq
    import torch

    from bio_inspired_nanochat.checkpoint_manager import load_model
    from bio_inspired_nanochat.common import get_base_dir
    from bio_inspired_nanochat.conformal_abstention import (
        _THRESHOLDS,
        _upper_risk,
        ConformalAbstentionEngine,
    )
    from bio_inspired_nanochat.dataset import parquet_paths_for_split
    from bio_inspired_nanochat.matrix_manifest import digest, file_digest

    base = Path(get_base_dir())
    model, tokenizer, meta = load_model("base", torch.device("cpu"), "eval", model_tag=tag, step=step)
    if model.training or meta["step"] != step or meta.get("synapses"):
        raise ValueError("expected an eval-mode vanilla checkpoint at the requested step")
    validation = parquet_paths_for_split("val", data_dir=str(base / "base_data"))
    training = parquet_paths_for_split("train", data_dir=str(base / "base_data"))
    if len(validation) != 1 or set(validation).intersection(training):
        raise ValueError("expected a separate native validation shard")
    val_path = Path(validation[0])
    inputs: list[list[int]] = []
    targets: list[int] = []
    document_indices: list[int] = []
    seen: set[tuple[int, ...]] = set()
    examined = 0
    for batch in pq.ParquetFile(val_path).iter_batches(batch_size=64, columns=["text"]):
        for text in batch.column(0).to_pylist():
            examined += 1
            ids = tokenizer.encode(text)
            prefix = tuple(ids[:16])
            if len(ids) < 17 or prefix in seen:
                continue
            seen.add(prefix)
            inputs.append(list(prefix))
            targets.append(ids[16])
            document_indices.append(examined - 1)
            if len(inputs) == 384:
                break
        if len(inputs) == 384:
            break
    if len(inputs) != 384:
        raise ValueError(f"validation shard supplied {len(inputs)} distinct eligible documents, need 384")
    probs_parts = []
    with torch.inference_mode():
        for batch in torch.tensor(inputs, dtype=torch.long).split(8):
            logits = model(batch)[:, -1, :]
            probs_parts.append(logits.float().softmax(dim=-1))
    probs = torch.cat(probs_parts)
    labels = torch.tensor(targets, dtype=torch.long)
    draws = torch.randint(256, (512,), generator=torch.Generator().manual_seed(1337))
    engine = ConformalAbstentionEngine(target_alpha=0.10, delta=0.05)
    threshold = engine.calibrate(probs[draws], labels[draws])
    decisions = [engine.predict(row) for row in probs]
    keep = torch.tensor([not d.abstained for d in decisions])
    scores = torch.tensor([d.non_conformity_score for d in decisions], dtype=torch.float64)
    predicted = probs.argmax(dim=1)
    errors = predicted != labels

    def counts(indices):
        n = int(indices.numel())
        answered = int(keep[indices].sum())
        wrong = int((errors[indices] & keep[indices]).sum())
        return {"queries": n, "answered": answered, "errors_among_answered": wrong,
                "abstained": n - answered, "coverage": answered / n,
                "answered_error_rate": wrong / answered if answered else None,
                "unconditional_top1_errors": int(errors[indices].sum())}

    # Expose the nearest attempted risk bound even when the chosen policy abstains on all inputs.
    tail = math.log(len(_THRESHOLDS) / engine.delta)
    candidates = []
    for cutoff in _THRESHOLDS:
        accepted = scores[draws] <= cutoff
        n = int(accepted.sum())
        if n:
            wrong = int(errors[draws][accepted].sum())
            candidates.append({"cutoff": cutoff, "answered": n, "errors": wrong,
                               "upper_bound": _upper_risk(wrong, n, tail)})
    eligible = [c for c in candidates if c["upper_bound"] <= engine.target_alpha]
    if threshold != max((c["cutoff"] for c in eligible), default=-1.0):
        raise AssertionError("enumerated calibration grid disagrees with deployed threshold")
    checkpoint = base / "base_checkpoints" / tag
    encoding = base / "tokenizer" / "tokenizer.json"
    if not encoding.is_file():
        encoding = base / "tokenizer" / "tokenizer.pkl"
    files = [checkpoint / f"model_{step:06d}.pt", checkpoint / f"meta_{step:06d}.json",
             encoding, base / "tokenizer" / "token_bytes.pt", val_path]
    return {
        "study": "fixed next-token abstention on a uniform finite population",
        "protocol": {
            "document_order": "native validation shard, parquet row order, one query per document",
            "query": "first 16 encoded text tokens, no inserted BOS, predict the 17th token",
            "eligibility": "at least 17 tokens and a prefix not used in any earlier accepted document",
            "population": "first 256 eligible distinct prefixes, equiprobable",
            "calibration": "512 uniform draws with replacement from that population",
            "calibration_rng": "torch.randint with CPU torch.Generator, seed 1337",
            "separate_description": "next 128 distinct eligible prefixes in shard order",
            "alpha": engine.target_alpha, "delta": engine.delta,
            "score": "one minus maximum predicted token probability",
            "threshold_grid": "0.00 through 1.00 inclusive in increments of 0.01",
        },
        "checkpoint": {"tag": tag, "step": step, "loader": "load_model(base, cpu, eval)",
                       "model_class": type(model).__name__, "model_config": meta["model_config"],
                       "training_provenance": meta.get("provenance"), "torch_version": str(torch.__version__)},
        "file_sha256": {str(p.relative_to(base)): file_digest(p) for p in files},
        "source_sha256": {name: file_digest(Path(__file__).resolve().parents[2] / name) for name in (
            "bio_inspired_nanochat/conformal_abstention.py", "bio_inspired_nanochat/checkpoint_manager.py",
            "bio_inspired_nanochat/gpt.py", "scripts/e2e/uncertainty_calibration.py")},
        "sample_identity": {
            "documents_examined": examined, "population_first_last_doc_indices": document_indices[:256:255],
            "descriptive_first_last_doc_indices": [document_indices[256], document_indices[-1]],
            "ordered_document_prefix_target_sha256": digest(list(zip(document_indices, inputs, targets, strict=True))),
            "calibration_draws_sha256": digest(draws.tolist()), "calibration_distinct_queries": int(draws.unique().numel()),
        },
        "selected_policy": {"threshold": threshold, "population_risk_upper_bound": engine.answered_risk_upper_bound,
                            "candidates_with_answers": len(candidates), "admissible_candidates": len(eligible),
                            "lowest_candidate_upper_bound": min(candidates, key=lambda c: c["upper_bound"])},
        "calibration_sample": counts(draws), "finite_population_census": counts(torch.arange(256)),
        "separate_descriptive_sample": counts(torch.arange(256, 384)),
        "scope": "Calibration risk bound applies to fresh uniform draws from the first 256 queries. "
                 "The next 128 queries are descriptive. Conditional error is undefined for zero answers.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uncertainty/calibration E2E + JSONL evidence")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig.train_steps)
    parser.add_argument("--mc-samples", type=int, default=ExperimentConfig.mc_samples)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--abstention-checkpoint-tag", help="Run the fixed trained-checkpoint abstention study")
    parser.add_argument("--checkpoint-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.abstention_checkpoint_tag or args.checkpoint_step is not None:
        if not args.abstention_checkpoint_tag or args.checkpoint_step is None or not args.run_dir:
            raise SystemExit("checkpoint study requires --abstention-checkpoint-tag, --checkpoint-step and --run-dir")
        result = run_checkpoint_abstention(args.abstention_checkpoint_tag, args.checkpoint_step)
        out = Path(args.run_dir) / "checkpoint_abstention.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        Console().print(f"Wrote checkpoint abstention result to {out}", markup=False)
        return
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
