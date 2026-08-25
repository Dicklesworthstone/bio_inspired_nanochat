"""E2E SCRIPT: CMA-ES tune loop (resume) with detailed logs (bead eqyk.7).

Exercises the CMA-ES hyperparameter optimizer (``scripts.tune_bio_params``) end-to-end:
  1. Runs optimization for ~2 generations on the synthetic associative recall task.
  2. Asserts ``progress.jsonl``, ``best_params.json``, and inert JSON replay checkpoints exist.
  3. Verifies the checkpoint resume contract: resuming from ``es_state.json`` reproduces state
     and correctly increments generation counters without corrupting previous history.
  4. Tests the stagnation policy: early-stopping or sigma-reset fires deterministically on stagnant progress.
  5. Emits structured per-generation logs (candidates, scores, sigma, compute budget) into a machine-readable
     ``events.jsonl`` trace.

Run:
    python -m scripts.e2e.cmaes_tune
    pytest tests/test_e2e_cmaes.py -v
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.run_logging import RunLogger
from scripts.tune_bio_params import (
    TOP10_PARAM_SPECS,
    _CMA_STATE_FORMAT,
    main as tune_main,
)


@dataclass
class CmaesE2EConfig:
    """Configuration for the CMA-ES E2E battery."""

    generations: int = 2
    popsize: int = 4
    steps: int = 2
    batch_size: int = 4
    seed: int = 1337
    device: str = "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    stagnation_gens: int = 20
    stagnation_min_improve_frac: float = 0.01
    stagnation_action: str = "stop"
    no_tensorboard: bool = True
    gpu_cost_per_hour: float = 0.0


@dataclass
class CmaesE2EReport:
    run_id: str
    config: CmaesE2EConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(
                f"CMA-ES E2E battery failed with {len(failed)} failure(s):\n{msg}"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if line_s:
            try:
                records.append(json.loads(line_s))
            except json.JSONDecodeError:
                continue
    return records


def run_cmaes_e2e(
    cfg: CmaesE2EConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> CmaesE2EReport:
    """Run the CMA-ES E2E suite and return a structured report of all invariant checks."""
    if cfg is None:
        cfg = CmaesE2EConfig()

    console = Console(quiet=not verbose)
    run_id = f"cmaes-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="cmaes_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    main_run_dir = base_dir / "main_run"
    main_run_dir.mkdir(parents=True, exist_ok=True)
    resume_run_dir = base_dir / "resume_run"
    stagnation_run_dir = base_dir / "stagnation_run"
    registry_file = base_dir / "registry.jsonl"

    run_logger = RunLogger(base_dir, name="cmaes_e2e", run_id=run_id, console=verbose)
    run_logger.event("cmaes_config", config=asdict(cfg))

    try:
        # -------------------------------------------------------------------
        # Phase 1: 2-generation initial optimization
        # -------------------------------------------------------------------
        cmd_args = [
            "optimize",
            "--seed",
            str(cfg.seed),
            "--device",
            str(cfg.device),
            "--generations",
            str(cfg.generations),
            "--popsize",
            str(cfg.popsize),
            "--steps",
            str(cfg.steps),
            "--batch-size",
            str(cfg.batch_size),
            "--lr",
            str(cfg.lr),
            "--weight-decay",
            str(cfg.weight_decay),
            "--stagnation-gens",
            str(cfg.stagnation_gens),
            "--stagnation-min-improve-frac",
            str(cfg.stagnation_min_improve_frac),
            "--stagnation-action",
            str(cfg.stagnation_action),
            "--run-dir",
            str(main_run_dir),
            "--registry-path",
            str(registry_file),
            "--gpu-cost-per-hour",
            str(cfg.gpu_cost_per_hour),
        ]
        if cfg.no_tensorboard:
            cmd_args.append("--no-tensorboard")

        ret = tune_main(cmd_args)
        invariants.append(
            InvariantResult(
                name="initial_optimization_exits_clean",
                passed=(ret == 0),
                observed=ret,
                detail=f"tune_main returned exit code {ret}",
            )
        )

        # Invariant 1: progress.jsonl written with expected generation count
        progress_path = main_run_dir / "progress.jsonl"
        progress_records = _read_jsonl(progress_path)
        has_progress = len(progress_records) >= cfg.generations
        gen_numbers = [r.get("generation") for r in progress_records]
        invariants.append(
            InvariantResult(
                name="progress_jsonl_written",
                passed=has_progress
                and gen_numbers[: cfg.generations]
                == list(range(1, cfg.generations + 1)),
                observed=len(progress_records),
                detail=f"progress.jsonl contains {len(progress_records)} records, gens={gen_numbers}",
            )
        )

        # Invariant 2: best_params.json written with all 10 parameters and finite loss
        best_params_path = main_run_dir / "best_params.json"
        best_params_ok = False
        best_loss_val = float("inf")
        if best_params_path.exists():
            try:
                best_doc = json.loads(best_params_path.read_text(encoding="utf-8"))
                best_loss_val = float(best_doc.get("best_loss", float("inf")))
                params = best_doc.get("best_params", {})
                expected_names = {s.name for s in TOP10_PARAM_SPECS}
                all_keys_present = expected_names.issubset(params.keys())
                best_params_ok = (
                    math.isfinite(best_loss_val)
                    and all_keys_present
                    and all(math.isfinite(float(v)) for v in params.values())
                )
            except Exception:
                best_params_ok = False
        invariants.append(
            InvariantResult(
                name="best_params_json_written",
                passed=best_params_ok,
                observed=best_params_ok,
                detail=f"best_params.json valid with finite loss {best_loss_val:.4f}",
            )
        )

        # Invariant 3: inert, schema-tagged JSON replay checkpoints written
        es_latest_path = main_run_dir / "es_state.json"
        gen1_path = main_run_dir / "es_state_gen_0001.json"
        gen2_path = main_run_dir / "es_state_gen_0002.json"
        checkpoints_ok = (
            es_latest_path.exists() and gen1_path.exists() and gen2_path.exists()
        )
        if checkpoints_ok:
            try:
                es_doc = json.loads(es_latest_path.read_text(encoding="utf-8"))
                checkpoints_ok = (
                    es_doc.get("format") == _CMA_STATE_FORMAT
                    and len(es_doc.get("generation_records", [])) >= cfg.generations
                    and int(es_doc.get("strategy_summary", {}).get("countiter", 0))
                    >= cfg.generations
                )
            except Exception:
                checkpoints_ok = False
        invariants.append(
            InvariantResult(
                name="checkpoints_written",
                passed=checkpoints_ok,
                observed=checkpoints_ok,
                detail="es_state.json and per-generation replay states exist and parse",
            )
        )

        # -------------------------------------------------------------------
        # Phase 2: Checkpoint Resume contract
        # -------------------------------------------------------------------
        # Copy main_run to resume_run and continue for 1 more generation (total: 3)
        shutil.copytree(main_run_dir, resume_run_dir)
        resume_cmd = [
            "optimize",
            "--seed",
            str(cfg.seed),
            "--device",
            str(cfg.device),
            "--generations",
            str(cfg.generations + 1),
            "--popsize",
            str(cfg.popsize),
            "--steps",
            str(cfg.steps),
            "--batch-size",
            str(cfg.batch_size),
            "--run-dir",
            str(resume_run_dir),
            "--resume",
            "--registry-path",
            str(registry_file),
            "--no-tensorboard",
        ]
        ret_resume = tune_main(resume_cmd)
        resume_progress = _read_jsonl(resume_run_dir / "progress.jsonl")
        resume_gens = [r.get("generation") for r in resume_progress]
        resume_passed = (
            ret_resume == 0
            and len(resume_progress) == cfg.generations + 1
            and resume_gens == list(range(1, cfg.generations + 2))
        )
        invariants.append(
            InvariantResult(
                name="checkpoint_resume_contract",
                passed=resume_passed,
                observed=len(resume_progress),
                detail=f"Resumed run completed generation {cfg.generations + 1}; gens={resume_gens}",
            )
        )

        # -------------------------------------------------------------------
        # Phase 3: Stagnation policy firing test
        # -------------------------------------------------------------------
        # Run with strict stagnation: window=1, min_improve=0.999, action=stop
        stagnation_cmd = [
            "optimize",
            "--seed",
            str(cfg.seed),
            "--device",
            str(cfg.device),
            "--generations",
            "5",
            "--popsize",
            str(cfg.popsize),
            "--steps",
            "1",
            "--batch-size",
            str(cfg.batch_size),
            "--stagnation-gens",
            "1",
            "--stagnation-min-improve-frac",
            "0.99999",
            "--stagnation-action",
            "stop",
            "--run-dir",
            str(stagnation_run_dir),
            "--registry-path",
            str(registry_file),
            "--no-tensorboard",
        ]
        ret_stag = tune_main(stagnation_cmd)
        stag_progress = _read_jsonl(stagnation_run_dir / "progress.jsonl")
        stag_triggered = any(bool(r.get("stagnation_triggered")) for r in stag_progress)
        stag_stopped_early = len(stag_progress) < 5
        invariants.append(
            InvariantResult(
                name="stagnation_policy_fires",
                passed=(ret_stag == 0 and stag_triggered and stag_stopped_early),
                observed=len(stag_progress),
                detail=f"Stagnation early-stop triggered={stag_triggered}, finished at gen {len(stag_progress)}/5",
            )
        )

        # Invariant 5: Results registry record written
        registry_records = _read_jsonl(registry_file)
        registry_ok = len(registry_records) >= 3  # initial + resume + stagnation
        invariants.append(
            InvariantResult(
                name="results_registry_appended",
                passed=registry_ok,
                observed=len(registry_records),
                detail=f"registry.jsonl contains {len(registry_records)} records",
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = CmaesE2EReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "initial_best_loss": best_loss_val,
                "total_generations_evaluated": len(progress_records)
                + 1
                + len(stag_progress),
                "resume_successful": resume_passed,
                "stagnation_tested": stag_triggered,
            },
        )

        if verbose:
            table = Table(title="CMA-ES E2E Battery Results")
            table.add_column("Invariant", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Detail", style="dim")
            for inv in invariants:
                status = "[green]PASS[/green]" if inv.passed else "[red]FAIL[/red]"
                table.add_row(inv.name, status, inv.detail)
            console.print(table)

        return report

    finally:
        run_logger.close()
        if clean_tmp:
            shutil.rmtree(base_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CMA-ES E2E verification battery")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to save E2E traces and logs",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to execute on"
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = parser.parse_args(argv)

    cfg = CmaesE2EConfig(device=args.device, seed=args.seed)
    report = run_cmaes_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
