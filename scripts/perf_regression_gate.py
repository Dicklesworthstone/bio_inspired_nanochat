r"""Performance-Regression Gate CLI (beads eqyk.15, r2d).

Runs standard performance benchmarks against committed baselines,
verifying that throughput does not regress beyond the configured tolerance.

Usage:
    python -m scripts.perf_regression_gate --mode check --tolerance 0.30
    python -m scripts.perf_regression_gate --mode update --tolerance 0.25
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.perf_regression import (
    DEFAULT_BASELINES_PATH,
    DEFAULT_THROUGHPUT_BUDGET,
    STANDARD_BENCHMARK_CONFIGS,
    PerfRegressionHarness,
    ThroughputBudget,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run performance regression gates")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="Run every incumbent and candidate on this device")
    parser.add_argument("--observations-json", type=Path,
                        help="Save actual per-arm measurements and relative budget comparisons")
    parser.add_argument(
        "--mode",
        choices=["check", "update", "profile"],
        default="check",
        help="Check or update baselines, or annotate CUDA measurement windows for Nsight Systems",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=str(DEFAULT_BASELINES_PATH),
        help="Path to committed baselines JSON file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.30,
        help="Allowable performance degradation ratio before failing gate (default 0.30)",
    )
    parser.add_argument(
        "--check-budget",
        action="store_true",
        help="Also enforce the relative synaptic-vs-standard ThroughputBudget in the same run",
    )
    parser.add_argument(
        "--train-ratio-max",
        type=float,
        default=DEFAULT_THROUGHPUT_BUDGET.train_ratio_max,
        help="Maximum allowable standard/synaptic training throughput ratio (default 2.0)",
    )
    parser.add_argument(
        "--decode-ratio-max",
        type=float,
        default=DEFAULT_THROUGHPUT_BUDGET.decode_ratio_max,
        help="Maximum allowable standard/synaptic decode latency ratio (default 1.5)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output gate comparison JSON",
    )
    parser.add_argument(
        "--record-registry",
        action="store_true",
        help="Append benchmark results to experiment registry",
    )
    args = parser.parse_args(argv)
    if (args.mode == "update" and args.device == "cuda"
            and Path(args.baselines).resolve() == Path(DEFAULT_BASELINES_PATH).resolve()):
        parser.error("CUDA baseline updates require a separate --baselines file for the measured GPU")

    console = Console()
    harness = PerfRegressionHarness(baselines_path=args.baselines)

    if args.mode == "profile":
        if args.device != "cuda" or args.check_budget or args.record_registry or args.output_json or args.observations_json:
            parser.error("profile requires --device cuda and writes the profiler trace only")
        for cfg in STANDARD_BENCHMARK_CONFIGS:
            console.print(f"Profiling {cfg.name}")
            harness.run_benchmark(replace(cfg, device="cuda"), nvtx=True)
        return 0

    console.print("[bold cyan]Running Performance Regression Benchmarks...[/bold cyan]")
    if args.device is None:
        results = harness.run_all()
    else:
        results = harness.run_all([replace(cfg, device=args.device) for cfg in STANDARD_BENCHMARK_CONFIGS])

    if args.mode == "update":
        harness.save_baselines(results, tolerance=args.tolerance)
        console.print(f"[bold green]Updated baseline numbers committed to {args.baselines}[/bold green]")
        return 0

    # Mode: check
    comparisons = harness.evaluate_gates(results, override_tolerance=args.tolerance)

    table = Table(title="Performance Regression Gate Results")
    table.add_column("Benchmark Config", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Observed (tok/s)", justify="right")
    table.add_column("Baseline (tok/s)", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Status", style="bold")

    all_passed = True
    for comp in comparisons:
        status_str = "[green]PASS[/green]" if comp.passed else "[red]FAIL[/red]"
        if not comp.passed:
            all_passed = False
        table.add_row(
            comp.name,
            comp.mode,
            f"{comp.observed_tok_per_sec:.1f}",
            f"{comp.baseline_tok_per_sec:.1f}",
            f"{comp.speed_ratio:.1%}",
            status_str,
        )

    console.print(table)

    budget_comparisons = []
    if args.check_budget:
        budget = ThroughputBudget(
            train_ratio_max=args.train_ratio_max,
            decode_ratio_max=args.decode_ratio_max,
        )
        budget_comparisons = harness.evaluate_budget_gates(results, budget=budget)
        budget_table = Table(title="Relative Throughput Budget Gate Results")
        budget_table.add_column("Mode", style="magenta")
        budget_table.add_column("Standard (tok/s)", justify="right")
        budget_table.add_column("Synaptic (tok/s)", justify="right")
        budget_table.add_column("Slowdown", justify="right")
        budget_table.add_column("Budget Max", justify="right")
        budget_table.add_column("Status", style="bold")
        if not budget_comparisons:
            all_passed = False
        for bcomp in budget_comparisons:
            status_str = "[green]PASS[/green]" if bcomp.passed else "[red]FAIL[/red]"
            if not bcomp.passed:
                all_passed = False
            budget_table.add_row(
                bcomp.mode,
                f"{bcomp.standard_tok_per_sec:.1f}",
                f"{bcomp.synaptic_tok_per_sec:.1f}",
                f"{bcomp.slowdown_ratio:.2f}x",
                f"<= {bcomp.max_ratio:.2f}x",
                status_str,
            )
        console.print(budget_table)

    if args.record_registry:
        harness.record_to_registry(results)

    if args.observations_json:
        args.observations_json.parent.mkdir(parents=True, exist_ok=True)
        args.observations_json.write_text(json.dumps({
            "results": [asdict(result) for result in results],
            "relative_budget": [asdict(comparison) for comparison in budget_comparisons],
        }, indent=2) + "\n", encoding="utf-8")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name": c.name,
                "mode": c.mode,
                "observed_tok_per_sec": c.observed_tok_per_sec,
                "baseline_tok_per_sec": c.baseline_tok_per_sec,
                "speed_ratio": c.speed_ratio,
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in comparisons
        ]
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not all_passed:
        console.print("[bold red]Performance regression gate failed![/bold red]")
        return 1

    console.print("[bold green]All performance regression gates passed![/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
