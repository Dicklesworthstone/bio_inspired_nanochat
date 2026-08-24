r"""Performance-Regression Gate CLI (beads eqyk.15, r2d).

Runs standard performance benchmarks against committed baselines,
verifying that throughput does not regress beyond the configured tolerance.

Usage:
    python -m scripts.perf_regression_gate --mode check --tolerance 0.30
    python -m scripts.perf_regression_gate --mode update --tolerance 0.25
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.perf_regression import (
    DEFAULT_BASELINES_PATH,
    PerfRegressionHarness,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run performance regression gates")
    parser.add_argument(
        "--mode",
        choices=["check", "update"],
        default="check",
        help="Whether to check against baselines or update baseline JSON file",
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

    console = Console()
    harness = PerfRegressionHarness(baselines_path=args.baselines)

    console.print("[bold cyan]Running Performance Regression Benchmarks...[/bold cyan]")
    results = harness.run_all()

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

    if args.record_registry:
        harness.record_to_registry(results)

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
