"""CLI tool to summarize GPU-hours and financial budget across runs (bead 2a7)."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.budgeting import DEFAULT_BUDGET_LOG_PATH, load_budget_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize compute budget accounting.")
    parser.add_argument("--log-path", type=str, default=DEFAULT_BUDGET_LOG_PATH, help="Path to budget JSONL log")
    args = parser.parse_args()

    console = Console()
    entries = load_budget_entries(args.log_path)
    if not entries:
        console.print(f"[yellow]No budget entries found at {args.log_path}[/yellow]")
        return 0

    table = Table(title=f"Compute Budget Accounting ({len(entries)} runs)")
    table.add_column("Run ID", style="cyan")
    table.add_column("Purpose", style="magenta")
    table.add_column("GPUs", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("GPU Hours", justify="right", style="green")
    table.add_column("Est. Cost ($)", justify="right", style="bold green")
    table.add_column("Objective", style="yellow")

    total_gpu_hours = 0.0
    total_cost_usd = 0.0

    for entry in entries:
        total_gpu_hours += entry.gpu_hours
        total_cost_usd += entry.estimated_cost_usd
        table.add_row(
            entry.run_id,
            entry.purpose,
            str(entry.num_gpus),
            f"{entry.duration_seconds:.1f}",
            f"{entry.gpu_hours:.4f}",
            f"${entry.estimated_cost_usd:.2f}",
            entry.objective_type,
        )

    console.print(table)
    console.print(f"\n[bold]Total GPU Hours:[/bold] {total_gpu_hours:.4f} hrs")
    console.print(f"[bold]Total Estimated Cost:[/bold] ${total_cost_usd:.2f} USD\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
